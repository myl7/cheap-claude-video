#!/usr/bin/env python3
"""Transcribe a video via Doubao (Volcano Engine) streaming ASR 2.0.

Strategy: extract audio (mono 16kHz PCM s16le), open a WebSocket to the Doubao
streaming ASR endpoint, stream the audio in ~200ms packets, and collect the
utterances the server returns. Returns segments in the same shape as
transcribe.parse_vtt ({start, end, text}) so the rest of the pipeline
(filter_range, format_transcript) doesn't care where the transcript came from.

Why streaming and not the cheaper file-recognition API: file recognition only
accepts a public audio URL, and /watch's audio is a local temp file. Streaming
accepts raw bytes, so it needs no object storage. Cost is ~1 CNY/hour.

Pure stdlib — no `pip install websocket-client`. The WebSocket handshake and
framing (RFC 6455) plus Volcano's binary sub-protocol are implemented by hand
to keep the skill dependency-free.

Docs: https://www.volcengine.com/docs/6561/1354869
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import select
import shutil
import socket
import ssl
import struct
import subprocess
import sys
import uuid
from pathlib import Path


HOST = "openspeech.bytedance.com"
PORT = 443
# Streaming-input mode (higher accuracy, batches results) — best fit for an
# offline file where we care about accuracy over first-word latency.
WS_PATH = "/api/v3/sauc/bigmodel_nostream"

# Doubao streaming ASR 2.0 resource id (小时版 / pay-by-duration). The 1.0 model
# is volc.bigasr.sauc.duration; 2.0 (seedasr) is the newer, cheaper tier.
DEFAULT_RESOURCE_ID = "volc.seedasr.sauc.duration"

BACKEND_LABEL = "doubao (streaming 2.0)"

# 16kHz mono s16le → 32000 bytes/sec. 200ms per packet = 6400 bytes, the size
# the docs recommend for streaming packets.
PCM_RATE = 16000
PCM_BYTES_PER_SEC = PCM_RATE * 2  # 16-bit mono
PACKET_MS = 200
PACKET_BYTES = PCM_BYTES_PER_SEC * PACKET_MS // 1000

# Compress payloads with gzip. Every official demo does; the server echoes the
# client's compression choice. Flip to False only for protocol debugging.
USE_GZIP = True

# Volcano binary-protocol message types (4-bit).
_MSG_FULL_CLIENT = 0b0001   # client: request params (JSON)
_MSG_AUDIO_ONLY = 0b0010    # client: raw audio bytes
_MSG_FULL_SERVER = 0b1001   # server: recognition result (JSON)
_MSG_ERROR = 0b1111         # server: protocol/processing error

# message-type-specific flags (4-bit).
_FLAG_NONE = 0b0000
_FLAG_LAST_NO_SEQ = 0b0010  # last (negative) packet, no sequence number
_FLAG_LAST_RESULT = 0b0011  # server: final result for last packet

_SERIAL_NONE = 0b0000
_SERIAL_JSON = 0b0001
_COMPRESS_NONE = 0b0000
_COMPRESS_GZIP = 0b0001


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def _from_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def _from_dotenv(path: Path, name: str) -> str | None:
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() != name:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                value = value[1:-1]
            return value or None
    except OSError:
        return None
    return None


def _lookup(name: str) -> str | None:
    """Env var first, then ~/.config/watch/.env, then ./.env."""
    value = _from_env(name)
    if value:
        return value
    for candidate in (Path.home() / ".config" / "watch" / ".env", Path.cwd() / ".env"):
        value = _from_dotenv(candidate, name)
        if value:
            return value
    return None


def load_credentials() -> dict | None:
    """Return auth headers for the Doubao WebSocket, or None if not configured.

    Two credential shapes are supported:
      - New console: a single API key (DOUBAO_ASR_API_KEY) → X-Api-Key.
      - DOUBAO_ASR_ACCESS_TOKEN with an APP ID uses the old
        X-Api-App-Key + X-Api-Access-Key pair. Without an APP ID, the token is
        treated as a new-console key and sent as X-Api-Key.
    """
    resource_id = _lookup("DOUBAO_ASR_RESOURCE_ID") or DEFAULT_RESOURCE_ID

    api_key = _lookup("DOUBAO_ASR_API_KEY")
    if api_key:
        return {
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": resource_id,
        }

    app_id = _lookup("DOUBAO_ASR_APP_ID") or _lookup("DOUBAO_ASR_APP_KEY")
    access_token = _lookup("DOUBAO_ASR_ACCESS_TOKEN") or _lookup("DOUBAO_ASR_ACCESS_KEY")
    if access_token:
        if app_id:
            return {
                "X-Api-App-Key": app_id,
                "X-Api-Access-Key": access_token,
                "X-Api-Resource-Id": resource_id,
            }
        return {
            "X-Api-Key": access_token,
            "X-Api-Resource-Id": resource_id,
        }
    return None


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
def extract_pcm(
    video_path: str,
    out_path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> Path:
    """Extract mono 16kHz signed-16-bit little-endian PCM (headerless).

    Raw PCM lets us slice the stream into exact-duration packets: every byte
    offset is a valid sample boundary, unlike mp3 frames.

    When start/end are given, only that window is extracted so the ASR only
    processes (and bills for) the focus range. Timestamps in the returned
    transcript are then relative to the window; the caller shifts them back to
    absolute time.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    start = start_seconds if (start_seconds and start_seconds > 0) else 0.0
    if start:
        cmd += ["-ss", f"{start:.3f}"]  # input seeking (fast)
    cmd += ["-i", str(Path(video_path).resolve())]
    if end_seconds is not None and end_seconds > start:
        cmd += ["-t", f"{end_seconds - start:.3f}"]  # duration, unambiguous
    cmd += [
        "-vn",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", str(PCM_RATE),
        "-ac", "1",
        str(out_path.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg PCM extraction failed: {result.stderr.strip()}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit("ffmpeg produced no audio — video may have no audio track")
    return out_path


# --------------------------------------------------------------------------- #
# Volcano binary sub-protocol
# --------------------------------------------------------------------------- #
def _header(msg_type: int, flags: int, serialization: int, compression: int) -> bytes:
    # byte0: protocol version 1 (0b0001) << 4 | header size 1 (=4 bytes)
    return bytes([
        (0b0001 << 4) | 0b0001,
        (msg_type << 4) | flags,
        (serialization << 4) | compression,
        0x00,
    ])


def _compress(payload: bytes) -> tuple[bytes, int]:
    if USE_GZIP:
        return gzip.compress(payload), _COMPRESS_GZIP
    return payload, _COMPRESS_NONE


def build_full_client_request(params: dict) -> bytes:
    payload, comp = _compress(json.dumps(params, ensure_ascii=False).encode("utf-8"))
    return (
        _header(_MSG_FULL_CLIENT, _FLAG_NONE, _SERIAL_JSON, comp)
        + struct.pack(">I", len(payload))
        + payload
    )


def build_audio_request(chunk: bytes, last: bool) -> bytes:
    payload, comp = _compress(chunk)
    flags = _FLAG_LAST_NO_SEQ if last else _FLAG_NONE
    return (
        _header(_MSG_AUDIO_ONLY, flags, _SERIAL_NONE, comp)
        + struct.pack(">I", len(payload))
        + payload
    )


def parse_server_message(data: bytes) -> dict:
    """Parse a Volcano binary server message into a dict.

    Returns {type, flags, seq?, payload?(dict), code?, error?}.
    """
    header_size = (data[0] & 0x0F) * 4
    msg_type = (data[1] >> 4) & 0x0F
    flags = data[1] & 0x0F
    compression = data[2] & 0x0F
    serialization = (data[2] >> 4) & 0x0F
    rest = data[header_size:]

    out: dict = {"type": msg_type, "flags": flags}

    if msg_type == _MSG_FULL_SERVER:
        out["seq"] = struct.unpack(">i", rest[:4])[0]
        size = struct.unpack(">I", rest[4:8])[0]
        body = rest[8:8 + size]
        if compression == _COMPRESS_GZIP and body:
            body = gzip.decompress(body)
        if serialization == _SERIAL_JSON:
            out["payload"] = json.loads(body.decode("utf-8")) if body else {}
        else:
            out["payload"] = body
    elif msg_type == _MSG_ERROR:
        out["code"] = struct.unpack(">I", rest[:4])[0]
        size = struct.unpack(">I", rest[4:8])[0]
        body = rest[8:8 + size]
        if compression == _COMPRESS_GZIP and body:
            try:
                body = gzip.decompress(body)
            except OSError:
                pass
        out["error"] = body.decode("utf-8", errors="replace")
    return out


# --------------------------------------------------------------------------- #
# Minimal WebSocket client (RFC 6455) over TLS
# --------------------------------------------------------------------------- #
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class _WSConn:
    """A tiny client-side WebSocket connection: TLS + framing only."""

    def __init__(self, sock: ssl.SSLSocket, leftover: bytes):
        self._sock = sock
        self._buf = leftover

    # -- framing --------------------------------------------------------- #
    def send_binary(self, data: bytes) -> None:
        self._send_frame(0x2, data)

    def _send_frame(self, opcode: int, data: bytes) -> None:
        length = len(data)
        frame = bytes([0x80 | opcode])  # FIN + opcode
        if length < 126:
            frame += bytes([0x80 | length])
        elif length < 65536:
            frame += bytes([0x80 | 126]) + struct.pack(">H", length)
        else:
            frame += bytes([0x80 | 127]) + struct.pack(">Q", length)
        mask = os.urandom(4)
        frame += mask
        frame += bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        self._sock.sendall(frame)

    def _read_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("websocket closed by server")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def recv_message(self) -> tuple[int, bytes]:
        """Return (opcode, payload) for one complete message.

        Handles fragmentation and answers ping/close automatically. opcode 0x8
        means the server closed the connection.
        """
        frags = b""
        first_opcode: int | None = None
        while True:
            b0 = self._read_exact(1)[0]
            fin = b0 & 0x80
            opcode = b0 & 0x0F
            b1 = self._read_exact(1)[0]
            masked = b1 & 0x80
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._read_exact(8))[0]
            mask = self._read_exact(4) if masked else b""
            payload = self._read_exact(length) if length else b""
            if masked and payload:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

            if opcode == 0x8:  # close
                return 0x8, payload
            if opcode == 0x9:  # ping → pong
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:  # pong
                continue
            if opcode == 0x0:  # continuation
                frags += payload
            else:
                first_opcode = opcode
                frags = payload
            if fin:
                return first_opcode if first_opcode is not None else opcode, frags

    def readable(self, timeout: float) -> bool:
        if self._buf:
            return True
        r, _, _ = select.select([self._sock], [], [], timeout)
        return bool(r)

    def settimeout(self, seconds: float | None) -> None:
        self._sock.settimeout(seconds)

    def close(self) -> None:
        try:
            self._send_frame(0x8, b"")
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass


def _ws_connect(headers: dict, connect_timeout: float = 15.0) -> tuple[_WSConn, str]:
    """Open a TLS WebSocket to HOST/WS_PATH with the given extra headers.

    Returns (conn, logid). Raises SystemExit on handshake failure.
    """
    key = base64.b64encode(os.urandom(16)).decode()
    lines = [
        f"GET {WS_PATH} HTTP/1.1",
        f"Host: {HOST}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
        f"X-Api-Connect-Id: {uuid.uuid4()}",
        f"X-Api-Request-Id: {uuid.uuid4()}",
    ]
    for name, value in headers.items():
        lines.append(f"{name}: {value}")
    request = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii")

    ctx = ssl.create_default_context()
    raw = socket.create_connection((HOST, PORT), timeout=connect_timeout)
    sock = ctx.wrap_socket(raw, server_hostname=HOST)

    sock.sendall(request)

    # Read the handshake response headers (up to the blank line).
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise SystemExit("Doubao ASR: connection closed during WebSocket handshake")
        buf += chunk
    head, _, leftover = buf.partition(b"\r\n\r\n")
    head_text = head.decode("latin-1", errors="replace")
    status_line = head_text.splitlines()[0] if head_text else ""

    logid = ""
    accept = ""
    for line in head_text.splitlines()[1:]:
        name, _, value = line.partition(":")
        name, value = name.strip().lower(), value.strip()
        if name == "x-tt-logid":
            logid = value
        elif name == "sec-websocket-accept":
            accept = value

    if "101" not in status_line:
        sock.close()
        raise SystemExit(
            f"Doubao ASR handshake failed: {status_line or 'no status line'} "
            f"(logid={logid or 'n/a'}). Check credentials/resource id."
        )
    expected = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
    if accept and accept != expected:
        sock.close()
        raise SystemExit("Doubao ASR handshake failed: bad Sec-WebSocket-Accept")

    return _WSConn(sock, leftover), logid


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
def _segments_from_utterances(utterances: list[dict]) -> list[dict]:
    out: list[dict] = []
    for utt in utterances or []:
        text = (utt.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "start": round(float(utt.get("start_time") or 0) / 1000.0, 2),
            "end": round(float(utt.get("end_time") or 0) / 1000.0, 2),
            "text": text,
        })
    out.sort(key=lambda s: s["start"])
    return out


def _build_params(language: str | None) -> dict:
    request = {
        "model_name": "bigmodel",
        "show_utterances": True,   # required to get per-utterance timestamps
        "result_type": "full",     # each response carries the full result so far
        "enable_punc": True,
        "enable_itn": True,
    }
    audio = {"format": "pcm", "codec": "raw", "rate": PCM_RATE, "bits": 16, "channel": 1}
    if language:
        audio["language"] = language
    return {"user": {"uid": "watch-skill"}, "audio": audio, "request": request}


def transcribe_pcm(
    pcm_bytes: bytes,
    headers: dict,
    language: str | None = None,
    read_timeout: float | None = None,
) -> list[dict]:
    """Stream PCM to Doubao and return {start,end,text} segments.

    Raises SystemExit on any failure — including a dropped connection or a
    read timeout — so callers (watch.py) can catch one exception type and
    fall back gracefully, the same way the Whisper backends do.
    """
    if read_timeout is None:
        # Bounded by audio length so a long/whole-video transcription doesn't
        # trip a fixed timeout while the server is still processing; +30s
        # covers connection/model latency on top of that.
        seconds = len(pcm_bytes) / PCM_BYTES_PER_SEC
        read_timeout = max(30.0, seconds + 30.0)

    try:
        conn, logid = _ws_connect(headers)
    except OSError as exc:
        raise SystemExit(f"Doubao ASR: connection failed: {exc}") from exc

    conn.settimeout(read_timeout)
    try:
        conn.send_binary(build_full_client_request(_build_params(language)))

        # Fail fast on an immediate auth/param rejection, without deadlocking:
        # only read if the server already has something waiting for us.
        if conn.readable(2.0):
            opcode, raw = conn.recv_message()
            if opcode == 0x8:
                raise SystemExit(f"Doubao ASR: server closed before audio (logid={logid})")
            msg = parse_server_message(raw)
            if msg["type"] == _MSG_ERROR:
                raise SystemExit(
                    f"Doubao ASR error {msg.get('code')}: {msg.get('error')} (logid={logid})"
                )

        # Stream audio packets; flag the final one.
        total = len(pcm_bytes)
        if total == 0:
            conn.send_binary(build_audio_request(b"", last=True))
        else:
            offset = 0
            while offset < total:
                chunk = pcm_bytes[offset:offset + PACKET_BYTES]
                offset += PACKET_BYTES
                conn.send_binary(build_audio_request(chunk, last=offset >= total))

        # Drain responses until the final-result frame or a close.
        final_utterances: list[dict] = []
        while True:
            opcode, raw = conn.recv_message()
            if opcode == 0x8:
                break
            msg = parse_server_message(raw)
            if msg["type"] == _MSG_ERROR:
                raise SystemExit(
                    f"Doubao ASR error {msg.get('code')}: {msg.get('error')} (logid={logid})"
                )
            payload = msg.get("payload") or {}
            result = payload.get("result") or {}
            utterances = result.get("utterances")
            if utterances:
                final_utterances = utterances
            if msg["flags"] == _FLAG_LAST_RESULT:
                break

        return _segments_from_utterances(final_utterances)
    except OSError as exc:
        raise SystemExit(f"Doubao ASR: connection error while streaming (logid={logid}): {exc}") from exc
    finally:
        conn.close()


def _shift_segments(segments: list[dict], offset: float) -> list[dict]:
    if not offset:
        return segments
    return [
        {
            "start": round(seg["start"] + offset, 2),
            "end": round(seg["end"] + offset, 2),
            "text": seg["text"],
        }
        for seg in segments
    ]


def transcribe_video(
    video_path: str,
    audio_out: Path,
    headers: dict | None = None,
    language: str | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> tuple[list[dict], str]:
    """Full flow: extract PCM → stream to Doubao → parse segments.

    When start/end are given, only that window is transcribed (and billed), and
    the returned segment timestamps are shifted back to absolute source time.

    Returns (segments, backend_label). Raises SystemExit on any failure.
    """
    if headers is None:
        headers = load_credentials()
    if not headers:
        setup_py = Path(__file__).resolve().parent / "setup.py"
        raise SystemExit(
            "No Doubao ASR credentials. Set DOUBAO_ASR_ACCESS_TOKEN "
            "(optionally DOUBAO_ASR_APP_ID), or set DOUBAO_ASR_API_KEY, in the "
            "environment or in ~/.config/watch/.env. "
            f"Run `python3 {setup_py}` to configure."
        )
    if language is None:
        language = _lookup("DOUBAO_ASR_LANGUAGE")

    offset = start_seconds if (start_seconds and start_seconds > 0) else 0.0
    print("[watch] extracting audio for Doubao ASR…", file=sys.stderr)
    pcm_path = extract_pcm(video_path, audio_out, start_seconds, end_seconds)
    pcm_bytes = pcm_path.read_bytes()
    seconds = len(pcm_bytes) / PCM_BYTES_PER_SEC
    scope = f" (focus {offset:.0f}-{offset + seconds:.0f}s)" if offset or end_seconds else ""
    print(
        f"[watch] streaming {seconds:.0f}s of audio to Doubao streaming ASR 2.0{scope}…",
        file=sys.stderr,
    )

    segments = _shift_segments(transcribe_pcm(pcm_bytes, headers, language=language), offset)
    if not segments:
        raise SystemExit("Doubao ASR returned no transcript segments")

    print(f"[watch] transcribed {len(segments)} segments via Doubao", file=sys.stderr)
    return segments, BACKEND_LABEL


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: doubao.py <video-path> [<audio-out.pcm>] [--language zh-CN]", file=sys.stderr)
        raise SystemExit(2)
    video = sys.argv[1]
    audio_out = (
        Path(sys.argv[2])
        if len(sys.argv) > 2 and not sys.argv[2].startswith("--")
        else Path("audio.pcm")
    )
    lang = None
    if "--language" in sys.argv:
        lang = sys.argv[sys.argv.index("--language") + 1]
    segments, backend = transcribe_video(video, audio_out, language=lang)
    print(json.dumps({"backend": backend, "segments": segments}, indent=2, ensure_ascii=False))
