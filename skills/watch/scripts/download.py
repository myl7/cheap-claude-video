#!/usr/bin/env python3
"""Download a video via yt-dlp, or resolve a local file path.

Also fetches subtitles (manual first, then auto-generated) in VTT format so
transcribe.py can parse them without needing Whisper.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from config import read_env_file


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}

# A modern desktop-browser UA. Some gated sites (Bilibili) reject yt-dlp's
# default UA with HTTP 412; a browser UA plus cookies clears it.
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _ytdlp_auth_args() -> list[str]:
    """Cookie / UA args for yt-dlp, from env or ~/.config/watch/.env.

    - WATCH_YTDLP_COOKIES_FROM_BROWSER: e.g. `chrome`, `firefox`, `chrome:Default`
    - WATCH_YTDLP_COOKIES_FILE: path to a Netscape cookies.txt
    - WATCH_YTDLP_USER_AGENT: override the UA (defaults to a browser UA when any
      cookie source is set, which is what gated sites like Bilibili need)
    """
    file_values = read_env_file()

    def get(name: str) -> str | None:
        return os.environ.get(name) or file_values.get(name)

    args: list[str] = []
    from_browser = get("WATCH_YTDLP_COOKIES_FROM_BROWSER")
    if from_browser:
        args += ["--cookies-from-browser", from_browser]
    cookies_file = get("WATCH_YTDLP_COOKIES_FILE")
    if cookies_file:
        args += ["--cookies", cookies_file]

    user_agent = get("WATCH_YTDLP_USER_AGENT")
    if user_agent:
        args += ["--user-agent", user_agent]
    elif from_browser or cookies_file:
        args += ["--user-agent", _BROWSER_UA]
    return args


# Languages to fetch captions for, in no particular order (the pick below ranks
# them). Chinese variants + English cover the common cases; native captions are
# free, so grabbing them avoids paying an ASR backend. Configurable so a user
# whose videos are in another language can add it.
_DEFAULT_SUB_LANGS = "zh-Hans,zh-Hant,zh,zh-CN,zh-TW,zh-HK,yue,en-orig,en,en-US,en-GB"


def _sub_langs() -> str:
    file_values = read_env_file()
    return os.environ.get("WATCH_SUB_LANGS") or file_values.get("WATCH_SUB_LANGS") or _DEFAULT_SUB_LANGS


def _is_youtube(url: str) -> bool:
    host = urlparse(url).netloc.lower().split("@")[-1].split(":")[0]
    return host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")


def _ytdlp_site_args(url: str) -> list[str]:
    """Site-specific yt-dlp args. YouTube's newer anti-bot sometimes needs a
    remote JS challenge solver, available via yt-dlp's `--remote-components`
    — but that fetches and runs a solver component from yt-dlp's GitHub, so
    it's opt-in rather than on by default. Off for every other site.

    Set WATCH_YTDLP_REMOTE_COMPONENTS to a value like `ejs:github` to enable
    it; unset (the default) or `off`/`0`/`false`/empty leaves it disabled.
    """
    if not _is_youtube(url):
        return []
    file_values = read_env_file()
    value = os.environ.get("WATCH_YTDLP_REMOTE_COMPONENTS")
    if value is None:
        value = file_values.get("WATCH_YTDLP_REMOTE_COMPONENTS")
    if value is None:
        return []  # disabled by default — see the hint in download_url's error message
    value = value.strip()
    if not value or value.lower() in ("off", "0", "false", "none"):
        return []
    return ["--remote-components", value]


def is_url(source: str) -> bool:
    if source.startswith("-"):
        return False
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def resolve_local(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    if p.suffix.lower() not in VIDEO_EXTS:
        print(
            f"[watch] warning: {p.suffix} is not a known video extension, proceeding anyway",
            file=sys.stderr,
        )
    return {
        "video_path": str(p),
        "subtitle_path": None,
        "info": {"title": p.name, "url": str(p)},
        "downloaded": False,
    }


def _subtitle_lang(name: str) -> str:
    """`video.zh-Hans.vtt` → `zh-Hans`, `video.en-orig.vtt` → `en-orig`."""
    parts = name.split(".")
    return parts[-2] if len(parts) >= 3 else ""


def _pick_subtitle(out_dir: Path, prefer_lang: str | None = None) -> Path | None:
    """Pick the best caption file.

    Rank: (1) the video's own language (`prefer_lang` from info.json) — this
    beats a machine-translated caption in another language; (2) Chinese; (3)
    English; (4) anything. Native captions are free, so preferring the original
    language keeps quality high and avoids paying an ASR backend.
    """
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None

    prefer_base = (prefer_lang or "").lower().split("-")[0]

    def rank(path: Path) -> tuple[int, int, str]:
        lang = _subtitle_lang(path.name).lower()
        base = lang.split("-")[0]
        if prefer_base and base == prefer_base:
            tier = 0
        elif base in ("zh", "yue"):
            tier = 1
        elif base == "en":
            tier = 2
        else:
            tier = 3
        orig_bonus = 0 if lang.endswith("-orig") else 1  # prefer original auto-subs
        return (tier, orig_bonus, path.name)

    return sorted(candidates, key=rank)[0]


def _pick_video(out_dir: Path) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3", ".opus"):
        for candidate in out_dir.glob(f"video*{ext}"):
            return candidate
    for candidate in out_dir.glob("video.*"):
        if candidate.suffix.lower() in VIDEO_EXTS:
            return candidate
    return None


def fetch_captions(url: str, out_dir: Path) -> dict:
    """Fetch metadata and best available VTT captions without downloading video."""
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")
    cmd = [
        "yt-dlp",
        *_ytdlp_auth_args(),
        *_ytdlp_site_args(url),
        "--skip-download",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", _sub_langs(),
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]
    subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
    info = _read_info(out_dir / "video.info.json", url)
    subtitle = _pick_subtitle(out_dir, prefer_lang=info.get("language"))
    return {
        "video_path": None,
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": False,
    }


def _read_info(info_path: Path, url: str) -> dict:
    info: dict = {}
    if info_path.exists():
        try:
            raw = json.loads(info_path.read_text(encoding="utf-8"))
            info = {
                "title": raw.get("title"),
                "uploader": raw.get("uploader") or raw.get("channel"),
                "duration": raw.get("duration"),
                "language": raw.get("language"),
                "url": raw.get("webpage_url") or url,
            }
        except Exception as exc:
            print(f"[watch] info.json parse failed: {exc}", file=sys.stderr)
            info = {"url": url}
    return info


def download_url(
    url: str,
    out_dir: Path,
    audio_only: bool = False,
) -> dict:
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")

    fmt = "ba/bestaudio" if audio_only else "bv*[height<=720]+ba/b[height<=720]/bv+ba/b"
    cmd = [
        "yt-dlp",
        *_ytdlp_auth_args(),
        *_ytdlp_site_args(url),
        "-N", "8",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", _sub_langs(),
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]

    # yt-dlp may exit non-zero if a subtitle variant fails (e.g. 429) even when
    # the video itself downloaded fine. Treat "video file present" as success.
    result = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
    video = _pick_video(out_dir)
    if video is None:
        hint = (
            " If this looks like a YouTube anti-bot/JS-challenge failure, set "
            "WATCH_YTDLP_REMOTE_COMPONENTS=ejs:github in ~/.config/watch/.env to let "
            "yt-dlp fetch+run its remote JS solver (disabled by default)."
            if _is_youtube(url) else ""
        )
        raise SystemExit(
            f"yt-dlp did not produce a video file in {out_dir} (exit {result.returncode}).{hint}"
        )

    info = _read_info(out_dir / "video.info.json", url)
    subtitle = _pick_subtitle(out_dir, prefer_lang=info.get("language"))

    return {
        "video_path": str(video),
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": True,
    }


def download(
    source: str,
    out_dir: Path,
    audio_only: bool = False,
) -> dict:
    if is_url(source):
        return download_url(source, out_dir, audio_only=audio_only)
    return resolve_local(source)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: download.py <url-or-path> <out-dir>", file=sys.stderr)
        raise SystemExit(2)
    result = download(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
