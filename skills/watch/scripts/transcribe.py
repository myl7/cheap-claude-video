#!/usr/bin/env python3
"""Parse a WebVTT subtitle file into a clean, timestamped transcript.

YouTube auto-subs emit rolling-duplicate cues (each line appears 2-3 times as it
scrolls). We dedupe consecutive identical cues and merge their time ranges.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    segments: list[dict] = []
    i = 0
    while i < len(lines):
        match = TS_RE.match(lines[i])
        if not match:
            i += 1
            continue

        start = _to_seconds(*match.groups()[:4])
        end = _to_seconds(*match.groups()[4:])
        i += 1

        cue_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            cleaned = TAG_RE.sub("", lines[i]).strip()
            if cleaned:
                cue_lines.append(cleaned)
            i += 1

        cue_text = " ".join(cue_lines).strip()
        if cue_text:
            segments.append({"start": round(start, 2), "end": round(end, 2), "text": cue_text})
        i += 1

    return _dedupe(segments)


def _dedupe(segments: list[dict]) -> list[dict]:
    """Collapse rolling duplicates common in YouTube auto-subs."""
    out: list[dict] = []
    for seg in segments:
        if out and seg["text"] == out[-1]["text"]:
            out[-1]["end"] = seg["end"]
            continue
        if out and seg["text"].startswith(out[-1]["text"] + " "):
            out[-1]["text"] = seg["text"]
            out[-1]["end"] = seg["end"]
            continue
        out.append(seg)
    return out


def filter_range(
    segments: list[dict],
    start_seconds: float | None,
    end_seconds: float | None,
) -> list[dict]:
    """Return segments whose time range overlaps [start, end]."""
    if start_seconds is None and end_seconds is None:
        return segments
    lo = start_seconds if start_seconds is not None else float("-inf")
    hi = end_seconds if end_seconds is not None else float("inf")
    return [seg for seg in segments if seg["end"] >= lo and seg["start"] <= hi]


# In focus mode the frames stay inside the exact [--start, --end] window, but a
# transcript clipped to that same window reads without lead-in/lead-out — Claude
# loses the sentence that set up the moment and the one that resolves it. So the
# transcript window is widened by a pad on each side. The pad scales with the
# focus length (longer focus, more surrounding context worth showing) but is
# clamped so a tiny focus still gets a useful cushion and a big one doesn't
# balloon the token cost.
TRANSCRIPT_PAD_RATIO = 0.25
TRANSCRIPT_PAD_MIN_SECONDS = 10.0
TRANSCRIPT_PAD_MAX_SECONDS = 30.0


def context_window(
    start_seconds: float | None,
    end_seconds: float | None,
    full_duration: float | None = None,
) -> tuple[float | None, float | None]:
    """Widen a focus range into the transcript window (frames stay unpadded).

    Returns (padded_start, padded_end). A bound stays None when the focus left
    it open, so downstream filter_range / audio extraction keep treating it as
    unbounded (0 for the start, end-of-video for the end). padded_start is
    clamped to 0 and padded_end to full_duration when that duration is known.
    """
    if start_seconds is None and end_seconds is None:
        return None, None

    lo = start_seconds if start_seconds is not None else 0.0
    hi = end_seconds if end_seconds is not None else (full_duration or lo)
    span = max(0.0, hi - lo)
    pad = min(
        TRANSCRIPT_PAD_MAX_SECONDS,
        max(TRANSCRIPT_PAD_MIN_SECONDS, span * TRANSCRIPT_PAD_RATIO),
    )

    padded_start = None
    if start_seconds is not None:
        padded_start = max(0.0, start_seconds - pad)

    padded_end = None
    if end_seconds is not None:
        padded_end = end_seconds + pad
        if full_duration and full_duration > 0:
            padded_end = min(full_duration, padded_end)

    return padded_start, padded_end


def format_transcript(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        start = int(seg["start"])
        stamp = f"[{start // 60:02d}:{start % 60:02d}]"
        lines.append(f"{stamp} {seg['text']}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: transcribe.py <vtt-path>", file=sys.stderr)
        raise SystemExit(2)
    print(format_transcript(parse_vtt(sys.argv[1])))
