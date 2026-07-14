"""yt-dlp argv construction for download.py.

Regression guard: ``--sub-langs all`` makes yt-dlp fetch YouTube's hundreds of
auto-translated caption tracks, which can take minutes and stalls before the
video download even starts. The configured list must stay an explicit,
bounded set of language codes rather than the wildcard.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import config  # noqa: E402
import download  # noqa: E402

URL = "https://www.youtube.com/watch?v=rlOpbu3Enkw"
NON_YOUTUBE_URL = "https://www.bilibili.com/video/BV1xx411c7mD"


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    """Point config.CONFIG_FILE at an empty file so a developer's real
    ~/.config/watch/.env (cookies, WATCH_SUB_LANGS, etc.) can't leak into
    these argv-construction assertions."""
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    monkeypatch.delenv("WATCH_SUB_LANGS", raising=False)
    monkeypatch.delenv("WATCH_YTDLP_REMOTE_COMPONENTS", raising=False)


def _capture_argv(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Stub subprocess.run inside download.py and record every argv."""
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    return calls


def _sub_langs(argv: list[str]) -> str:
    idx = argv.index("--sub-langs")
    return argv[idx + 1]


def _assert_bounded_sub_langs(langs: str) -> None:
    tokens = langs.split(",")
    assert "all" not in tokens, f"sub-langs must not request all languages, got {langs!r}"
    assert tokens, "sub-langs must not be empty"


def test_fetch_captions_requests_bounded_langs(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    _assert_bounded_sub_langs(_sub_langs(calls[0]))


def test_download_url_requests_bounded_langs(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    # _pick_video returns None with no real file, which raises SystemExit after
    # the yt-dlp argv is already built — that's all we need to inspect.
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    _assert_bounded_sub_langs(_sub_langs(calls[0]))


def test_remote_components_off_by_default_for_youtube():
    assert download._ytdlp_site_args(URL) == []


def test_remote_components_opt_in_via_env(monkeypatch):
    monkeypatch.setenv("WATCH_YTDLP_REMOTE_COMPONENTS", "ejs:github")
    assert download._ytdlp_site_args(URL) == ["--remote-components", "ejs:github"]


def test_remote_components_explicit_off_stays_off(monkeypatch):
    monkeypatch.setenv("WATCH_YTDLP_REMOTE_COMPONENTS", "off")
    assert download._ytdlp_site_args(URL) == []


def test_remote_components_never_applies_to_non_youtube(monkeypatch):
    monkeypatch.setenv("WATCH_YTDLP_REMOTE_COMPONENTS", "ejs:github")
    assert download._ytdlp_site_args(NON_YOUTUBE_URL) == []


def test_download_failure_hints_at_remote_components_for_youtube(monkeypatch, tmp_path):
    _capture_argv(monkeypatch)
    with pytest.raises(SystemExit, match="WATCH_YTDLP_REMOTE_COMPONENTS"):
        download.download_url(URL, tmp_path / "download")


def test_download_failure_has_no_hint_for_non_youtube(monkeypatch, tmp_path):
    _capture_argv(monkeypatch)
    with pytest.raises(SystemExit) as exc_info:
        download.download_url(NON_YOUTUBE_URL, tmp_path / "download")
    assert "WATCH_YTDLP_REMOTE_COMPONENTS" not in str(exc_info.value)
