"""Doubao ASR: credential resolution and utterance/segment shaping.

No network or websocket tests here — transcribe_pcm/_ws_connect need a live
Volcano Engine connection, so they're exercised manually, not in CI.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import doubao

_DOUBAO_ENV_KEYS = (
    "DOUBAO_ASR_API_KEY",
    "DOUBAO_ASR_APP_ID",
    "DOUBAO_ASR_APP_KEY",
    "DOUBAO_ASR_ACCESS_TOKEN",
    "DOUBAO_ASR_ACCESS_KEY",
    "DOUBAO_ASR_RESOURCE_ID",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Isolate HOME/cwd so a developer's real ~/.config/watch/.env or ./.env
    can't leak Doubao credentials into these assertions."""
    for key in _DOUBAO_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)


def _write_dotenv(tmp_path: Path, body: str) -> None:
    cfg = tmp_path / ".config" / "watch"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / ".env").write_text(body, encoding="utf-8")


class TestLoadCredentials:
    def test_no_credentials_returns_none(self):
        assert doubao.load_credentials() is None

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("DOUBAO_ASR_API_KEY", "sk-test")
        creds = doubao.load_credentials()
        assert creds == {
            "X-Api-Key": "sk-test",
            "X-Api-Resource-Id": doubao.DEFAULT_RESOURCE_ID,
        }

    def test_app_id_and_token_from_dotenv(self, tmp_path):
        _write_dotenv(tmp_path, "DOUBAO_ASR_APP_ID=app123\nDOUBAO_ASR_ACCESS_TOKEN=tok456\n")
        creds = doubao.load_credentials()
        assert creds == {
            "X-Api-App-Key": "app123",
            "X-Api-Access-Key": "tok456",
            "X-Api-Resource-Id": doubao.DEFAULT_RESOURCE_ID,
        }

    def test_app_id_without_token_is_incomplete(self, monkeypatch):
        monkeypatch.setenv("DOUBAO_ASR_APP_ID", "app123")
        assert doubao.load_credentials() is None

    def test_api_key_takes_priority_over_app_id_pair(self, monkeypatch):
        monkeypatch.setenv("DOUBAO_ASR_API_KEY", "sk-test")
        monkeypatch.setenv("DOUBAO_ASR_APP_ID", "app123")
        monkeypatch.setenv("DOUBAO_ASR_ACCESS_TOKEN", "tok456")
        assert doubao.load_credentials() == {
            "X-Api-Key": "sk-test",
            "X-Api-Resource-Id": doubao.DEFAULT_RESOURCE_ID,
        }

    def test_custom_resource_id_overrides_default(self, monkeypatch):
        monkeypatch.setenv("DOUBAO_ASR_API_KEY", "sk-test")
        monkeypatch.setenv("DOUBAO_ASR_RESOURCE_ID", "volc.custom.tier")
        creds = doubao.load_credentials()
        assert creds["X-Api-Resource-Id"] == "volc.custom.tier"


class TestSegmentsFromUtterances:
    def test_converts_ms_to_seconds(self):
        segs = doubao._segments_from_utterances(
            [{"text": "hello", "start_time": 1000, "end_time": 2500}]
        )
        assert segs == [{"start": 1.0, "end": 2.5, "text": "hello"}]

    def test_drops_empty_text(self):
        segs = doubao._segments_from_utterances(
            [{"text": "  ", "start_time": 0, "end_time": 1000}, {"text": "hi", "start_time": 1000, "end_time": 2000}]
        )
        assert segs == [{"start": 1.0, "end": 2.0, "text": "hi"}]

    def test_sorts_by_start(self):
        segs = doubao._segments_from_utterances(
            [
                {"text": "second", "start_time": 2000, "end_time": 3000},
                {"text": "first", "start_time": 0, "end_time": 1000},
            ]
        )
        assert [s["text"] for s in segs] == ["first", "second"]

    def test_none_utterances_is_empty(self):
        assert doubao._segments_from_utterances(None) == []


class TestShiftSegments:
    def test_adds_offset(self):
        segs = [{"start": 0.0, "end": 2.0, "text": "hi"}]
        assert doubao._shift_segments(segs, 100.0) == [{"start": 100.0, "end": 102.0, "text": "hi"}]

    def test_zero_offset_is_identity(self):
        segs = [{"start": 1.0, "end": 2.0, "text": "x"}]
        assert doubao._shift_segments(segs, 0.0) == segs
