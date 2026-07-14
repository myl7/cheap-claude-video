"""setup.py --json surfaces the resolved watch detail and transcriber."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SETUP = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "setup.py"


def _run(args, *, home=None, extra_env=None):
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    env.pop("WATCH_TRANSCRIBER", None)
    # Don't let a real key in the developer's shell env leak into the test.
    env.pop("GROQ_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.pop("DOUBAO_ASR_APP_ID", None)
    env.pop("DOUBAO_ASR_ACCESS_TOKEN", None)
    env.pop("DOUBAO_ASR_API_KEY", None)
    env.pop("SETUP_COMPLETE", None)
    if home is not None:
        env["HOME"] = str(home)
        env["USERPROFILE"] = str(home)  # Windows
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(SETUP), *args],
        capture_output=True, text=True, env=env,
    )


def _write_env(home: Path, body: str) -> None:
    cfg = home / ".config" / "watch"
    cfg.mkdir(parents=True, exist_ok=True)
    f = cfg / ".env"
    f.write_text(body, encoding="utf-8")
    f.chmod(0o600)


def test_json_reports_watch_detail(tmp_path):
    # Needs an isolated HOME: an unrelated real ~/.config/watch/.env on the
    # machine running the suite would otherwise leak its WATCH_DETAIL in.
    proc = _run(["--json"], home=tmp_path)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["watch_detail"] == "balanced"


def test_keyless_completed_setup_proceeds_silently(tmp_path):
    """A user who finished setup without a key must NOT be nagged forever."""
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\nSETUP_COMPLETE=true\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 0, f"keyless-complete should pass --check; got {chk.returncode}: {chk.stderr}"
    assert chk.stdout == "" and chk.stderr == ""

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["can_proceed"] is True
    assert js["first_run"] is False
    assert js["setup_complete"] is True
    # status still encourages a key even though we can proceed
    assert js["status"] == "needs_key"


def test_keyless_first_run_is_encouraged(tmp_path):
    """Genuine first run with no key: --check reports exit 3 (encourage a key)."""
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 3, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["can_proceed"] is False
    assert js["first_run"] is True


def test_key_present_is_ready(tmp_path):
    _write_env(tmp_path, "GROQ_API_KEY=sk-test-abc\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 0, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["status"] == "ready"
    assert js["can_proceed"] is True
    assert js["whisper_backend"] == "groq"


def test_doubao_creds_alone_are_not_ready_under_auto_transcriber(tmp_path):
    """Regression guard: watch.py's `auto` transcriber (the default) only ever
    tries Whisper (Groq -> OpenAI), never Doubao — so credentials alone must
    NOT report `ready`/`has_api_key` without WATCH_TRANSCRIBER=doubao, or the
    agent skips setup per SKILL.md's `can_proceed` branch and /watch silently
    comes back frames-only on the first video with no captions."""
    _write_env(tmp_path, "DOUBAO_ASR_APP_ID=app\nDOUBAO_ASR_ACCESS_TOKEN=token\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 3, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["has_api_key"] is False
    assert js["status"] == "needs_key"
    assert js["can_proceed"] is False


def test_doubao_creds_are_ready_when_transcriber_is_doubao(tmp_path):
    _write_env(
        tmp_path,
        "DOUBAO_ASR_APP_ID=app\nDOUBAO_ASR_ACCESS_TOKEN=token\nWATCH_TRANSCRIBER=doubao\n",
    )
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 0, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["status"] == "ready"
    assert js["can_proceed"] is True
    assert js["whisper_backend"] == "doubao"


def test_groq_key_ignored_when_transcriber_pinned_to_openai(tmp_path):
    """Explicitly selecting one backend must not fall through to another —
    matches watch.py, which never silently substitutes a different backend
    for an explicit --whisper/WATCH_TRANSCRIBER choice."""
    _write_env(tmp_path, "GROQ_API_KEY=sk-test-abc\nWATCH_TRANSCRIBER=openai\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 3, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["has_api_key"] is False
