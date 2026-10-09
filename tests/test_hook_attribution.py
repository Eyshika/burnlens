"""Hook-injected context is attributed instead of silently dropped."""

from __future__ import annotations

import json
from pathlib import Path

from burnlens.aggregate import aggregate
from burnlens.findings import Thresholds, detect
from burnlens.transcripts import parse_session

BOARD = "<system-reminder>The task board:\n- task one (you)\n- task two (them)</system-reminder>"
NUDGE = "<system-reminder>compact your context</system-reminder>"


def _write_transcript(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    path = root / "sess1.jsonl"
    lines = [
        {
            "type": "user",
            "timestamp": "2026-10-09T10:00:00Z",
            "message": {"content": "fix the login bug"},
            "cwd": "/tmp/proj",
            "uuid": "u1",
        },
        {
            "type": "assistant",
            "timestamp": "2026-10-09T10:00:01Z",
            "uuid": "a1",
            "message": {
                "id": "m1",
                "model": "claude-opus",
                "usage": {"input_tokens": 8000, "output_tokens": 200},
                "content": [{"type": "text", "text": "on it"}],
            },
        },
        # SessionStart hook board, injected before the human typed anything
        {
            "type": "user",
            "timestamp": "2026-10-09T10:00:02Z",
            "message": {"content": BOARD},
            "uuid": "u2",
        },
        # list-form hook reminder
        {
            "type": "user",
            "timestamp": "2026-10-09T10:00:03Z",
            "message": {"content": [{"type": "text", "text": NUDGE}]},
            "uuid": "u3",
        },
        {
            "type": "assistant",
            "timestamp": "2026-10-09T10:00:04Z",
            "uuid": "a2",
            "message": {
                "id": "m2",
                "model": "claude-opus",
                "usage": {"input_tokens": 16000, "output_tokens": 400},
                "content": [{"type": "text", "text": "board seen"}],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    return path


def test_hook_injection_is_attributed(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    session = parse_session(path, tmp_path)
    assert session.hook_injections == 2
    assert session.hook_chars == len(BOARD) + len(NUDGE)
    # hook content is never counted as a human prompt
    assert session.first_prompt == "fix the login bug"
    assert len(session.prompts) == 1


def test_hook_injection_finding(tmp_path: Path) -> None:
    path = _write_transcript(tmp_path)
    session = parse_session(path, tmp_path)
    agg = aggregate([session], context_threshold=150_000, large_payload_bytes=50_000)
    assert agg.sessions[0].hook_chars == len(BOARD) + len(NUDGE)
    findings = detect(agg, Thresholds(hook_share=0.0))
    assert "hook-injection" in [f.rule for f in findings]


def test_no_hooks_no_finding(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    path = root / "sess2.jsonl"
    lines = [
        {
            "type": "user",
            "timestamp": "2026-10-09T10:00:00Z",
            "message": {"content": "hello"},
            "cwd": "/tmp/proj",
            "uuid": "u1",
        },
        {
            "type": "assistant",
            "timestamp": "2026-10-09T10:00:01Z",
            "uuid": "a1",
            "message": {
                "id": "m1",
                "model": "claude-opus",
                "usage": {"input_tokens": 100, "output_tokens": 10},
                "content": [{"type": "text", "text": "hi"}],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    session = parse_session(path, tmp_path)
    assert session.hook_injections == 0
    assert session.hook_chars == 0
    agg = aggregate([session], context_threshold=150_000, large_payload_bytes=50_000)
    findings = detect(agg, Thresholds(hook_share=0.0))
    assert "hook-injection" not in [f.rule for f in findings]
