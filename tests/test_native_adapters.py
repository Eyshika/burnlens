import json
from pathlib import Path
from datetime import datetime, timezone

from burnlens.adapters import load, discover_sources

TS = "2026-09-10T10:00:00Z"


def write_records(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records))


def record(kind: str, payload: dict) -> dict:
    return {"timestamp": TS, "type": kind, "payload": payload}


def test_codex_detailed_usage_deduplicates_legacy_events_and_results(tmp_path: Path) -> None:
    usage = {"input_tokens": 100, "cached_input_tokens": 60, "output_tokens": 20, "reasoning_output_tokens": 5}
    counted = record("token_usage_record", {"response_id": "r1", "usage": usage})
    write_records(tmp_path / "rollout.jsonl", [
        record("session_meta", {"id": "native", "cwd": "/project"}),
        record("turn_context", {"model": "configured-model"}),
        record("response_item", {"type": "function_call", "call_id": "c", "name": "exec_command", "arguments": '{"cmd":"test"}'}),
        record("response_item", {"type": "function_call_output", "call_id": "c", "output": "é"}),
        record("event_msg", {"type": "token_count", "info": {"total_token_usage": usage, "last_token_usage": usage}}),
        counted, counted,
    ])
    session = load("codex", tmp_path)[0]
    assert session.agent == "codex"
    assert session.usage.total == 120
    assert session.usage.input_tokens == 40
    assert len(session.turns) == 1
    assert session.turns[0].model == "configured-model"
    call = session.turns[0].tool_calls[0]
    assert call.name == "Bash" and call.command_head == "test"
    assert call.result_bytes == 2


def test_codex_cumulative_events_not_counted_twice(tmp_path: Path) -> None:
    usage = {"input_tokens": 100, "cached_input_tokens": 25, "output_tokens": 10}
    event = record("event_msg", {"type": "token_count", "info": {"total_token_usage": usage, "last_token_usage": usage}})
    write_records(tmp_path / "old.jsonl", [event, event])
    assert load("codex", tmp_path)[0].usage.total == 110
    assert load("codex", tmp_path, since=datetime(2027, 1, 1, tzinfo=timezone.utc)) == []


def test_gemini_json_and_jsonl_dedup_and_rewind(tmp_path: Path) -> None:
    metadata = {"sessionId": "gem", "projectHash": "project"}
    message = {"id": "m", "type": "gemini", "timestamp": TS, "model": "configured", "tokens": {"input": 100, "cached": 30, "output": 10, "thoughts": 5}, "toolCalls": [{"id": "t", "name": "run_shell_command", "args": {"command": "test"}, "result": "é"}]}
    path = tmp_path / "session-one.json"
    path.write_text(json.dumps({**metadata, "messages": [message]}, indent=2))
    session = load("gemini-cli", path)[0]
    assert session.usage.total == 115
    assert session.turns[0].tool_calls[0].result_bytes == 2
    stream = tmp_path / "session-two.jsonl"
    write_records(stream, [metadata, message, message, {**message, "id": "gone"}, {"$rewindTo": "gone"}])
    assert len(load("gemini-cli", stream)[0].turns) == 1


def test_discovery_honors_codex_home(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "sessions").mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert ("codex", tmp_path / "sessions") in discover_sources()


def test_codex_command_events_and_unknown_usage(tmp_path: Path, caplog) -> None:
    write_records(tmp_path / "rollout.jsonl", [
        record("event_msg", {"type": "item_completed", "item": {"type": "CommandExecution", "id": "shell", "command": "test", "formatted_output": "é", "aggregated_output": "larger unformatted"}}),
        record("token_usage_record", {"response_id": "unknown", "usage": {"total_tokens": 500}}),
        record("token_usage_record", {"response_id": "known", "usage": {"input_tokens": 10, "output_tokens": 2}}),
    ])
    session = load("codex", tmp_path)[0]
    assert session.usage.total == 12
    assert len(session.turns) == 1
    assert session.turns[0].tool_calls[0].result_bytes == 2
    assert "components unavailable" in caplog.text


def test_codex_copied_response_not_counted_across_files(tmp_path: Path) -> None:
    common = record("token_usage_record", {"response_id": "shared", "usage": {"input_tokens": 10}})
    write_records(tmp_path / "a.jsonl", [common])
    write_records(tmp_path / "b.jsonl", [common, record("token_usage_record", {"response_id": "new", "usage": {"input_tokens": 20}})])
    sessions = load("codex", tmp_path)
    assert sum(session.usage.total for session in sessions) == 30
    assert sum(len(session.turns) for session in sessions) == 2
