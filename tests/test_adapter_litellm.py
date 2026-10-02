"""LiteLLM adapter: StandardLoggingPayload files and /spend/logs rows into sessions."""

from __future__ import annotations

import json
from pathlib import Path

from burnlens.adapter_litellm import load_litellm
from burnlens.adapters import load_all
from burnlens.aggregate import aggregate
from burnlens.findings import Thresholds, detect


def _payload(i: int, session: str, user: str, cache_read: int, tool: dict | None = None, result_len: int = 0) -> dict:
    messages = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "fix the payment retry bug in payment_flow.py"}]
    response = {"choices": [{"message": {"role": "assistant", "content": "Reading the file first.", "tool_calls": []}}]}
    if tool:
        response["choices"][0]["message"]["tool_calls"] = [{"id": f"call_{i}", "type": "function", "function": {"name": tool["name"], "arguments": json.dumps(tool["args"])}}]
        messages.append({"role": "assistant", "content": None, "tool_calls": response["choices"][0]["message"]["tool_calls"]})
        messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "x" * result_len})
    return {
        "id": f"chatcmpl-{session}-{i}",
        "trace_id": f"trace-{session}",
        "session_id": session,
        "call_type": "acompletion",
        "status": "success",
        "startTime": 1757066400.0 + i * 30,
        "endTime": 1757066405.0 + i * 30,
        "model": "claude-sonnet-5",
        "prompt_tokens": 1_000 + cache_read,
        "completion_tokens": 120,
        "total_tokens": 1_120 + cache_read,
        "response_cost": 0.01,
        "metadata": {
            "user_api_key_alias": "codex-key", "user_api_key_team_alias": "payments", "user_api_key_user_id": user,
            "usage_object": {"input_tokens": 1_000, "output_tokens": 120, "cache_read_input_tokens": cache_read, "cache_creation_input_tokens": 0},
        },
        "messages": messages,
        "response": response,
        "request_tags": [],
    }


def test_standard_logging_payloads(tmp_path: Path) -> None:
    root = tmp_path / "litellm"
    root.mkdir()
    (root / "one.json").write_text(json.dumps(_payload(0, "s-a", "sam", 200_000, {"name": "read_file", "args": {"path": "services/payment_flow.py"}}, 60_000)))
    (root / "batch.json").write_text(json.dumps([
        _payload(1, "s-a", "sam", 240_000, {"name": "execute_command", "args": {"cmd": "cat logs/checkout.log"}}, 120_000),
        _payload(2, "s-a", "sam", 260_000),
        _payload(0, "s-b", "priya", 5_000),
    ]))
    (root / "stream.jsonl").write_text(json.dumps(_payload(3, "s-a", "sam", 280_000)) + "\nnot json\n")
    sessions = {s.session_id: s for s in load_litellm(root)}
    assert set(sessions) == {"s-a", "s-b"}
    a = sessions["s-a"]
    assert len(a.turns) == 4 and a.agent == "codex-key" and a.user == "sam" and a.project == "payments"
    assert a.first_prompt.startswith("fix the payment retry bug")
    first = a.turns[0]
    assert first.usage.cache_read_input_tokens == 200_000 and first.usage.input_tokens == 1_000 and first.usage.output_tokens == 120
    assert first.tool_calls[0].name == "Read" and first.tool_calls[0].file_path == "services/payment_flow.py" and first.tool_calls[0].result_bytes == 60_000
    second = a.turns[1]
    assert second.tool_calls[0].name == "Bash" and second.tool_calls[0].command_head == "cat logs/checkout.log" and second.tool_calls[0].result_bytes == 120_000
    assert first.text_preview == "Reading the file first."
    th = Thresholds()
    agg = aggregate(list(sessions.values()), th.context_tokens, th.large_payload_bytes, th.premium_markers)
    assert "large-payloads" in {f.rule for f in detect(agg, th)}
    assert set(agg.by_agent) == {"codex-key"} and set(agg.by_user) == {"sam", "priya"}


def test_spend_log_rows_with_openai_cached_tokens_and_fallback_grouping(tmp_path: Path) -> None:
    rows = {"data": [
        {"request_id": "r1", "call_type": "acompletion", "api_key": "hash", "spend": 0.002, "total_tokens": 50_120, "prompt_tokens": 50_000, "completion_tokens": 120,
         "startTime": "2026-09-05T10:00:00.000Z", "endTime": "2026-09-05T10:00:03.000Z", "model": "gpt-5", "model_group": "gpt-5", "user": "prod-svc", "team_id": "support",
         "metadata": json.dumps({"user_api_key_alias": "support-bot", "usage_object": {"prompt_tokens": 50_000, "completion_tokens": 120, "prompt_tokens_details": {"cached_tokens": 48_000}}}),
         "request_tags": ["env:prod"], "cache_hit": "False"},
        {"request_id": "r2", "prompt_tokens": 50_000, "completion_tokens": 100, "startTime": "2026-09-05T10:05:00Z", "model": "gpt-5", "user": "prod-svc",
         "metadata": "{}", "cache_hit": "True"},
        {"request_id": "r3", "prompt_tokens": 10, "completion_tokens": 5, "startTime": "2026-09-06T09:00:00Z", "model": "gpt-5-mini", "user": "prod-svc", "metadata": {}},
    ]}
    path = tmp_path / "spend_logs.json"
    path.write_text(json.dumps(rows))
    sessions = {s.session_id: s for s in load_litellm(path)}
    assert set(sessions) == {"prod-svc-2026-09-05", "prod-svc-2026-09-06"}, "no session id: one session per user per day"
    day1 = sessions["prod-svc-2026-09-05"]
    assert day1.agent == "support-bot" and day1.project == "support" and day1.user == "prod-svc"
    u1, u2 = day1.turns[0].usage, day1.turns[1].usage
    assert u1.cache_read_input_tokens == 48_000 and u1.input_tokens == 2_000
    assert u2.cache_read_input_tokens == 50_000 and u2.input_tokens == 0, "a proxy cache hit sent nothing new"
    assert sessions["prod-svc-2026-09-06"].agent == "litellm", "no key alias: default application"


def test_load_all_accepts_litellm_extras(tmp_path: Path) -> None:
    from test_profiler import _assistant, _write

    cc = tmp_path / "cc"
    _write(cc / "-p" / "sess-a.jsonl", [_assistant("a0", "2026-09-05T09:00:01Z", "claude-opus-5", 5_000, 10)])
    lite = tmp_path / "lite"
    lite.mkdir()
    (lite / "p.json").write_text(json.dumps(_payload(0, "s-z", "sam", 1_000)))
    sessions = load_all("claude-code", cc, [("litellm", lite)])
    assert {s.agent for s in sessions} == {"claude-code", "codex-key"}
