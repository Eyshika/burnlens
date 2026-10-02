"""Langfuse adapter and the OTLP receiver."""

from __future__ import annotations

import json
from pathlib import Path

from burnlens.adapter_langfuse import load_langfuse
from burnlens.adapters import load_all
from burnlens.aggregate import aggregate
from burnlens.findings import Thresholds
from burnlens.otlp import OtlpIngest
from burnlens.server import DashboardServer


def test_langfuse_traces_and_observations(tmp_path: Path) -> None:
    root = tmp_path / "lf"
    root.mkdir()
    traces = {"data": [
        {"id": "t1", "timestamp": "2026-09-05T10:00:00.000Z", "name": "support-agent", "userId": "lee", "sessionId": "conv-9", "tags": ["prod"], "metadata": {"project": "support"}, "input": {"messages": [{"role": "user", "content": "where is my refund"}]}},
        {"id": "t2", "timestamp": "2026-09-05T11:00:00.000Z", "name": "support-agent", "userId": "lee", "sessionId": "conv-9", "tags": [], "metadata": {}},
    ], "meta": {"page": 1}}
    (root / "traces.json").write_text(json.dumps(traces))
    obs = [
        {"id": "o1", "traceId": "t1", "type": "GENERATION", "name": "chat", "startTime": "2026-09-05T10:00:01Z", "endTime": "2026-09-05T10:00:03Z", "model": "claude-sonnet-5",
         "usageDetails": {"input": 90_000, "output": 200, "total": 90_200, "cache_read_input_tokens": 80_000}, "input": [{"role": "user", "content": "where is my refund"}], "output": {"role": "assistant", "content": "Let me check."}},
        {"id": "o2", "traceId": "t1", "type": "TOOL", "name": "retrieve", "startTime": "2026-09-05T10:00:02Z", "input": {"query": "refund policy"}, "output": "k" * 70_000},
        {"id": "o3", "traceId": "t2", "type": "GENERATION", "name": "chat", "startTime": "2026-09-05T11:00:01Z", "model": "gpt-5", "usage": {"input": 1_000, "output": 50, "total": 1_050, "unit": "TOKENS"}, "usageDetails": {"input": 1_000, "output": 50, "input_cached_tokens": 600}},
        {"id": "o4", "traceId": "t2", "type": "SPAN", "name": "pipeline", "startTime": "2026-09-05T11:00:00Z"},
    ]
    (root / "observations.jsonl").write_text("\n".join(json.dumps(o) for o in obs) + "\n")
    sessions = {s.session_id: s for s in load_langfuse(root)}
    assert set(sessions) == {"conv-9"}, "two traces in one Langfuse session become one session"
    s = sessions["conv-9"]
    assert s.user == "lee" and s.agent == "support-agent" and s.project == "support"
    assert len(s.turns) == 2 and s.first_prompt == "where is my refund"
    t1, t2 = s.turns
    assert t1.usage.cache_read_input_tokens == 80_000 and t1.usage.input_tokens == 10_000 and t1.usage.output_tokens == 200
    assert t1.tool_calls[0].name == "Retrieve" and t1.tool_calls[0].result_bytes == 70_000 and t1.text_preview == "Let me check."
    assert t2.usage.cache_read_input_tokens == 600 and t2.usage.input_tokens == 400
    th = Thresholds()
    agg = aggregate(list(sessions.values()), th.context_tokens, th.large_payload_bytes, th.premium_markers)
    assert set(agg.by_user) == {"lee"} and set(agg.by_agent) == {"support-agent"}
    assert load_all("langfuse", root, []) and load_all("claude-code", tmp_path, [("langfuse", root)]) if (tmp_path / "x").mkdir() is None else True


def _otlp(records: list[tuple[str, dict, int]]) -> dict:
    def val(v):
        return {"intValue": str(v)} if isinstance(v, int) else {"stringValue": str(v)}
    return {"resourceLogs": [{
        "resource": {"attributes": [{"key": "session.id", "value": {"stringValue": "sess-42"}}, {"key": "user.email", "value": {"stringValue": "sam@acme.com"}}, {"key": "organization.id", "value": {"stringValue": "acme"}}]},
        "scopeLogs": [{"logRecords": [
            {"timeUnixNano": str(ts), "body": {"stringValue": name}, "attributes": [{"key": k, "value": val(v)} for k, v in attrs.items()]}
            for name, attrs, ts in records
        ]}],
    }]}


def test_otlp_folds_events_into_generic_records(tmp_path: Path) -> None:
    spool = tmp_path / "otlp"
    ingest = OtlpIngest(spool)
    base = 1_757_066_400_000_000_000
    payload = _otlp([
        ("claude_code.user_prompt", {"event.name": "claude_code.user_prompt", "prompt": "fix the loader", "prompt_length": 14}, base),
        ("claude_code.tool_result", {"event.name": "claude_code.tool_result", "tool_name": "Read", "tool_use_id": "tu1", "tool_result_size_bytes": 60_000, "tool_parameters": json.dumps({"file_path": "/repo/loader.py"})}, base + 1_000_000_000),
        ("claude_code.tool_result", {"event.name": "claude_code.tool_result", "tool_name": "Bash", "tool_use_id": "tu2", "tool_result_size_bytes": 300, "tool_parameters": json.dumps({"bash_command": "pytest -q", "full_command": "pytest -q tests"})}, base + 2_000_000_000),
        ("claude_code.api_request", {"event.name": "claude_code.api_request", "model": "claude-opus-5", "input_tokens": 12, "output_tokens": 400, "cache_read_tokens": 250_000, "cache_creation_tokens": 800, "request_id": "req-1"}, base + 3_000_000_000),
        ("claude_code.api_request", {"event.name": "claude_code.api_request", "model": "claude-sonnet-5", "input_tokens": 5, "output_tokens": 50, "cache_read_tokens": 1_000, "cache_creation_tokens": 0, "agent.name": "researcher", "request_id": "req-2"}, base + 4_000_000_000),
    ])
    assert ingest.ingest_logs(payload) == 2
    files = list(spool.glob("*.jsonl"))
    assert len(files) == 1
    rows = [json.loads(l) for l in files[0].read_text().splitlines()]
    main = rows[0]
    assert main["session_id"] == "sess-42" and main["user"] == "sam@acme.com" and main["agent"] == "claude-code" and main["project"] == "acme"
    assert main["usage"] == {"input_tokens": 12, "output_tokens": 400, "cache_read_input_tokens": 250_000, "cache_creation_input_tokens": 800}
    assert main["prompt"] == "fix the loader" and [t["name"] for t in main["tools"]] == ["Read", "Bash"]
    assert main["tools"][0]["input"]["file_path"] == "/repo/loader.py" and main["tools"][1]["input"]["command"] == "pytest -q"
    sub = rows[1]
    assert sub["parent_session_id"] == "sess-42" and sub["session_id"] == "sess-42:researcher" and sub["tools"] == []
    # And it is readable by the generic adapter with per-person identity.
    from burnlens.adapters import load_generic

    sessions = {s.session_id: s for s in load_generic(spool)}
    assert sessions["sess-42"].user == "sam@acme.com" and sessions["sess-42:researcher"].is_subagent


def test_server_accepts_otlp_post(tmp_path: Path, monkeypatch) -> None:
    import threading
    import urllib.request

    from test_profiler import _assistant, _write

    root = tmp_path / "cc"
    _write(root / "-p" / "s.jsonl", [_assistant("a0", "2026-09-05T09:00:01Z", "claude-opus-5", 5_000, 10)])
    monkeypatch.setattr("burnlens.server.OtlpIngest", lambda: OtlpIngest(tmp_path / "spool"))
    server = DashboardServer("127.0.0.1", 0, root, Thresholds(), notify=False)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        body = json.dumps(_otlp([("claude_code.api_request", {"event.name": "claude_code.api_request", "model": "m", "input_tokens": 1, "output_tokens": 1}, 1_757_066_400_000_000_000)])).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/logs", data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as res:
            assert res.status == 200
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/report?days=0", timeout=5) as res:
            report = json.loads(res.read())
        assert "claude-code" in report["by_agent"] and report["sessions"] == 2, "the OTLP spool joins the same view"
        assert report["by_user"].get("") is None
    finally:
        server.shutdown()
        server.server_close()
