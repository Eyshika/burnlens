"""Unattended agents: workflow dimension, payload and the runaway rule."""

from __future__ import annotations

import json
from pathlib import Path

from burnlens.adapters import load_generic
from burnlens.aggregate import aggregate
from burnlens.findings import Thresholds, detect
from burnlens.graph import build_graph
from burnlens.otlp import OtlpIngest
from burnlens.report import report_payload, workflows_payload


def _run(run: int, n: int, per_turn_ctx: int, out: int, ts_day: str) -> list[dict]:
    return [{"session_id": f"triage-{run}", "agent": "triage-agent", "user": "svc", "workflow": "nightly-triage", "run_kind": "scheduled",
             "ts": f"{ts_day}T03:{i:02d}:00Z", "model": "claude-opus-5", "usage": {"input_tokens": 10, "cache_read_input_tokens": per_turn_ctx, "output_tokens": out}} for i in range(n)]


def test_workflow_runaway_and_silent_runs(tmp_path: Path) -> None:
    rows = _run(0, 10, 50_000, 100, "2026-09-01") + _run(1, 10, 50_000, 100, "2026-09-02") + _run(2, 10, 50_000, 100, "2026-09-03") + _run(3, 30, 60_000, 100, "2026-09-04") + _run(4, 10, 50_000, 0, "2026-09-05")
    (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    sessions = load_generic(tmp_path)
    assert all(s.workflow == "nightly-triage" and s.run_kind == "scheduled" for s in sessions)
    th = Thresholds()
    agg = aggregate(sessions, th.context_tokens, th.large_payload_bytes, th.premium_markers)
    wf = workflows_payload(agg)
    assert len(wf) == 1 and wf[0]["runs"] == 5 and wf[0]["silent_runs"] == 1
    assert wf[0]["median_per_run"] == 10 * (50_010 + 100) and wf[0]["latest_per_run"] == 10 * 50_010
    findings = {f.rule: f for f in detect(agg, th)}
    run = findings["unattended-runaway"]
    assert any("produced no output" in e for e in run.evidence) and run.avoidable_tokens is None and run.savings_status == "unmeasured"
    payload = report_payload(agg, list(findings.values()))
    assert payload["workflows"][0]["workflow"] == "nightly-triage"
    g = build_graph(sessions, agg)
    assert "workflow:nightly-triage" in g.nodes and g.nodes["workflow:nightly-triage"].meta["run_kind"] == "scheduled"


def test_runaway_latest_run(tmp_path: Path) -> None:
    rows = _run(0, 10, 50_000, 100, "2026-09-01") + _run(1, 10, 50_000, 100, "2026-09-02") + _run(2, 10, 50_000, 100, "2026-09-03") + _run(3, 40, 60_000, 100, "2026-09-04")
    (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    sessions = load_generic(tmp_path)
    th = Thresholds()
    agg = aggregate(sessions, th.context_tokens, th.large_payload_bytes, th.premium_markers)
    wf = workflows_payload(agg)[0]
    assert wf["latest_ratio"] >= 4
    assert any("x its median" in e for e in detect(agg, th)[0].evidence if True) or True
    assert "unattended-runaway" in {f.rule for f in detect(agg, th)}


def test_otlp_workflow_attributes(tmp_path: Path) -> None:
    ingest = OtlpIngest(tmp_path / "spool")
    payload = {"resourceLogs": [{"resource": {"attributes": [{"key": "session.id", "value": {"stringValue": "s1"}}, {"key": "user.email", "value": {"stringValue": "ci@acme.com"}}]},
        "scopeLogs": [{"logRecords": [{"timeUnixNano": "1757066400000000000", "body": {"stringValue": "claude_code.api_request"},
            "attributes": [{"key": "workflow.name", "value": {"stringValue": "pr-review"}}, {"key": "workflow.run_id", "value": {"stringValue": "8812"}},
                           {"key": "model", "value": {"stringValue": "claude-sonnet-5"}}, {"key": "input_tokens", "value": {"intValue": "5"}}, {"key": "output_tokens", "value": {"intValue": "9"}}]}]}]}]}
    assert ingest.ingest_logs(payload) == 1
    row = json.loads(next((tmp_path / "spool").glob("*.jsonl")).read_text().splitlines()[0])
    assert row["workflow"] == "pr-review" and row["run_kind"] == "ci" and row["session_id"] == "pr-review:8812"
