"""Health score, tokens-per-commit and the handoff brief."""

from __future__ import annotations

from pathlib import Path

from burnlens.aggregate import SessionStat, aggregate, health_score
from burnlens.cli import main
from burnlens.findings import Thresholds
from burnlens.handoff import build_handoff
from burnlens.model import Usage
from burnlens.report import commits_in_session, report_payload
from burnlens.transcripts import load_sessions

from test_profiler import _assistant, _prompt, _tool_result, _tool_use, _write, tree  # noqa: F401


def _stat(**over) -> SessionStat:
    base = dict(session_id="s", project="p", model="claude-opus-5", agent="claude-code", user="", workflow="", run_kind="interactive", is_subagent=False, parent_session_id=None, first_prompt="", turns=100,
                usage=Usage(), peak_context=0, turns_over_threshold=0)
    from datetime import datetime, timezone

    base.update(start=datetime.now(timezone.utc), end=datetime.now(timezone.utc))
    base.update(over)
    return SessionStat(**base)


def test_health_score_bounds_and_penalties() -> None:
    assert health_score(_stat()) == 100
    assert health_score(_stat(turns_over_threshold=100)) == 50, "all turns bloated costs the full bloat weight"
    worst = _stat(turns_over_threshold=100, payload_bytes=100 * 20_000, repeated_read_calls=50,
                  subagent_usage=Usage(cache_read_input_tokens=10), premium_subagent_tokens=10)
    assert health_score(worst) == 0
    assert 0 <= health_score(_stat(repeated_read_calls=10)) < 100


def test_aggregate_fills_health_inputs(tree: Path) -> None:
    th = Thresholds()
    agg = aggregate(load_sessions(tree), th.context_tokens, th.large_payload_bytes)
    main = next(s for s in agg.sessions if s.session_id == "sess-main")
    assert main.repeated_read_calls == 58, "30 reads of two files: 29 repeats each"
    assert main.payload_bytes == 30 * 60_000
    assert main.premium_subagent_tokens == main.subagent_usage.total > 0
    assert main.health < 30
    healthy = next(s for s in agg.sessions if s.session_id == "sess-healthy")
    assert healthy.health == 100
    payload = report_payload(agg, [])
    assert payload["health_median"] is not None and payload["sessions_detail"][0]["health"] == main.health


def test_commits_split_tokens(tmp_path: Path) -> None:
    root = tmp_path / "projects"
    recs = [_prompt("ship it", "2026-09-01T10:00:00Z")]
    for i in range(6):
        ts = f"2026-09-01T10:0{i}:00Z"
        cmd = "git add -A && git commit -m 'step'" if i in (2, 5) else "pytest -q"
        recs.append(_assistant(f"m{i}", ts, "claude-opus-5", 10_000, 10, [_tool_use(f"b{i}", "Bash", {"command": cmd})]))
        recs.append(_tool_result(f"b{i}", ts, "ok"))
    _write(root / "-p" / "sess-c.jsonl", recs)
    session = load_sessions(root)[0]
    commits = commits_in_session(session)
    assert [c["turn"] for c in commits] == [2, 5] and [c["turns"] for c in commits] == [3, 3]
    assert commits[0]["tokens"] == 3 * (10_000 + 1_000 + 5 + 10)


def test_handoff_brief_and_cli(tree: Path, capsys) -> None:
    sessions = {s.session_id: s for s in load_sessions(tree)}
    brief = build_handoff(sessions["sess-main"], [sessions["agent-abc"]])
    assert "**Goal:** fix the bug in server.py" in brief
    assert "/repo/server.py" in brief and "cat big.log" in brief
    assert "Paste this to start the new session" in brief and "line ranges" in brief
    assert main(["--root", str(tree), "handoff", "sess-m"]) == 0
    assert "# Handoff from session" in capsys.readouterr().out
