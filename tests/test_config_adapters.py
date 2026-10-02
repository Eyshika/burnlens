"""Config file (the adjustable algorithm) and the generic ingest adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from burnlens.adapters import load, load_generic
from burnlens.aggregate import aggregate
from burnlens.cli import main
from burnlens.coach import coach_prompt
from burnlens.config import ConfigError, load_settings
from burnlens.findings import detect


def test_settings_from_toml_and_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "burnlens.toml"
    cfg.write_text('''
[thresholds]
context_tokens = 90000
live_burn_warn_per_min = 500000
[rules]
disabled = ["long-session"]
[models]
premium_markers = ["opus", "gpt-5"]
[tiers]
lookup = "sonnet"
''')
    s = load_settings(cfg)
    assert s.thresholds.context_tokens == 90000 and s.thresholds.live_burn_warn_per_min == 500000
    assert s.thresholds.is_premium("gpt-5-pro") and not s.thresholds.is_premium("claude-fable-5")
    assert s.disabled_rules == frozenset({"long-session"}) and s.task_tiers == {"lookup": "sonnet"}
    s2 = load_settings(cfg, overrides={"context_tokens": 200000, "large_payload_bytes": None})
    assert s2.thresholds.context_tokens == 200000 and s2.thresholds.large_payload_bytes == 50000
    monkeypatch.setenv("BURNLENS_CONFIG", str(cfg))
    assert load_settings().source == cfg
    bad = tmp_path / "bad.toml"
    bad.write_text("[thresholds]\nnot_a_threshold = 1\n")
    with pytest.raises(ConfigError):
        load_settings(bad)
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "missing.toml")
    c = coach_prompt("find where the loader is", 1000, "claude-opus-5", s.thresholds, s.task_tiers)
    assert c.recommended_tier == "sonnet", "tier table from config wins"


def test_generic_adapter_maps_tools_and_groups_sessions(tmp_path: Path) -> None:
    src = tmp_path / "traces"
    src.mkdir()
    rows = []
    for i in range(20):
        rows.append({
            "session_id": "s1", "project": "checkout", "user": "sam", "agent": "codex", "ts": f"2026-09-05T10:{i:02d}:00Z", "model": "gpt-5",
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 200000, "output_tokens": 50},
            "prompt": "fix the flaky loader test" if i == 0 else None,
            "tools": [{"name": "read_file", "input": {"file_path": "src/loader.py"}, "result_bytes": 60000}, {"name": "shell", "input": {"command": "cat logs/app.log"}, "result_bytes": 90000}],
        })
    rows.append({"session_id": "sub1", "parent_session_id": "s1", "ts": "2026-09-05T10:02:30Z", "model": "gpt-5", "usage": {"input_tokens": 5, "output_tokens": 5}})
    rows.append({"session_id": "s2", "project": "billing", "ts": 1757066400, "model": "gpt-5-mini", "usage": {"input_tokens": 5, "output_tokens": 5}, "tools": [{"name": "custom_tool", "result_bytes": 10}]})
    (src / "day1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nbroken line\n")
    sessions = {s.session_id: s for s in load_generic(src)}
    assert set(sessions) == {"s1", "sub1", "s2"}
    s1 = sessions["s1"]
    assert len(s1.turns) == 20 and s1.first_prompt == "fix the flaky loader test" and s1.project == "checkout"
    names = {c.name for t in s1.turns for c in t.tool_calls}
    assert names == {"Read", "Bash"}, "vendor tool names map onto the canonical set"
    assert sessions["sub1"].is_subagent and sessions["sub1"].parent_session_id == "s1"
    assert sessions["s2"].turns[0].tool_calls[0].name == "custom_tool", "unknown names pass through"
    from burnlens.findings import Thresholds

    th = Thresholds(premium_markers=("gpt-5",))
    agg = aggregate(list(sessions.values()), th.context_tokens, th.large_payload_bytes, th.premium_markers)
    rules = {f.rule for f in detect(agg, th)}
    assert {"context-bloat", "large-payloads"} <= rules, "the same algorithm runs on foreign traces"
    assert load("generic", src) and len(load("generic", src / "day1.jsonl")) == 3


def test_cli_source_and_config_flags(tmp_path: Path, capsys) -> None:
    src = tmp_path / "t.jsonl"
    src.write_text(json.dumps({"session_id": "x", "ts": "2026-09-05T10:00:00Z", "model": "m", "usage": {"input_tokens": 1, "output_tokens": 1}}) + "\n")
    assert main(["--source", "generic", "--root", str(src), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["sessions"] == 1
    assert main(["--source", "generic", "--root", str(tmp_path / "nope")]) == 2


def test_load_all_merges_sources_and_reports_by_application_and_person(tmp_path: Path) -> None:
    from burnlens.adapters import load_all
    from burnlens.report import report_payload
    from burnlens.findings import Thresholds
    from test_profiler import _assistant, _prompt, _write

    cc = tmp_path / "cc"
    _write(cc / "-p" / "sess-a.jsonl", [_prompt("hi", "2026-09-05T09:00:00Z"), _assistant("a0", "2026-09-05T09:00:01Z", "claude-opus-5", 5_000, 10)])
    gen = tmp_path / "gen"
    gen.mkdir()
    rows = [
        {"session_id": "sess-a", "agent": "codex", "user": "sam", "project": "checkout", "ts": "2026-09-05T10:00:00Z", "model": "gpt-5", "usage": {"input_tokens": 1, "cache_read_input_tokens": 300_000, "output_tokens": 1}},
        {"session_id": "b", "agent": "support-bot", "user": "prod", "ts": "2026-09-05T10:00:00Z", "model": "claude-sonnet-5", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    (gen / "t.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    sessions = load_all("claude-code", cc, [("generic", gen)])
    ids = sorted(s.session_id for s in sessions)
    assert ids == ["b", "codex:sess-a", "sess-a"], "id clash across sources is disambiguated by application"
    th = Thresholds()
    agg = aggregate(sessions, th.context_tokens, th.large_payload_bytes, th.premium_markers)
    assert set(agg.by_agent) == {"claude-code", "codex", "support-bot"} and set(agg.by_user) == {"sam", "prod"}
    payload = report_payload(agg, [])
    people = {p["user"]: p for p in payload["people"]}
    assert people["sam"]["agents"] == ["codex"] and people["sam"]["bloated_share"] == 1.0 and people["prod"]["health_median"] == 100


def test_report_payload_derives_repo_from_path_like_project(tmp_path: Path) -> None:
    from burnlens.findings import Thresholds
    from burnlens.report import report_payload

    src = tmp_path / "t.jsonl"
    src.write_text(
        json.dumps({
            "session_id": "x",
            "project": "/srv/repos/checkout-service",
            "ts": "2026-09-05T10:00:00Z",
            "model": "gpt-5",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }) + "\n" +
        json.dumps({
            "session_id": "y",
            "project": "checkout-service/",
            "ts": "2026-09-05T10:01:00Z",
            "model": "gpt-5",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }) + "\n" +
        json.dumps({
            "session_id": "z",
            "project": "C:\\work\\checkout-service\\",
            "ts": "2026-09-05T10:02:00Z",
            "model": "gpt-5",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }) + "\n"
    )
    th = Thresholds()
    agg = aggregate(load_generic(src), th.context_tokens, th.large_payload_bytes, th.premium_markers)
    payload = report_payload(agg, [])
    assert set(payload["by_project"]) == {"/srv/repos/checkout-service", "checkout-service/", "C:\\work\\checkout-service\\"}
    assert set(payload["by_repo"]) == {"/srv/repos/checkout-service", "checkout-service", "C:/work/checkout-service"}


def test_cli_propagates_hook_tiers_and_install_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    from unittest.mock import Mock

    config = tmp_path / "team.toml"
    config.write_text('[tiers]\nlookup = "team-model"\n')
    hook = Mock(return_value={})
    install = Mock(return_value="installed")
    monkeypatch.setattr("burnlens.cli.run_hook", hook)
    monkeypatch.setattr("burnlens.cli.install_hooks", install)
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    assert main(["hook", "--config", str(config)]) == 0
    assert hook.call_args.kwargs["task_tiers"] == {"lookup": "team-model"}
    assert main(["install-hooks", "--config", str(config)]) == 0
    assert install.call_args.kwargs["config_path"] == config
