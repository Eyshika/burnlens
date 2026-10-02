"""End-to-end test on a synthetic transcript tree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from burnlens.aggregate import aggregate
from burnlens.cli import main
from burnlens.findings import Thresholds, detect
from burnlens.transcripts import load_sessions

BIG_CONTEXT = 200_000
SMALL_CONTEXT = 10_000
IMAGE_B64_BYTES = 8_000


def _assistant(msg_id: str, ts: str, model: str, context: int, output: int, tools: list[dict] | None = None) -> dict:
    return {
        "type": "assistant",
        "timestamp": ts,
        "message": {
            "id": msg_id,
            "model": model,
            "usage": {"input_tokens": 5, "cache_creation_input_tokens": 1_000, "cache_read_input_tokens": context, "output_tokens": output},
            "content": [{"type": "text", "text": "ok"}, *(tools or [])],
        },
    }


def _tool_use(call_id: str, name: str, inp: dict) -> dict:
    return {"type": "tool_use", "id": call_id, "name": name, "input": inp}


def _tool_result(call_id: str, ts: str, content) -> dict:
    return {"type": "user", "timestamp": ts, "message": {"content": [{"type": "tool_result", "tool_use_id": call_id, "content": content}]}}


def _prompt(text: str, ts: str) -> dict:
    return {"type": "user", "timestamp": ts, "message": {"content": text}}


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "projects"
    main_records = [_prompt("fix the bug in server.py", "2026-09-01T10:00:00Z")]
    # 30 bloated turns, each re-reading the same file and an image; the first assistant
    # message is streamed twice with the same id and a partial usage first.
    partial = _assistant("m0", "2026-09-01T10:00:01Z", "claude-opus-5", 0, 0)
    main_records.append(partial)
    for i in range(30):
        ts = f"2026-09-01T10:{i:02d}:05Z"
        read_id, img_id, bash_id = f"r{i}", f"i{i}", f"b{i}"
        tools = [
            _tool_use(read_id, "Read", {"file_path": "/repo/server.py"}),
            _tool_use(img_id, "Read", {"file_path": "/repo/diagram.png"}),
            _tool_use(bash_id, "Bash", {"command": "cat big.log\n# second line"}),
        ]
        main_records.append(_assistant(f"m{i}", ts, "claude-opus-5", BIG_CONTEXT, 100, tools))
        main_records.append(_tool_result(read_id, ts, "x" * 2_000))
        main_records.append(_tool_result(img_id, ts, [{"type": "image", "source": {"type": "base64", "data": "A" * IMAGE_B64_BYTES}}]))
        main_records.append(_tool_result(bash_id, ts, [{"type": "text", "text": "y" * 60_000}]))
    _write(root / "-Users-me-repo" / "sess-main.jsonl", main_records)

    # A subagent on a premium model, credited to the main session.
    sub_records = [_assistant(f"s{i}", f"2026-09-01T10:{i:02d}:30Z", "claude-opus-5", 40_000, 50) for i in range(30)]
    _write(root / "-Users-me-repo" / "sess-main" / "subagents" / "agent-abc.jsonl", sub_records)

    # A small, healthy session on another day (outside a 3-day window from 2026-09-01).
    healthy = [_prompt("hello", "2026-08-20T09:00:00Z"), _assistant("h0", "2026-08-20T09:00:01Z", "claude-sonnet-5", SMALL_CONTEXT, 20)]
    _write(root / "-Users-me-other" / "sess-healthy.jsonl", healthy)
    return root


def test_parse_dedupes_streamed_messages_and_joins_results(tree: Path) -> None:
    sessions = {s.session_id: s for s in load_sessions(tree)}
    main_session = sessions["sess-main"]
    assert len(main_session.turns) == 30, "streamed duplicate of m0 must collapse into one turn"
    assert main_session.turns[0].usage.cache_read_input_tokens == BIG_CONTEXT, "latest usage wins"
    assert main_session.first_prompt == "fix the bug in server.py"
    assert not main_session.is_subagent
    sub = sessions["agent-abc"]
    assert sub.is_subagent and sub.parent_session_id == "sess-main"
    reads = [c for t in main_session.turns for c in t.tool_calls if c.name == "Read"]
    assert sum(c.result_bytes for c in reads if c.file_path == "/repo/server.py") == 30 * 2_000
    assert all(c.result_has_image for c in reads if c.file_path == "/repo/diagram.png")


def test_aggregate_and_findings(tree: Path) -> None:
    th = Thresholds()
    agg = aggregate(load_sessions(tree), th.context_tokens, th.large_payload_bytes)
    assert agg.session_count == 2 and agg.subagent_count == 1
    assert agg.turn_count == 61
    main_stat = next(s for s in agg.sessions if s.session_id == "sess-main")
    assert main_stat.subagent_usage.total == agg.subagent_usage.total > 0
    assert main_stat.share_over_threshold == 1.0
    assert agg.context_histogram[">400k"] == 0 and agg.context_histogram["150-400k"] == 30
    assert agg.file_reads["/repo/server.py"].count == 30
    assert agg.commands["cat big.log"].count == 30
    assert agg.large_payloads["Bash"].count == 30

    rules = {f.rule: f for f in detect(agg, th)}
    assert {"context-bloat", "repeated-reads", "image-rereads", "large-payloads", "subagent-premium-model"} <= set(rules)
    assert "long-session" not in rules, "30 turns is under the long-session threshold"
    assert rules["context-bloat"].severity == "high"


def test_window_filter_drops_old_turns(tree: Path) -> None:
    from datetime import datetime, timezone

    sessions = load_sessions(tree, since=datetime(2026, 8, 25, tzinfo=timezone.utc))
    assert {s.session_id for s in sessions} == {"sess-main", "agent-abc"}


def test_cli_report_and_json(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--root", str(tree), "report"]) == 0
    out = capsys.readouterr().out
    assert "== findings ==" in out and "context-bloat" in out
    assert main(["--root", str(tree), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["turns"] == 61 and payload["findings"]


def test_cli_session_detail(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--root", str(tree), "session", "sess-m"]) == 0
    out = capsys.readouterr().out
    assert "subagents 1" in out and "agent-abc" in out
    assert main(["--root", str(tree), "session", "nope"]) == 2


def test_missing_root_is_an_error(tmp_path: Path) -> None:
    assert main(["--root", str(tmp_path / "missing")]) == 2


def test_price_weighting_moves_cache_write_from_two_percent_to_a_fifth() -> None:
    """The claim the table exists to make, on the real shape of a heavy user's data."""
    from burnlens.config import PriceTable
    from burnlens.model import Usage
    from burnlens.report import price_weighted_payload

    usage = Usage(cache_creation_input_tokens=520_515_338, cache_read_input_tokens=23_766_395_145)
    assert 0.020 < usage.cache_creation_input_tokens / usage.context < 0.022

    payload = price_weighted_payload({"claude-opus-5": usage}, PriceTable())
    assert 0.20 < payload["shares"]["cache_write"] < 0.23
    assert "not a bill" in payload["basis"]


def test_one_ratio_across_a_mixed_fleet_gives_the_wrong_share() -> None:
    """Cache-read is not a fixed fraction of base input across models, so the fleet is weighed per model."""
    from burnlens.config import PriceTable, TokenPrices
    from burnlens.report import price_weighted_payload
    from burnlens.model import Usage

    half = Usage(cache_creation_input_tokens=250_000_000, cache_read_input_tokens=11_000_000_000)
    by_model = {"claude-opus-5": half, "claude-fable-5-1": half}

    flat = price_weighted_payload(by_model, PriceTable())
    per_model = price_weighted_payload(by_model, PriceTable(by_marker={"fable": TokenPrices(cache_read=0.025)}))

    # fable reads are cheaper, so the same tokens put a larger share of the cost on cache write
    assert per_model["shares"]["cache_write"] > flat["shares"]["cache_write"] + 0.05
    assert per_model["models"] == ["claude-fable-5-1", "claude-opus-5"]


def test_config_override_beats_the_shipped_price_table(tmp_path: Path) -> None:
    """A scalar sets the default; a [prices.<marker>] table overrides that model's published ratios."""
    from burnlens.config import load_settings

    cfg = tmp_path / "burnlens.toml"
    cfg.write_text('[prices]\ncache_write = 2.5\n\n[prices."claude-fable"]\ncache_read = 0.9\n')
    table = load_settings(cfg).prices

    assert table.default.cache_write == 2.5
    assert table.for_model("some-unshipped-model").cache_write == 2.5

    fable = table.for_model("claude-fable-5-1")
    assert fable.cache_read == 0.9, "user override must beat the shipped entry"
    assert fable.cache_write == 1.25, "unoverridden fields keep that model's own published ratio"

    assert table.for_model("claude-opus-5").cache_read == 0.1


def test_repo_key_survives_subdirectories_and_credentials(tmp_path: Path) -> None:
    """The repo is the unit findings attach to, so two sessions in one checkout must share a key."""
    from burnlens import repo

    checkout = tmp_path / "work" / "svc"
    (checkout / ".git").mkdir(parents=True)
    (checkout / ".git" / "config").write_text('[remote "origin"]\n\turl = git@github.com:acme/svc.git\n')
    nested = checkout / "src" / "deep"
    nested.mkdir(parents=True)

    repo._cache.clear()
    assert repo.resolve(str(checkout)) == "github.com/acme/svc"
    assert repo.resolve(str(nested)) == "github.com/acme/svc"
    assert repo.normalise_remote("https://user:tok@github.com/acme/svc.git") == "github.com/acme/svc"

    # degrades instead of failing: no checkout, then no path at all
    assert repo.resolve(str(tmp_path / "loose")) == "loose"
    assert repo.resolve("") == "unknown"


def test_sessions_aggregate_by_repo(tmp_path: Path) -> None:
    from burnlens import repo
    from burnlens.aggregate import aggregate
    from burnlens.transcripts import load_sessions

    checkout = tmp_path / "acme"
    (checkout / ".git").mkdir(parents=True)
    (checkout / ".git" / "config").write_text('[remote "origin"]\n\turl = https://github.com/acme/svc.git\n')

    root = tmp_path / "projects"
    for sid in ("s1", "s2"):
        recs = [
            {"type": "user", "cwd": str(checkout), "gitBranch": "main", "timestamp": "2026-09-12T10:00:00Z",
             "message": {"content": "go"}},
            _assistant(f"{sid}m0", "2026-09-12T10:01:00Z", "claude-opus-5", 50_000, 100),
        ]
        _write(root / "-acme" / f"{sid}.jsonl", recs)

    repo._cache.clear()
    sessions = load_sessions(root)
    assert {s.repo for s in sessions} == {"github.com/acme/svc"}
    assert {s.git_branch for s in sessions} == {"main"}

    th = Thresholds()
    agg = aggregate(sessions, th.context_tokens, th.large_payload_bytes)
    assert list(agg.by_repo) == ["github.com/acme/svc"]
    assert agg.by_repo["github.com/acme/svc"].total == agg.usage.total


def test_fable_5_1_cache_read_is_cheaper_than_fable_5() -> None:
    """5.1 ships a cheaper cache read at the same base price, so the longer marker has to win."""
    from burnlens.config import load_published_prices

    table = load_published_prices()
    assert table.for_model("claude-fable-5-1").cache_read == 0.025
    assert table.for_model("claude-fable-5").cache_read == 0.1
    assert table.for_model("claude-opus-5").cache_read == 0.1


def test_edit_then_verify_pairs_are_attributed_not_measured(tmp_path: Path) -> None:
    """An edit followed by a turn that only runs its check costs that turn's whole request."""
    root = tmp_path / "projects"
    records: list[dict] = [_prompt("tighten the loader", "2026-09-02T09:00:00Z")]
    for i in range(6):
        ts_edit, ts_run = f"2026-09-02T09:{i:02d}:00Z", f"2026-09-02T09:{i:02d}:30Z"
        edit_id, bash_id = f"e{i}", f"v{i}"
        records.append(_assistant(f"a{i}", ts_edit, "claude-opus-5", 80_000, 40, [_tool_use(edit_id, "Edit", {"file_path": "/repo/loader.py"})]))
        records.append(_tool_result(edit_id, ts_edit, "edited"))
        records.append(_assistant(f"b{i}", ts_run, "claude-opus-5", 90_000, 10, [_tool_use(bash_id, "Bash", {"command": "pytest -q tests/test_loader.py"})]))
        records.append(_tool_result(bash_id, ts_run, "1 passed"))
    # an edit followed by more real work is not a fusable pair
    records.append(_assistant("a9", "2026-09-02T09:30:00Z", "claude-opus-5", 90_000, 40, [_tool_use("e9", "Edit", {"file_path": "/repo/loader.py"})]))
    records.append(_tool_result("e9", "2026-09-02T09:30:00Z", "edited"))
    records.append(_assistant("b9", "2026-09-02T09:31:00Z", "claude-opus-5", 95_000, 10, [_tool_use("r9", "Read", {"file_path": "/repo/loader.py"})]))
    records.append(_tool_result("r9", "2026-09-02T09:31:00Z", "body"))
    _write(root / "-Users-me-repo" / "sess-fuse.jsonl", records)

    th = Thresholds()
    agg = aggregate(load_sessions(root), th.context_tokens, th.large_payload_bytes, th.premium_markers)
    assert len(agg.fuse_pairs) == 6
    # each verify turn read 5 fresh + 1,000 written + 90,000 cached and wrote 10
    assert {p.avoidable_tokens for p in agg.fuse_pairs} == {91_015}

    finding = next(f for f in detect(agg, th) if f.rule == "edit-then-verify")
    assert finding.avoidable_tokens == 6 * 91_015
    assert finding.savings_status == "attributed" and "not a ledger row" in finding.savings_basis
    assert "pytest -q tests/test_loader.py" in " ".join(finding.evidence)
    assert all(f.rule != "edit-then-verify" for f in detect(agg, Thresholds(fuse_pair_min_count=99)))
