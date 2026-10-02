"""Prompt coaching, habits, and the UserPromptSubmit hook."""

from __future__ import annotations

import json
from pathlib import Path

from burnlens.aggregate import aggregate
from burnlens.coach import classify_task, coach_prompt, habits
from burnlens.findings import Thresholds, detect
from burnlens.hook import install_hooks, run_hook
from burnlens.transcripts import load_sessions

from test_profiler import _assistant, _write, tree  # noqa: F401

TH = Thresholds()


def test_classify_task() -> None:
    assert classify_task("where is the retry logic defined") == "lookup"
    assert classify_task("research how others handle rate limits, compare options") == "research"
    assert classify_task("why does the export script double count cache reads") == "debug"
    assert classify_task("refactor step_model.py into smaller modules") == "design"
    assert classify_task("write tests for the loader") == "tests"
    assert classify_task("hello") == "general"


def test_coach_recommends_cheaper_tier_and_flags_waste() -> None:
    c = coach_prompt("find where CANONICAL_WH is defined in the whole repo", context_now=20_000, current_model="claude-opus-5", th=TH)
    assert c.task == "lookup" and c.recommended_tier == "haiku" and not c.stop
    assert any("Haiku is a candidate" in m and "not been measured" in m for m in c.messages)
    assert all("gives the same answer" not in m for m in c.messages)
    assert any("whole repo" in m.lower() for m in c.messages)
    assert c.rewrite.startswith("find where CANONICAL_WH") and "Files:" in c.rewrite
    assert any("model='haiku'" in n for n in c.agent_notes)
    lean = coach_prompt("fix the off-by-one in burnlens/aggregate.py line 142, run pytest -q", context_now=20_000, current_model="claude-sonnet-5", th=TH)
    assert lean.messages == [] and lean.rewrite == ""


def test_red_context_stops_only_with_waste() -> None:
    clean = coach_prompt("add the export endpoint to api/export.py", context_now=400_000, current_model="claude-opus-5", th=TH)
    assert not clean.stop and clean.messages[0].startswith("Expensive") and "mid-feature" in clean.messages[0] and clean.rewrite == ""
    c = coach_prompt("continue", context_now=400_000, current_model="claude-opus-5", th=TH, waste=["server.py read 6x"])
    assert c.stop and c.messages[0].startswith("STOP") and "server.py read 6x" in c.messages[0] and "not starting over" in c.messages[0]
    assert any("what changed" in m for m in c.messages)


def test_habits_from_tree(tree: Path) -> None:
    from datetime import datetime, timezone

    sessions = load_sessions(tree)
    agg = aggregate(sessions, TH.context_tokens, TH.large_payload_bytes)
    rows = habits(sessions, agg, TH, now=datetime(2026, 9, 3, tzinfo=timezone.utc))
    keys = {h.key: h for h in rows}
    assert {"bloat", "reread", "image-reread", "payload"} <= set(keys)
    assert "premium-subagent" not in keys, "a premium child without a known task does not establish a cheaper-model opportunity"
    assert keys["reread"].this_week == 29 and keys["reread"].last_week == 0 and keys["reread"].trend == "new"
    assert keys["payload"].avoidable_tokens is None and "tail" in keys["payload"].say_this
    assert keys["payload"].as_dict()["savings_status"] == "unmeasured"
    findings = detect(agg, TH)
    assert findings and all(f.avoidable_tokens is None for f in findings)


def test_prompt_hook_coaches_and_blocks_only_in_strict(tmp_path: Path) -> None:
    transcript = tmp_path / "s.jsonl"
    _write(transcript, [_assistant("m1", "2026-09-01T10:01:00Z", "claude-opus-5", 400_000, 5)])
    payload = {"hook_event_name": "UserPromptSubmit", "session_id": "quiet-1", "prompt": "find the config loader", "transcript_path": str(transcript)}
    state = tmp_path / "state"
    out = run_hook(json.dumps(payload), TH, events_path=tmp_path / "e.jsonl", state_dir=state)
    assert out["systemMessage"].startswith("Burnlens: Expensive"), "big clean context is a cost note, said once"
    again = run_hook(json.dumps({**payload, "prompt": "now add the endpoint to api/export.py"}), TH, events_path=tmp_path / "e.jsonl", state_dir=state)
    assert "systemMessage" not in again, "the cost note is not repeated on the next prompt"
    third = run_hook(json.dumps(payload), TH, events_path=tmp_path / "e.jsonl", state_dir=state)
    assert "Expensive" not in third.get("systemMessage", "") and "Haiku" in third["systemMessage"], "prompt-specific advice still shows"
    assert "Haiku" in out["systemMessage"]
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit" and "haiku" in out["hookSpecificOutput"]["additionalContext"]
    assert "decision" not in out, "default mode warns, never blocks the prompt"
    strict = run_hook(json.dumps(payload), TH, events_path=tmp_path / "e.jsonl", strict=True, state_dir=state)
    assert "decision" not in strict, "strict does not block without waste evidence"
    wasteful = tmp_path / "w.jsonl"
    rows = [_assistant("m1", "2026-09-01T10:01:00Z", "claude-opus-5", 400_000, 5, [{"type": "tool_use", "id": f"r{i}", "name": "Read", "input": {"file_path": "/repo/server.py"}} for i in range(4)])]
    _write(wasteful, rows)
    strict = run_hook(json.dumps({**payload, "session_id": "waste-1", "transcript_path": str(wasteful)}), TH, events_path=tmp_path / "e.jsonl", strict=True, state_dir=state)
    assert strict["decision"] == "block" and strict["reason"].startswith("STOP") and "server.py read 4x" in strict["reason"]
    logged = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
    assert logged[0]["tool"] == "UserPromptSubmit" and logged[-1]["permission"] == "block" and "allow" in {e["permission"] for e in logged}
    quiet = run_hook(json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": "fix aggregate.py line 12", "transcript_path": str(transcript)}), TH, events_path=tmp_path / "e.jsonl")
    assert "Heads up" not in quiet.get("systemMessage", "") or "STOP" in quiet["systemMessage"]


def test_install_hooks_registers_prompt_event(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    install_hooks(settings)
    data = json.loads(settings.read_text())
    assert "UserPromptSubmit" in data["hooks"] and "matcher" not in data["hooks"]["UserPromptSubmit"][0]
    install_hooks(settings, remove=True)
    assert "hooks" not in json.loads(settings.read_text())


def test_complex_and_ambiguous_prompts_preserve_current_model() -> None:
    for prompt in (
        "find the root cause of a deadlock and design a safe fix",
        "fix a failing test caused by a production race condition",
        "find and summarize the loader",
        "hello",
    ):
        result = coach_prompt(prompt, 1000, "selected-premium", TH, {"debug": "cheap", "general": "cheap"})
        assert result.recommended_tier == "selected-premium"
        assert not any("candidate model=" in note for note in result.agent_notes)
    assert classify_task("find the root cause of a deadlock and design a safe fix") == "debug"
    assert classify_task("find and summarize the loader") == "general"


def test_habits_exclude_old_and_future_events(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    root = tmp_path / "history"
    _write(root / "-p" / "old.jsonl", [
        _assistant("old", "2026-08-01T10:00:00Z", "claude-opus-5", 400_000, 5),
        _assistant("future", "2026-10-01T10:00:00Z", "claude-opus-5", 400_000, 5),
    ])
    sessions = load_sessions(root)
    agg = aggregate(sessions, TH.context_tokens, TH.large_payload_bytes)
    assert habits(sessions, agg, TH, now=datetime(2026, 9, 10, tzinfo=timezone.utc)) == []


def test_premium_lookup_habit_has_counts_without_invented_savings(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    root = tmp_path / "history"
    _write(root / "-p" / "lookup.jsonl", [_assistant(
        "lookup", "2026-09-09T10:00:00Z", "claude-opus-5", 1000, 5,
        [{"type": "tool_use", "id": "agent", "name": "Agent", "input": {"prompt": "find the loader", "model": "opus"}}],
    )])
    sessions = load_sessions(root)
    agg = aggregate(sessions, TH.context_tokens, TH.large_payload_bytes)
    rows = habits(sessions, agg, TH, now=datetime(2026, 9, 10, tzinfo=timezone.utc))
    result = next(row for row in rows if row.key == "premium-subagent")
    assert result.this_week == 1 and result.last_week == 0
    assert result.avoidable_tokens is None


def test_habit_reads_respect_ranges_and_unknown_agent_tasks(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    tools = [
        {"type": "tool_use", "id": str(offset), "name": "Read", "input": {"file_path": "/repo/a.py", "offset": offset, "limit": 10}}
        for offset in (1, 11, 21)
    ]
    tools.append({"type": "tool_use", "id": "unknown", "name": "Agent", "input": {"prompt": "continue", "model": "opus"}})
    root = tmp_path / "history"
    _write(root / "-p" / "ranges.jsonl", [_assistant("read", "2026-09-09T10:00:00Z", "claude-opus-5", 1000, 5, tools)])
    sessions = load_sessions(root)
    agg = aggregate(sessions, TH.context_tokens, TH.large_payload_bytes)
    assert habits(sessions, agg, TH, now=datetime(2026, 9, 10, tzinfo=timezone.utc)) == []


def _turn(model: str, write: int, read: int, index: int = 0):
    from datetime import datetime, timezone

    from burnlens.model import Turn, Usage

    return Turn(
        message_id=f"m{index}",
        timestamp=datetime(2026, 9, 12, 10, index % 60, tzinfo=timezone.utc),
        model=model,
        usage=Usage(input_tokens=5, cache_creation_input_tokens=write, cache_read_input_tokens=read, output_tokens=100),
    )


def test_cache_reset_needs_a_cached_prefix_not_just_a_big_write() -> None:
    from burnlens.coach import cache_resets

    warm = [_turn("claude-opus-5", 1_000, 300_000, i) for i in range(4)]
    switched = warm + [_turn("claude-sonnet-5", 310_000, 0, 4)]
    resets = cache_resets(switched, TH)
    assert len(resets) == 1
    assert resets[0].turn_index == 4 and resets[0].rewritten == 310_000
    assert resets[0].previously_read == 300_000
    assert "model changed" in resets[0].cause

    # a session that writes its prefix on the first turn is not a reset: nothing was cached yet
    cold = [_turn("claude-opus-5", 310_000, 0, 0), *(_turn("claude-opus-5", 1_000, 300_000, i) for i in range(1, 4))]
    assert cache_resets(cold, TH) == []


def test_coach_reports_a_cache_reset_without_claiming_savings() -> None:
    from burnlens.coach import cache_resets

    turns = [_turn("claude-opus-5", 1_000, 300_000, i) for i in range(3)] + [_turn("claude-sonnet-5", 310_000, 0, 3)]
    c = coach_prompt(
        "fix the off-by-one in burnlens/aggregate.py line 142",
        context_now=310_000,
        current_model="claude-sonnet-5",
        th=TH,
        resets=cache_resets(turns, TH),
    )
    note = next(m for m in c.messages if "Cache reset" in m)
    assert "310,000 tokens were re-written" in note
    assert "cache-write class instead of cache-read" in note
    assert "saved" not in note.lower() and "$" not in note
    assert any("keep the model and effort as they are" in n for n in c.agent_notes)


def test_compaction_breakeven_depends_on_the_model_cache_prices() -> None:
    from burnlens.coach import compaction_breakeven
    from burnlens.config import load_published_prices

    prices = load_published_prices()
    opus = compaction_breakeven(400_000, prices.for_model("claude-opus-5"), TH)
    assert opus is not None
    # 21k retained; one-off = 11.5*21k re-cache + 400k summarising read + 50*1k summary output
    assert opus.cache_write_read_ratio == 12.5
    assert opus.saved_per_request == 379_000
    assert opus.one_off_read_equivalents == 691_500
    assert round(opus.breakeven_requests, 1) == 1.8

    # fable 5.1 sells cache reads at a quarter of the usual rate, so a rewrite takes longer to repay
    fable = compaction_breakeven(400_000, prices.for_model("claude-fable-5-1"), TH)
    assert fable is not None and fable.cache_write_read_ratio == 50.0
    assert round(fable.breakeven_requests, 1) == 4.3
    assert fable.breakeven_requests > opus.breakeven_requests

    assert compaction_breakeven(30_000, prices.for_model("claude-opus-5"), TH) is None
    assert opus.window_headroom_requests is None and opus.worth_it is None
    with_growth = compaction_breakeven(400_000, prices.for_model("claude-opus-5"), TH, growth_per_turn=8_000)
    assert with_growth is not None and with_growth.window_headroom_requests == 75 and with_growth.worth_it is True


def test_red_zone_note_carries_the_payback_number() -> None:
    c = coach_prompt("add the export endpoint to api/export.py", context_now=400_000, current_model="claude-opus-5", th=TH, growth_per_turn=8_000)
    assert c.compaction is not None and c.compaction.savings_status == "projected"
    note = c.messages[0]
    # the prefix is load-bearing: hook.py suppresses repeats of the cost note by matching it
    assert note.startswith("Expensive") and "mid-feature" in note
    assert "pays for itself after about 2 more turns" in note and "about 75 more turns before the window fills" in note
    assert "assumptions, not measurements" in c.compaction.basis


def test_diagnostic_command_without_a_limiter_is_coached() -> None:
    c = coach_prompt("run the full pytest suite and tell me what broke", context_now=20_000, current_model="claude-opus-5", th=TH)
    assert any("failing lines quoted back" in m for m in c.messages)
    assert any("first fatal line" in n for n in c.agent_notes)
    quiet = coach_prompt("run pytest -q on burnlens/aggregate.py", context_now=20_000, current_model="claude-opus-5", th=TH)
    assert all("failing lines quoted back" not in m for m in quiet.messages)


def test_dashboard_reads_only_fields_the_coach_returns() -> None:
    """The coach card renders payback from /api/coach; a rename here breaks it silently."""
    import re as _re
    from pathlib import Path as _Path

    c = coach_prompt("add the export endpoint", context_now=400_000, current_model="claude-opus-5", th=TH, growth_per_turn=8_000)
    payload = c.as_dict()["compaction"]
    assert isinstance(payload, dict)
    source = (_Path(__file__).resolve().parent.parent / "burnlens" / "static" / "app.js").read_text()
    for start, end in (("function paybackRow", "async function coachPrompt"), ("function paybackTag", "function renderLive")):
        body = source[source.index(start) : source.index(end)]
        assert set(_re.findall(r"\bp\.(\w+)", body)) <= set(payload), start
    assert "paybackRow(c.compaction)" in source and "paybackTag(s.compaction)" in source
