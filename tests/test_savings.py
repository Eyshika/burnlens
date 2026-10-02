"""Observed activity must not be presented as counterfactual savings."""
from datetime import datetime, timezone
from pathlib import Path

from burnlens.aggregate import aggregate
from burnlens.findings import Thresholds, detect
from burnlens.model import Session, ToolCall, Turn, Usage
from burnlens.teacher import lessons


def _session(sid: str, reads: list[ToolCall], subagent: bool = False) -> Session:
    return Session(sid, "/repo", Path(f"/{sid}.jsonl"), "parent" if subagent else None, "inspect", turns=[
        Turn(sid, datetime(2026, 9, 1, tzinfo=timezone.utc), "claude-opus", Usage(10, output_tokens=1), reads)
    ])


def _read(offset: int = 1, path: str = "/repo/a.py") -> ToolCall:
    return ToolCall("read", "Read", {"file_path": path, "offset": offset, "limit": 10}, 100)


def test_cross_session_reads_do_not_count_as_repeated_reads() -> None:
    sessions = [_session(str(i), [_read()]) for i in range(10)]
    agg = aggregate(sessions, 150_000, 50_000)
    assert agg.usage.total == 110
    assert "repeated-reads" not in {finding.rule for finding in detect(agg, Thresholds())}
    maps = [lesson for lesson in lessons(sessions) if lesson.feature == "project map"]
    assert len(maps) == 1
    assert maps[0].avoidable_tokens is None
    assert maps[0].as_dict()["savings_status"] == "unmeasured"


def test_different_ranges_are_not_repeat_reads() -> None:
    agg = aggregate([_session("ranges", [_read(i) for i in range(10)])], 150_000, 50_000)
    assert agg.sessions[0].repeated_read_calls == 0
    assert "repeated-reads" not in {finding.rule for finding in detect(agg, Thresholds())}


def test_same_range_is_qualified_opportunity_with_attribution() -> None:
    agg = aggregate([_session("repeat", [_read()] * 10)], 150_000, 50_000)
    finding = next(finding for finding in detect(agg, Thresholds()) if finding.rule == "repeated-reads")
    assert "session=repeat" in finding.evidence[0]
    assert "offset=1 limit=10" in finding.evidence[0]
    assert "revisions" in finding.evidence[-1]
    assert finding.avoidable_tokens is None
    assert finding.as_dict()["savings_basis"]


def test_model_switch_does_not_eliminate_tokens() -> None:
    agg = aggregate([_session("premium", [], subagent=True)], 150_000, 50_000)
    finding = next(finding for finding in detect(agg, Thresholds(subagent_min_tokens=1)) if finding.rule == "subagent-premium-model")
    assert finding.avoidable_tokens is None
    assert "quality" in finding.suggestion


def test_image_bytes_are_not_token_savings() -> None:
    agg = aggregate([_session("images", [_read(path="/repo/a.png")] * 2)], 150_000, 50_000)
    finding = next(finding for finding in detect(agg, Thresholds()) if finding.rule == "image-rereads")
    assert finding.avoidable_tokens is None
