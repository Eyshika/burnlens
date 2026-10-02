"""Render aggregates and findings as text tables or JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .aggregate import Aggregates, SessionStat, Tally
from .config import PriceTable
from .findings import Finding
from .model import Session, Usage

USAGE_COLUMNS = ("input", "cache_write", "cache_read", "output", "total")
PRICE_BASIS = "per-model class weights, fresh input = 1.0; compares classes, not a bill"


def render_report(agg: Aggregates, findings: list[Finding], top: int, prices: PriceTable | None = None) -> str:
    prices = prices or PriceTable()
    lines: list[str] = []
    lines.append(_header(agg))
    lines.append("")
    lines.append("== token split ==")
    lines.extend(_usage_split(agg.usage))
    lines.append("")
    lines.append("== price-weighted token split ==")
    lines.append(PRICE_BASIS)
    lines.extend(_price_split(agg.by_model, prices))
    lines.append("")
    lines.append("== per day ==")
    lines.extend(_usage_table(agg.by_day, "day"))
    lines.append("")
    lines.append("== per model ==")
    lines.extend(_usage_table(agg.by_model, "model"))
    lines.append("")
    lines.append("== main thread vs subagents ==")
    lines.extend(_main_vs_sub(agg))
    lines.append("")
    lines.append("== per repo ==")
    lines.extend(_usage_table(dict(list(agg.by_repo.items())[:top]), "repo"))
    lines.append("")
    lines.append("== per application ==")
    lines.extend(_usage_table(agg.by_agent, "application"))
    if agg.by_user:
        lines.append("")
        lines.append("== per person ==")
        lines.extend(_usage_table(dict(list(agg.by_user.items())[:top]), "person"))
    lines.append("")
    lines.append("== per project ==")
    lines.extend(_usage_table(dict(list(agg.by_project.items())[:top]), "project"))
    workflows = workflows_payload(agg)
    if workflows:
        lines.append("")
        lines.append("== unattended agents (scheduled / CI) ==")
        rows = [(w["workflow"], w["agent"], f"{w['runs']:,}", f"{w['tokens']:,}", f"{w['median_per_run']:,}", f"{w['max_ratio'] or 0:.1f}x", f"{w['runaway_runs']}", f"{w['silent_runs']}") for w in workflows[:top]]
        lines.extend(_table(("workflow", "application", "runs", "tokens", "median/run", "worst run vs median", "runaway runs", "silent runs"), rows))
    lines.append("")
    lines.append("== context size per turn ==")
    lines.extend(_histogram(agg))
    lines.append("")
    lines.append("== bytes returned into context, by tool ==")
    lines.extend(_tally_table(list(agg.tools.items())[:top], "tool"))
    lines.append("")
    lines.append(f"== top {top} files Read into context ==")
    lines.extend(_tally_table([(_short(p), t) for p, t in list(agg.file_reads.items())[:top]], "file"))
    lines.append("")
    lines.append(f"== top {top} shell commands by output size ==")
    lines.extend(_tally_table([(c[:72], t) for c, t in list(agg.commands.items())[:top]], "command"))
    lines.append("")
    lines.append(f"== top {top} sessions ==")
    lines.extend(_session_table([s for s in agg.sessions if not s.is_subagent][:top]))
    lines.append("")
    lines.append("== findings ==")
    lines.extend(_findings(findings))
    return "\n".join(lines)


def render_sessions(agg: Aggregates, top: int) -> str:
    return "\n".join([_header(agg), "", *_session_table([s for s in agg.sessions if not s.is_subagent][:top])])


def render_session_detail(session: Session, children: list[Session], top: int, prices: PriceTable | None = None) -> str:
    """Timeline of one session: where the context grew and what the tools pulled in."""
    prices = prices or PriceTable()
    lines = [
        f"session  {session.session_id}",
        f"project  {session.project}",
        f"prompt   {session.first_prompt}",
        f"turns    {len(session.turns)}   subagents {len(children)}",
        f"window   {session.start:%Y-%m-%d %H:%M} -> {session.end:%Y-%m-%d %H:%M} UTC",
        "",
        "== token split (this session) ==",
        *_usage_split(session.usage),
        "",
        "== price-weighted token split (this session) ==",
        PRICE_BASIS,
        *_price_split(_session_by_model(session), prices),
        "",
        "== context growth (every Nth turn) ==",
    ]
    step = max(1, len(session.turns) // 20)
    for index in range(0, len(session.turns), step):
        turn = session.turns[index]
        bar = "#" * min(60, turn.usage.context // 20_000)
        lines.append(f"{index:>5}  {turn.timestamp:%m-%d %H:%M}  {turn.usage.context:>9,}  {bar}")
    tools: dict[str, Tally] = {}
    files: dict[str, Tally] = {}
    for turn in session.turns:
        for call in turn.tool_calls:
            tools.setdefault(call.name, Tally()).add(call.result_bytes)
            if call.name == "Read" and call.file_path:
                files.setdefault(call.file_path, Tally()).add(call.result_bytes)
    lines.append("")
    lines.append("== bytes returned into context, by tool ==")
    lines.extend(_tally_table(sorted(tools.items(), key=lambda kv: kv[1].bytes, reverse=True)[:top], "tool"))
    lines.append("")
    lines.append(f"== top {top} files Read ==")
    lines.extend(_tally_table([(_short(p), t) for p, t in sorted(files.items(), key=lambda kv: kv[1].bytes, reverse=True)[:top]], "file"))
    if children:
        lines.append("")
        lines.append("== subagents ==")
        rows = [(c.session_id[:20], c.turns[0].model if c.turns else "?", f"{len(c.turns):,}", f"{c.usage.total:,}") for c in children]
        lines.extend(_table(("agent", "model", "turns", "tokens"), rows))
    return "\n".join(lines)


def render_json(agg: Aggregates, findings: list[Finding], prices: PriceTable | None = None) -> str:
    return json.dumps(report_payload(agg, findings, prices), indent=2)


def report_payload(agg: Aggregates, findings: list[Finding], prices: PriceTable | None = None) -> dict[str, object]:
    """Everything the report prints, as plain data for JSON or the dashboard."""
    prices = prices or PriceTable()
    return {
        "window": {"start": _iso(agg.window_start), "end": _iso(agg.window_end)},
        "price_weighted": price_weighted_payload(agg.by_model, prices),
        "sessions": agg.session_count,
        "subagents": agg.subagent_count,
        "turns": agg.turn_count,
        "usage": agg.usage.as_dict(),
        "main_usage": agg.main_usage.as_dict(),
        "subagent_usage": agg.subagent_usage.as_dict(),
        "by_day": {k: v.as_dict() for k, v in agg.by_day.items()},
        "by_model": {k: v.as_dict() for k, v in agg.by_model.items()},
        "subagent_by_model": {k: v.as_dict() for k, v in agg.subagent_by_model.items()},
        "by_project": {k: v.as_dict() for k, v in agg.by_project.items()},
        "by_repo": {k: v.as_dict() for k, v in agg.by_repo.items()},
        "by_agent": {k: v.as_dict() for k, v in agg.by_agent.items()},
        "by_user": {k: v.as_dict() for k, v in agg.by_user.items()},
        "people": people_payload(agg),
        "workflows": workflows_payload(agg),
        "heatmap": agg.heatmap,
        "context_histogram": agg.context_histogram,
        "tools": {k: vars(v) for k, v in agg.tools.items()},
        "file_reads": {_short(k): vars(v) for k, v in agg.file_reads.items()},
        "commands": {k: vars(v) for k, v in agg.commands.items()},
        "sessions_detail": [_session_dict(s) for s in agg.sessions],
        "health_median": _median([s.health for s in agg.sessions if not s.is_subagent]),
        "findings": [f.as_dict() for f in findings],
    }


def session_payload(session: Session, children: list[Session], max_points: int) -> dict[str, object]:
    """One session for the dashboard drawer: timeline, tools, files, subagents."""
    tools: dict[str, Tally] = {}
    files: dict[str, Tally] = {}
    commands: dict[str, Tally] = {}
    for turn in session.turns:
        for call in turn.tool_calls:
            tools.setdefault(call.name, Tally()).add(call.result_bytes)
            if call.name == "Read" and call.file_path:
                files.setdefault(_short(call.file_path), Tally()).add(call.result_bytes)
            head = call.command_head
            if call.name == "Bash" and head:
                commands.setdefault(head, Tally()).add(call.result_bytes)
    step = max(1, len(session.turns) // max_points)
    timeline = [
        {"turn": i, "ts": _iso(t.timestamp), "context": t.usage.context, "output": t.usage.output_tokens, "tools": [c.name for c in t.tool_calls]}
        for i, t in enumerate(session.turns)
        if i % step == 0 or i == len(session.turns) - 1
    ]
    return {
        "session_id": session.session_id,
        "project": session.project,
        "first_prompt": session.first_prompt,
        "last_prompt": session.last_prompt,
        "commits": commits_in_session(session),
        "start": _iso(session.start),
        "end": _iso(session.end),
        "turns": len(session.turns),
        "usage": session.usage.as_dict(),
        "peak_context": session.peak_context,
        "timeline": timeline,
        "tools": {k: vars(v) for k, v in sorted(tools.items(), key=lambda kv: kv[1].bytes, reverse=True)},
        "files": {k: vars(v) for k, v in sorted(files.items(), key=lambda kv: kv[1].bytes, reverse=True)},
        "commands": {k: vars(v) for k, v in sorted(commands.items(), key=lambda kv: kv[1].bytes, reverse=True)},
        "subagents": [
            {"agent_id": c.session_id, "model": c.turns[0].model if c.turns else "?", "turns": len(c.turns), "usage": c.usage.as_dict()}
            for c in children
        ],
    }


def workflows_payload(agg: Aggregates) -> list[dict[str, object]]:
    """Unattended agents: per workflow, runs, tokens, median per run, latest run vs median, runs that produced nothing."""
    rows: dict[str, list[SessionStat]] = {}
    for s in agg.sessions:
        if s.is_subagent or not s.workflow:
            continue
        rows.setdefault(s.workflow, []).append(s)
    out = []
    for name, runs in rows.items():
        runs.sort(key=lambda r: r.start)
        per_run = [r.usage.total + r.subagent_usage.total for r in runs]
        ordered = sorted(per_run)
        median = ordered[len(ordered) // 2]
        latest = per_run[-1]
        silent = sum(1 for r in runs if r.usage.output_tokens == 0)
        runaway = [t for t in per_run if median and t >= 2 * median]
        out.append({
            "workflow": name, "agent": runs[-1].agent, "run_kind": runs[-1].run_kind, "runs": len(runs), "tokens": sum(per_run),
            "median_per_run": median, "latest_per_run": latest, "latest_ratio": round(latest / median, 2) if median else None,
            "max_ratio": round(max(per_run) / median, 2) if median else None, "runaway_runs": len(runaway),
            "silent_runs": silent, "health_median": sorted(r.health for r in runs)[len(runs) // 2], "last_run": _iso(runs[-1].start),
        })
    out.sort(key=lambda r: r["tokens"], reverse=True)
    return out


def people_payload(agg: Aggregates) -> list[dict[str, object]]:
    """Per person: tokens, sessions, applications used, median health, bloated share. The team's learning curve."""
    rows: dict[str, dict[str, object]] = {}
    for s in agg.sessions:
        if s.is_subagent or not s.user:
            continue
        row = rows.setdefault(s.user, {"user": s.user, "tokens": 0, "sessions": 0, "agents": set(), "healths": [], "bloated_turns": 0, "turns": 0})
        row["tokens"] += s.usage.total + s.subagent_usage.total
        row["sessions"] += 1
        row["agents"].add(s.agent)
        row["healths"].append(s.health)
        row["bloated_turns"] += s.turns_over_threshold
        row["turns"] += s.turns
    out = []
    for row in rows.values():
        healths = sorted(row["healths"])
        out.append({
            "user": row["user"], "tokens": row["tokens"], "sessions": row["sessions"], "agents": sorted(row["agents"]),
            "health_median": healths[len(healths) // 2] if healths else None,
            "bloated_share": round(row["bloated_turns"] / row["turns"], 3) if row["turns"] else 0.0,
        })
    out.sort(key=lambda r: r["tokens"], reverse=True)
    return out


def commits_in_session(session: Session) -> list[dict[str, object]]:
    """Tokens spent between successive `git commit` calls: spend tied to output."""
    out: list[dict[str, object]] = []
    since = 0
    turns_since = 0
    for index, turn in enumerate(session.turns):
        since += turn.usage.total
        turns_since += 1
        for call in turn.tool_calls:
            head = call.command_head or ""
            if call.name == "Bash" and _looks_like_commit(head):
                out.append({"turn": index, "tokens": since, "turns": turns_since, "command": head[:90]})
                since = 0
                turns_since = 0
                break
    return out


def _looks_like_commit(head: str) -> bool:
    return "git commit" in head or head.startswith("git commit") or " && git commit" in head


def _median(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _header(agg: Aggregates) -> str:
    window = f"{agg.window_start:%Y-%m-%d} -> {agg.window_end:%Y-%m-%d}" if agg.window_start and agg.window_end else "no data"
    return f"burnlens  window {window} UTC  sessions {agg.session_count}  subagents {agg.subagent_count}  turns {agg.turn_count:,}"


def price_weighted_payload(by_model: dict[str, Usage], prices: PriceTable) -> dict[str, object]:
    """The same token classes, each weighed with the ratios of the model that produced them."""
    weighted = prices.weigh(by_model)
    total = sum(weighted.values())
    tokens = _sum_usage(by_model)
    return {
        "weights": prices.as_dict(),
        "tokens": tokens.as_dict(),
        "weighted": {name: round(value) for name, value in weighted.items()},
        "shares": {name: (value / total if total else 0.0) for name, value in weighted.items()},
        "weighted_total": round(total),
        "models": sorted(by_model),
        "basis": PRICE_BASIS,
    }


def _price_split(by_model: dict[str, Usage], prices: PriceTable) -> list[str]:
    weighted = prices.weigh(by_model)
    total = sum(weighted.values()) or 1.0
    usage = _sum_usage(by_model)
    order = (
        ("input (fresh)", usage.input_tokens, "input"),
        ("cache write", usage.cache_creation_input_tokens, "cache_write"),
        ("cache read", usage.cache_read_input_tokens, "cache_read"),
        ("output", usage.output_tokens, "output"),
    )
    rows = [(name, f"{tokens:,}", f"{weighted[key]:,.0f}", f"{weighted[key] / total:.1%}") for name, tokens, key in order]
    return _table(("class", "tokens", "weighted", "share"), rows)


def _sum_usage(by_model: dict[str, Usage]) -> Usage:
    total = Usage()
    for usage in by_model.values():
        total = total + usage
    return total


def _session_by_model(session: Session) -> dict[str, Usage]:
    out: dict[str, Usage] = {}
    for turn in session.turns:
        out[turn.model] = out.get(turn.model, Usage()) + turn.usage
    return out


def _usage_split(usage: Usage) -> list[str]:
    total = usage.total or 1
    rows = [
        ("input (fresh)", usage.input_tokens),
        ("cache write", usage.cache_creation_input_tokens),
        ("cache read", usage.cache_read_input_tokens),
        ("output", usage.output_tokens),
        ("total", usage.total),
    ]
    return [f"{name:<16}{value:>18,}  {value / total:>6.1%}" for name, value in rows]


def _usage_table(table: dict[str, Usage], key: str) -> list[str]:
    rows = [(k[:36], f"{u.input_tokens:,}", f"{u.cache_creation_input_tokens:,}", f"{u.cache_read_input_tokens:,}", f"{u.output_tokens:,}", f"{u.total:,}") for k, u in table.items()]
    return _table((key, *USAGE_COLUMNS), rows)


def _main_vs_sub(agg: Aggregates) -> list[str]:
    total = agg.usage.total or 1
    rows = [("main", agg.main_usage.total), ("subagents", agg.subagent_usage.total)]
    lines = [f"{name:<12}{value:>18,}  {value / total:>6.1%}" for name, value in rows]
    if agg.subagent_by_model:
        lines.append("  subagent tokens by model:")
        lines.extend(f"    {model:<26}{u.total:>16,}" for model, u in agg.subagent_by_model.items())
    return lines


def _histogram(agg: Aggregates) -> list[str]:
    total = sum(agg.context_histogram.values()) or 1
    return [f"{name:<10}{count:>8,}  {count / total:>6.1%}  {'#' * int(40 * count / total)}" for name, count in agg.context_histogram.items()]


def _tally_table(items: Iterable[tuple[str, Tally]], key: str) -> list[str]:
    rows = [(k, f"{t.count:,}", _human(t.bytes), f"~{t.bytes // 4:,}") for k, t in items]
    return _table((key, "calls", "bytes", "~tokens"), rows)


def _session_table(stats: list[SessionStat]) -> list[str]:
    rows = [
        (
            s.session_id[:8],
            s.project[-24:],
            f"{s.turns:,}",
            f"{s.usage.total:,}",
            f"{s.subagent_usage.total:,}",
            f"{s.peak_context:,}",
            f"{s.share_over_threshold:.0%}",
            f"{s.health}",
            s.first_prompt[:40],
        )
        for s in stats
    ]
    return _table(("session", "project", "turns", "tokens", "sub-tokens", "peak ctx", "bloated", "health", "first prompt"), rows)


def _findings(findings: list[Finding]) -> list[str]:
    if not findings:
        return ["no findings above thresholds"]
    lines: list[str] = []
    for f in findings:
        avoid = "  (savings unmeasured)" if f.avoidable_tokens is None else f"  ({f.avoidable_tokens:,} tokens; {f.savings_status})"
        lines.append(f"[{f.severity.upper():<6}] {f.rule}: {f.title}{avoid}")
        lines.extend(f"    {e}" for e in f.evidence)
        lines.append(f"    -> {f.suggestion}")
        lines.append("")
    return lines


def _table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    if not rows:
        return ["(none)"]
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" if i == 0 or i == len(headers) - 1 else f"{{:>{w}}}" for i, w in enumerate(widths))
    return [fmt.format(*headers), fmt.format(*("-" * w for w in widths)), *(fmt.format(*r) for r in rows)]


def _session_dict(s: SessionStat) -> dict[str, object]:
    return {
        "session_id": s.session_id,
        "project": s.project,
        "agent": s.agent,
        "user": s.user,
        "is_subagent": s.is_subagent,
        "parent_session_id": s.parent_session_id,
        "first_prompt": s.first_prompt,
        "start": _iso(s.start),
        "end": _iso(s.end),
        "turns": s.turns,
        "usage": s.usage.as_dict(),
        "subagent_usage": s.subagent_usage.as_dict(),
        "peak_context": s.peak_context,
        "share_over_threshold": round(s.share_over_threshold, 3),
        "health": s.health,
    }


def _iso(value) -> str | None:  # noqa: ANN001 - datetime | None
    return value.isoformat() if value else None


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _short(path: str) -> str:
    home = str(Path.home())
    return path.replace(home, "~", 1) if path.startswith(home) else path
