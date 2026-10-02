"""Roll parsed sessions up into the numbers the report prints."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .model import DIAGNOSTIC_COMMAND, MUTATION_TOOLS, Session, Usage

CONTEXT_BUCKETS: tuple[tuple[str, int], ...] = (
    ("<50k", 50_000),
    ("50-150k", 150_000),
    ("150-400k", 400_000),
    (">400k", 1 << 62),
)


@dataclass
class Tally:
    """Count and byte total for one grouping key."""

    count: int = 0
    bytes: int = 0

    def add(self, size: int) -> None:
        self.count += 1
        self.bytes += size


@dataclass
class FusePair:
    """An edit whose verification ran as a separate model request."""

    project: str
    session_id: str
    file_path: str
    command: str
    at: datetime
    avoidable_tokens: int


@dataclass
class ReadGroup:
    project: str
    session_id: str
    file_path: str
    offset: str
    limit: str
    pages: str
    tally: Tally = field(default_factory=Tally)


@dataclass
class SessionStat:
    session_id: str
    project: str
    model: str
    agent: str
    user: str
    workflow: str
    run_kind: str
    is_subagent: bool
    parent_session_id: str | None
    first_prompt: str
    start: datetime
    end: datetime
    turns: int
    usage: Usage
    peak_context: int
    turns_over_threshold: int
    payload_bytes: int = 0
    repeated_read_calls: int = 0
    subagent_usage: Usage = field(default_factory=Usage)
    premium_subagent_tokens: int = 0

    @property
    def share_over_threshold(self) -> float:
        return self.turns_over_threshold / self.turns if self.turns else 0.0

    @property
    def health(self) -> int:
        """0-100. 100 = lean session. Penalties for bloat, oversized payloads, re-reads and premium subagents."""
        return health_score(self)


HEALTH_WEIGHTS: dict[str, float] = {"bloat": 50.0, "payload": 20.0, "reread": 15.0, "premium_subagents": 15.0}
PAYLOAD_BYTES_PER_TURN_FULL_PENALTY = 20_000
REREAD_CALLS_PER_TURN_FULL_PENALTY = 0.5


def health_score(stat: "SessionStat") -> int:
    turns = max(stat.turns, 1)
    bloat = stat.share_over_threshold
    payload = min(1.0, (stat.payload_bytes / turns) / PAYLOAD_BYTES_PER_TURN_FULL_PENALTY)
    reread = min(1.0, (stat.repeated_read_calls / turns) / REREAD_CALLS_PER_TURN_FULL_PENALTY)
    total_sub = stat.subagent_usage.total
    premium = (stat.premium_subagent_tokens / total_sub) if total_sub else 0.0
    penalty = bloat * HEALTH_WEIGHTS["bloat"] + payload * HEALTH_WEIGHTS["payload"] + reread * HEALTH_WEIGHTS["reread"] + premium * HEALTH_WEIGHTS["premium_subagents"]
    return int(round(max(0.0, 100.0 - penalty)))


@dataclass
class Aggregates:
    window_start: datetime | None
    window_end: datetime | None
    session_count: int
    subagent_count: int
    turn_count: int
    usage: Usage
    main_usage: Usage
    subagent_usage: Usage
    by_day: dict[str, Usage]
    by_model: dict[str, Usage]
    subagent_by_model: dict[str, Usage]
    by_project: dict[str, Usage]
    by_repo: dict[str, Usage]
    by_agent: dict[str, Usage]
    by_user: dict[str, Usage]
    heatmap: list[list[int]]  # 7 weekdays (Mon=0) x 24 local hours, tokens
    context_histogram: dict[str, int]
    tools: dict[str, Tally]
    file_reads: dict[str, Tally]
    commands: dict[str, Tally]
    large_payloads: dict[str, Tally]
    sessions: list[SessionStat]
    read_groups: list[ReadGroup] = field(default_factory=list)
    fuse_pairs: list[FusePair] = field(default_factory=list)


def aggregate(sessions: list[Session], context_threshold: int, large_payload_bytes: int, premium_markers: tuple[str, ...] = ("opus", "fable", "mythos")) -> Aggregates:
    """Compute every roll-up in one pass over the sessions."""
    by_day: dict[str, Usage] = defaultdict(Usage)
    by_model: dict[str, Usage] = defaultdict(Usage)
    subagent_by_model: dict[str, Usage] = defaultdict(Usage)
    by_project: dict[str, Usage] = defaultdict(Usage)
    by_repo: dict[str, Usage] = defaultdict(Usage)
    by_agent: dict[str, Usage] = defaultdict(Usage)
    by_user: dict[str, Usage] = defaultdict(Usage)
    heatmap = [[0] * 24 for _ in range(7)]
    histogram: Counter[str] = Counter({name: 0 for name, _ in CONTEXT_BUCKETS})
    tools: dict[str, Tally] = defaultdict(Tally)
    file_reads: dict[str, Tally] = defaultdict(Tally)
    commands: dict[str, Tally] = defaultdict(Tally)
    large_payloads: dict[str, Tally] = defaultdict(Tally)
    stats: list[SessionStat] = []
    read_groups: list[ReadGroup] = []
    fuse_pairs: list[FusePair] = []
    total = Usage()
    main_total = Usage()
    sub_total = Usage()
    turn_count = 0
    starts: list[datetime] = []
    ends: list[datetime] = []

    for session in sessions:
        if not session.turns:
            continue
        session_usage = Usage()
        over = 0
        payload_bytes = 0
        session_reads: dict[tuple[str, str, str, str], ReadGroup] = {}
        repeated_reads = 0
        for turn in session.turns:
            usage = turn.usage
            session_usage = session_usage + usage
            turn_count += 1
            by_day[turn.timestamp.date().isoformat()] += usage
            local = turn.timestamp.astimezone()
            heatmap[local.weekday()][local.hour] += usage.total
            by_model[turn.model] += usage
            if session.is_subagent:
                subagent_by_model[turn.model] += usage
            if usage.context > context_threshold:
                over += 1
            histogram[_bucket(usage.context)] += 1
            for call in turn.tool_calls:
                tools[call.name].add(call.result_bytes)
                if call.result_bytes > large_payload_bytes:
                    large_payloads[call.name].add(call.result_bytes)
                    payload_bytes += call.result_bytes
                if call.name == "Read" and call.file_path:
                    file_reads[call.file_path].add(call.result_bytes)
                    key = (call.file_path, str(call.input.get("offset", "")), str(call.input.get("limit", "")), str(call.input.get("pages", "")))
                    group = session_reads.setdefault(key, ReadGroup(session.project, session.session_id, *key))
                    if group.tally.count:
                        repeated_reads += 1
                    group.tally.add(call.result_bytes)
                head = call.command_head
                if call.name == "Bash" and head:
                    commands[head].add(call.result_bytes)
        read_groups.extend(session_reads.values())
        fuse_pairs.extend(_fuse_pairs(session))
        total = total + session_usage
        if session.is_subagent:
            sub_total = sub_total + session_usage
        else:
            main_total = main_total + session_usage
        by_project[_project_label(session)] += session_usage
        by_repo[_repo_label(session)] += session_usage
        by_agent[session.agent or "unknown"] += session_usage
        if session.user:
            by_user[session.user] += session_usage
        starts.append(session.turns[0].timestamp)
        ends.append(session.turns[-1].timestamp)
        stats.append(
            SessionStat(
                session_id=session.session_id,
                project=session.project,
                model=session.turns[-1].model,
                agent=session.agent,
                user=session.user,
                workflow=session.workflow,
                run_kind=session.run_kind,
                is_subagent=session.is_subagent,
                parent_session_id=session.parent_session_id,
                first_prompt=session.first_prompt,
                start=session.turns[0].timestamp,
                end=session.turns[-1].timestamp,
                turns=len(session.turns),
                usage=session_usage,
                peak_context=session.peak_context,
                turns_over_threshold=over,
                payload_bytes=payload_bytes,
                repeated_read_calls=repeated_reads,
            )
        )

    _attach_subagent_usage(stats, premium_markers)
    stats.sort(key=lambda s: s.usage.total, reverse=True)
    return Aggregates(
        window_start=min(starts) if starts else None,
        window_end=max(ends) if ends else None,
        session_count=sum(1 for s in stats if not s.is_subagent),
        subagent_count=sum(1 for s in stats if s.is_subagent),
        turn_count=turn_count,
        usage=total,
        main_usage=main_total,
        subagent_usage=sub_total,
        by_day=dict(sorted(by_day.items())),
        by_model=_sorted_by_total(by_model),
        subagent_by_model=_sorted_by_total(subagent_by_model),
        by_project=_sorted_by_total(by_project),
        by_repo=_sorted_by_total(by_repo),
        by_agent=_sorted_by_total(by_agent),
        by_user=_sorted_by_total(by_user),
        heatmap=heatmap,
        context_histogram=dict(histogram),
        tools=_sorted_by_bytes(tools),
        file_reads=_sorted_by_bytes(file_reads),
        commands=_sorted_by_bytes(commands),
        large_payloads=_sorted_by_bytes(large_payloads),
        sessions=stats,
        read_groups=read_groups,
        fuse_pairs=fuse_pairs,
    )


def _attach_subagent_usage(stats: list[SessionStat], premium_markers: tuple[str, ...]) -> None:
    """Credit each subagent's tokens to the main session that spawned it."""
    by_id = {s.session_id: s for s in stats if not s.is_subagent}
    for stat in stats:
        if stat.is_subagent and stat.parent_session_id in by_id:
            parent = by_id[stat.parent_session_id]
            parent.subagent_usage = parent.subagent_usage + stat.usage
            if any(m in stat.model.lower() for m in premium_markers):
                parent.premium_subagent_tokens += stat.usage.total


def _project_label(session: Session) -> str:
    return session.project


def _fuse_pairs(session: Session) -> list[FusePair]:
    """Edits followed by a turn that only ran a verification command.

    Fusing the two removes that second request, so its context read and output are what the
    separate turn cost. Whether the pair could have been fused is an inference, not a measurement.
    """
    pairs: list[FusePair] = []
    for index in range(len(session.turns) - 1):
        edited = [c.file_path for c in session.turns[index].tool_calls if c.name in MUTATION_TOOLS and c.file_path]
        if not edited:
            continue
        follow = session.turns[index + 1]
        if not follow.tool_calls or any(c.name != "Bash" for c in follow.tool_calls):
            continue
        names = {Path(path).name for path in edited}
        for call in follow.tool_calls:
            head = call.command_head or ""
            if not head or not (DIAGNOSTIC_COMMAND.search(head) or any(name in head for name in names)):
                continue
            pairs.append(
                FusePair(
                    project=session.project,
                    session_id=session.session_id,
                    file_path=edited[0],
                    command=head[:80],
                    at=follow.timestamp,
                    avoidable_tokens=follow.usage.context + follow.usage.output_tokens,
                )
            )
            break
    return pairs


def _repo_label(session: Session) -> str:
    """The git repo when the source recorded a cwd, else the project path normalised."""
    if session.repo:
        return session.repo
    project = session.project.replace("\\", "/").rstrip("/")
    return project or "unknown"


def _bucket(context: int) -> str:
    for name, upper in CONTEXT_BUCKETS:
        if context < upper:
            return name
    return CONTEXT_BUCKETS[-1][0]


def _sorted_by_total(table: dict[str, Usage]) -> dict[str, Usage]:
    return dict(sorted(table.items(), key=lambda kv: kv[1].total, reverse=True))


def _sorted_by_bytes(table: dict[str, Tally]) -> dict[str, Tally]:
    return dict(sorted(table.items(), key=lambda kv: kv[1].bytes, reverse=True))
