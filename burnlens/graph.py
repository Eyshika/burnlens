"""Knowledge graph of where tokens flow.

Nodes are the things tokens attach to: projects, sessions, subagents, models, files,
commands. Edges are how tokens moved between them: a session read a file (bytes),
ran a command (bytes), spawned a subagent (tokens), ran on a model (tokens). Node
weight is the tokens that thing accounts for, so a picture of the graph IS the
answer to "where did the tokens go", and every other view (findings, habits,
handoff) is a query over the same structure.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .aggregate import Aggregates
from .model import Session

BYTES_PER_TOKEN = 4
EDIT_WEIGHT_TOKENS = 500  # an edit is cheap in bytes but expensive in attention; count it as a fixed nudge
MAX_SESSIONS = 40
MAX_FILES = 40
MAX_COMMANDS = 15
MAX_SUBAGENTS = 60
EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})


@dataclass
class Node:
    id: str
    type: str  # project | session | subagent | model | file | command
    label: str
    weight: int = 0
    meta: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {"id": self.id, "type": self.type, "label": self.label, "weight": self.weight, "meta": self.meta}


@dataclass
class Edge:
    source: str
    target: str
    kind: str  # in_project | read | edited | ran | spawned | on_model
    weight: int = 0

    def as_dict(self) -> dict[str, object]:
        return {"source": self.source, "target": self.target, "kind": self.kind, "weight": self.weight}


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], Edge] = field(default_factory=dict)

    def node(self, node_id: str, type_: str, label: str, weight: int = 0, **meta: object) -> Node:
        existing = self.nodes.get(node_id)
        if existing is None:
            existing = Node(id=node_id, type=type_, label=label, weight=0, meta={})
            self.nodes[node_id] = existing
        existing.weight += weight
        existing.meta.update(meta)
        return existing

    def edge(self, source: str, target: str, kind: str, weight: int) -> None:
        key = (source, target, kind)
        if key in self.edges:
            self.edges[key].weight += weight
        else:
            self.edges[key] = Edge(source, target, kind, weight)

    def as_dict(self) -> dict[str, object]:
        return {"nodes": [n.as_dict() for n in self.nodes.values()], "edges": [e.as_dict() for e in self.edges.values()]}


def build_graph(sessions: list[Session], agg: Aggregates) -> Graph:
    """Build the graph for the sessions in the window, trimmed to what a person can read."""
    stats = {s.session_id: s for s in agg.sessions}
    mains = sorted((s for s in sessions if not s.is_subagent and s.turns), key=lambda s: s.usage.total, reverse=True)[:MAX_SESSIONS]
    kept = {s.session_id for s in mains}
    children = [s for s in sessions if s.is_subagent and s.parent_session_id in kept and s.turns]
    children = sorted(children, key=lambda s: s.usage.total, reverse=True)[:MAX_SUBAGENTS]

    g = Graph()
    file_weight: dict[str, int] = defaultdict(int)
    command_weight: dict[str, int] = defaultdict(int)
    file_edges: list[tuple[str, str, str, int]] = []
    command_edges: list[tuple[str, str, int]] = []

    for s in mains:
        stat = stats.get(s.session_id)
        sid = f"session:{s.session_id}"
        g.node("project:" + s.project, "project", _project_label(s.project))
        g.node("agent:" + s.agent, "agent", s.agent, s.usage.total)
        g.edge("agent:" + s.agent, sid, "ran_in", s.usage.total)
        if s.user:
            g.node("user:" + s.user, "user", s.user, s.usage.total)
            g.edge("user:" + s.user, sid, "owned_by", s.usage.total)
        if s.workflow:
            g.node("workflow:" + s.workflow, "workflow", s.workflow, s.usage.total, run_kind=s.run_kind)
            g.edge("workflow:" + s.workflow, sid, "scheduled", s.usage.total)
        g.node(
            sid, "session", (s.first_prompt or s.session_id)[:60], s.usage.total,
            session_id=s.session_id, turns=len(s.turns), health=stat.health if stat else None,
            peak_context=s.peak_context, start=s.start.isoformat() if s.start else None,
        )
        g.edge("project:" + s.project, sid, "in_project", s.usage.total)
        model_id = f"model:{s.turns[-1].model}"
        g.node(model_id, "model", s.turns[-1].model, s.usage.total)
        g.edge(sid, model_id, "on_model", s.usage.total)
        per_file_read: dict[str, int] = defaultdict(int)
        per_file_edit: dict[str, int] = defaultdict(int)
        per_command: dict[str, int] = defaultdict(int)
        for turn in s.turns:
            for call in turn.tool_calls:
                if call.name == "Read" and call.file_path:
                    per_file_read[call.file_path] += call.result_bytes // BYTES_PER_TOKEN
                elif call.name in EDIT_TOOLS and call.file_path:
                    per_file_edit[call.file_path] += 1
                elif call.name == "Bash" and call.command_head:
                    per_command[call.command_head] += call.result_bytes // BYTES_PER_TOKEN
        for path, tokens in per_file_read.items():
            file_weight[path] += tokens
            file_edges.append((sid, path, "read", tokens))
        for path, edits in per_file_edit.items():
            file_weight[path] += edits * EDIT_WEIGHT_TOKENS
            file_edges.append((sid, path, "edited", edits * EDIT_WEIGHT_TOKENS))
        for head, tokens in per_command.items():
            command_weight[head] += tokens
            command_edges.append((sid, head, tokens))

    for c in children:
        cid = f"subagent:{c.session_id}"
        model = c.turns[-1].model
        g.node(cid, "subagent", c.session_id[:14], c.usage.total, turns=len(c.turns), model=model, parent=c.parent_session_id)
        g.edge(f"session:{c.parent_session_id}", cid, "spawned", c.usage.total)
        g.node(f"model:{model}", "model", model, c.usage.total)
        g.edge(cid, f"model:{model}", "on_model", c.usage.total)

    top_files = {p for p, _ in sorted(file_weight.items(), key=lambda kv: kv[1], reverse=True)[:MAX_FILES]}
    for path in top_files:
        g.node(f"file:{path}", "file", Path(path).name, file_weight[path], path=_short(path), is_image=Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"})
    for sid, path, kind, tokens in file_edges:
        if path in top_files:
            g.edge(sid, f"file:{path}", kind, tokens)
    top_commands = {c for c, _ in sorted(command_weight.items(), key=lambda kv: kv[1], reverse=True)[:MAX_COMMANDS]}
    for head in top_commands:
        g.node(f"command:{head}", "command", head[:48], command_weight[head], command=head)
    for sid, head, tokens in command_edges:
        if head in top_commands:
            g.edge(sid, f"command:{head}", "ran", tokens)
    return g


def _project_label(project: str) -> str:
    return project.split("-")[-1] if "-" in project else project


def _short(path: str) -> str:
    home = str(Path.home())
    return path.replace(home, "~", 1) if path.startswith(home) else path
