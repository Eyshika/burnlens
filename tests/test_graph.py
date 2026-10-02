"""Graph model: nodes, edges, weights, trimming."""

from __future__ import annotations

from pathlib import Path

from burnlens.aggregate import aggregate
from burnlens.findings import Thresholds
from burnlens.graph import build_graph
from burnlens.transcripts import load_sessions

from test_profiler import tree  # noqa: F401


def test_graph_has_expected_nodes_and_weighted_edges(tree: Path) -> None:
    th = Thresholds()
    sessions = load_sessions(tree)
    g = build_graph(sessions, aggregate(sessions, th.context_tokens, th.large_payload_bytes))
    types = {n.type for n in g.nodes.values()}
    assert {"project", "session", "subagent", "model", "file", "command"} <= types
    main = g.nodes["session:sess-main"]
    assert main.weight == sum(t.usage.total for s in sessions if s.session_id == "sess-main" for t in s.turns)
    assert main.meta["health"] is not None and main.meta["turns"] == 30
    server = g.nodes["file:/repo/server.py"]
    assert server.weight == 30 * 2_000 // 4 and server.meta["path"] == "/repo/server.py"
    assert g.nodes["file:/repo/diagram.png"].meta["is_image"] is True
    read_edge = g.edges[("session:sess-main", "file:/repo/server.py", "read")]
    assert read_edge.weight == server.weight
    assert ("session:sess-main", "subagent:agent-abc", "spawned") in g.edges
    assert ("subagent:agent-abc", "model:claude-opus-5", "on_model") in g.edges
    payload = g.as_dict()
    assert {"nodes", "edges"} == set(payload) and all({"id", "type", "label", "weight", "meta"} <= set(n) for n in payload["nodes"])
