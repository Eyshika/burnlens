# Design

## Data source

Claude Code writes one JSONL file per session:

```
~/.claude/projects/<project-slug>/<session-id>.jsonl
~/.claude/projects/<project-slug>/<session-id>/subagents/<agent-id>.jsonl
```

The project slug is the working directory with `/` replaced by `-`. Subagent transcripts sit in
a `subagents/` directory named after the parent session, which is how they are credited back.

Relevant record types:

| `type` | What we take |
|---|---|
| `assistant` | `message.id`, `message.model`, `message.usage`, `tool_use` blocks in `message.content`, `timestamp` |
| `user` | first plain-string prompt (for labelling), `tool_result` blocks with `tool_use_id` and `content` |

`message.usage` fields used: `input_tokens`, `cache_creation_input_tokens`,
`cache_read_input_tokens`, `output_tokens`. Others (`service_tier`, `iterations`, cache TTL
split) are ignored for now.

## Dedupe

One assistant message is written as several lines sharing the same `message.id`, one per
content block (thinking, text, each tool_use), each carrying the full usage. Records are merged
by id; the latest usage wins; `tool_use` blocks are unioned by their own id. Without this, totals double or triple.
On one machine 75,702 assistant lines collapsed to 36,353 unique turns.

## Attribution

- **Context per turn** = `input + cache_creation + cache_read`. This is what the model had to
  read to answer. It is the cost driver on subscriptions.
- **Tool payload bytes**: each `tool_result` is joined to its `tool_use` by id. Size is the
  length of text blocks, plus the base64 length of image blocks. Bytes divided by four is a
  rough token estimate for text and is labelled as such. Image tokens depend on pixel size,
  so image bytes are a relative signal, not a count.
- **Files**: `Read` calls grouped by `file_path`. **Commands**: `Bash` calls grouped by the
  first line of the command, whitespace-collapsed.
- **Subagents**: a transcript under `subagents/` is a subagent; its usage is added to the
  parent's `subagent_usage`. The subagent's model comes from its own assistant records, which
  is how "subagents inherited Opus" becomes visible.

## Findings

Rules are pure functions `(Aggregates, Thresholds) -> Finding | None`. Each finding carries
the rule name, a severity, a one-line title with the headline number, up to five evidence
lines, and a suggestion. Thresholds are one frozen dataclass; the two most useful are CLI flags.

## Caveats, stated plainly

- Token counts are the API's own accounting, so they are exact. How those map onto the
  subscription's 5-hour and weekly limits is not public. Shares of tokens are not shares of
  quota.
- `--days` and `--since` filter by turn timestamp but skip whole files by mtime first, so a
  file last touched before the cutoff is never opened. A session that spans the cutoff is
  included with only its in-window turns.
- Session ids are file stems. A resumed session appends to the same file.
- Malformed lines are skipped and logged at debug level.

## App server

`server.py` is a stdlib `ThreadingHTTPServer` bound to 127.0.0.1. `SessionCache` keeps parsed
sessions per window and re-reads when the newest transcript mtime changes, so the dashboard
always reflects the live tree without re-parsing on every request. Endpoints: `/api/meta`,
`/api/report?days=`, `/api/session/<prefix>?days=`, `/api/live`. Static files are served from
`burnlens/static` with a path-traversal guard.

## Live monitor

`live.py` parses only transcripts whose mtime is within `live_active_minutes`, then keeps a
session only if its last turn timestamp is that recent (mtime alone is wrong after a sync or
restore). Per session: last prompt, last assistant text, last six tool calls, context of the
latest turn, tokens in the last `live_window_minutes` (main + subagents) as tokens/min.
Alerts are computed only for sessions that moved inside the window. `AlertNotifier` runs a
daemon thread, snapshots every 5 s, and notifies once per alert key; a key that clears can
fire again later.

## Pre-execution hook

`hook.py` implements Claude Code's `PreToolUse` protocol: the call arrives as JSON on stdin
(`tool_name`, `tool_input`, `transcript_path`, `session_id`); we answer JSON on stdout with
`hookSpecificOutput.permissionDecision` (`allow` / `ask` / `deny`), a `permissionDecisionReason`
the agent reads, optional `additionalContext`, and a `systemMessage` for the user. The zone
comes from the tail (256 KB) of the transcript, so the hook costs milliseconds. Estimates:
Read uses the file size on disk; Bash uses a pattern list of verbose commands and a list of
limiters; Agent looks at `model` and `subagent_type`. `install_hooks` edits
`~/.claude/settings.json` idempotently, backs it up, and points the hook at the current
interpreter so it works from any shell without the env activated.

## Module map

| Module | Responsibility |
|---|---|
| `model.py` | `Usage`, `ToolCall`, `Turn`, `Session` dataclasses |
| `transcripts.py` | walk root, parse and merge records, join results to calls |
| `aggregate.py` | one pass over sessions into `Aggregates` and `SessionStat` |
| `findings.py` | `Thresholds`, rules, `detect()` |
| `report.py` | text tables, session timeline, JSON |
| `cli.py` | argparse: `report` (default), `sessions`, `session <id>`, `ui`, `hook`, `install-hooks` |
| `server.py` | local HTTP API + static dashboard, session cache |
| `live.py` | active sessions, alerts, notifier thread |
| `hook.py` | PreToolUse guard and settings installer |
| `static/` | dashboard (index.html, app.js, style.css, logo.svg) |
