# Ingest: one view across your AI applications

Burnlens reads Claude Code transcripts natively. For everything else, export your own traces
into the **generic JSONL** shape below and point Burnlens at the directory. One record per
model call. The same algorithm, health score, graph, habits and handoff apply.

```bash
burnlens --source generic --root /path/to/traces                  # only foreign traces
burnlens ui --generic-root /path/to/codex --generic-root /path/to/bot   # Claude Code + others, one view
```

With `user` and `agent` set on the records, the dashboard shows "Across applications and people":
tokens by application, tokens by person, and a people table with sessions, bloated share and median
health. Click a person to see their habits.

## LiteLLM: native, no export

If your applications go through a LiteLLM proxy, point Burnlens at what it already logs:

```bash
# StandardLoggingPayload files from S3 / GCS bucket logging or a custom callback (JSON, JSON list, or JSONL)
burnlens ui --litellm-root /path/to/litellm-logs

# or a dump of the proxy's spend logs
curl -s "http://proxy:4000/spend/logs?summarize=false" -H "Authorization: Bearer $LITELLM_MASTER_KEY" > spend_logs.json
burnlens ui --litellm-root spend_logs.json
```

What maps where: person = `metadata.user_api_key_user_id` (else `user`, else `end_user`);
application = the virtual key alias `metadata.user_api_key_alias` (else team alias); project = team
alias. Cache tokens come from `metadata.usage_object` (Anthropic `cache_read_input_tokens` /
`cache_creation_input_tokens`, OpenAI `prompt_tokens_details.cached_tokens`); a proxy `cache_hit`
counts the whole prompt as a cache read. Tool calls come from `response.choices[].message.tool_calls`,
sized by the matching `role: "tool"` message. Sessions group by `session_id`, then `trace_id`, then a
`session:<id>` request tag, then one session per user per day. Turn on
`store_prompts_in_spend_logs` if you want prompts, tool calls and the coach's rewrite; token
accounting works without it.

## Langfuse: native

Export the traces and observations tables from the UI (JSON), or save API pages
(`GET /api/public/traces`, `GET /api/public/observations`) into one directory:

```bash
burnlens ui --langfuse-root /path/to/langfuse-export
```

Mapping: a GENERATION observation is one turn (`usageDetails` with the provider's cache keys, else
`usage`); TOOL and RETRIEVER observations attach to the generation before them in the same trace,
sized by their `output`; session = trace `sessionId`, else the trace id; person = trace `userId`;
application = trace `name`. Field names follow the public API objects; if your export differs,
send one row and the mapping is a five-line change.

## Claude Code telemetry: local OTLP receiver

Burnlens accepts Claude Code OpenTelemetry logs at `/v1/logs`. The dashboard binds to
`127.0.0.1` by default, so this endpoint is available only on the same machine:

```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_LOGS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/json
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:8765
export OTEL_LOG_TOOL_DETAILS=1        # optional: file paths and commands for the rules
export OTEL_LOG_USER_PROMPTS=1        # optional: prompt text for the coach
```

Claude Code then streams `claude_code.api_request`, `claude_code.tool_result`,
`claude_code.user_prompt` and `claude_code.assistant_response` events, each carrying `session.id`,
`user.email` and `organization.id`. Burnlens folds them into generic records under
`~/.burnlens/otlp/` (override with `BURNLENS_OTLP_DIR`) and reads them like any other source.
Metrics and traces posted to `/v1/metrics` and `/v1/traces` are accepted and ignored.

This receiver has no authentication. Although `burnlens ui --host` can bind to another interface,
do not expose the dashboard or receiver directly to a network or use it as a shared fleet server.
Remote collection needs an authenticated, access-controlled boundary that Burnlens does not provide;
see the README's privacy section for the server's current limits.

## Unattended agents

Scheduled jobs, CI agents and routines run with nobody watching. Label their records with a
`workflow` name (generic JSONL), a `workflow:<name>` request tag or `spend_logs_metadata.workflow`
(LiteLLM), `metadata.workflow` on the trace (Langfuse), or nothing at all for headless Claude Code,
whose telemetry already carries `workflow.name` and `workflow.run_id`. Burnlens then shows tokens per
run, the median, the latest run against it, runs that produced nothing, and raises
`unattended-runaway` when a run blows past its own history.

## Copilot, Cursor, Windsurf

Copilot exposes no per-message data; its premium-requests usage report gives requests per user and
model, not tokens, so it will land in a seat-usage panel, not the waste engine. Cursor's team Admin
API returns per-request token usage with cache read and write per user email, which maps directly
onto Burnlens's unit; that adapter is next. Windsurf offers per-user credit analytics for enterprise.

## Record shape

```json
{"session_id": "abc", "project": "checkout-service", "user": "sam", "agent": "codex",
 "ts": "2026-09-07T10:00:00Z", "model": "gpt-5",
 "usage": {"input_tokens": 12, "cache_read_input_tokens": 90000, "cache_creation_input_tokens": 0, "output_tokens": 400},
 "prompt": "fix the flaky loader test",
 "text": "I'll start with the failing test",
 "tools": [{"name": "read_file", "input": {"file_path": "src/loader.py"}, "result_bytes": 18000}],
 "parent_session_id": null}
```

| Field | Required | Notes |
|---|---|---|
| `session_id` | yes | one agent run; subagent runs get their own id and `parent_session_id` |
| `ts` | yes | ISO-8601 or unix seconds |
| `usage` | yes | any subset of the four token fields; the API's own numbers |
| `model` | no | used for premium detection and per-model views |
| `project`, `user`, `agent` | no | `project` (else `agent`) groups sessions |
| `prompt` | no | the user message that started this turn; first and last are kept |
| `text` | no | what the agent said; shows in the live panel and digest |
| `tools[]` | no | `name`, `input` (`file_path` and `command` are understood), `result_bytes` |
| `workflow`, `run_kind` | no | name of the scheduled job or CI workflow, and `scheduled` / `ci` / `interactive`; unattended runs get their own view and rule |

Tool names are mapped to the canonical set so the rules apply: `read_file`, `view`, `cat` become
`Read`; `shell`, `exec`, `run_command` become `Bash`; `apply_patch`, `str_replace_editor` become
`Edit`; `task`, `spawn`, `delegate` become `Agent`. Unknown names pass through.

## Where the records come from

- **Claude Code**: native, nothing to do.
- **Agent SDK (Claude) / Codex SDK**: one record per assistant message from the stream.
- **A gateway or proxy** (LiteLLM, your own): one record per upstream call; `tools` from the
  request's tool results.
- **Langfuse / Helicone / OTel exports**: map spans to records; usage attributes are the same
  four fields.

## The algorithm is configuration

`burnlens.example.toml` lists every threshold, the rules you can disable, which model names count
as premium, and the task-to-tier table the coach uses. Copy it, change it, pass `--config`.
