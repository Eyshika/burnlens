"""Synthetic sessions for `burnlens ui --demo`, screenshots and tests. Nothing here is real data."""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROMPTS = [
    "fix the flaky loader test and clean up imports",
    "add pagination to the sessions endpoint",
    "why does the export script double count cache reads",
    "refactor checkout.py into smaller modules",
    "investigate the 3am pipeline failure",
    "write docs for the new profiler CLI",
    "migrate settings to pydantic BaseSettings",
    "speed up the JSONL parser",
]
PROJECTS = ["-Users-demo-code-webshop", "-Users-demo-code-ml-pipeline", "-Users-demo-code-api-gateway"]
FILES = [
    "/Users/demo/code/webshop/src/server.py",
    "/Users/demo/code/webshop/src/checkout.py",
    "/Users/demo/code/ml-pipeline/train.py",
    "/Users/demo/code/webshop/docs/architecture.png",
    "/Users/demo/code/api-gateway/src/router.py",
]
COMMANDS = [
    "cd ~/code/webshop && pytest -q",
    'python3 -c "import json;print(1)"',
    "cat logs/pipeline.log",
    "git status",
    "python train.py --epochs 3",
]
SESSION_LENGTHS = [40, 90, 180, 600, 1400]
IMAGE_B64 = "A" * 600_000


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def assistant(msg_id: str, ts: str, model: str, context: int, output: int, tools: list[dict]) -> dict:
    return {
        "type": "assistant",
        "timestamp": ts,
        "message": {
            "id": msg_id,
            "model": model,
            "usage": {"input_tokens": 5, "cache_creation_input_tokens": 1_000, "cache_read_input_tokens": context, "output_tokens": output},
            "content": [{"type": "text", "text": "ok"}, *tools],
        },
    }


def tool_use(call_id: str, name: str, inp: dict) -> dict:
    return {"type": "tool_use", "id": call_id, "name": name, "input": inp}


def tool_result(call_id: str, ts: str, content) -> dict:
    return {"type": "user", "timestamp": ts, "message": {"content": [{"type": "tool_result", "tool_use_id": call_id, "content": content}]}}


def prompt(text: str, ts: str) -> dict:
    return {"type": "user", "timestamp": ts, "message": {"content": text}}


def write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def one_session(rng: random.Random, sid: str, start: datetime, n: int, model: str) -> list[dict]:
    recs = [prompt(rng.choice(PROMPTS), iso(start))]
    ctx = rng.randint(20_000, 60_000)
    for i in range(n):
        ts = iso(start + timedelta(seconds=25 * i))
        ctx = min(980_000, ctx + rng.randint(500, 9_000))
        tools: list[dict] = []
        results: list[dict] = []
        roll = rng.random()
        if roll < 0.45:
            cid = f"{sid}b{i}"
            tools.append(tool_use(cid, "Bash", {"command": rng.choice(COMMANDS)}))
            results.append(tool_result(cid, ts, "x" * rng.choice([300, 2_000, 9_000, 80_000])))
        elif roll < 0.75:
            cid = f"{sid}r{i}"
            path = rng.choice(FILES)
            tools.append(tool_use(cid, "Read", {"file_path": path}))
            body = [{"type": "image", "source": {"type": "base64", "data": IMAGE_B64}}] if path.endswith(".png") else "y" * rng.choice([3_000, 12_000, 40_000])
            results.append(tool_result(cid, ts, body))
        elif roll < 0.9:
            cid = f"{sid}e{i}"
            tools.append(tool_use(cid, "Edit", {"file_path": rng.choice(FILES[:3])}))
            results.append(tool_result(cid, ts, "ok"))
        recs.append(assistant(f"{sid}m{i}", ts, model, ctx, rng.randint(60, 900), tools))
        recs.extend(results)
    return recs


def build(root: Path, seed: int = 7) -> int:
    rng = random.Random(seed)
    shutil.rmtree(root, ignore_errors=True)
    now = datetime.now(timezone.utc)
    count = 0
    for day_offset in range(14):
        day = (now - timedelta(days=13 - day_offset)).replace(hour=9, minute=0, second=0, microsecond=0)
        for k in range(rng.randint(1, 3)):
            sid = f"{day_offset:02d}{k}-{rng.getrandbits(32):08x}"
            project = rng.choice(PROJECTS)
            n = rng.choice(SESSION_LENGTHS)
            model = rng.choice(["claude-opus-5-5", "claude-opus-5-5", "claude-sonnet-5-5"])
            start = day + timedelta(minutes=90 * k)
            write(root / project / f"{sid}.jsonl", one_session(rng, sid, start, n, model))
            count += 1
            if n >= 600:
                for a in range(3):
                    sub = [assistant(f"{sid}s{a}{i}", iso(start + timedelta(minutes=60, seconds=30 * i)), "claude-opus-5-5", rng.randint(30_000, 120_000), 200, []) for i in range(40)]
                    write(root / project / sid / "subagents" / f"agent-{a}.jsonl", sub)
    # One session working right now, in the red zone, with a premium subagent.
    sid = f"live-{rng.getrandbits(32):08x}"
    recs = [prompt("check any bugs first and ui improvement", iso(now - timedelta(minutes=4)))]
    for i in range(14):
        ts = iso(now - timedelta(minutes=4) + timedelta(seconds=15 * i))
        cid = f"{sid}e{i}"
        recs.append(assistant(f"{sid}m{i}", ts, "claude-opus-5-5", 320_000 + i * 4_000, 300, [tool_use(cid, "Edit", {"file_path": FILES[0]})]))
        recs.append(tool_result(cid, ts, "ok"))
    recs.append(prompt("now run the tests and fix what breaks", iso(now - timedelta(seconds=20))))
    write(root / PROJECTS[0] / f"{sid}.jsonl", recs)
    sub = [assistant(f"{sid}s{i}", iso(now - timedelta(seconds=90 - i * 10)), "claude-opus-5-5", 60_000, 100, []) for i in range(6)]
    write(root / PROJECTS[0] / sid / "subagents" / "agent-research.jsonl", sub)
    return count + 1


def build_generic(root: Path, seed: int = 11) -> int:
    """Two other applications' traces in the generic JSONL shape: a Codex coding agent used by two
    people, and a production support bot whose retrieval stuffs the same knowledge base into every call."""
    rng = random.Random(seed)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    count = 0
    codex = root / "codex.jsonl"
    with codex.open("w") as fh:
        for day_offset in range(10):
            day = (now - timedelta(days=9 - day_offset)).replace(hour=14, minute=0, second=0, microsecond=0)
            for user, n, waste in (("sam", rng.choice([60, 120, 400]), True), ("priya", rng.choice([30, 60, 90]), False)):
                sid = f"cx-{user}-{day_offset}"
                ctx = 30_000
                for i in range(n):
                    ctx = min(600_000, ctx + (rng.randint(3_000, 12_000) if waste else rng.randint(300, 1_500)))
                    tools = []
                    if waste and rng.random() < 0.5:
                        tools.append({"name": "read_file", "input": {"file_path": "services/checkout/payment_flow.py"}, "result_bytes": 42_000})
                    if waste and rng.random() < 0.3:
                        tools.append({"name": "shell", "input": {"command": "cat logs/checkout.log"}, "result_bytes": 120_000})
                    if not waste and rng.random() < 0.6:
                        tools.append({"name": "apply_patch", "input": {"file_path": f"services/billing/{rng.choice(['invoice', 'tax', 'ledger'])}.py"}, "result_bytes": 200})
                    rec = {
                        "session_id": sid, "project": "checkout" if user == "sam" else "billing", "user": user, "agent": "codex",
                        "ts": iso(day + timedelta(seconds=30 * i)), "model": "gpt-5" if user == "sam" else "gpt-5-mini",
                        "usage": {"input_tokens": 20, "cache_read_input_tokens": ctx, "cache_creation_input_tokens": 800, "output_tokens": rng.randint(80, 600)},
                        "tools": tools,
                    }
                    if i == 0:
                        rec["prompt"] = "fix the payment retry bug" if user == "sam" else "add tax rounding tests"
                    fh.write(json.dumps(rec) + "\n")
                count += 1
    bot = root / "support-bot.jsonl"
    with bot.open("w") as fh:
        for day_offset in range(10):
            day = (now - timedelta(days=9 - day_offset)).replace(hour=8, minute=0, second=0, microsecond=0)
            for k in range(3):
                sid = f"sb-{day_offset}-{k}"
                for i in range(40):
                    rec = {
                        "session_id": sid, "project": "support-bot", "user": "prod", "agent": "support-bot",
                        "ts": iso(day + timedelta(minutes=k * 60, seconds=20 * i)), "model": "claude-sonnet-5-5",
                        "usage": {"input_tokens": 400, "cache_read_input_tokens": 180_000, "cache_creation_input_tokens": 0, "output_tokens": 150},
                        "tools": [{"name": "retrieve", "input": {}, "result_bytes": 95_000}],
                    }
                    if i == 0:
                        rec["prompt"] = "customer: where is my refund"
                    fh.write(json.dumps(rec) + "\n")
                count += 1
    # A scheduled agent: nightly triage over the API. Steady runs, then one runaway run and one that produced nothing.
    nightly = root / "nightly-triage.jsonl"
    with nightly.open("w") as fh:
        for day_offset in range(10):
            day = (now - timedelta(days=9 - day_offset)).replace(hour=3, minute=0, second=0, microsecond=0)
            runaway = day_offset == 8
            silent = day_offset == 9
            n = 260 if runaway else 40
            for i in range(n):
                rec = {
                    "session_id": f"triage-run-{day_offset}", "project": "platform", "user": "svc-triage", "agent": "triage-agent", "workflow": "nightly-triage", "run_kind": "scheduled",
                    "ts": iso(day + timedelta(seconds=20 * i)), "model": "claude-opus-5-5",
                    "usage": {"input_tokens": 30, "cache_read_input_tokens": 60_000 + i * (1_500 if runaway else 400), "cache_creation_input_tokens": 500, "output_tokens": 0 if silent else rng.randint(40, 200)},
                    "tools": [{"name": "shell", "input": {"command": "gh issue list --limit 200 --json body"}, "result_bytes": 180_000}] if runaway and i % 3 == 0 else [],
                }
                if i == 0:
                    rec["prompt"] = "triage new issues and label them"
                fh.write(json.dumps(rec) + "\n")
            count += 1

    # A LiteLLM proxy log in the StandardLoggingPayload shape: an internal RAG assistant behind a virtual key.
    lite = root / "litellm"
    lite.mkdir()
    payloads = []
    for day_offset in range(10):
        day = (now - timedelta(days=9 - day_offset)).replace(hour=11, minute=0, second=0, microsecond=0)
        for i in range(25):
            payloads.append({
                "id": f"chatcmpl-{day_offset}-{i}", "trace_id": f"rag-{day_offset}", "session_id": f"rag-{day_offset}", "call_type": "acompletion", "status": "success",
                "startTime": (day + timedelta(seconds=40 * i)).timestamp(), "model": "gpt-5", "prompt_tokens": 140_000, "completion_tokens": 300, "total_tokens": 140_300,
                "metadata": {"user_api_key_alias": "docs-assistant", "user_api_key_team_alias": "platform", "user_api_key_user_id": "lee",
                             "usage_object": {"prompt_tokens": 140_000, "completion_tokens": 300, "prompt_tokens_details": {"cached_tokens": 120_000}}},
                "messages": [{"role": "user", "content": "how do I rotate the signing key"}, {"role": "tool", "tool_call_id": f"c{i}", "content": "k" * 70_000}],
                "response": {"choices": [{"message": {"role": "assistant", "content": "Here is the procedure.", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "retrieve", "arguments": "{}"}}]}}]},
            })
        count += 1
    (lite / "standard_logging.json").write_text(json.dumps(payloads))
    return count


def generate(dest: Path) -> tuple[Path, list[Path], list[Path]]:
    """Write the demo set under dest; returns the Claude Code root, generic roots and LiteLLM roots."""
    claude, generic = dest / "claude-code", dest / "generic"
    build(claude)
    build_generic(generic)
    return claude, [generic], [generic / "litellm"]
