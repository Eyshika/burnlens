"""Model-assisted session analysis: the "why" behind a wasteful session.

Rules say what happened. This module sends a compact DIGEST of one session
(never raw tool output) to a text model and asks for the story: where the
agent looped, re-read, drifted, where a new session should have started, and
what the human should do differently, as Don't / Do lines.

Provider-agnostic on purpose. Configuration is environment only, nothing is
written to disk except the cached answer:

    BURNLENS_LLM_API_KEY     (falls back to OPENROUTER_API_KEY)
    BURNLENS_LLM_BASE_URL    default https://openrouter.ai/api/v1
    BURNLENS_LLM_MODEL       default claude-haiku-4-5 on an Anthropic key, openai/gpt-oss-20b on OpenRouter
    BURNLENS_LLM_FORMAT      "openai" (default, OpenRouter / vLLM / Foundry OpenAI route)
                             or "anthropic" (Anthropic Messages API, e.g. Haiku on Foundry)
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .model import Session

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"          # slug valid on the default OpenRouter transport
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
DIGEST_MAX_LINES = 320
PROMPT_CHARS = 140
TEXT_CHARS = 100
CACHE_DIR = Path.home() / ".burnlens" / "explanations"
REQUEST_TIMEOUT_SECONDS = 90
MAX_OUTPUT_TOKENS = 1200

SYSTEM_PROMPT = """You are Burnlens, an analyst of AI coding-agent sessions. You receive a digest of one
session: one line per turn with the context size the model had to read, the tools it called,
how many bytes each tool pushed back into context, and the user's prompts. You never see file
contents. Cost on a subscription is context size x number of turns, so re-reading a large
context every turn is the main waste; retry loops, re-reading the same files, oversized shell
output, images, and subagents on premium models are the usual causes.

Answer ONLY with a JSON object with these keys:
  "narrative": 3-5 sentences telling what this session was doing and where tokens went, in plain English for a non-technical reader.
  "drift": list of {"turns": "a-b", "kind": one of ["retry-loop","re-reading","stale-context","re-explaining","runaway-subagents","large-output","healthy"], "evidence": one sentence}.
  "cut_point": integer turn index where a fresh session would have been cheaper, or null.
  "dont": list of 2-4 short imperative sentences starting with "Don't".
  "do": list of 2-4 short imperative sentences starting with "Do", including at least one thing the HUMAN should do themselves (open a file and paste the function, run the command and paste the failing line, write a five-line handoff and start a new session).
  "better_prompt": a rewrite of the user's last prompt that scopes the task, names files or line ranges, and asks for limited output.
Be concrete: quote turn numbers, file names and commands from the digest. No markdown."""


class AnalysisError(RuntimeError):
    """Raised when the model cannot be reached or returns nothing usable."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    wire_format: str = "openai"  # "openai" | "anthropic"
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS
    max_output_tokens: int = MAX_OUTPUT_TOKENS

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "LLMConfig | None":
        env = env if env is not None else dict(os.environ)
        explicit, openrouter = env.get("BURNLENS_LLM_API_KEY"), env.get("OPENROUTER_API_KEY")
        key = explicit or openrouter or env.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        # an Anthropic key implies Anthropic's endpoint and model id; the OpenRouter slug differs
        native = not (explicit or openrouter)
        return cls(
            api_key=key,
            base_url=env.get("BURNLENS_LLM_BASE_URL", ANTHROPIC_BASE_URL if native else DEFAULT_BASE_URL).rstrip("/"),
            model=env.get("BURNLENS_LLM_MODEL", ANTHROPIC_MODEL if native else DEFAULT_MODEL),
            wire_format=env.get("BURNLENS_LLM_FORMAT", "anthropic" if native else "openai").lower(),
        )

    def public(self) -> dict[str, object]:
        return {"configured": True, "model": self.model, "base_url": self.base_url, "format": self.wire_format}


@dataclass
class Explanation:
    session_id: str
    model: str
    created_at: str
    narrative: str
    drift: list[dict[str, Any]] = field(default_factory=list)
    cut_point: int | None = None
    dont: list[str] = field(default_factory=list)
    do: list[str] = field(default_factory=list)
    better_prompt: str = ""
    digest_lines: int = 0
    raw: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "model": self.model,
            "created_at": self.created_at,
            "narrative": self.narrative,
            "drift": self.drift,
            "cut_point": self.cut_point,
            "dont": self.dont,
            "do": self.do,
            "better_prompt": self.better_prompt,
            "digest_lines": self.digest_lines,
        }


Transport = Callable[[str, dict[str, str], dict[str, Any], int], dict[str, Any]]


def build_digest(session: Session, children: list[Session], max_lines: int = DIGEST_MAX_LINES) -> str:
    """Compact, privacy-conscious view of a session: prompts, tools, sizes. No tool output."""
    turns = session.turns
    header = [
        f"session {session.session_id} project {session.project}",
        f"turns {len(turns)}  peak_context {session.peak_context:,}  subagents {len(children)}",
        f"first_prompt: {session.first_prompt[:PROMPT_CHARS]}",
        f"last_prompt: {session.last_prompt[:PROMPT_CHARS]}",
    ]
    for child in children[:8]:
        model = child.turns[-1].model if child.turns else "?"
        header.append(f"subagent {child.session_id[:12]} model {model} turns {len(child.turns)} tokens {child.usage.total:,}")
    body: list[str] = []
    for index, turn in enumerate(turns):
        tools = " ".join(_tool_token(c.name, c.file_path, c.command_head, c.result_bytes) for c in turn.tool_calls)
        text = f' say:"{turn.text_preview[:TEXT_CHARS]}"' if turn.text_preview and not turn.tool_calls else ""
        body.append(f"t{index} ctx={turn.usage.context // 1000}k out={turn.usage.output_tokens} {tools}{text}".rstrip())
    if len(body) > max_lines:
        step = len(body) / max_lines
        keep = sorted({int(i * step) for i in range(max_lines)} | {len(body) - 1})
        body = [body[i] for i in keep]
        header.append(f"(digest sampled: {len(body)} of {len(turns)} turns shown)")
    return "\n".join(header + body)


def _tool_token(name: str, file_path: str | None, command_head: str | None, result_bytes: int) -> str:
    target = Path(file_path).name if file_path else (command_head or "")[:40]
    size = f"<{result_bytes // 1024}KB" if result_bytes >= 1024 else ""
    return f"{name}({target}){size}"


class SessionNarrator:
    """Ask a model to explain one session; cache the answer per session id."""

    def __init__(self, config: LLMConfig, cache_dir: Path = CACHE_DIR, transport: Transport | None = None) -> None:
        self._cfg = config
        self._cache_dir = cache_dir
        self._post = transport or _post_json

    def explain(self, session: Session, children: list[Session], refresh: bool = False) -> Explanation:
        cached = None if refresh else self._read_cache(session.session_id)
        if cached is not None:
            return cached
        digest = build_digest(session, children)
        raw = self._complete(digest)
        explanation = _parse_explanation(raw, session.session_id, self._cfg.model, digest.count("\n") + 1)
        self._write_cache(explanation)
        return explanation

    def _complete(self, digest: str) -> str:
        if self._cfg.wire_format == "anthropic":
            url = f"{self._cfg.base_url}/messages"
            headers = {"x-api-key": self._cfg.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            body = {
                "model": self._cfg.model,
                "max_tokens": self._cfg.max_output_tokens,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": digest}],
            }
            data = self._post(url, headers, body, self._cfg.timeout_seconds)
            parts = data.get("content") or []
            return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        url = f"{self._cfg.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._cfg.api_key}", "content-type": "application/json", "X-Title": "Burnlens"}
        body = {
            "model": self._cfg.model,
            "max_tokens": self._cfg.max_output_tokens,
            "temperature": 0.2,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": digest}],
        }
        data = self._post(url, headers, body, self._cfg.timeout_seconds)
        choices = data.get("choices") or []
        if not choices:
            raise AnalysisError(f"model returned no choices: {json.dumps(data)[:300]}")
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")

    def _cache_path(self, session_id: str) -> Path:
        return self._cache_dir / f"{session_id}.json"

    def _read_cache(self, session_id: str) -> Explanation | None:
        path = self._cache_path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return Explanation(**{k: v for k, v in data.items() if k in Explanation.__dataclass_fields__})
        except (OSError, ValueError, TypeError) as exc:
            logger.debug("ignoring bad explanation cache %s: %s", path, exc)
            return None

    def _write_cache(self, explanation: Explanation) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(explanation.session_id).write_text(json.dumps(explanation.as_dict(), indent=2))
        except OSError as exc:
            logger.debug("could not cache explanation: %s", exc)


def _post_json(url: str, headers: dict[str, str], body: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        logger.error("model request failed url=%s status=%s body=%s", url, exc.code, detail)
        raise AnalysisError(f"model request failed ({exc.code}): {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.error("model request failed url=%s error=%s", url, exc)
        raise AnalysisError(f"model request failed: {exc}") from exc


def _parse_explanation(raw: str, session_id: str, model: str, digest_lines: int) -> Explanation:
    payload = _extract_json(raw)
    if payload is None:
        raise AnalysisError(f"model did not return JSON: {raw[:200]!r}")
    cut = payload.get("cut_point")
    return Explanation(
        session_id=session_id,
        model=model,
        created_at=datetime.now(timezone.utc).isoformat(),
        narrative=str(payload.get("narrative") or "").strip(),
        drift=[d for d in payload.get("drift") or [] if isinstance(d, dict)],
        cut_point=int(cut) if isinstance(cut, (int, float)) else None,
        dont=[str(x) for x in payload.get("dont") or []],
        do=[str(x) for x in payload.get("do") or []],
        better_prompt=str(payload.get("better_prompt") or "").strip(),
        digest_lines=digest_lines,
        raw=raw,
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def render_explanation(explanation: Explanation) -> str:
    lines = [f"session {explanation.session_id}  ({explanation.model}, digest {explanation.digest_lines} lines)", "", explanation.narrative, ""]
    if explanation.drift:
        lines.append("== drift ==")
        lines.extend(f"  turns {d.get('turns', '?'):<10} {d.get('kind', '?'):<18} {d.get('evidence', '')}" for d in explanation.drift)
        lines.append("")
    if explanation.cut_point is not None:
        lines.append(f"cut point: a new session at turn {explanation.cut_point} would have been cheaper")
        lines.append("")
    lines.append("== don't ==")
    lines.extend(f"  - {x}" for x in explanation.dont)
    lines.append("== do ==")
    lines.extend(f"  - {x}" for x in explanation.do)
    if explanation.better_prompt:
        lines.extend(["", "== better prompt ==", f"  {explanation.better_prompt}"])
    return "\n".join(lines)
