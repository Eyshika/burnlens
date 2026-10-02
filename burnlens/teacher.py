"""The teacher: turn recurring waste into the platform feature the person never used.

People re-type the same conventions because they do not know CLAUDE.md exists. They ask for
the same multi-step task every day because they have never written a skill. They research the
same question three weeks running because nothing saved the answer. Each pattern is visible in
the prompts, each has a cost, and each has a concrete artifact as the fix. This module finds
the patterns and drafts the artifact.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .coach import FILE_HINT, classify_task
from .model import Session

BYTES_PER_TOKEN = 4
MIN_SESSIONS = 3
HEAVY_CONTEXT_MD_TOKENS = 3_000  # a CLAUDE.md above this is re-read every turn of every session; the post says keep it lightweight
RULE_LINE = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?(never|always|must|do not|don'?t)\b", re.I | re.M)
INSTRUCTION = re.compile(r"^(always|never|remember|from now on|make sure|use \w+ for|don'?t|do not|please always|rule:)", re.I)
LONG_PROMPT_CHARS = 3_000
LONG_THREAD_TURNS = 120          # past this a thread pays for its own history on every turn
NOISY_RESULT_BYTES = 50_000      # one result this size is re-read for the rest of the session
NOISY_MIN_CALLS = 5
MIN_SEARCH_CALLS = 4
VAGUE_FILE = re.compile(r"\b(?:the|that|this) (config|settings?|main|test|schema|model|router?|handler|readme|script|module|class|function|endpoint|migration|file)\b", re.I)
SEARCH_TOOLS = frozenset({"Glob", "Grep", "LS"})
STOPWORDS = frozenset("the a an and or to of in on for with this that these those it is are be as at by from into then just please can you we our my your all any some check make run fix add new file files code tests test".split())


@dataclass
class Lesson:
    feature: str          # "CLAUDE.md" | "skill" | "saved research" | "file reference" | "project map" | "hook"
    title: str
    occurrences: int      # distinct sessions showing the pattern
    people: list[str]
    avoidable_tokens: int | None
    why: str
    examples: list[str] = field(default_factory=list)
    draft: str = ""       # the artifact, ready to paste
    savings_status: str = "unmeasured"
    savings_basis: str = "No matched before/after measurement; potential savings are unknown."

    def as_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature, "title": self.title, "occurrences": self.occurrences, "people": self.people,
            "savings_status": self.savings_status, "savings_basis": self.savings_basis,
            "avoidable_tokens": self.avoidable_tokens, "why": self.why, "examples": self.examples, "draft": self.draft,
        }


def lessons(sessions: list[Session], min_sessions: int = MIN_SESSIONS) -> list[Lesson]:
    mains = [s for s in sessions if not s.is_subagent and s.turns]
    out: list[Lesson] = []
    out.extend(_repeated_instructions(mains, min_sessions))
    out.extend(_repeated_tasks(mains, min_sessions))
    out.extend(_repeated_research(mains, min_sessions))
    out.extend(_pasted_content(mains))
    out.extend(_project_map(mains, min_sessions))
    out.extend(_heavy_context_files(mains))
    out.extend(_long_threads(mains, min_sessions))
    out.extend(_unnamed_files(mains, min_sessions))
    out.extend(_noisy_commands(mains, min_sessions))
    out.sort(key=lambda l: l.occurrences, reverse=True)
    return out


def _repeated_instructions(mains: list[Session], min_sessions: int) -> list[Lesson]:
    """The same standing instruction typed into many sessions -> CLAUDE.md."""
    by_key: dict[str, dict[str, object]] = {}
    for s in mains:
        seen: set[str] = set()
        for prompt in s.prompts:
            for line in re.split(r"[.\n]", prompt):
                line = line.strip()
                if len(line) < 12 or not INSTRUCTION.match(line):
                    continue
                key = _norm(line)[:80]
                if key in seen:
                    continue
                seen.add(key)
                entry = by_key.setdefault(key, {"sessions": set(), "people": set(), "example": line[:140]})
                entry["sessions"].add(s.session_id)
                entry["people"].add(s.user or "me")
    out: list[Lesson] = []
    hits = [(k, e) for k, e in by_key.items() if len(e["sessions"]) >= min_sessions]
    if not hits:
        return out
    hits.sort(key=lambda kv: len(kv[1]["sessions"]), reverse=True)
    gotchas = [e for _, e in hits if not RULE_LINE.match(e["example"])]
    rules = [e for _, e in hits if RULE_LINE.match(e["example"])]
    total_sessions = len({sid for _, e in hits for sid in e["sessions"]})
    draft_lines = []
    if gotchas:
        draft_lines += ["# CLAUDE.md (keep it lightweight: gotchas and principles only)", ""] + [f"- {e['example'].rstrip('.')}." for e in gotchas[:6]]
    if rules:
        draft_lines += ["", "# Rules -> principles or a skill", "Claude 5 models do better with judgement than with never/always lists.", ""]
        draft_lines += [f"- Instead of: '{e['example'][:90]}'  ->  state the principle behind it, or put the procedure in a skill that loads when relevant." for e in rules[:5]]
    draft_lines += ["", "# Or let auto-memory keep it", "Say it once with 'remember: ...' and Claude Code saves it to its memory directory; no file editing."]
    out.append(Lesson(
        feature="CLAUDE.md", title=f"{len(hits)} standing instruction(s) re-typed across {total_sessions} sessions",
        occurrences=total_sessions, people=sorted({p for _, e in hits for p in e["people"]}),
        avoidable_tokens=None,
        why=("Anything told to the agent every session should live where it loads automatically: a repository gotcha or principle in a lightweight CLAUDE.md, "
             "a procedure in a skill that loads only when relevant, a personal preference in auto-memory. Repeated instructions suggest a reusable-guidance opportunity, but do not establish eliminated turns; "
             "and Anthropic's Claude 5 guidance is to replace never/always rules with principles."),
        examples=[e["example"] for _, e in hits[:4]],
        draft="\n".join(draft_lines) + "\n",
    ))
    return out


def _repeated_tasks(mains: list[Session], min_sessions: int) -> list[Lesson]:
    """The same multi-step ask across many sessions -> a skill (slash command)."""
    clusters: dict[str, dict[str, object]] = {}
    for s in mains:
        seen: set[str] = set()
        for prompt in s.prompts:
            task = classify_task(prompt)
            if task in ("general", "lookup"):
                continue
            words = [w for w in re.findall(r"[a-z][a-z0-9_\-]{3,}", prompt.lower()) if w not in STOPWORDS][:6]
            if len(words) < 2:
                continue
            key = task + ":" + " ".join(sorted(set(words[:3])))
            if key in seen:
                continue
            seen.add(key)
            entry = clusters.setdefault(key, {"task": task, "sessions": set(), "people": set(), "example": " ".join(prompt.split())[:140]})
            entry["sessions"].add(s.session_id)
            entry["people"].add(s.user or "me")
    out: list[Lesson] = []
    for key, e in sorted(clusters.items(), key=lambda kv: len(kv[1]["sessions"]), reverse=True):
        if len(e["sessions"]) < min_sessions or e["task"] == "research":
            continue
        name = "-".join(re.findall(r"[a-z0-9]+", key.split(":", 1)[1]))[:40] or e["task"]
        out.append(Lesson(
            feature="skill", title=f"'{e['example'][:60]}' asked in {len(e['sessions'])} sessions",
            occurrences=len(e["sessions"]), people=sorted(e["people"]), avoidable_tokens=None,
            why="A recurring multi-step ask is a skill: the steps, files, checks and output format written once and loaded only when the task comes up (progressive disclosure), so the agent stops rediscovering them and you stop retyping them.",
            examples=[e["example"]],
            draft=(
                f"# .claude/skills/{name}/SKILL.md\n---\nname: {name}\ndescription: {e['example'][:100]}\n---\n\n"
                "## Steps\n1. <the first thing you always tell it to do>\n2. <files to read, as line ranges>\n3. <the command to run, with -q / tail -50>\n\n"
                "## Done when\n- <the check that proves it worked>\n\n## Output\n- <diff only / summary under 200 words>\n"
            ),
        ))
        if len(out) >= 3:
            break
    return out


def _repeated_research(mains: list[Session], min_sessions: int) -> list[Lesson]:
    """The same research question across sessions -> save the answer once."""
    clusters: dict[str, dict[str, object]] = {}
    for s in mains:
        for prompt in s.prompts:
            if classify_task(prompt) != "research":
                continue
            words = sorted({w for w in re.findall(r"[a-z][a-z0-9\-]{3,}", prompt.lower()) if w not in STOPWORDS})[:4]
            if len(words) < 2:
                continue
            key = " ".join(words[:3])
            entry = clusters.setdefault(key, {"sessions": set(), "people": set(), "example": " ".join(prompt.split())[:140]})
            if s.session_id not in entry["sessions"]:
                entry["sessions"].add(s.session_id)
                entry["people"].add(s.user or "me")
    out: list[Lesson] = []
    for key, e in clusters.items():
        if len(e["sessions"]) < min_sessions:
            continue
        out.append(Lesson(
            feature="saved research", title=f"Researched '{e['example'][:50]}' in {len(e['sessions'])} sessions",
            occurrences=len(e["sessions"]), people=sorted(e["people"]), avoidable_tokens=None,
            why="Similar research prompts recur; prior answers may still need updating. Save a dated answer with sources, check freshness before reuse, and compare effort and quality on later tasks.",
            examples=[e["example"]],
            draft=f"# docs/research/{'-'.join(key.split())}.md\n\n<paste the last answer here; date it; link the sources>\n\nThen in CLAUDE.md: `- Before researching {key}, read docs/research/{'-'.join(key.split())}.md`\n",
        ))
    return out


def _pasted_content(mains: list[Session]) -> list[Lesson]:
    """Very long prompts (pasted logs, files) -> put it in a file and reference a line range."""
    hits = [(s, p, n) for s in mains for p, n in zip(s.prompts, s.prompt_chars, strict=True) if n >= LONG_PROMPT_CHARS]
    if len(hits) < 2:
        return []
    return [Lesson(
        feature="file reference", title=f"{len(hits)} prompt(s) pasted more than {LONG_PROMPT_CHARS // 1000}k characters",
        occurrences=len({s.session_id for s, _, _ in hits}), people=sorted({s.user or "me" for s, _, _ in hits}),
        avoidable_tokens=None,
        why="Large pasted inputs may increase context. Save the source and request a relevant range when the full content is unnecessary; verify that narrowing the input preserves the needed evidence.",
        examples=[p[:100] + "…" for _, p, _ in hits[:3]],
        draft="Save the paste as scratch/<name>.log, then: 'grep -n ERROR scratch/<name>.log | tail -20' or 'read scratch/<name>.log lines 120-180'.\n",
    )]


def _project_map(mains: list[Session], min_sessions: int) -> list[Lesson]:
    """The same files read at the start of most sessions -> a project map in CLAUDE.md."""
    first_reads: Counter[str] = Counter()
    per_session: dict[str, set[str]] = defaultdict(set)
    for s in mains:
        for turn in s.turns[:6]:
            for call in turn.tool_calls:
                if call.name == "Read" and call.file_path:
                    per_session[s.session_id].add(call.file_path)
    for files in per_session.values():
        first_reads.update(files)
    common = [(p, n) for p, n in first_reads.most_common(6) if n >= max(min_sessions, len(mains) // 3)]
    if not common:
        return []
    draft = "# Project map (add to CLAUDE.md)\n\n" + "\n".join(f"- `{Path(p).name}`: <one line on what it owns and where the entry points are>" for p, _ in common)
    return [Lesson(
        feature="project map", title=f"{len(common)} file(s) read near the start of {common[0][1]}+ sessions: possible project-map opportunity",
        occurrences=common[0][1], people=sorted({s.user or "me" for s in mains}),
        avoidable_tokens=None,
        why="The same files appear in early reads across sessions. A five-line map in CLAUDE.md may reduce orientation work. Confirm those reads were for orientation, keep the map current, and measure task outcomes before claiming savings.",
        examples=[Path(p).name for p, _ in common[:4]],
        draft=draft + "\n",
    )]


def _heavy_context_files(mains: list[Session]) -> list[Lesson]:
    """CLAUDE.md files that are big and rule-dense: paid for on every turn of every session."""
    candidates: dict[Path, int] = {}
    home_md = Path.home() / ".claude" / "CLAUDE.md"
    if home_md.is_file():
        candidates[home_md] = len(mains)
    for s in mains:
        root = _project_root(s.project)
        if root is not None:
            md = root / "CLAUDE.md"
            if md.is_file():
                candidates[md] = candidates.get(md, 0) + 1
    rows = []
    for path, sessions in candidates.items():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        tokens = len(text) // BYTES_PER_TOKEN
        if tokens < HEAVY_CONTEXT_MD_TOKENS:
            continue
        rules = len(RULE_LINE.findall(text))
        heads = [(m.group(1).strip(), m.start()) for m in re.finditer(r"^##\s+(.+)$", text, re.M)]
        sizes = []
        for i, (h, st) in enumerate(heads):
            en = heads[i + 1][1] if i + 1 < len(heads) else len(text)
            sizes.append((h, (en - st) // BYTES_PER_TOKEN))
        sizes.sort(key=lambda kv: kv[1], reverse=True)
        rows.append((path, tokens, rules, sizes, sessions))
    if not rows:
        return []
    out: list[Lesson] = []
    for path, tokens, rules, sizes, sessions in rows:
        movable = [h for h, n in sizes[:5] if n >= 200]
        draft = [f"# {_short(path)}: {tokens:,} tokens, {rules} never/always/must lines", "",
                 "Keep here (lightweight): repository gotchas, the two or three principles, pointers.",
                 "Move to skills (load only when relevant):"] + [f"- '{h}'  ->  .claude/skills/<name>/SKILL.md with a description that says when to load it" for h in movable] + [
                 "Rewrite never/always lists as principles the model can apply with judgement.",
                 "Run `claude doctor` to right-size the rest."]
        out.append(Lesson(
            feature="lean CLAUDE.md", title=f"{_short(path)} is approximately {tokens:,} tokens by character heuristic; review loading scope",
            occurrences=sessions, people=sorted({s.user or "me" for s in mains}),
            avoidable_tokens=None,
            why=("The file is large by a character-based estimate; actual loading, caching and compaction depend on the runtime. Consider keeping general guidance lightweight, "
                 "prefer principles over rules, and load procedures through skills only when the task needs them."),
            examples=[f"{h}: {n:,} tok" for h, n in sizes[:4]],
            draft="\n".join(draft) + "\n",
        ))
    return out


def _long_threads(mains: list[Session], min_sessions: int) -> list[Lesson]:
    """Threads that ran well past a natural break -> /compact, /clear and the handoff brief."""
    long_ones = sorted((s for s in mains if len(s.turns) >= LONG_THREAD_TURNS), key=lambda s: -len(s.turns))
    if len(long_ones) < min_sessions:
        return []
    return [Lesson(
        feature="session reset",
        title=f"{len(long_ones)} sessions ran past {LONG_THREAD_TURNS} turns without a reset",
        occurrences=len(long_ones),
        people=sorted({s.user for s in long_ones if s.user}),
        avoidable_tokens=None,
        why="Every turn re-sends the whole thread, so a long session keeps paying for its own history. "
            "/compact condenses the thread and carries on; /clear starts a new one. Neither is free to follow "
            "from memory, which is what the handoff brief is for.",
        examples=[f"{s.session_id[:8]} {len(s.turns)} turns, peak context {s.peak_context:,}" for s in long_ones[:3]],
        draft="At the end of a task:\n"
              "  /compact    keep going on the same thread, condensed\n"
              "  /clear      next task is unrelated, carry nothing\n"
              "Carry the thread across a /clear:\n"
              "  burnlens handoff <session-id>\n",
    )]


def _unnamed_files(mains: list[Session], min_sessions: int) -> list[Lesson]:
    """Prompts that describe a file instead of naming it, then a search pass -> @-mention."""
    hits: list[tuple[Session, int, int]] = []
    for s in mains:
        vague = [p for p in s.prompts if VAGUE_FILE.search(p) and not FILE_HINT.search(p)]
        searches = sum(1 for turn in s.turns for call in turn.tool_calls if call.name in SEARCH_TOOLS)
        if vague and searches >= MIN_SEARCH_CALLS:
            hits.append((s, len(vague), searches))
    if len(hits) < min_sessions:
        return []
    return [Lesson(
        feature="@ mention",
        title=f"{len(hits)} sessions described a file instead of naming it",
        occurrences=len(hits),
        people=sorted({s.user for s, _, _ in hits if s.user}),
        avoidable_tokens=None,
        why="'the config file' makes the agent go looking, and every Glob and Grep result stays in context "
            "for the rest of the session. @-mentioning the path puts the file in directly.",
        examples=[f"{s.session_id[:8]} {vague} vague prompts, {searches} search calls" for s, vague, searches in sorted(hits, key=lambda h: -h[2])[:3]],
        draft="Instead of: 'update the config file to add the new flag'\n"
              "Write:      'update @src/app/config.py to add the new flag'\n"
              "Type @ in Claude Code and the path completes.\n",
    )]


def _noisy_commands(mains: list[Session], min_sessions: int) -> list[Lesson]:
    """Big command output landing in the main thread -> run it in a subagent."""
    hits: list[tuple[Session, int, str]] = []
    for s in mains:
        noisy = [c for turn in s.turns for c in turn.tool_calls if c.name == "Bash" and c.result_bytes >= NOISY_RESULT_BYTES]
        if len(noisy) >= NOISY_MIN_CALLS:
            worst = max(noisy, key=lambda c: c.result_bytes)
            hits.append((s, len(noisy), worst.command_head or ""))
    if len(hits) < min_sessions:
        return []
    return [Lesson(
        feature="subagent",
        title=f"{len(hits)} sessions dumped large command output into the main thread",
        occurrences=len(hits),
        people=sorted({s.user for s, _, _ in hits if s.user}),
        avoidable_tokens=None,
        why="A big result stays in the main thread and is re-sent on every later turn. A subagent runs the "
            "command in its own context and returns only the conclusion, so the output is read once.",
        examples=[f"{s.session_id[:8]} {count} results over {NOISY_RESULT_BYTES // 1000} KB, worst: {head[:60]}" for s, count, head in sorted(hits, key=lambda h: -h[1])[:3]],
        draft="Instead of running the noisy command in the main thread, say:\n"
              "  'Use a subagent to run <command> and report only the failures.'\n"
              "Keep the command itself intact; the subagent, not a pipe, is what bounds the context.\n",
    )]


def _project_root(slug: str) -> Path | None:
    """Claude Code project slugs are the cwd with '/' -> '-'. Recover the path when it exists on disk."""
    if not slug.startswith("-"):
        return None
    parts = slug[1:].split("-")
    # greedy: try joining hyphenated segments back together when the plain split does not exist
    candidate = Path("/" + "/".join(parts))
    if candidate.is_dir():
        return candidate
    for i in range(len(parts) - 1, 0, -1):
        joined = Path("/" + "/".join(parts[:i]) + "/" + "-".join(parts[i:]))
        if joined.is_dir():
            return joined
    return None


def _short(path: Path) -> str:
    home = str(Path.home())
    text = str(path)
    return text.replace(home, "~", 1) if text.startswith(home) else text


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
