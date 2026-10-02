/* Burnlens dashboard. Vanilla JS, no build step. Talks to the local /api. */
(() => {
  "use strict";

  const SERIES = [
    { key: "cache_read_input_tokens", label: "Cache read", color: "var(--series-1)" },
    { key: "cache_creation_input_tokens", label: "Cache write", color: "var(--series-2)" },
    { key: "output_tokens", label: "Output", color: "var(--series-3)" },
    { key: "input_tokens", label: "Fresh input", color: "var(--series-4)" },
  ];
  const TOP_ROWS = 10;
  const BLOAT_BAD = 0.5;
  const BLOAT_WARN = 0.25;

  const $ = (id) => document.getElementById(id);
  const state = { days: readStore("days", 7), meta: null, report: null };

  // ---------- formatting ----------
  const fmt = (n) => {
    if (n == null) return "–";
    const abs = Math.abs(n);
    if (abs >= 1e9) return (n / 1e9).toFixed(abs >= 1e10 ? 0 : 1) + "B";
    if (abs >= 1e6) return (n / 1e6).toFixed(abs >= 1e7 ? 0 : 1) + "M";
    if (abs >= 1e3) return (n / 1e3).toFixed(abs >= 1e4 ? 0 : 1) + "k";
    return String(Math.round(n));
  };
  const full = (n) => (n == null ? "–" : Math.round(n).toLocaleString());
  const pct = (a, b) => { if (!b) return "0%"; const v = 100 * a / b; return v.toFixed((v > 0 && v < 10) || (v > 99 && v < 100) ? 1 : 0) + "%"; };
  const bytes = (b) => {
    if (b < 1024) return b + " B";
    if (b < 1024 ** 2) return (b / 1024).toFixed(1) + " KB";
    if (b < 1024 ** 3) return (b / 1024 ** 2).toFixed(1) + " MB";
    return (b / 1024 ** 3).toFixed(2) + " GB";
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const attr = (s) => String(s ?? "").replace(/"/g, "&quot;"); // for HTML strings stored in attributes
  const short = (s, n) => (s && s.length > n ? s.slice(0, n - 1) + "…" : s || "");
  const day = (iso) => (iso ? iso.slice(0, 10) : "–");

  function readStore(key, fallback) {
    try { const v = localStorage.getItem("burnlens." + key); return v == null ? fallback : JSON.parse(v); } catch { return fallback; }
  }
  function writeStore(key, value) {
    try { localStorage.setItem("burnlens." + key, JSON.stringify(value)); } catch { /* private mode */ }
  }

  // ---------- tooltip ----------
  const tip = $("tooltip");
  function showTip(html, x, y) {
    tip.innerHTML = html;
    tip.classList.add("show");
    const r = tip.getBoundingClientRect();
    const left = Math.min(x + 14, window.innerWidth - r.width - 8);
    const top = y - r.height - 12 < 8 ? y + 16 : y - r.height - 12;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }
  const hideTip = () => tip.classList.remove("show");
  function bindTips(root) {
    root.querySelectorAll("[data-tip]").forEach((el) => {
      el.addEventListener("mousemove", (e) => showTip(el.dataset.tip, e.clientX, e.clientY));
      el.addEventListener("mouseleave", hideTip);
    });
  }

  // ---------- data ----------
  async function api(path) {
    const res = await fetch(path, { cache: "no-store" });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || res.statusText);
    return body;
  }

  async function load() {
    $("meta").textContent = "Reading transcripts…";
    try {
      const [meta, report] = await Promise.all([api("/api/meta?days=" + state.days), api("/api/report?days=" + state.days)]);
      state.meta = meta;
      state.report = report;
      document.title = meta.app;
      $("app-name").textContent = meta.app;
      render();
    } catch (err) {
      $("meta").innerHTML = `Could not load: <code>${esc(err.message)}</code>`;
    }
  }

  // ---------- render ----------
  function render() {
    const r = state.report;
    const m = state.meta;
    const th = m.thresholds || {};
    const rangeLabel = state.days ? `last ${state.days} days` : "all time";
    $("meta").innerHTML =
      `${rangeLabel} · ${day(r.window.start)} → ${day(r.window.end)} · reading <code>${esc(m.root.includes("burnlens-demo-") ? "synthetic demo data" : m.root)}</code>` +
      (r.loaded_at ? ` · parsed ${new Date(r.loaded_at).toLocaleTimeString()}` : "");

    if (!r.turns) {
      $("tiles").innerHTML = "";
      $("findings").innerHTML = `<div class="empty">No transcripts with activity in this range. Use Claude Code, then Refresh.</div>`;
      ["day-chart", "hist", "models", "tools", "files", "commands", "sessions"].forEach((id) => ($(id).innerHTML = ""));
      return;
    }

    renderTiles(r, th);
    renderOrg(r);
    renderWorkflows(r.workflows || []);
    loadMap();
    renderHeatmap(r.heatmap || []);
    renderTreemap(r.sessions_detail || []);
    renderHabits(r.habits || []);
    renderLessons(r.lessons || []);
    renderFindings(r.findings);
    renderDayChart(r.by_day);
    renderHistogram(r.context_histogram, th);
    renderModels(r);
    renderTools(r.tools);
    renderFiles(r.file_reads);
    renderCommands(r.commands);
    renderSessions(r.sessions_detail, th);
  }

  function renderTiles(r, th) {
    const u = r.usage;
    const turns = r.turns || 1;
    const hist = r.context_histogram || {};
    const bloatTurns = (hist["150-400k"] || 0) + (hist[">400k"] || 0);
    const bloatShare = bloatTurns / turns;
    const subShare = r.subagent_usage.total / (u.total || 1);
    const tiles = [
      { label: "Tokens processed", value: fmt(u.total), sub: `${full(u.total)} tokens`, tip: "Input + cache write + cache read + output, across every turn." },
      { label: "Cache re-reads", value: pct(u.cache_read_input_tokens, u.total), sub: "share of all tokens", tip: "Context the model had to read again because it was already in the conversation." },
      { label: "Bloated turns", value: pct(bloatTurns, turns), sub: `turns above ${fmt(th.context_tokens || 150000)} context`, status: bloatShare >= BLOAT_BAD ? "critical" : bloatShare >= BLOAT_WARN ? "serious" : "good" },
      { label: "Subagent share", value: pct(r.subagent_usage.total, u.total), sub: `${r.subagents} subagent runs`, status: subShare >= 0.3 ? "serious" : undefined },
      { label: "Session health", value: r.health_median == null ? "–" : String(r.health_median), sub: "median score, 100 = lean", status: r.health_median == null ? undefined : r.health_median < 50 ? "critical" : r.health_median < 75 ? "serious" : "good", tip: "Per-session score: penalties for bloated context, oversized tool output, re-reads and premium-model subagents." },
      { label: "Sessions", value: full(r.sessions), sub: `${full(r.turns)} turns` },
      { label: "Model output", value: pct(u.output_tokens, u.total), sub: "what the model actually wrote", tip: "Output tokens as a share of everything processed. Usually tiny." },
    ];
    $("tiles").innerHTML = tiles
      .map((t) => `<div class="tile ${t.status ? "status-" + t.status : ""}" ${t.tip ? `data-tip="${esc(t.tip)}"` : ""}>
        <div class="label">${esc(t.label)}</div><div class="value">${esc(t.value)}</div><div class="sub">${esc(t.sub)}</div></div>`)
      .join("");
    bindTips($("tiles"));
  }

  function renderLessons(rows) {
    $("teacher-card").hidden = !rows.length;
    if (!rows.length) return;
    $("lessons").innerHTML = rows.map((l, i) => `<article class="lesson">
      <span class="feat">${esc(l.feature)}</span>
      <div>
        <div class="title">${esc(l.title)}</div>
        <div class="nums">${full(l.occurrences)} sessions · savings unmeasured${l.people.length ? " · " + esc(l.people.join(", ")) : ""}</div>
        <div class="why">${esc(l.why)}</div>
        ${l.examples.length ? `<div class="ex" title="${esc(l.examples.join(" · "))}">e.g. ${esc(l.examples.slice(0, 2).join(" · "))}</div>` : ""}
        ${l.draft ? `<details><summary>template to complete and review</summary><pre>${esc(l.draft)}</pre><button class="ghost" data-copy="${i}">Copy</button></details>` : ""}
      </div></article>`).join("");
    $("lessons").querySelectorAll("[data-copy]").forEach((b) => b.addEventListener("click", async () => { try { await navigator.clipboard.writeText(rows[Number(b.dataset.copy)].draft); b.textContent = "Copied"; } catch { b.textContent = "Select and copy"; } }));
  }

  function renderFindings(findings) {
    if (!findings.length) {
      $("findings").innerHTML = `<div class="empty good">Nothing above thresholds in this range. Keep it that way.</div>`;
      return;
    }
    $("findings").innerHTML = findings
      .map((f) => `<article class="finding ${esc(f.severity)}">
        <div class="badge">${esc(f.severity)}</div>
        <div>
          <div class="headline"><div class="title">${esc(f.title)}</div><div class="avoid">Savings unmeasured</div></div>
          <div class="fix">${esc(f.suggestion).replace(/Don't:/g, "<strong>Don't:</strong>").replace(/Do:/g, "<strong>Do:</strong>")}</div>
          ${f.evidence.length ? `<details><summary>evidence (${f.evidence.length})</summary><ul class="evidence">${f.evidence.map((e) => `<li>${esc(e)}</li>`).join("")}</ul></details>` : ""}
        </div></article>`)
      .join("");
  }

  const MAP_TYPES = [["session", "Sessions"], ["subagent", "Subagents"], ["file", "Files"], ["command", "Commands"], ["model", "Models"], ["agent", "Applications"], ["user", "People"], ["workflow", "Workflows"], ["project", "Projects"]];
  const MAP_COLORS = { session: "var(--series-1)", subagent: "var(--series-2)", file: "var(--series-3)", command: "var(--series-4)", model: "var(--series-7)", agent: "var(--series-6)", user: "var(--series-5)", workflow: "var(--series-8)", project: "var(--text-muted)" };

  function renderOrg(r) {
    const agents = Object.entries(r.by_agent || {});
    const users = Object.entries(r.by_user || {});
    const multiUser = users.length > 1;
    const show = agents.length > 1 || multiUser;
    $("org-card").hidden = !show;
    if (!show) return;
    $("org-card").querySelector("h2").textContent = multiUser ? "Across applications and people" : "Across applications";
    $("by-user").parentElement.hidden = !multiUser;
    $("people").hidden = !multiUser;
    $("by-agent").closest(".grid2").style.gridTemplateColumns = multiUser ? "" : "1fr";
    const maxA = Math.max(...agents.map(([, u]) => u.total)) || 1;
    $("by-agent").innerHTML = hbars(agents.map(([a, u]) => ({ name: a, value: u.total, share: u.total / maxA, tip: `<b>${esc(a)}</b><br>${fmt(u.total)} tokens · ${pct(u.cache_read_input_tokens, u.total)} cache re-reads` })), (row) => fmt(row.value));
    const maxU = Math.max(...users.map(([, u]) => u.total)) || 1;
    $("by-user").innerHTML = hbars(users.map(([n, u]) => ({ name: n, value: u.total, share: u.total / maxU, tip: `<b>${esc(n)}</b><br>${fmt(u.total)} tokens` })), (row) => fmt(row.value));
    const people = r.people || [];
    $("people").innerHTML = people.length ? table(["Person", "Applications", "Sessions", "Tokens", "Bloated turns", "Health"], people.map((p) => [
      `<td><span class="chip-user" data-user="${esc(p.user)}" title="Show this person's habits">${esc(p.user)}</span></td>`, `<td class="num">${esc(p.agents.join(", "))}</td>`, `<td class="num">${full(p.sessions)}</td>`, `<td class="num">${fmt(p.tokens)}</td>`,
      `<td class="num"><span class="pct ${p.bloated_share >= BLOAT_BAD ? "bad" : p.bloated_share >= BLOAT_WARN ? "warn" : ""}">${(p.bloated_share * 100).toFixed(0)}%</span></td>`,
      `<td class="num"><span class="pct ${p.health_median < 50 ? "bad" : p.health_median < 75 ? "warn" : ""}">${p.health_median ?? "–"}</span></td>`])) : "";
    bindTips($("org-card"));
    $("people").querySelectorAll(".chip-user").forEach((el) => el.addEventListener("click", () => loadHabitsFor(el.dataset.user)));
  }

  function renderWorkflows(rows) {
    $("unattended-card").hidden = !rows.length;
    if (!rows.length) return;
    $("workflows").innerHTML = table(["Workflow", "Application", "Runs", "Tokens", "Median / run", "Worst run", "Runaway runs", "Silent runs", "Health", "Last run"], rows.map((w) => [
      `<td class="mono">${esc(w.workflow)}</td>`, `<td class="num">${esc(w.agent)}</td>`, `<td class="num">${full(w.runs)}</td>`, `<td class="num">${fmt(w.tokens)}</td>`, `<td class="num">${fmt(w.median_per_run)}</td>`,
      `<td class="num"><span class="pct ${w.max_ratio >= 2 ? "bad" : w.max_ratio >= 1.5 ? "warn" : ""}">${w.max_ratio == null ? "–" : w.max_ratio.toFixed(1) + "x"}</span></td>`,
      `<td class="num"><span class="pct ${w.runaway_runs ? "bad" : ""}">${full(w.runaway_runs)}</span></td>`,
      `<td class="num"><span class="pct ${w.silent_runs ? "bad" : ""}">${full(w.silent_runs)}</span></td>`,
      `<td class="num"><span class="pct ${w.health_median < 50 ? "bad" : w.health_median < 75 ? "warn" : ""}">${w.health_median}</span></td>`, `<td class="num">${esc(day(w.last_run))}</td>`]));
  }

  async function loadHabitsFor(user) {
    try {
      const r = await api("/api/report?days=" + state.days + "&user=" + encodeURIComponent(user));
      renderHabits(r.habits || []);
      $("coach-card").querySelector(".hint").textContent = `Habits for ${user}. Click Refresh to return to everyone.`;
      $("coach-card").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) { /* keep current view */ }
  }
  let mapHandle = null;
  let map3d = null;
  const want3d = () => readStore("map", "3d") === "3d" && typeof window.ForceGraph3D === "function" && !window.__no3d;
  function render3d(g) {
    const host = $("map3d");
    host.hidden = false; $("map").hidden = true;
    const W = Math.max(320, host.parentElement.clientWidth), H = Math.max(360, Math.round(W * 0.55));
    const maxW = Math.max(1, ...g.nodes.map((n) => n.weight));
    const nodes = g.nodes.map((n) => ({ ...n, val: 1 + 30 * Math.sqrt(n.weight / maxW) }));
    const ids = new Set(nodes.map((n) => n.id));
    const links = g.edges.filter((e) => ids.has(e.source) && ids.has(e.target)).map((e) => ({ source: e.source, target: e.target, weight: e.weight }));
    const color = (n) => cssVar(({ session: "--series-1", subagent: "--series-2", file: "--series-3", command: "--series-4", model: "--series-7", agent: "--series-6", user: "--series-5", workflow: "--series-8", project: "--text-muted" })[n.type] || "--text-muted");
    if (!map3d) map3d = window.ForceGraph3D()(host);
    map3d.width(W).height(H).backgroundColor("rgba(0,0,0,0)")
      .graphData({ nodes, links })
      .nodeVal("val").nodeColor(color).nodeOpacity(0.92)
      .nodeLabel((n) => `${n.label} · ${fmt(n.weight)} tokens`)
      .linkColor(() => cssVar("--border")).linkOpacity(0.35).linkWidth((l) => 0.2 + 1.5 * Math.sqrt(l.weight / maxW))
      .onNodeClick((n) => { if (n.type === "session") openSession(n.meta.session_id); });
  }
  async function loadMap() {
    $("map-legend").innerHTML = MAP_TYPES.map(([t, l]) => `<span style="--c:${MAP_COLORS[t]}">${l}</span>`).join("");
    document.querySelectorAll("#map-mode button").forEach((b) => b.setAttribute("aria-selected", String((b.dataset.mode === "3d") === want3d())));
    try {
      const g = await api("/api/graph?days=" + state.days);
      state.graph = g;
      if (!g.nodes.length) { $("map-card").hidden = true; return; }
      $("map-card").hidden = false;
      if (want3d()) {
        try { render3d(g); return; } catch (err) { window.__no3d = true; map3d = null; $("map3d").innerHTML = ""; }
      }
      document.querySelectorAll("#map-mode button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === "2d")));
      $("map3d").hidden = true; $("map").hidden = false;
      mapHandle = window.BurnlensGraph.render($("map"), g, {
        tip: (n, x, y) => showTip(`<b>${esc(n.label)}</b> <span style="opacity:.7">${esc(n.type)}</span><br>${fmt(n.weight)} tokens${n.meta.turns ? `<br>${full(n.meta.turns)} turns` : ""}${n.meta.health != null ? `<br>health ${n.meta.health}` : ""}${n.meta.model ? `<br>${esc(n.meta.model)}` : ""}${n.meta.path ? `<br>${esc(n.meta.path)}` : ""}${n.meta.command ? `<br>${esc(n.meta.command)}` : ""}`, x, y),
        untip: hideTip,
        click: (n) => { if (n.type === "session") openSession(n.meta.session_id); },
      });
    } catch (err) {
      $("map-card").hidden = true;
    }
  }

  // ---------- heatmap: weekday x hour ----------
  const HEAT_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
  const HEAT_RAMP_DARK = ["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#86b6ef"];
  function renderHeatmap(grid) {
    const el = $("heatmap");
    if (!grid.length) { el.innerHTML = ""; return; }
    const dark = document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    const ramp = dark ? HEAT_RAMP_DARK : HEAT_RAMP;
    const max = Math.max(1, ...grid.flat());
    const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    let html = `<div></div>` + Array.from({ length: 24 }, (_, h) => `<div class="hh">${h % 3 === 0 ? h : ""}</div>`).join("");
    grid.forEach((row, d) => {
      html += `<div class="hl">${days[d]}</div>`;
      row.forEach((v, h) => {
        const step = v ? Math.min(ramp.length - 1, Math.floor(Math.sqrt(v / max) * ramp.length)) : -1;
        html += `<div class="cell" style="${step >= 0 ? `background:${ramp[step]}` : ""}" data-tip="<b>${days[d]} ${String(h).padStart(2, "0")}:00</b><br>${fmt(v)} tokens"></div>`;
      });
    });
    el.innerHTML = html;
    bindTips(el);
  }

  // ---------- treemap: application > project > session, colour = health ----------
  function squarify(items, x, y, w, h, out) {
    if (!items.length) return;
    const total = items.reduce((a, it) => a + it.value, 0);
    if (total <= 0) return;
    let row = [], rest = items.slice();
    const vertical = w >= h;
    const side = vertical ? h : w;
    const worst = (r) => { const s = r.reduce((a, it) => a + it.value, 0); const len = (s / total) * (vertical ? w : h); return Math.max(...r.map((it) => { const l = (it.value / s) * side; return Math.max(len / l, l / len); })); };
    while (rest.length) {
      const cand = row.concat(rest[0]);
      if (!row.length || worst(cand) <= worst(row)) { row = cand; rest.shift(); } else break;
    }
    const s = row.reduce((a, it) => a + it.value, 0);
    const len = (s / total) * (vertical ? w : h);
    let off = 0;
    for (const it of row) {
      const l = (it.value / s) * side;
      const rect = vertical ? { x, y: y + off, w: len, h: l } : { x: x + off, y, w: l, h: len };
      out.push({ ...it, ...rect });
      off += l;
    }
    if (vertical) squarify(rest, x + len, y, w - len, h, out); else squarify(rest, x, y + len, w, h - len, out);
  }
  function healthColor(hp) {
    if (hp == null) return cssVar("--text-muted");
    return hp >= 75 ? cssVar("--status-good") : hp >= 50 ? cssVar("--status-warning") : hp >= 25 ? cssVar("--status-serious") : cssVar("--status-critical");
  }
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function renderTreemap(sessions) {
    const canvas = $("treemap");
    const host = canvas.parentElement;
    const W = Math.max(300, host.clientWidth), H = Math.max(260, Math.round(W * 0.62));
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr; canvas.height = H * dpr; canvas.style.height = H + "px";
    const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
    const mains = sessions.filter((s) => !s.is_subagent && s.usage.total > 0);
    if (!mains.length) return;
    const groups = {};
    for (const s of mains) {
      const g = groups[s.agent] || (groups[s.agent] = { name: s.agent, value: 0, kids: {} });
      g.value += s.usage.total;
      const pj = g.kids[s.project] || (g.kids[s.project] = { name: s.project.replace(/^-Users-[^-]+-/, ""), value: 0, kids: [] });
      pj.value += s.usage.total;
      pj.kids.push({ name: s.first_prompt || s.session_id.slice(0, 8), value: s.usage.total, health: s.health, id: s.session_id });
    }
    const pad = 4;
    const rects = [];
    const level1 = [];
    squarify(Object.values(groups).sort((a, b) => b.value - a.value), 0, 0, W, H, level1);
    const bg = cssVar("--surface-0"), ink = cssVar("--text-primary"), muted = cssVar("--text-muted"), border = cssVar("--border");
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
    for (const g of level1) {
      ctx.strokeStyle = border; ctx.lineWidth = 1; ctx.strokeRect(g.x + 1, g.y + 1, g.w - 2, g.h - 2);
      const level2 = [];
      squarify(Object.values(g.kids).sort((a, b) => b.value - a.value), g.x + pad, g.y + pad + 14, g.w - 2 * pad, g.h - 2 * pad - 14, level2);
      for (const pj of level2) {
        const leaves = [];
        squarify(pj.kids.sort((a, b) => b.value - a.value), pj.x + 2, pj.y + 2, Math.max(0, pj.w - 4), Math.max(0, pj.h - 4), leaves);
        for (const leaf of leaves) {
          ctx.fillStyle = healthColor(leaf.health);
          ctx.globalAlpha = 0.85;
          ctx.fillRect(leaf.x + 1, leaf.y + 1, Math.max(0, leaf.w - 2), Math.max(0, leaf.h - 2));
          ctx.globalAlpha = 1;
          if (leaf.w > 70 && leaf.h > 26) {
            ctx.fillStyle = "#fff"; ctx.font = "11px -apple-system, Inter, sans-serif";
            ctx.fillText(leaf.name.length > leaf.w / 6.5 ? leaf.name.slice(0, Math.floor(leaf.w / 6.5) - 1) + "…" : leaf.name, leaf.x + 6, leaf.y + 16);
            ctx.fillStyle = "rgba(255,255,255,.8)"; ctx.font = "10px ui-monospace, Menlo, monospace";
            ctx.fillText(`${fmt(leaf.value)} · h${leaf.health}`, leaf.x + 6, leaf.y + 29);
          }
          rects.push(leaf);
        }
      }
      ctx.fillStyle = ink; ctx.font = "600 11px ui-monospace, Menlo, monospace";
      ctx.fillText(`${g.name} · ${fmt(g.value)}`, g.x + pad + 2, g.y + pad + 9);
    }
    canvas.onmousemove = (e) => {
      const r = canvas.getBoundingClientRect(); const x = e.clientX - r.left, y = e.clientY - r.top;
      const hit = rects.find((l) => x >= l.x && x <= l.x + l.w && y >= l.y && y <= l.y + l.h);
      if (hit) showTip(`<b>${esc(hit.name)}</b><br>${fmt(hit.value)} tokens · health ${hit.health}`, e.clientX, e.clientY); else hideTip();
      canvas.style.cursor = hit ? "pointer" : "default";
    };
    canvas.onmouseleave = hideTip;
    canvas.onclick = (e) => {
      const r = canvas.getBoundingClientRect(); const x = e.clientX - r.left, y = e.clientY - r.top;
      const hit = rects.find((l) => x >= l.x && x <= l.x + l.w && y >= l.y && y <= l.y + l.h);
      if (hit) openSession(hit.id);
    };
  }

  function renderHabits(rows) {
    if (!rows.length) { $("habits").innerHTML = `<div class="empty good">No recurring waste in this range.</div>`; return; }
    $("habits").innerHTML = rows.map((h) => `<article class="habit">
      <span class="trend ${esc(h.trend)}">${esc(h.trend)}</span>
      <div class="title">${esc(h.title)}</div>
      <div class="nums"><b>${full(h.this_week)}</b> this week · <b>${full(h.last_week)}</b> last week · savings unmeasured</div>
      <div class="say"><strong>Say:</strong> ${esc(h.say_this.replace(/^Say:\s*/, ""))}</div>
      ${h.examples.length ? `<div class="ex" title="${esc(h.examples.join(" · "))}">e.g. ${esc(h.examples.join(" · "))}</div>` : ""}
    </article>`).join("");
  }

  function paybackRow(p) {
    if (!p) return "";
    const turns = Math.ceil(p.breakeven_requests);
    const room = p.window_headroom_requests === null || p.window_headroom_requests === undefined
      ? ""
      : ` \u00b7 room for about ${p.window_headroom_requests} turns before the window fills`;
    const verdict = p.worth_it === true ? " \u00b7 worth doing now" : p.worth_it === false ? " \u00b7 not worth it yet" : "";
    return `<div class="payback">
      <div><b>Compaction payback</b> \u00b7 breaks even after about ${turns} more turn${turns === 1 ? "" : "s"} \u00b7 saves ${fmt(p.saved_per_request)} a turn${room}${verdict} <span class="tag">projected</span></div>
      <div class="hint">${esc(p.basis)}</div>
    </div>`;
  }

  async function coachPrompt() {
    const text = $("coach-input").value.trim();
    if (!text) return;
    const live = state.live || {};
    const active = (live.sessions || []).length === 1 ? live.sessions[0] : null;
    const params = new URLSearchParams({ prompt: text, model: active ? active.model : "", context: active ? String(active.context_now) : "0" });
    const out = $("coach-out");
    try {
      const c = await api("/api/coach?" + params.toString());
      out.innerHTML = `<div class="coach-result">
        <div class="kind">task <b>${esc(c.task)}</b> · candidate tier (unvalidated) <b>${esc(c.recommended_tier)}</b> · zone <b>${esc(c.zone)}</b>${active ? ` · judged against your active session (${fmt(c.context_now)} context, ${esc(c.current_model)})` : " · no unique active session; context is unknown"}</div>
        ${c.messages.length ? `<ul>${c.messages.map((m) => `<li class="${m.startsWith("STOP") ? "stop" : ""}">${esc(m)}</li>`).join("")}</ul>` : `<div class="hint">Looks lean. Go ahead.</div>`}
        ${c.rewrite ? `<div class="rewrite">${esc(c.rewrite)}</div>` : ""}
        ${paybackRow(c.compaction)}
      </div>`;
    } catch (err) {
      out.innerHTML = `<div class="empty">${esc(err.message)}</div>`;
    }
  }
  $("coach-btn").addEventListener("click", coachPrompt);
  $("coach-input").addEventListener("keydown", (e) => { if (e.key === "Enter") coachPrompt(); });

  function renderDayChart(byDay) {
    const days = Object.keys(byDay);
    const host = $("day-chart");
    $("day-legend").innerHTML = SERIES.map((s) => `<span style="--c:${s.color}">${s.label}</span>`).join("");
    if (!days.length) { host.innerHTML = ""; return; }
    const W = Math.max(320, host.clientWidth || 600);
    const H = 240, padL = 44, padR = 8, padT = 10, padB = 28;
    const innerW = W - padL - padR, innerH = H - padT - padB;
    const max = Math.max(...days.map((d) => byDay[d].total)) || 1;
    const slot = innerW / days.length;
    const barW = Math.max(3, Math.min(28, slot - 4));
    const y = (v) => padT + innerH - (v / max) * innerH;
    let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="height:${H}px">`;
    svg += `<g class="grid">` + [0, 0.25, 0.5, 0.75, 1].map((f) => `<line x1="${padL}" x2="${W - padR}" y1="${y(max * f)}" y2="${y(max * f)}"/>`).join("") + `</g>`;
    svg += `<g class="axis">` + [0, 0.5, 1].map((f) => `<text x="${padL - 6}" y="${y(max * f) + 4}" text-anchor="end">${fmt(max * f)}</text>`).join("") + `</g>`;
    days.forEach((d, i) => {
      const u = byDay[d];
      const x = padL + i * slot + (slot - barW) / 2;
      let cursor = y(0);
      const tipHtml = `<b>${d}</b><br>` + SERIES.map((s) => `${s.label}: ${fmt(u[s.key])}`).join("<br>") + `<br>Total: ${fmt(u.total)}`;
      SERIES.forEach((s) => {
        const h = (u[s.key] / max) * innerH;
        if (h <= 0) return;
        const top = cursor - h;
        svg += `<rect class="seg" x="${x}" y="${top}" width="${barW}" height="${Math.max(0, h - 2)}" fill="${s.color}" data-tip="${esc(tipHtml)}"/>`;
        cursor = top;
      });
      const every = Math.ceil(days.length / Math.max(1, Math.floor(innerW / 64)));
      if (i % every === 0 || i === days.length - 1) svg += `<text class="axis" x="${x + barW / 2}" y="${H - 8}" text-anchor="middle" fill="var(--text-muted)" font-size="11">${d.slice(5)}</text>`;
    });
    svg += `</svg>`;
    host.innerHTML = svg;
    bindTips(host);
  }

  function renderHistogram(hist, th) {
    const total = Object.values(hist).reduce((a, b) => a + b, 0) || 1;
    const rows = Object.entries(hist).map(([name, count]) => ({
      name: name + " tokens", value: count, share: count / total,
      cls: name === ">400k" ? "bad" : name === "150-400k" ? "warn" : "",
      tip: `<b>${esc(name)}</b> context per turn<br>${full(count)} turns · ${pct(count, total)}`,
    }));
    $("hist").innerHTML = hbars(rows, (r) => `${pct(r.value, total)} · ${fmt(r.value)} turns`);
    bindTips($("hist"));
  }

  function renderModels(r) {
    const entries = Object.entries(r.by_model).filter(([m]) => m !== "<synthetic>");
    const max = Math.max(...entries.map(([, u]) => u.total)) || 1;
    const rows = entries.map(([model, u]) => {
      const sub = (r.subagent_by_model[model] || {}).total || 0;
      return { name: model, value: u.total, share: u.total / max, tip: `<b>${esc(model)}</b><br>Total: ${fmt(u.total)}<br>Output: ${fmt(u.output_tokens)}<br>In subagents: ${fmt(sub)} (${pct(sub, u.total)})` };
    });
    $("models").innerHTML = hbars(rows, (row) => fmt(row.value));
    bindTips($("models"));
    const subTotal = r.subagent_usage.total;
    $("sub-hint").textContent = subTotal
      ? `Subagents used ${fmt(subTotal)} (${pct(subTotal, r.usage.total)}). They inherit the parent's model unless told otherwise.`
      : "No subagent runs in this range.";
  }

  function renderTools(tools) {
    const entries = Object.entries(tools).slice(0, TOP_ROWS);
    const max = Math.max(...entries.map(([, t]) => t.bytes)) || 1;
    const rows = entries.map(([name, t]) => ({ name, value: t.bytes, share: t.bytes / max, tip: `<b>${esc(name)}</b><br>${full(t.count)} calls<br>${bytes(t.bytes)} returned · ~${fmt(t.bytes / 4)} tokens` }));
    $("tools").innerHTML = hbars(rows, (row) => bytes(row.value));
    bindTips($("tools"));
  }

  function hbars(rows, valueText) {
    if (!rows.length) return `<div class="empty">Nothing here.</div>`;
    return rows
      .map((r) => `<div class="hbar" data-tip="${attr(r.tip || "")}">
        <div class="name ${r.mono ? "mono" : ""}" title="${esc(r.name)}">${esc(r.name)}</div>
        <div class="track"><div class="fill ${r.cls || ""}" style="width:${Math.max(1, r.share * 100).toFixed(2)}%"></div></div>
        <div class="num">${esc(valueText(r))}</div></div>`)
      .join("");
  }

  function renderFiles(files) {
    const rows = Object.entries(files).slice(0, TOP_ROWS);
    $("files").innerHTML = table(["File", "Reads", "Bytes"], rows.map(([p, t]) => [
      `<td class="mono" title="${esc(p)}">${esc(short(p, 70))}</td>`, `<td class="num">${full(t.count)}</td>`, `<td class="num">${bytes(t.bytes)}</td>`]));
  }

  function renderCommands(commands) {
    const rows = Object.entries(commands).slice(0, TOP_ROWS);
    $("commands").innerHTML = table(["Command", "Runs", "Output"], rows.map(([c, t]) => [
      `<td class="mono" title="${esc(c)}">${esc(short(c, 70))}</td>`, `<td class="num">${full(t.count)}</td>`, `<td class="num">${bytes(t.bytes)}</td>`]));
  }

  function renderSessions(sessions, th) {
    const rows = sessions.filter((s) => !s.is_subagent).slice(0, 25);
    const body = rows.map((s) => {
      const share = s.share_over_threshold;
      const cls = share >= BLOAT_BAD ? "bad" : share >= BLOAT_WARN ? "warn" : "";
      return `<tr data-session="${esc(s.session_id)}">
        <td>${esc(day(s.start))}</td>
        <td title="${esc(s.first_prompt)}">${esc(short(s.first_prompt, 60)) || "<span class=muted>(no prompt)</span>"}<div class="muted">${esc(short(s.project.replace(/^-Users-[^-]+-/, ""), 40))}</div></td>
        <td class="num">${full(s.turns)}</td>
        <td class="num">${fmt(s.usage.total)}</td>
        <td class="num">${fmt(s.subagent_usage.total)}</td>
        <td class="num">${fmt(s.peak_context)}</td>
        <td class="num"><span class="pct ${cls}">${(share * 100).toFixed(0)}%</span></td>
        <td class="num"><span class="pct ${s.health < 50 ? "bad" : s.health < 75 ? "warn" : ""}">${s.health}</span></td></tr>`;
    });
    const head = ["Date", "Session", "Turns", "Tokens", "Subagents", "Peak context", `Turns > ${fmt(th.context_tokens || 150000)}`, "Health"];
    $("sessions").innerHTML = `<thead><tr>${head.map((h, i) => `<th class="${i >= 2 ? "num" : ""}">${h}</th>`).join("")}</tr></thead><tbody>${body.join("")}</tbody>`;
    $("sessions").querySelectorAll("tbody tr").forEach((tr) => tr.addEventListener("click", () => openSession(tr.dataset.session)));
  }

  function table(headers, rows) {
    if (!rows.length) return `<tbody><tr><td class="muted">Nothing here.</td></tr></tbody>`;
    return `<thead><tr>${headers.map((h, i) => `<th class="${i ? "num" : ""}">${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.join("")}</tr>`).join("")}</tbody>`;
  }

  // ---------- session drawer ----------
  async function openSession(id) {
    const drawer = $("drawer");
    $("drawer-title").textContent = "Loading…";
    $("drawer-sub").textContent = id;
    $("drawer-body").innerHTML = "";
    drawer.classList.add("open");
    $("scrim").classList.add("open");
    drawer.setAttribute("aria-hidden", "false");
    try {
      const s = await api("/api/session/" + encodeURIComponent(id) + "?days=" + state.days);
      renderSession(s);
    } catch (err) {
      $("drawer-body").innerHTML = `<div class="empty">${esc(err.message)}</div>`;
    }
  }
  function closeSession() {
    $("drawer").classList.remove("open");
    $("scrim").classList.remove("open");
    $("drawer").setAttribute("aria-hidden", "true");
  }

  function renderSession(s) {
    const th = (state.meta && state.meta.thresholds) || {};
    $("drawer-title").textContent = short(s.first_prompt, 90) || "Session";
    $("drawer-sub").textContent = `${s.session_id} · ${s.project} · ${day(s.start)} → ${day(s.end)}`;
    const u = s.usage;
    const subTotal = s.subagents.reduce((a, c) => a + c.usage.total, 0);
    const tiles = [
      ["Turns", full(s.turns)], ["Tokens", fmt(u.total)], ["Peak context", fmt(s.peak_context)],
      ["Cache re-reads", pct(u.cache_read_input_tokens, u.total)], ["Subagents", `${s.subagents.length} · ${fmt(subTotal)}`],
    ];
    const tools = Object.entries(s.tools).slice(0, 8);
    const maxTool = Math.max(...tools.map(([, t]) => t.bytes)) || 1;
    const files = Object.entries(s.files).slice(0, 8);
    $("drawer-body").innerHTML = `
      <div class="tiles">${tiles.map(([l, v]) => `<div class="tile"><div class="label">${l}</div><div class="value" style="font-size:20px">${v}</div></div>`).join("")}</div>
      <section><h3>Context per turn</h3><div class="chart" id="ctx-chart"></div>
        <div class="hint">Red line: the ${fmt(th.context_tokens || 150000)} threshold. Everything above it is re-read on every turn.</div></section>
      <section><h3>Bytes pushed into context, by tool</h3><div class="hbars">${hbars(tools.map(([n, t]) => ({ name: n, value: t.bytes, share: t.bytes / maxTool, tip: `<b>${esc(n)}</b><br>${full(t.count)} calls · ${bytes(t.bytes)}` })), (r) => bytes(r.value))}</div></section>
      <section><h3>Files read</h3><table class="table">${table(["File", "Reads", "Bytes"], files.map(([p, t]) => [`<td class="mono">${esc(short(p, 60))}</td>`, `<td class="num">${full(t.count)}</td>`, `<td class="num">${bytes(t.bytes)}</td>`]))}</table></section>
      <section><h3>Start fresh, with everything carried over</h3>
        <div class="explain-row"><button class="ghost" id="handoff-btn">Generate handoff brief</button><span class="hint">Goal, files in play, commands, last statements and a ready-to-paste opening prompt. Built locally, no model.</span></div>
        <div id="handoff-out"></div></section>
      ${s.commits.length ? `<section><h3>Tokens per commit</h3><table class="table">${table(["Turn", "Turns since previous", "Tokens", "Command"], s.commits.map((c) => [`<td class="num">${full(c.turn)}</td>`, `<td class="num">${full(c.turns)}</td>`, `<td class="num">${fmt(c.tokens)}</td>`, `<td class="mono">${esc(short(c.command, 60))}</td>`]))}</table></section>` : ""}
      <section id="explain-section"><h3>Why, in plain English</h3>${explainBlock(s.session_id)}</section>
      ${s.subagents.length ? `<section><h3>Subagents spawned</h3><table class="table">${table(["Agent", "Model", "Turns", "Tokens"], s.subagents.map((c) => [`<td class="mono">${esc(short(c.agent_id, 28))}</td>`, `<td>${esc(c.model)}</td>`, `<td class="num">${full(c.turns)}</td>`, `<td class="num">${fmt(c.usage.total)}</td>`]))}</table></section>` : ""}
    `;
    renderContextChart($("ctx-chart"), s.timeline, th.context_tokens || 150000);
    bindTips($("drawer-body"));
    const btn = $("explain-btn");
    if (btn) btn.addEventListener("click", () => explainSession(s.session_id, btn.dataset.refresh === "1"));
    $("handoff-btn").addEventListener("click", () => handoffSession(s.session_id));
  }

  async function handoffSession(sessionId) {
    const out = $("handoff-out");
    out.innerHTML = `<div class="hint">Building…</div>`;
    try {
      const h = await api("/api/handoff/" + encodeURIComponent(sessionId) + "?days=" + state.days);
      out.innerHTML = `<textarea class="handoff" readonly rows="14">${esc(h.markdown)}</textarea>
        <div class="explain-row"><button class="ghost" id="handoff-copy">Copy to clipboard</button><span class="hint" id="handoff-msg">Paste the last block into a new session.</span></div>`;
      $("handoff-copy").addEventListener("click", async () => {
        try { await navigator.clipboard.writeText(h.markdown); $("handoff-msg").textContent = "Copied."; } catch { $("handoff-msg").textContent = "Select the text and copy manually."; }
      });
    } catch (err) {
      out.innerHTML = `<div class="empty">${esc(err.message)}</div>`;
    }
  }

  function explainBlock(sessionId) {
    const llm = (state.meta && state.meta.llm) || {};
    if (!llm.configured) {
      return `<div class="hint">Set <code>OPENROUTER_API_KEY</code> (or <code>BURNLENS_LLM_API_KEY</code>) and restart to ask a model why this session burned tokens. Only a digest is sent: prompts, file names, command heads, sizes. Never file contents.</div>`;
    }
    return `<div class="explain-row"><button class="ghost" id="explain-btn">Explain with ${esc(llm.model)}</button>
      <span class="hint">Sends a digest of this session (prompts, file names, commands, sizes) to ${esc(llm.model)}. No file contents.</span></div>
      <div id="explain-out"></div>`;
  }

  async function explainSession(sessionId, refresh) {
    const out = $("explain-out");
    const btn = $("explain-btn");
    btn.disabled = true;
    out.innerHTML = `<div class="hint">Asking the model…</div>`;
    try {
      const e = await api("/api/explain/" + encodeURIComponent(sessionId) + (refresh ? "?refresh=1" : ""));
      out.innerHTML = `
        <p class="narrative">${esc(e.narrative)}</p>
        ${e.drift.length ? `<table class="table">${table(["Turns", "Kind", "Evidence"], e.drift.map((d) => [`<td class="mono">${esc(d.turns)}</td>`, `<td>${esc(d.kind)}</td>`, `<td>${esc(d.evidence)}</td>`]))}</table>` : ""}
        ${e.cut_point != null ? `<div class="cut">A new session at turn <b>${esc(e.cut_point)}</b> would have been cheaper.</div>` : ""}
        <div class="dodont"><div><h4>Don't</h4><ul>${e.dont.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div><div><h4>Do</h4><ul>${e.do.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div></div>
        ${e.better_prompt ? `<div class="better"><h4>Better prompt</h4><div class="prompt">${esc(e.better_prompt)}</div></div>` : ""}
        <div class="hint">${esc(e.model)} · ${new Date(e.created_at).toLocaleString()} · <a href="#" id="explain-again">ask again</a></div>`;
      $("explain-again").addEventListener("click", (ev) => { ev.preventDefault(); explainSession(sessionId, true); });
    } catch (err) {
      out.innerHTML = `<div class="empty">${esc(err.message)}</div>`;
    } finally {
      btn.disabled = false;
    }
  }

  function renderContextChart(host, points, threshold) {
    if (!points.length) { host.innerHTML = ""; return; }
    const W = Math.max(320, host.clientWidth || 600), H = 220, padL = 48, padR = 12, padT = 12, padB = 26;
    const innerW = W - padL - padR, innerH = H - padT - padB;
    const maxTurn = points[points.length - 1].turn || 1;
    const max = Math.max(threshold * 1.1, ...points.map((p) => p.context)) || 1;
    const x = (t) => padL + (t / maxTurn) * innerW;
    const y = (v) => padT + innerH - (v / max) * innerH;
    const path = points.map((p, i) => `${i ? "L" : "M"}${x(p.turn).toFixed(1)},${y(p.context).toFixed(1)}`).join(" ");
    const area = `${path} L${x(points[points.length - 1].turn).toFixed(1)},${y(0)} L${x(points[0].turn).toFixed(1)},${y(0)} Z`;
    let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="height:${H}px">`;
    svg += `<g class="grid">` + [0, 0.5, 1].map((f) => `<line x1="${padL}" x2="${W - padR}" y1="${y(max * f)}" y2="${y(max * f)}"/>`).join("") + `</g>`;
    svg += `<g class="axis">` + [0, 0.5, 1].map((f) => `<text x="${padL - 6}" y="${y(max * f) + 4}" text-anchor="end">${fmt(max * f)}</text>`).join("") + `</g>`;
    svg += `<line class="threshold" x1="${padL}" x2="${W - padR}" y1="${y(threshold)}" y2="${y(threshold)}"/>`;
    svg += `<text class="threshold-label" x="${W - padR}" y="${y(threshold) - 4}" text-anchor="end">${fmt(threshold)}</text>`;
    svg += `<path class="area" d="${area}"/><path class="line" d="${path}"/>`;
    svg += `<line class="crosshair" id="ctx-cross" x1="0" x2="0" y1="${padT}" y2="${padT + innerH}" style="display:none"/>`;
    svg += `<circle class="marker" id="ctx-dot" r="4" style="display:none"/>`;
    svg += [0, 0.5, 1].map((f) => `<text class="axis" x="${x(maxTurn * f)}" y="${H - 8}" text-anchor="${f === 0 ? "start" : f === 1 ? "end" : "middle"}" fill="var(--text-muted)" font-size="11">turn ${Math.round(maxTurn * f)}</text>`).join("");
    svg += `<rect class="hit" x="${padL}" y="${padT}" width="${innerW}" height="${innerH}" fill="transparent"/></svg>`;
    host.innerHTML = svg;
    const svgEl = host.querySelector("svg"), cross = host.querySelector("#ctx-cross"), dot = host.querySelector("#ctx-dot");
    host.querySelector(".hit").addEventListener("mousemove", (e) => {
      const rect = svgEl.getBoundingClientRect();
      const turn = ((e.clientX - rect.left) / rect.width * W - padL) / innerW * maxTurn;
      let best = points[0];
      for (const p of points) if (Math.abs(p.turn - turn) < Math.abs(best.turn - turn)) best = p;
      cross.setAttribute("x1", x(best.turn)); cross.setAttribute("x2", x(best.turn)); cross.style.display = "";
      dot.setAttribute("cx", x(best.turn)); dot.setAttribute("cy", y(best.context)); dot.style.display = "";
      showTip(`<b>turn ${best.turn}</b> · ${best.ts ? new Date(best.ts).toLocaleString() : ""}<br>context ${full(best.context)}<br>output ${full(best.output)}${best.tools.length ? `<br>tools: ${esc(best.tools.join(", "))}` : ""}`, e.clientX, e.clientY);
    });
    host.querySelector(".hit").addEventListener("mouseleave", () => { cross.style.display = "none"; dot.style.display = "none"; hideTip(); });
  }

  // ---------- live ----------
  const LIVE_POLL_MS = 5000;
  const ago = (sec) => (sec < 60 ? `${Math.round(sec)}s ago` : sec < 3600 ? `${Math.round(sec / 60)}m ago` : `${(sec / 3600).toFixed(1)}h ago`);

  async function pollLive() {
    try {
      const live = await api("/api/live");
      renderLive(live);
    } catch (err) {
      $("live-hint").textContent = "Live view unavailable: " + err.message;
    } finally {
      setTimeout(pollLive, LIVE_POLL_MS);
    }
  }

  function paybackTag(p) {
    if (!p) return "";
    const turns = Math.ceil(p.breakeven_requests);
    const room = p.window_headroom_requests === null || p.window_headroom_requests === undefined
      ? ""
      : `, room for about ${p.window_headroom_requests}`;
    return `<div class="payback-tag" title="${esc(p.basis)}">compact: breaks even in about ${turns} turn${turns === 1 ? "" : "s"}${room} <span class="tag">projected</span></div>`;
  }

  function renderLive(live) {
    state.live = live;
    const th = (state.meta && state.meta.thresholds) || {};
    const ctxTh = th.context_tokens || 150000;
    const burnWarn = th.live_burn_warn_per_min || 1e6;
    const zone = $("zone");
    zone.dataset.zone = live.zone;
    const busy = live.sessions.filter((s) => s.seconds_idle < 120).length;
    $("zone-text").textContent = { green: "Green", amber: "Amber", red: "Red" }[live.zone] + (live.sessions.length ? ` · ${busy} working, ${live.sessions.length} active` : " · idle");
    renderBrake(live);
    const MAX_ALERTS = 5;
    $("alerts").innerHTML = live.alerts.slice(0, MAX_ALERTS).map((a) => `<div class="alert ${esc(a.level)}"><span class="lvl">${esc(a.level)}</span><span>${esc(a.message)}</span></div>`).join("")
      + (live.alerts.length > MAX_ALERTS ? `<div class="more">+${live.alerts.length - MAX_ALERTS} more alerts, see the session cards</div>` : "");
    if (!live.sessions.length) {
      $("live").innerHTML = `<div class="empty">No session has been active in the last ${th.live_active_minutes || 10} minutes.</div>`;
      return;
    }
    $("live").innerHTML = live.sessions.map((s) => {
      const ctxCls = s.context_now >= ctxTh * (th.live_context_high_multiplier || 2) ? "bad" : s.context_now >= ctxTh ? "warn" : "";
      const burnCls = s.tokens_per_min >= (th.live_burn_high_per_min || 3e6) ? "bad" : s.tokens_per_min >= burnWarn ? "warn" : "";
      const working = s.seconds_idle < 120;
      return `<article class="live-session" data-zone="${esc(s.zone)}" data-session="${esc(s.session_id)}">
        <div class="row"><span class="proj">${esc(short(s.project.replace(/^-Users-[^-]+-/, ""), 34))}</span><span class="when">${working ? "working · " : "idle · "}${ago(s.seconds_idle)}</span></div>
        <div class="prompt" title="${esc(s.last_prompt)}">${esc(short(s.last_prompt, 110)) || "<span class=muted>(no prompt yet)</span>"}</div>
        ${s.last_text ? `<div class="doing">${esc(short(s.last_text, 140))}</div>` : ""}
        <div class="stats">
          <div>context<b class="${ctxCls}">${fmt(s.context_now)}</b></div>
          <div>tokens/min<b class="${burnCls}">${fmt(s.tokens_per_min)}</b></div>
          <div>turns<b>${full(s.turns_total)}</b></div>
        </div>
        <div class="tools">${s.recent_tools.map((t) => `<span title="${esc(t)}">${esc(t)}</span>`).join("")}</div>
        ${s.subagents_active ? `<div class="subs">${s.subagents_active} subagent${s.subagents_active > 1 ? "s" : ""} on ${esc(s.subagent_models.join(", "))}</div>` : ""}
        ${Object.keys(s.brake || {}).length ? `<div class="brake-tag">brake: ${Object.entries(s.brake).map(([k, v]) => `<b class="${esc(k)}">${v} ${esc(k)}</b>`).join(" · ")}</div>` : ""}
        ${paybackTag(s.compaction)}
        <div class="muted" style="font-size:11px">${esc(s.model)} · ${esc(s.session_id.slice(0, 8))}</div>
      </article>`;
    }).join("");
    $("live").querySelectorAll(".live-session").forEach((el) => el.addEventListener("click", () => openSession(el.dataset.session)));
  }

  function renderBrake(live) {
    const counts = live.hook_counts || {};
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    const th = (state.meta && state.meta.thresholds) || {};
    if (!total) {
      $("brake").innerHTML = `<div class="brake-counts"><span class="label">brake</span><span class="hint">No hook decisions in the last ${th.live_active_minutes || 10} min. Install with <code>burnlens install-hooks</code> if you have not.</span></div>`;
      return;
    }
    const order = ["deny", "block", "ask", "coach", "allow"];
    const labels = { deny: "denied", block: "stopped", ask: "asked you", coach: "coached", allow: "allowed" };
    const chips = order.filter((k) => counts[k]).map((k) => `<span class="chip ${k}">${full(counts[k])} ${labels[k]}</span>`).join("");
    const rows = (live.hook_events || []).map((e) => {
      const t = e.ts ? new Date(e.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "";
      return `<div class="brake-row" title="${esc(e.reason || "")}"><span class="t">${esc(t)}</span><span class="p ${esc(e.permission)}">${esc(e.permission)}</span><span class="what">${esc(e.tool || "")}${e.target ? " " + esc(e.target) : ""}</span><span class="why">${esc(e.reason || "")}</span></div>`;
    }).join("");
    $("brake").innerHTML = `<div class="brake-counts"><span class="label">brake · last ${th.live_active_minutes || 10} min</span>${chips}</div>${rows ? `<div class="brake-list">${rows}</div>` : ""}`;
  }

  // ---------- controls ----------
  function setRange(days) {
    state.days = days;
    writeStore("days", days);
    document.querySelectorAll("#range button").forEach((b) => b.setAttribute("aria-selected", String(Number(b.dataset.days) === days)));
    load();
  }
  $("range").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setRange(Number(b.dataset.days)); });
  $("refresh").addEventListener("click", load);
  $("map-mode").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; writeStore("map", b.dataset.mode); if (state.graph) loadMap(); });
  $("theme").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme === "dark" || (!root.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    root.dataset.theme = dark ? "light" : "dark";
    writeStore("theme", root.dataset.theme);
  });
  $("drawer-close").addEventListener("click", closeSession);
  $("scrim").addEventListener("click", closeSession);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSession(); });
  let resizeTimer;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (state.report) { renderDayChart(state.report.by_day); renderTreemap(state.report.sessions_detail || []); } if (state.graph) loadMap(); }, 200); });

  const urlTheme = new URLSearchParams(location.search).get("theme");
  const savedTheme = urlTheme || readStore("theme", "dark"); // dark tech look by default; Theme button flips it
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  document.querySelectorAll("#range button").forEach((b) => b.setAttribute("aria-selected", String(Number(b.dataset.days) === state.days)));
  load();
  pollLive();
})();
