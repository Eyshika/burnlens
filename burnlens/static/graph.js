/* Token flow map: force-directed graph on a canvas. Vanilla JS, no dependencies.
   Exposes window.BurnlensGraph.render(canvas, graph, handlers). */
(() => {
  "use strict";

  const COLORS = {
    session: "--series-1", file: "--series-3", command: "--series-4", subagent: "--series-2", model: "--series-7", agent: "--series-6", user: "--series-5", workflow: "--series-8", project: "--text-muted",
  };
  const ITERATIONS = 320;
  const REPULSION = 14000;
  const SPRING = 0.012;
  const GRAVITY = 0.0035;
  const DAMPING = 0.85;
  const MIN_R = 3, MAX_R = 24;
  const LABEL_R = 13;

  function cssVar(name, el) { return getComputedStyle(el).getPropertyValue(name).trim() || "#888"; }

  function layout(nodes, edges, W, H) {
    const idx = new Map(nodes.map((n, i) => [n.id, i]));
    const maxW = Math.max(1, ...nodes.map((n) => n.weight));
    nodes.forEach((n, i) => {
      const a = (i / nodes.length) * Math.PI * 2;
      n.x = W / 2 + Math.cos(a) * W * 0.38 + (Math.random() - 0.5) * 60;
      n.y = H / 2 + Math.sin(a) * H * 0.38 + (Math.random() - 0.5) * 60;
      n.vx = 0; n.vy = 0;
      n.r = MIN_R + (MAX_R - MIN_R) * Math.sqrt(n.weight / maxW);
    });
    const links = edges.map((e) => ({ s: idx.get(e.source), t: idx.get(e.target), w: e.weight })).filter((l) => l.s != null && l.t != null);
    for (let it = 0; it < ITERATIONS; it++) {
      const cool = 1 - it / ITERATIONS;
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          let dx = a.x - b.x, dy = a.y - b.y;
          let d2 = dx * dx + dy * dy + 0.01;
          const minD = a.r + b.r + 6;
          if (d2 < minD * minD) d2 = minD * minD * 0.5;
          const f = REPULSION / d2;
          const d = Math.sqrt(d2);
          dx /= d; dy /= d;
          a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
        }
        a.vx += (W / 2 - a.x) * GRAVITY; a.vy += (H / 2 - a.y) * GRAVITY;
      }
      for (const l of links) {
        const a = nodes[l.s], b = nodes[l.t];
        const dx = b.x - a.x, dy = b.y - a.y;
        const d = Math.sqrt(dx * dx + dy * dy) + 0.01;
        const rest = 110 + a.r + b.r;
        const f = (d - rest) * SPRING;
        a.vx += (dx / d) * f; a.vy += (dy / d) * f; b.vx -= (dx / d) * f; b.vy -= (dy / d) * f;
      }
      for (const n of nodes) {
        n.vx *= DAMPING; n.vy *= DAMPING;
        n.x = Math.max(n.r + 4, Math.min(W - n.r - 4, n.x + n.vx * cool));
        n.y = Math.max(n.r + 4, Math.min(H - n.r - 4, n.y + n.vy * cool));
      }
    }
    return links;
  }

  function render(canvas, graph, handlers) {
    const wrap = canvas.parentElement;
    const W = Math.max(320, wrap.clientWidth), H = Math.max(360, Math.round(W * 0.55));
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr; canvas.height = H * dpr; canvas.style.width = W + "px"; canvas.style.height = H + "px";
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const nodes = graph.nodes.map((n) => ({ ...n }));
    const links = layout(nodes, graph.edges, W, H);
    const colorOf = (n) => cssVar(COLORS[n.type] || "--text-muted", canvas);
    const ink = cssVar("--text-primary", canvas), muted = cssVar("--text-muted", canvas), border = cssVar("--border", canvas);
    let hover = null, pinned = null;

    function neighbors(n) {
      const set = new Set([n.id]);
      for (const e of graph.edges) { if (e.source === n.id) set.add(e.target); if (e.target === n.id) set.add(e.source); }
      return set;
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const focus = pinned || hover;
      const near = focus ? neighbors(focus) : null;
      const maxL = Math.max(1, ...links.map((l) => l.w));
      for (const l of links) {
        const a = nodes[l.s], b = nodes[l.t];
        const active = !near || (near.has(a.id) && near.has(b.id));
        ctx.strokeStyle = active ? border : "rgba(128,128,128,0.06)";
        ctx.lineWidth = active ? 0.6 + 2.4 * Math.sqrt(l.w / maxL) : 0.5;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      for (const n of nodes) {
        const active = !near || near.has(n.id);
        ctx.globalAlpha = active ? 1 : 0.18;
        const c = colorOf(n);
        if (active && focus && n.id === focus.id) { ctx.shadowColor = c; ctx.shadowBlur = 18; } else { ctx.shadowBlur = 0; }
        ctx.fillStyle = c;
        ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0;
        const focused = focus && n.id === focus.id;
        if (n.r >= LABEL_R || focused || (near && near.has(n.id) && n.type !== "session")) {
          ctx.fillStyle = active ? ink : muted;
          ctx.font = `${n.type === "session" ? 600 : 400} 11px ui-monospace, Menlo, monospace`;
          ctx.textAlign = "center";
          ctx.fillText(n.label.length > 26 ? n.label.slice(0, 25) + "…" : n.label, n.x, n.y + n.r + 12);
        }
        ctx.globalAlpha = 1;
      }
    }

    function hit(x, y) {
      let best = null, bd = 1e9;
      for (const n of nodes) { const d = Math.hypot(n.x - x, n.y - y); if (d < n.r + 6 && d < bd) { best = n; bd = d; } }
      return best;
    }
    canvas.onmousemove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const n = hit(e.clientX - rect.left, e.clientY - rect.top);
      if (n !== hover) { hover = n; draw(); }
      canvas.style.cursor = n ? "pointer" : "default";
      if (n && handlers.tip) handlers.tip(n, e.clientX, e.clientY); else if (handlers.untip) handlers.untip();
    };
    canvas.onmouseleave = () => { hover = null; draw(); if (handlers.untip) handlers.untip(); };
    canvas.onclick = (e) => {
      const rect = canvas.getBoundingClientRect();
      const n = hit(e.clientX - rect.left, e.clientY - rect.top);
      pinned = n && pinned && pinned.id === n.id ? null : n;
      draw();
      if (n && handlers.click) handlers.click(n);
    };
    draw();
    return { redraw: draw };
  }

  window.BurnlensGraph = { render };
})();
