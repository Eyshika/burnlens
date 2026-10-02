/* Local, explicit review/apply/measure workflow. No commands execute in the browser. */
(() => {
  "use strict";
  const host = document.getElementById("actions");
  const message = document.getElementById("action-message");
  const escape = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
  const number = n => Math.round(n || 0).toLocaleString();
  let token = "";
  let loading = false;
  let actions = [];

  async function request(path, body) {
    const options = body ? {method:"POST", headers:{"Content-Type":"application/json", "X-Burnlens-Token":token}, body:JSON.stringify(body)} : {cache:"no-store"};
    const response = await fetch(path, options);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not load workflow");
    return result;
  }

  function renderRun(action, run) {
    return `<li class="action-run" data-run="${escape(run.run_id)}">
      <div><b>${escape(run.task_label)}</b> · ${run.baseline ? "Baseline" : "Trial"} · exit ${escape(run.exit_code)} · ${Number(run.duration_seconds).toFixed(2)}s</div>
      <p>${number(run.displayed_bytes)} / ${number(run.output_bytes)} command-output bytes displayed. Full log: <code>${escape(run.output_path)}</code></p>
      <label>Task outcome (your assessment) <select aria-label="Outcome for ${escape(run.task_label)}">
        ${["unknown", "success", "rework", "failed"].map(v => `<option value="${v}" ${v === run.outcome ? "selected" : ""}>${v}</option>`).join("")}
      </select></label>
      <label>Evidence or rework notes <input maxlength="4096" value="${escape(run.notes)}" aria-label="Outcome notes for ${escape(run.task_label)}"></label>
      <button class="ghost" data-operation="outcome" data-action="${escape(action.id)}">Save outcome</button>
    </li>`;
  }

  const VISIBLE_ACTIONS = 2;

  function render() {
    if (!actions.length) {
      host.innerHTML = '<p class="empty">No oversized shell output found in the available transcripts. Once a Bash result exceeds your configured limit, a reviewable policy will appear here.</p>';
      return;
    }
    const card = action => {
      const m = action.measurement;
      return `<article class="action-card" data-id="${escape(action.id)}">
        <div class="card-head"><h3>Capture large shell output</h3><span class="badge">${escape(action.state)}</span></div>
        <p class="action-project">${escape(action.project)} · ${escape(action.agent)}</p>
        <p><b>${number(action.observed_count)}</b> oversized results returned <b>${number(action.observed_bytes)} bytes</b> in the source history. This is an opportunity to evaluate, not proven waste.</p>
        <details class="action-review"><summary>1. Review the exact policy and scope</summary>
          <pre>${escape(action.policy_text)}</pre>
          <p>Scope: ${action.transcript_dirs.map(escape).join(" · ")}</p>
          <p>${action.agent === "claude-code" ? "Apply activates guidance for installed Claude Code hooks in the matching scope." : "Apply enables explicit capture trials. This agent has no automatic guidance hook; copy the policy into your task instructions."} Commands run only when you explicitly invoke capture. Revert removes guidance; measurements and logs remain.</p>
          <button class="ghost" data-operation="copy" data-action="${escape(action.id)}">Copy policy and command</button>
          ${action.state !== "applied" ? `<label class="action-consent"><input type="checkbox" data-reviewed="${escape(action.id)}"> I reviewed the policy and scope</label><button class="ghost" disabled data-operation="apply" data-action="${escape(action.id)}">2. Apply reviewed policy</button>` : ""}
        </details>
        ${action.state === "applied" ? `<button class="ghost" data-operation="revert" data-action="${escape(action.id)}">Revert policy</button>` : ""}
        <div class="action-measure"><h4>3. Measure comparable work</h4>
          <p>Use the command above with <code>--baseline</code> for full output, then without it for a trial. Use the same task label for comparable work. Do not repeat a destructive command for measurement.</p>
          <p><b>${number(m.output_bytes_reduced)}</b> command-output bytes omitted from previews · ${number(m.baseline_runs)} baseline runs · ${number(m.trial_runs)} trials.</p>
          <p class="hint">This excludes the capture footer and later log retrieval. Token and dollar savings are unmeasured. ${escape(m.quality)}</p>
          ${m.task_comparisons.length ? `<div class="action-comparisons">${m.task_comparisons.map(c => `<p><b>${escape(c.task_label)}</b>: ${number(c.baseline_runs)} baseline / ${number(c.trial_runs)} trial runs. Mean original output: ${c.baseline_mean_output_bytes == null ? "—" : number(c.baseline_mean_output_bytes)} baseline / ${c.trial_mean_output_bytes == null ? "—" : number(c.trial_mean_output_bytes)} trial bytes; trial preview: ${c.trial_mean_displayed_bytes == null ? "—" : number(c.trial_mean_displayed_bytes)} bytes.</p>`).join("")}</div>` : ""}
          <ul class="action-runs">${action.runs.map(run => renderRun(action,run)).join("")}</ul>
        </div>
      </article>`;
    };
    const rest = actions.slice(VISIBLE_ACTIONS);
    host.innerHTML = actions.slice(0, VISIBLE_ACTIONS).map(card).join("")
      + (rest.length ? `<details class="action-more"><summary>Show ${rest.length} more</summary>${rest.map(card).join("")}</details>` : "");
  }

  async function load() {
    if (loading) return;
    loading = true;
    try {
      const [meta,result] = await Promise.all([request("/api/meta"),request("/api/actions?days=0")]);
      token = meta.action_token;
      actions = result.actions;
      render();
    } catch(error) { message.textContent = error.message; }
    finally { loading = false; }
  }

  host.addEventListener("change", event => {
    const checkbox = event.target.closest("[data-reviewed]");
    if (checkbox) checkbox.closest("article").querySelector('[data-operation="apply"]').disabled = !checkbox.checked;
  });
  host.addEventListener("click", async event => {
    const button = event.target.closest("button[data-operation]");
    if (!button) return;
    const action = actions.find(a => a.id === button.dataset.action);
    if (!action) return;
    const operation = button.dataset.operation;
    button.disabled = true;
    message.textContent = "";
    try {
      if (operation === "copy") {
        await navigator.clipboard.writeText(action.policy_text);
        message.textContent = "Policy copied. Replace the task label and command before running it.";
        return;
      }
      const body = {action_id:action.id,revision:action.revision};
      if (operation === "outcome") {
        const row = button.closest("[data-run]");
        body.run_id = row.dataset.run;
        body.outcome = row.querySelector("select").value;
        body.notes = row.querySelector("input").value;
      }
      const result = await request("/api/actions/" + operation, body);
      actions = actions.map(a => a.id === result.action.id ? result.action : a);
      render();
      message.textContent = operation === "apply" ? "Policy applied. Run an explicit capture trial to measure its effect." : operation === "revert" ? "Policy reverted. Existing measurements remain available." : "Your outcome assessment was saved.";
    } catch(error) { message.textContent = error.message; }
    finally { button.disabled = false; }
  });
  document.getElementById("refresh-actions").addEventListener("click", load);
  document.getElementById("refresh").addEventListener("click", load);
  load();
})();
