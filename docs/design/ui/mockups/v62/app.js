// Standalone design exploration. Synthetic data; no network or Server mutations.
const params = new URLSearchParams(location.search);
const variant = document.body.dataset.variant;
const state = ["preparing", "running", "failed"].includes(params.get("state"))
  ? params.get("state")
  : "running";
const theme = ["light", "dark", "black"].includes(params.get("theme"))
  ? params.get("theme")
  : "light";
const urlMode = params.get("mode") === "url";
document.documentElement.dataset.theme = theme;
const names = {
  a: "Guided overview",
  b: "Endpoint workspace",
  c: "Coverage first",
};
const checkMode = urlMode ? "URL template checks" : "Request checks";
const engine = urlMode ? "Nuclei" : "SQLMap";
const pill = (text, tone = "") =>
  `<span class="pill ${tone}"><span class="dot"></span>${text}</span>`;
const action = (text, extra = "") => `<button ${extra}>${text}</button>`;
const paths = [
  "/api/orders/{orderId}",
  "/api/products",
  "/api/customers/{customerId}",
  "/api/orders",
  "/api/payments",
  "/api/uploads",
];
const methods = ["GET", "GET", "GET", "POST", "POST", "POST"];
const statuses = [
  "Completed",
  "Checking",
  "Needs approval",
  "Completed",
  "Completed",
  "Not checked",
];
const tones = ["", "info", "warn", "", "", "warn"];
const endpointName = (i) =>
  `<span class="method">${urlMode ? "URL" : methods[i]}</span> <span class="mono row-path">${paths[i]}</span>`;
const exampleUrl = "https://shop.example/api/orders/1042";
const toolbar = `<div class="concept"><strong>V62 · Design exploration</strong><div class="choices">${Object.entries(
  names,
)
  .map(
    ([key, name]) =>
      `<a class="${variant === key ? "selected" : ""}" href="${key}.html?state=${state}&theme=${theme}${urlMode ? "&mode=url" : ""}">${key.toUpperCase()} · ${name}</a>`,
  )
  .join(
    "",
  )}</div><span class="grow spacer"></span><label>State <select data-query="state"><option value="running">Checking</option><option value="preparing">Preparing</option><option value="failed">Preparation failed</option></select></label><label>Theme <select data-query="theme"><option>light</option><option>dark</option><option>black</option></select></label></div>`;
const sidebar = `<aside class="sidebar"><div class="brand"><span class="brand-mark">c</span>contractor</div><nav class="nav"><a class="optional" href="#">Inbox</a><a href="#">Projects</a><a class="active" href="#">Checks</a><a class="optional" href="#">Issues</a><a class="optional" href="#">Reports</a></nav><div class="sidebar-foot">Research workspace<br><strong>Shop API</strong><br><br>Design sample · no live data</div></aside>`;
const status =
  state === "failed"
    ? pill("Failed", "danger")
    : state === "preparing"
      ? pill("Preparing", "info")
      : pill("Running", "info");
const header = `<div class="breadcrumb">Projects <span>/</span> Shop API <span>/</span> Checks <span>/</span> API security check</div><header class="between heading"><div><h1>API security check ${status}</h1><p class="subtitle">Shop API · Source → ${checkMode.toLowerCase()} · ${state === "preparing" ? "Started 2 minutes ago" : state === "failed" ? "Stopped during preparation" : "Started 18 minutes ago"}</p></div><div class="actions">${state === "failed" ? action("Start another check") : action("Pause")}${state === "failed" ? action("View diagnostics") : state === "preparing" ? action("More actions") : action("Review 3 approvals", variant === "b" ? "" : 'class="primary"')}</div></header>`;
const steps = `<nav class="steps" aria-label="Check progress"><div class="step ${state === "running" ? "done" : "current"}"><span class="number">${state === "running" ? "✓" : "1"}</span><div><strong>Prepare API spec</strong><small>${state === "running" ? "Accepted · 32 endpoints" : state === "failed" ? "Needs attention" : "In progress · no endpoints yet"}</small></div></div><div class="step ${state === "running" ? "current" : ""}"><span class="number">2</span><div><strong>Check endpoints</strong><small>${state === "running" ? "24 of 32 completed" : "Waiting for preparation"}</small></div></div><div class="step"><span class="number">3</span><div><strong>Review & report</strong><small>Available after checks settle</small></div></div></nav>`;
const technical = `<details class="technical"><summary>Technical details</summary><p>Sample check ID: audit_sample_v62 · exact API spec revision retained with the accepted inventory. Published results refer to pinned source, settings and evidence revisions. This exploration is not connected to a Server.</p></details>`;
function files(phase = "running") {
  return `<div class="file"><span class="file-icon">ZIP</span><div><a href="#">Source code</a><small>shop-api.zip · version 3 · pinned at start</small></div></div>${phase === "running" ? `<div class="file"><span class="file-icon">API</span><div><a href="#">Generated API spec</a><small>Version 2 · accepted preparation output</small><small>Generated from Source code · version 3</small></div></div>` : phase === "failed" ? `<div class="file"><span class="file-icon">LOG</span><div><a href="#">Preparation diagnostics</a><small>Failed attempt · diagnostic only</small><small>No generated API spec was accepted</small></div></div>` : `<div class="file"><span class="file-icon">API</span><div><strong style="font-size:12px">API spec is being prepared</strong><small>Accepted output will appear here</small></div></div>`}<div class="file"><span class="file-icon">CFG</span><div><a href="#">Scan settings</a><small>Version 1 · ${checkMode.toLowerCase()}</small></div></div>`;
}
const materials = (phase = "running") =>
  `<section class="card"><div class="card-head"><h2>Materials & origin</h2></div><div class="card-body">${files(phase)}</div><div class="card-foot">Original materials stay pinned. Generated files keep their own versions.</div></section>`;
const scanNotice = urlMode
  ? "URL templates only. The OpenAPI method, request body and authenticated session are not replayed."
  : "Request checks use the prepared method, URL and selected test parameters. A completed scan is not a clean security verdict.";
const metrics = `<div class="stats"><section class="card stat"><div class="label">Execution progress</div><div class="value">24 <span>/ 32 endpoints</span></div><div class="progress"><span></span></div><small>4 checking · 3 need approval · 1 not checked</small></section><section class="card stat"><div class="label">Scanner observations</div><div class="value">3 <span>to inspect</span></div><small>Observations require independent verification</small></section><section class="card stat"><div class="label">Security coverage</div><div class="value" style="font-size:23px">Inconclusive</div><small>No confirmed security conclusion from scans alone</small></section></div>`;
function endpointTable(limit = 5, indices = null) {
  return `<table class="rows"><thead><tr><th>Endpoint</th><th>Execution</th><th class="evidence-col">Evidence</th></tr></thead><tbody>${(indices ?? Array.from({ length: limit }, (_, i) => i)).map((i) => `<tr class="${i === 0 ? "selected" : ""}"><td>${endpointName(i)}<small>${i === 5 ? (urlMode ? "Required URL value missing" : "Unsupported request representation") : i === 2 ? (urlMode ? "Active URL check requires your decision" : "Active request check requires your decision") : checkMode + " · " + engine}</small></td><td>${pill(statuses[i], tones[i])}</td><td class="evidence-col">${i === 0 ? '<a href="#">1 observation ↗</a>' : i === 3 || i === 4 ? '<a href="#">Scanner report ↗</a>' : '<span class="muted">Not available yet</span>'}</td></tr>`).join("")}</tbody></table>`;
}
const filters = `<div class="filterbar">${action("All 32", 'class="small active"')}${action("Completed 24", 'class="small"')}${action("Checking 4", 'class="small"')}${action("Needs approval 3", 'class="small"')}${action("Not checked 1", 'class="small"')}</div>`;
const activity = `<section class="card"><div class="card-head"><h2>Latest activity</h2><small>Live updates</small></div><div class="card-body timeline"><div class="event"><time>10:42</time><div>${urlMode ? "URL template check" : "Request check"} completed<small>${urlMode ? "URL" : "GET"} /api/orders/{orderId} · scanner report retained</small></div></div><div class="event"><time>10:40</time><div>Endpoint list accepted<small>32 endpoints · Generated API spec, version 2</small></div></div><div class="event"><time>10:38</time><div>API spec prepared<small>Source code, version 3 → API spec, version 2</small></div></div></div></section>`;
function preparing() {
  return `${steps}<div class="columns"><div class="stack"><section class="card"><div class="waiting"><span class="activity-icon" aria-hidden="true"></span><div><h2>Preparing the API spec</h2><p class="muted">Building the API description from your source code.<br>The accepted endpoint list will appear after preparation finishes.</p><div style="margin-top:15px">${pill("Preparation running", "info")} <small style="margin-left:6px">Attempt 1 of 2</small></div></div></div><div class="card-foot">No endpoint checks have started. Preparation progress has no estimated percentage.</div></section><div class="empty-coverage"><strong>Coverage is not available yet</strong>The endpoint count is unknown until the generated API spec and worklist are accepted.</div><section class="card"><div class="card-head"><h2>What happens next</h2></div><div class="card-body timeline"><div class="event"><span class="pill">2</span><div>Validate and accept the endpoint list<small>The check keeps the exact generated API spec revision.</small></div></div><div class="event"><span class="pill">3</span><div>Review requested active checks<small>Required approvals remain separate from execution progress.</small></div></div><div class="event"><span class="pill">4</span><div>Inspect scanner evidence and the final report<small>Unchecked endpoints and limitations remain visible.</small></div></div></div></section></div>${materials("preparing")}</div>${technical}`;
}
function failed() {
  return `${steps}<div class="failure"><section class="card"><div class="card-body"><div class="label" style="color:var(--danger);margin-bottom:12px">Preparation stopped</div><h2>The API spec could not be validated</h2><p class="muted">The generated document is missing a required field. No endpoint list was accepted, and no scanner checks were started.</p><div class="notice error" style="margin-top:22px"><strong>Required field is missing: info.version</strong>Inspect the failed attempt output before starting another check.</div><div class="actions">${action("Open preparation diagnostics", 'class="primary"')}${action("Start another check")}</div></div><div class="card-foot">Failed-attempt output is diagnostic evidence, not an accepted API spec.</div></section><div class="empty-coverage" style="margin:18px 0"><strong>Coverage unavailable</strong>No endpoint denominator exists. This check has no successful security outcome.</div>${materials("failed")}</div>${technical}`;
}
function overview() {
  return `${steps}${metrics}<div class="columns"><div class="stack"><section class="card"><div class="card-head"><div><h2>Endpoint checks</h2><small>${checkMode} · ${engine}</small></div><a href="#" style="font-size:12px">View all 32 ↗</a></div>${filters}${endpointTable()}<div class="card-foot">Showing 5 of 32 accepted endpoints. Execution and security outcomes are separate.</div></section><div class="notice">${scanNotice}</div></div><aside class="stack">${materials()}<section class="card"><div class="card-head"><h2>Needs your attention</h2></div><div class="card-body"><strong>3 active checks need approval</strong><p class="muted" style="font-size:12px;margin:8px 0 14px">${urlMode ? "Review the exact target and template policy before allowing execution." : "Review the exact target and request before allowing execution."}</p>${action("Review approvals", 'class="small"')}</div></section></aside></div>${technical}`;
}
function workspace() {
  return `<div class="notice" style="margin-bottom:18px">Preparation accepted · <strong style="display:inline">32 endpoints</strong> from Generated API spec, version 2. ${urlMode ? "URL template checks" : "Request checks"} are running.</div><div class="workspace"><aside class="endpoint-list"><div class="list-head"><div class="between"><h2>Endpoints</h2>${pill("24 / 32")}</div><small>4 checking · 3 approvals · 1 gap</small></div>${paths.map((p, i) => `<div class="endpoint ${i === 0 ? "active" : ""}">${endpointName(i)}<small>${checkMode} · ${engine}</small>${pill(statuses[i], tones[i])}</div>`).join("")}<div style="padding:16px;font-size:11px" class="muted">Showing 6 of 32 · <a href="#">Load more</a></div></aside><section class="detail"><div class="between"><span class="label">Selected endpoint</span>${pill("Execution completed")}</div><h2 class="mono" style="margin:13px 0 7px">${urlMode ? "URL" : "GET"} /api/orders/{orderId}</h2><p class="muted" style="font-size:12px">${checkMode} · ${engine} · <strong>Security outcome: inconclusive</strong></p><nav class="tabs"><a class="active" href="#">Observations <strong>1</strong></a><a href="#">Evidence</a><a href="#">Origin</a><a href="#">Activity</a></nav><div class="notice warning"><strong>Scanner observation needs verification</strong>This result records what the scanner reported. It is not a confirmed finding.</div><section class="finding"><div><div class="between"><h3>${urlMode ? "Template match reported" : "Potential injection signal reported"}</h3>${pill("Unverified", "warn")}</div><p class="muted">${urlMode ? "A selected URL template produced a match. Inspect the retained scanner report before drawing a conclusion." : "The scanner reported a response difference for the selected parameter. Reproduce the behavior and inspect its evidence before treating it as an issue."}</p><div style="margin-top:15px">${action("Open scanner report", 'class="small primary"')}</div></div></section><h3 style="margin-top:22px">Evidence & exact origin</h3><div class="evidence-chain"><div><span class="label">Source</span><br><a href="#">API spec · v2</a><small>Accepted preparation output<br>Source code · v3</small></div><div><span class="label">Prepared input</span><br><a href="#">${urlMode ? "Target URL" : "HTTP request"} · v1</a><small>${urlMode ? "Concrete URL and limitations" : "Method, URL and test parameters"}</small></div><div><span class="label">Result</span><br><a href="#">Scanner report · v1</a><small>Exact output retained<br>Execution completed</small></div></div><div class="mono" style="padding:12px 14px;background:var(--sunken);border-radius:7px;overflow-wrap:anywhere">${urlMode ? "URL" : "GET"} ${exampleUrl}</div><p class="muted" style="font-size:11px;margin-top:12px">${scanNotice}</p>${technical}</section></div>`;
}
function coverage() {
  return `<div class="coverage-layout"><div class="stack"><section class="card"><div class="card-head"><div><h2>Endpoint execution</h2><small>Accepted API spec · version 2 · ${checkMode.toLowerCase()}</small></div><a href="#">View endpoint list ↗</a></div><div class="card-body"><div class="between"><div><span class="label">Execution progress</span><h2 style="font-size:27px;margin-top:7px">24 of 32 completed</h2></div>${pill("8 still need execution", "info")}</div><div class="coverage-grid">${Array.from({ length: 32 }, (_, i) => `<div class="cell ${i < 24 ? "completed" : i < 28 ? "running" : i < 31 ? "approval" : "gap"}" title="Endpoint ${i + 1}: ${i < 24 ? "completed" : i < 28 ? "checking" : i < 31 ? "needs approval" : "not checked"}">${i < 24 ? "✓" : i < 28 ? "…" : i < 31 ? "!" : "—"}</div>`).join("")}</div><div class="legend"><span><i></i>Completed 24</span><span class="running"><i></i>Checking 4</span><span class="approval"><i></i>Needs approval 3</span><span class="gap"><i></i>Not checked 1</span></div><div class="metric-strip"><div><strong>3</strong><small>Scanner observations</small></div><div><strong>0</strong><small>Confirmed findings</small></div><div><strong>1</strong><small>Preparation limitation</small></div></div><div class="notice warning"><strong>Security coverage remains inconclusive</strong>Completed scans and empty outputs do not establish that an endpoint is secure.</div></div></section><section class="card"><div class="card-head"><h2>Focus on what needs attention</h2>${pill("3 approvals", "warn")}</div>${endpointTable(0, [2, 5])}<div class="card-foot">Showing 2 of 4 items that need attention · <a href="#">View all</a></div></section><div class="notice">${scanNotice}</div></div><aside class="stack"><section class="card"><div class="card-head"><h2>Check journey</h2></div><div class="card-body"><div class="mini-step done"><span class="number">✓</span><div><strong>Prepare API spec</strong><small>Accepted · exact revision retained</small><a href="#" style="font-size:11px">View generated API spec ↗</a></div></div><div class="mini-step active"><span class="number">2</span><div><strong>Check endpoints</strong><small>24 of 32 completed</small><small>3 approvals need your decision</small></div></div><div class="mini-step"><span class="number">3</span><div><strong>Review & report</strong><small>Waiting for all checks to settle</small></div></div></div></section>${activity}${materials()}</aside></div>${technical}`;
}
document.body.innerHTML = `${toolbar}<div class="shell">${sidebar}<main><div class="page">${header}${state === "preparing" ? preparing() : state === "failed" ? failed() : variant === "a" ? overview() : variant === "b" ? workspace() : coverage()}<p class="bottom-note">${variant.toUpperCase()} · ${names[variant]} · Synthetic design sample · UI alternatives are not an implementation claim.</p></div></main></div>`;
for (const select of document.querySelectorAll("[data-query]")) {
  select.value = select.dataset.query === "state" ? state : theme;
  select.addEventListener("change", () => {
    params.set(select.dataset.query, select.value);
    location.search = params.toString();
  });
}
