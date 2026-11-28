(() => {
  const params = new URLSearchParams(location.search);
  const screens = ["materials", "material", "check"];
  let screen = screens.includes(params.get("screen"))
    ? params.get("screen")
    : "materials";
  history.scrollRestoration = "manual";
  document.documentElement.dataset.theme = params.get("theme") || "dark";
  const examples = [...window.materialExamples];
  const labels = {
    all: "All",
    source: "Source code",
    api: "API specs",
    engagement: "Engagement",
    docs: "Docs",
    architecture: "Architecture",
    diffs: "Diffs",
    other: "Other",
  };
  const kinds = Object.keys(labels);
  const state = {
    category: kinds.includes(params.get("category"))
      ? params.get("category")
      : "all",
    search: params.get("search") || "",
    selected: params.get("material") || "api-main",
    readerTab: "preview",
    type: "source",
    objective: "Can one customer read or export another customer’s invoices?",
    source: "source-main",
    api: "api-main",
    engagement: "",
    time: "24 hours",
    scope: "Invoice ownership and cross-account exports",
    pickerSlot: "source",
    pickerSelected: "source-main",
    pickerSearch: "",
    pickerPreview: false,
    uploadKind: "source",
    uploadFile: null,
    uploadFromPicker: false,
    viewState:
      screen === "materials" ? params.get("state") || "normal" : "normal",
    listScroll: 0,
  };
  const paths = {
    projects:
      '<path d="M3 7V5a2 2 0 0 1 2-2h4l3 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"/>',
    source:
      '<path d="M8 3H5v18h14V3h-3M9 3h6v4H9zM8 12l-2 2 2 2m8-4 2 2-2 2m-5-5-1 6"/>',
    api: '<path d="M8 4H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h3m8-16h3a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-3M10 8l-2 4 2 4m4-8 2 4-2 4"/>',
    engagement:
      '<path d="m12 3 8 4v5c0 5-8 9-8 9s-8-4-8-9V7l8-4Zm-4 9 3 3 5-6"/>',
    docs: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Zm0 0v6h6M8 13h8m-8 4h5"/>',
    architecture:
      '<rect x="8" y="3" width="8" height="5" rx="1"/><rect x="2" y="16" width="8" height="5" rx="1"/><rect x="14" y="16" width="8" height="5" rx="1"/><path d="M12 8v4H6v4m6-4h6v4"/>',
    diffs:
      '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Zm0 0v6h6M8 12h6m-3-3v6m-3 3h6"/>',
    search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    close: '<path d="m6 6 12 12M6 18 18 6"/>',
    arrow: '<path d="M5 12h14m-5-5 5 5-5 5"/>',
    back: '<path d="M19 12H5m5-5-5 5 5 5"/>',
    chevron: '<path d="m9 5 7 7-7 7"/>',
    copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M15 8V4H4v11h4"/>',
    download: '<path d="M12 3v12m-5-5 5 5 5-5M4 17v4h16v-4"/>',
    check: '<path d="m5 12 4 4L19 6"/>',
    upload: '<path d="M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5"/>',
    refresh: '<path d="M20 12a8 8 0 1 1-2.35-5.65M20 4v5h-5"/>',
    runs: '<circle cx="12" cy="12" r="9"/><path d="m10 8 6 4-6 4Z"/>',
    inbox: '<path d="M6 5h12l3 9v6H3v-6L6 5Zm-3 9h5l2 3h4l2-3h5"/>',
    issues: '<path d="m12 3 10 9-10 9L2 12l10-9Zm0 5v5m0 3h.01"/>',
    library: '<path d="M4 3h4v18H4zM10 3h4v18h-4zM16 4l4-1 4 17-4 1z"/>',
    settings:
      '<path d="M3 6h18M3 12h18M3 18h18"/><circle cx="8" cy="6" r="2"/><circle cx="16" cy="12" r="2"/><circle cx="9" cy="18" r="2"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10h.01"/>',
  };
  const escape = (text) =>
    String(text).replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  const icon = (name) =>
    `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.docs}</svg>`;
  const itemOf = (id) => examples.find((item) => item.id === id);
  const iconButton = (action, name, label, extra = "") =>
    `<button type="button" class="btn icon ghost" data-action="${action}" ${extra} title="${label}" aria-label="${label}">${icon(name)}</button>`;
  const kindIcon = (kind) =>
    `<span class="kind-icon" data-kind="${kind}">${icon(kind)}</span>`;
  const button = (text, action, className = "", iconName = "") =>
    `<button type="button" class="btn ${className}" data-action="${action}">${iconName ? icon(iconName) : ""}${text}</button>`;
  const screenHref = (value) =>
    `?screen=${value}&theme=${document.documentElement.dataset.theme}`;
  function materialHref(item) {
    const values = new URLSearchParams(location.search);
    values.set("screen", "material");
    values.set("material", item.id);
    values.set("revision", item.revision);
    values.set("state", "normal");
    values.set("category", state.category);
    values.set("search", state.search);
    return `?${values}`;
  }
  function materialsHref() {
    const values = new URLSearchParams(location.search);
    values.set("screen", "materials");
    values.delete("material");
    values.delete("revision");
    values.set("state", state.viewState);
    values.set("category", state.category);
    values.set("search", state.search);
    return `?${values}`;
  }
  function navigate(href, fromList = false) {
    history.pushState({ materialFromList: fromList }, "", href);
    screen = new URLSearchParams(location.search).get("screen");
    render();
    window.scrollTo(0, screen === "materials" ? state.listScroll : 0);
    document
      .querySelector(screen === "material" ? "h1" : "#material-search")
      ?.focus({ preventScroll: true });
  }
  window.addEventListener("popstate", () => {
    const values = new URLSearchParams(location.search);
    screen = screens.includes(values.get("screen"))
      ? values.get("screen")
      : "materials";
    state.selected = values.get("material") || state.selected;
    state.category = kinds.includes(values.get("category"))
      ? values.get("category")
      : "all";
    state.search = values.get("search") || "";
    state.viewState =
      screen === "materials" ? values.get("state") || "normal" : "normal";
    state.listScroll = history.state?.listScroll ?? state.listScroll;
    state.readerTab = "preview";
    render();
    window.scrollTo(0, screen === "materials" ? state.listScroll : 0);
  });
  let toastTimeout;

  function toast(text) {
    const notice = document.getElementById("toast");
    notice.textContent = text;
    notice.hidden = false;
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
      notice.hidden = true;
    }, 3500);
  }

  function rail() {
    return `<nav class="rail" aria-label="Main navigation"><img class="brand" src="contractor-logo.png" alt="Contractor">${[
      ["inbox", "Inbox"],
      ["projects", "Projects"],
      ["engagement", "Checks"],
      ["issues", "Issues"],
      ["docs", "Reports"],
      ["runs", "Runs"],
      ["library", "Library"],
    ]
      .map(
        ([glyph, title]) =>
          `<a href="${screenHref(title === "Checks" ? "check" : "materials")}" class="${title === (screen === "check" ? "Checks" : "Projects") ? "active" : ""}">${icon(glyph)}<span>${title}</span></a>`,
      )
      .join(
        "",
      )}<a href="#" data-action="demo-link" class="bottom">${icon("settings")}<span>Operations</span></a></nav>`;
  }

  function context() {
    return `<aside class="context" aria-label="Projects"><div class="context-head"><h2>Projects</h2><div class="search">${icon("search")}<span>Filter projects</span></div></div><a href="#" class="context-row active"><strong>Invoice Lab</strong><small>2 possible issues to review</small></a><a href="#" data-action="demo-link" class="context-row"><strong>Payment Portal</strong><small>Last check completed</small></a><a href="#" data-action="demo-link" class="context-row"><strong>Customer Workspace</strong><small>Updated yesterday</small></a></aside>`;
  }

  function render() {
    document.title = `${screen === "check" ? "New check" : screen === "material" ? itemOf(state.selected)?.name || "Material" : "Materials"} · Contractor`;
    document.getElementById("app").innerHTML =
      `<div class="shell">${rail()}${context()}<main class="workspace"><div class="topbar"><span class="search-shortcut">${icon("search")}Search or start a check… <kbd>⌘ K</kbd></span></div><div class="page">${screen === "check" ? checkPage() : screen === "material" ? materialPage() : materialsPage()}</div></main></div>`;
    window.parent.postMessage(
      { type: "materials-preview-route", query: location.search },
      location.origin,
    );
  }

  function materialsPage() {
    return `<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#" data-action="demo-link">Projects</a>${icon("chevron")}<span>Invoice Lab</span></nav><div class="page-heading"><div><h1>Invoice Lab</h1><p>Security review of the invoice API and its supporting services.</p></div><div class="toolbar">${iconButton("refresh", "refresh", "Refresh project")}</div></div><nav class="project-tabs" aria-label="Project sections">${["Overview", "Materials", "Checks", "Issues", "Facts", "Settings"].map((label) => `<a href="${label === "Checks" ? screenHref("check") : "#"}" ${label === "Materials" ? 'class="active" aria-current="page"' : label === "Checks" ? "" : 'data-action="demo-link"'}>${label}</a>`).join("")}</nav><div class="section-title"><div><h2>Materials</h2><p>${state.viewState === "normal" ? `${examples.length} materials in this project` : state.viewState === "empty" ? "No materials yet" : state.viewState === "loading" ? "Loading materials…" : "Materials unavailable"}</p></div>${button("Add material", "add", "primary", "plus")}</div><div class="library library-list"><section class="library-main" aria-label="Project materials">${searchField("material-search", "Search materials", state.search)}${categoryFilters()}<div id="material-results">${materialResults()}</div></section></div>`;
  }

  function materialPage() {
    const item = itemOf(state.selected);
    const revision = new URLSearchParams(location.search).get("revision");
    const back = `<a class="btn ghost material-back" data-action="back-list" href="${escape(materialsHref())}">${icon("back")}Back to materials</a>`;
    if (!item || (revision && revision !== item.revision))
      return `${back}<div class="empty"><h1 tabindex="-1">Material version unavailable</h1><p>This saved version is not included in the demo.</p></div>`;
    return `<nav class="breadcrumb" aria-label="Breadcrumb"><a href="${escape(materialsHref())}" data-action="back-list">Invoice Lab</a>${icon("chevron")}<a href="${escape(materialsHref())}" data-action="back-list">Materials</a>${icon("chevron")}<span>${escape(item.name)}</span></nav>${back}<header class="material-heading"><div class="material-identity">${kindIcon(item.kind)}<div><span class="label">${labels[item.kind]}</span><h1 tabindex="-1">${escape(item.name)}</h1></div></div><div class="toolbar">${iconButton("copy", "copy", "Copy material identifier", `data-id="${item.id}"`)}${iconButton("download", "download", "Download material", `data-id="${item.id}"`)}</div></header><div class="material-version"><span class="pill success">Current</span><code>${escape(item.revision)}</code><span>${item.format}</span><span>${item.size}</span><span>${item.updated}</span>${item.commit ? `<span class="material-commit">Git <code>${item.commit}</code></span>` : ""}<button class="link-button" data-action="details" data-id="${item.id}" type="button">details →</button></div><section class="reader material-viewer" aria-label="Material content"><div class="reader-tabs" role="group" aria-label="Preview view"><button type="button" class="${state.readerTab === "preview" ? "active" : ""}" aria-pressed="${state.readerTab === "preview"}" data-action="preview-tab" data-tab="preview">Preview</button><button type="button" class="${state.readerTab === "raw" ? "active" : ""}" aria-pressed="${state.readerTab === "raw"}" data-action="preview-tab" data-tab="raw">Raw</button></div><div class="reader-body">${state.readerTab === "raw" ? `<pre>${escape(item.content)}</pre>` : previewContent(item)}</div></section>`;
  }

  function searchField(id, label, value) {
    return `<label class="search" for="${id}">${icon("search")}<span class="sr-only">${label}</span><input id="${id}" type="search" placeholder="${label}" value="${escape(value)}" autocomplete="off"></label>`;
  }

  function categoryFilters() {
    const primary = ["all", "source", "api", "engagement", "docs"];
    const secondary = ["architecture", "diffs", "other"];
    const count = (kind) =>
      kind === "all"
        ? examples.length
        : examples.filter((item) => item.kind === kind).length;
    return `<div class="filter-chips" role="group" aria-label="Material category">${primary.map((kind) => `<button type="button" data-action="category" data-kind="${kind}" aria-pressed="${state.category === kind}">${labels[kind]}${state.viewState === "normal" ? `<span class="count">${count(kind)}</span>` : ""}</button>`).join("")}<select id="more-kinds" aria-label="More material categories" class="${secondary.includes(state.category) ? "active" : ""}"><option value="">More</option>${secondary.map((kind) => `<option value="${kind}" ${state.category === kind ? "selected" : ""}>${labels[kind]}${state.viewState === "normal" ? ` · ${count(kind)}` : ""}</option>`).join("")}</select></div>`;
  }

  function matchingMaterials() {
    const needle = state.search.trim().toLowerCase();
    return examples.filter(
      (item) =>
        (state.category === "all" || item.kind === state.category) &&
        `${item.name} ${item.namespace} ${item.description}`
          .toLowerCase()
          .includes(needle),
    );
  }

  function materialResults() {
    if (state.viewState === "loading")
      return `<div class="empty" role="status">Loading materials…</div>`;
    if (state.viewState === "error")
      return `<div class="notice"><h3>Materials could not be loaded</h3><p>Try loading this project’s materials again.</p>${button("Try again", "recover")}</div>`;
    const needle = state.search.trim().toLowerCase();
    const shown = state.viewState === "empty" ? [] : matchingMaterials();
    if (!shown.length)
      return `<div class="empty">${icon("search")}<h3>${needle ? "No matching materials" : "No materials in this view"}</h3><p>${needle ? "Try another name or category." : "Add a material to use it in your checks."}</p>${button(needle ? "Clear filters" : "Add material", needle ? "clear" : "add")}</div>`;
    return kinds
      .filter((kind) => kind !== "all")
      .map((kind) => {
        const rows = shown.filter((item) => item.kind === kind);
        return rows.length
          ? `<section class="group" aria-label="${labels[kind]}"><h3 class="group-title"><span>${labels[kind]}</span><span>${rows.length}</span></h3><div class="material-rows">${rows.map((item) => materialRow(item, false)).join("")}</div></section>`
          : "";
      })
      .join("");
  }

  function materialRow(item, picker) {
    const selected =
      item.id === (picker ? state.pickerSelected : state.selected);
    return `<${picker ? 'button type="button"' : `a href="${escape(materialHref(item))}"`} class="material-row" data-action="${picker ? "pick-row" : "select"}" data-id="${item.id}" ${picker ? `aria-pressed="${selected}"` : ""} aria-label="${escape(item.name)}">${kindIcon(item.kind)}<span class="row-main"><span class="row-title"><span class="filename">${escape(item.name)}</span>${selected && picker ? `<span class="pill success">${icon("check")}Selected</span>` : ""}</span><span class="row-meta"><span>${item.format}</span><span>${item.size}</span><span>${item.updated}</span></span>${item.commit ? `<span class="row-meta"><code>${item.commit}</code><span>${escape(item.description)}</span></span>` : ""}</span><span class="row-action">${icon(picker && selected ? "check" : "chevron")}</span></${picker ? "button" : "a"}>`;
  }

  function reader(item, picker = false) {
    if (!item || state.viewState !== "normal")
      return `<aside class="reader"><div class="reader-body"><div class="empty">${icon("docs")}<h3>Material preview</h3><p>Choose a material to inspect its saved version.</p></div></div></aside>`;
    return `<aside class="reader" aria-label="Preview ${escape(item.name)}">${!picker ? button("Back to materials", "back-list", "ghost mobile-back", "back") : button("Back to choices", "picker-back", "ghost mobile-back", "back")}<div class="reader-head"><div class="toolbar spread"><span class="reader-kicker">${labels[item.kind]} preview</span><span class="toolbar">${iconButton("copy", "copy", "Copy material identifier", `data-id="${item.id}"`)}${iconButton("download", "download", "Download material", `data-id="${item.id}"`)}</span></div><h3 class="reader-title" tabindex="-1">${escape(item.name)}</h3><div class="reader-meta"><span class="pill success">Current</span><code>${item.revision}</code><span>${item.size}</span></div></div><div class="reader-tabs" role="group" aria-label="Preview view"><button type="button" class="${state.readerTab === "preview" ? "active" : ""}" aria-pressed="${state.readerTab === "preview"}" data-action="preview-tab" data-tab="preview">Preview</button><button type="button" class="${state.readerTab === "raw" ? "active" : ""}" aria-pressed="${state.readerTab === "raw"}" data-action="preview-tab" data-tab="raw">Raw</button></div><div class="reader-body">${state.readerTab === "raw" ? `<pre>${escape(item.content)}</pre>` : previewContent(item)}</div><footer class="reader-footer"><span>${escape(item.namespace)}/${escape(item.name)}</span><button class="link-button" data-action="details" data-id="${item.id}" type="button">details →</button></footer></aside>`;
  }

  function previewContent(item) {
    if (item.operations)
      return `<p>${item.description}</p><div class="label">Operations · ${item.operations.length}</div>${item.operations.map(([method, path]) => `<div class="endpoint"><span class="method" data-method="${method}">${method}</span><code>${escape(path)}</code></div>`).join("")}`;
    if (item.kind === "source")
      return `<p>${item.description}</p><div class="label">Files</div><div class="file-tree">${item.files.map((file) => `<button type="button" data-action="file">${icon("docs")}<code>${file}</code></button>`).join("")}</div><div class="label">${item.files[0]}</div><pre>${escape(item.content)}</pre>`;
    if (item.kind === "engagement")
      return `<dl class="fact-list"><div><dt>Allowed origin</dt><dd><code>https://staging.invoice.example</code></dd></div><div><dt>Routes</dt><dd><code>/api/*</code> · All HTTP methods</dd></div><div><dt>Identities</dt><dd><span class="pill">customer-a</span> <span class="pill">customer-b</span></dd></div><div><dt>Request limit</dt><dd>2 requests / second</dd></div></dl>`;
    if (item.format === "Markdown")
      return item.content
        .split("\n\n")
        .map((part) =>
          part.startsWith("## ")
            ? `<h3>${escape(part.slice(3))}</h3>`
            : part.startsWith("# ")
              ? `<h2>${escape(part.slice(2))}</h2>`
              : `<p>${escape(part).replaceAll("\n", "<br>")}</p>`,
        )
        .join("");
    return `<pre>${escape(item.content)}</pre>`;
  }

  function checkPage() {
    const missing =
      state.type === "source"
        ? !state.source
        : !state.api || (state.type === "active" && !state.engagement);
    const attached = [
      state.source,
      state.api,
      ...(state.type === "active" ? [state.engagement] : []),
    ].filter(Boolean).length;
    return `<nav class="breadcrumb" aria-label="Breadcrumb"><a href="${screenHref("materials")}">Invoice Lab</a>${icon("chevron")}<span>New check</span></nav><div class="page-heading"><div><h1>Start a check</h1></div><a class="btn ghost" href="${screenHref("materials")}">Cancel</a></div><label class="label" for="objective">Your objective, in your own words</label><textarea class="objective" id="objective" rows="2">${escape(state.objective)}</textarea><label class="mobile-type-select form-field" for="mobile-check-type">Check type<select class="control" id="mobile-check-type">${[
      ["source", "Source review"],
      ["api", "API review"],
      ["active", "Active HTTP checks"],
    ]
      .map(
        ([value, label]) =>
          `<option value="${value}" ${state.type === value ? "selected" : ""}>${label}</option>`,
      )
      .join(
        "",
      )}</select></label><div class="new-check"><aside class="check-types" aria-label="Check types">${[
      ["source", "Source review", "Review source code", "source"],
      ["api", "API review", "Review API boundaries", "api"],
      ["active", "Active HTTP checks", "Test a running service", "engagement"],
    ]
      .map(
        ([id, title, subtitle, glyph]) =>
          `<button type="button" class="check-type ${state.type === id ? "active" : ""}" data-action="check-type" data-type="${id}" aria-pressed="${state.type === id}">${icon(glyph)}<span><strong>${title}</strong><small>${subtitle}</small></span></button>`,
      )
      .join(
        "",
      )}</aside><section class="check-cards" aria-label="Check setup"><section class="card"><div class="card-head"><h2>Materials</h2>${button("Add", "add-check", "ghost", "plus")}</div>${slot("source", "Source code", state.type === "source")}${slot("api", "API spec", state.type !== "source")}${state.type === "active" ? slot("engagement", "Engagement", true) : ""}<p class="hint">Choose the saved version to use for this check. You can preview each material before attaching it.</p></section><section class="card"><div class="card-head"><h2>Scope</h2><span class="muted">Optional</span></div><label class="form-field" for="scope">What should this check focus on?<textarea class="control" id="scope" rows="3">${escape(state.scope)}</textarea></label></section><details class="card"><summary>Advanced options</summary><p class="hint">Use the project’s execution defaults.</p></details></section><aside class="card summary" aria-label="Check summary"><h2>${missing ? "Complete the materials" : "Ready to start"}</h2><dl class="fact-list"><div><dt>Check type</dt><dd>${state.type === "source" ? "Source review" : state.type === "api" ? "API review" : "Active HTTP checks"}</dd></div><div><dt>Materials</dt><dd>${attached} attached</dd></div></dl><label class="form-field" for="time-limit">Time limit<select class="control" id="time-limit">${["24 hours", "7 days", "No time limit"].map((option) => `<option ${state.time === option ? "selected" : ""}>${option}</option>`).join("")}</select></label><button type="button" class="btn primary" data-action="start" ${missing ? "disabled" : ""}>${icon("arrow")}Start check</button>${button("Save as draft", "save", "ghost")}<p class="hint">This preview uses synthetic data.</p></aside></div>`;
  }

  function slot(id, title, required) {
    const item = itemOf(state[id]);
    return `<div class="slot"><div class="slot-label">${title}<span class="optional">${required ? "Required" : "Optional"}</span></div>${item ? `<div class="attached">${kindIcon(item.kind)}<span class="row-main"><strong>${escape(item.name)}</strong><span class="row-meta"><code>${item.revision}</code><span>${item.format}</span><span>${item.size}</span></span>${item.commit ? `<span class="row-meta"><code>${item.commit}</code></span>` : ""}</span><span class="slot-actions">${iconButton("slot-preview", "docs", `Preview ${title.toLowerCase()}`, `data-slot="${id}"`)}<button type="button" class="btn ghost" data-action="choose" data-slot="${id}">Change</button></span></div>` : `<div class="slot-empty"><span>${required ? "Choose a material" : "No material attached"}</span><button type="button" class="btn" data-action="choose" data-slot="${id}">Choose</button></div>`}</div>`;
  }

  function openPicker(slotId, preview = false) {
    state.pickerSlot = slotId;
    state.pickerSelected = state[slotId] || "";
    state.pickerSearch = "";
    state.pickerPreview = preview;
    state.readerTab = "preview";
    renderPicker();
    const dialog = document.getElementById("picker");
    if (!dialog.open) dialog.showModal();
    if (!preview) document.getElementById("picker-search").focus();
  }

  function renderPicker() {
    const dialog = document.getElementById("picker");
    const selected = itemOf(state.pickerSelected);
    const matches = examples.filter(
      (item) =>
        item.kind === state.pickerSlot &&
        `${item.name} ${item.description} ${item.revision}`
          .toLowerCase()
          .includes(state.pickerSearch.trim().toLowerCase()),
    );
    dialog.dataset.preview = String(state.pickerPreview);
    dialog.innerHTML = `<header class="dialog-header"><div><h2 id="picker-title">Choose ${labels[state.pickerSlot].toLowerCase()}</h2><p>Invoice Lab · ${matches.length} compatible materials</p></div>${iconButton("close-picker", "close", "Close material chooser")}</header><div class="picker-grid"><section class="picker-list" aria-label="Compatible materials">${searchField("picker-search", "Search compatible materials", state.pickerSearch)}<div class="material-rows">${matches.length ? matches.map((item) => materialRow(item, true)).join("") : `<div class="empty"><h3>No matching materials</h3><p>Try another name, or add a material.</p></div>`}</div><p class="hint">${state.pickerSlot === "source" ? "ZIP source archives and Git imports." : state.pickerSlot === "api" ? "OpenAPI specifications in YAML or JSON." : "Saved egress rules, identities and request limits."}</p><button type="button" class="btn ghost" data-action="add-picker">${icon("plus")}Add material</button></section><section class="picker-preview" aria-label="Selected material preview">${reader(selected, true)}</section></div><footer class="dialog-footer"><div class="selection">${selected ? `<strong>${escape(selected.name)}</strong><code>${selected.revision}</code>` : "Select a material to attach"}</div><div class="toolbar">${button("Cancel", "close-picker", "ghost")}<button type="button" class="btn primary" data-action="attach" ${selected ? "" : "disabled"}>${icon("check")}Attach material</button></div></footer>`;
  }

  function openUpload(fromPicker = false) {
    state.uploadFromPicker = fromPicker;
    state.uploadKind = fromPicker
      ? state.pickerSlot
      : screen === "check" && state.type === "api"
        ? "api"
        : "source";
    state.uploadFile = null;
    renderUpload();
    document.getElementById("upload").showModal();
  }

  function renderUpload() {
    const dialog = document.getElementById("upload");
    dialog.innerHTML = `<header class="dialog-header"><div><h2 id="upload-title">Add material</h2><p>Invoice Lab${screen === "check" ? " · Your check setup stays open" : ""}</p></div>${iconButton("close-upload", "close", "Close upload")}</header><form id="upload-form"><div class="upload-body"><label class="form-field" for="upload-kind">Category<select class="control" id="upload-kind">${kinds
      .filter(
        (kind) =>
          kind !== "all" &&
          (!state.uploadFromPicker || kind === state.pickerSlot),
      )
      .map(
        (kind) =>
          `<option value="${kind}" ${kind === state.uploadKind ? "selected" : ""}>${labels[kind]}</option>`,
      )
      .join(
        "",
      )}</select></label><div class="file-drop">${icon("upload")}<label for="upload-file">Choose a file to add</label><input type="file" id="upload-file" required ${state.uploadFromPicker ? `accept="${state.pickerSlot === "source" ? ".zip" : ".yaml,.yml,.json"}"` : ""}></div><label class="form-field" for="upload-name">Name<input class="control" id="upload-name" placeholder="For example, invoice-service" required maxlength="120"></label><p id="upload-error" role="alert" class="hint" hidden></p><p class="hint">A new material is added to the project. Existing versions keep their exact references.</p></div><footer class="dialog-footer"><span class="muted">Synthetic preview</span><div class="toolbar">${button("Cancel", "close-upload", "ghost")}<button class="btn primary" type="submit">${state.uploadFromPicker ? "Add and attach" : "Add material"}</button></div></footer></form>`;
  }

  document.addEventListener("click", async (event) => {
    const control = event.target.closest("[data-action]");
    if (!control) return;
    if (
      control.tagName === "A" &&
      (event.metaKey ||
        event.ctrlKey ||
        event.shiftKey ||
        event.altKey ||
        event.button !== 0)
    )
      return;
    event.preventDefault();
    const action = control.dataset.action;
    if (action === "category") {
      state.category = control.dataset.kind;
      state.selected = matchingMaterials()[0]?.id || "";
      render();
    }
    if (action === "select") {
      state.selected = control.dataset.id;
      state.readerTab = "preview";
      state.listScroll = window.scrollY;
      history.replaceState(
        { listScroll: state.listScroll },
        "",
        materialsHref(),
      );
      navigate(control.getAttribute("href"), true);
    }
    if (action === "back-list") {
      if (history.state?.materialFromList) history.back();
      else navigate(control.getAttribute("href"));
    }
    if (action === "clear") {
      state.search = "";
      state.category = "all";
      render();
    }
    if (action === "refresh" || action === "recover") {
      state.viewState = "normal";
      render();
    }
    if (action === "add") openUpload(false);
    if (action === "add-check") openUpload(false);
    if (action === "choose" || action === "slot-preview")
      openPicker(control.dataset.slot, action === "slot-preview");
    if (action === "pick-row") {
      state.pickerSelected = control.dataset.id;
      state.pickerPreview = true;
      state.readerTab = "preview";
      renderPicker();
      const next =
        innerWidth <= 820
          ? document.querySelector("#picker .reader-title")
          : document.querySelector(
              `#picker [data-action="pick-row"][data-id="${state.pickerSelected}"]`,
            );
      next?.focus({ preventScroll: true });
    }
    if (action === "picker-back") {
      state.pickerPreview = false;
      renderPicker();
      document.getElementById("picker-search").focus();
    }
    if (action === "close-picker") document.getElementById("picker").close();
    if (action === "add-picker") openUpload(true);
    if (action === "close-upload") document.getElementById("upload").close();
    if (action === "attach") {
      state[state.pickerSlot] = state.pickerSelected;
      document.getElementById("picker").close();
      render();
      toast(`${itemOf(state.pickerSelected).name} attached`);
    }
    if (action === "preview-tab") {
      state.readerTab = control.dataset.tab;
      document.getElementById("picker").open ? renderPicker() : render();
    }
    if (action === "copy") {
      const item = itemOf(control.dataset.id);
      await navigator.clipboard.writeText(
        `${item.namespace}/${item.name}@${item.revision}`,
      );
      toast("Material identifier copied");
    }
    if (action === "download") {
      const item = itemOf(control.dataset.id);
      const href = URL.createObjectURL(
        new Blob([item.content], { type: "text/plain" }),
      );
      const link = document.createElement("a");
      link.href = href;
      link.download = item.name;
      link.click();
      setTimeout(() => URL.revokeObjectURL(href), 1000);
    }
    if (action === "details") {
      const item = itemOf(control.dataset.id);
      const detail = document.getElementById("details");
      detail.innerHTML = `<header class="dialog-header"><div><h2 id="details-title">Material details</h2><p>${escape(item.name)}</p></div>${iconButton("close-details", "close", "Close material details")}</header><div class="upload-body"><dl class="fact-list"><div><dt>Project</dt><dd>Invoice Lab</dd></div><div><dt>Exact reference</dt><dd><code>${escape(item.namespace)}/${escape(item.name)}@${item.revision}</code></dd></div><div><dt>Format</dt><dd>${item.format}</dd></div><div><dt>Size</dt><dd>${item.size}</dd></div><div><dt>Revision</dt><dd><span class="pill success">Current</span> <code>${item.revision}</code></dd></div>${item.commit ? `<div><dt>Git commit</dt><dd><code>${item.commit}</code></dd></div>` : ""}</dl></div><footer class="dialog-footer"><span class="muted">Saved material version</span>${button("Done", "close-details")}</footer>`;
      detail.showModal();
    }
    if (action === "close-details") document.getElementById("details").close();
    if (action === "file") toast("Showing the saved source excerpt");
    if (action === "check-type") {
      state.type = control.dataset.type;
      render();
    }
    if (action === "start")
      toast(
        state.type === "active" && !state.engagement
          ? "Attach an Engagement material before starting"
          : "Demo check started",
      );
    if (action === "save") toast("Demo draft saved");
    if (action === "demo-link")
      toast("This design preview focuses on Materials and New check");
  });

  document.addEventListener("input", (event) => {
    const input = event.target;
    if (input.id === "material-search") {
      state.search = input.value;
      document.getElementById("material-results").innerHTML = materialResults();
    }
    if (input.id === "picker-search") {
      const caret = input.selectionStart;
      state.pickerSearch = input.value;
      const matches = examples.filter(
        (item) =>
          item.kind === state.pickerSlot &&
          `${item.name} ${item.description} ${item.revision}`
            .toLowerCase()
            .includes(state.pickerSearch.trim().toLowerCase()),
      );
      if (!matches.some((item) => item.id === state.pickerSelected))
        state.pickerSelected = "";
      renderPicker();
      const replacement = document.getElementById("picker-search");
      replacement.focus();
      if (caret !== null) replacement.setSelectionRange(caret, caret);
    }
    if (input.id === "objective") state.objective = input.value;
    if (input.id === "scope") state.scope = input.value;
  });
  document.addEventListener("change", (event) => {
    if (event.target.id === "more-kinds") {
      state.category = event.target.value || "all";
      state.selected = matchingMaterials()[0]?.id || "";
      render();
      document.getElementById("more-kinds").focus();
    }
    if (event.target.id === "mobile-check-type") {
      state.type = event.target.value;
      render();
      document
        .getElementById("mobile-check-type")
        .focus({ preventScroll: true });
    }
    if (event.target.id === "time-limit") state.time = event.target.value;
    if (event.target.id === "upload-kind")
      state.uploadKind = event.target.value;
    if (event.target.id === "upload-file") {
      state.uploadFile = event.target.files[0] || null;
      if (state.uploadFile)
        document.getElementById("upload-name").value =
          state.uploadFile.name.replace(/\.(zip|yaml|yml|json|md|txt)$/i, "");
    }
  });
  document.addEventListener("submit", async (event) => {
    if (event.target.id !== "upload-form") return;
    event.preventDefault();
    const file = state.uploadFile;
    if (!file) return;
    const valid =
      state.uploadKind === "source"
        ? /\.zip$/i.test(file.name)
        : state.uploadKind === "api"
          ? /\.(yaml|yml|json)$/i.test(file.name)
          : state.uploadKind === "engagement"
            ? /\.json$/i.test(file.name)
            : true;
    if (!valid) {
      const error = document.getElementById("upload-error");
      error.textContent =
        state.uploadKind === "source"
          ? "Choose a ZIP archive for source code."
          : state.uploadKind === "api"
            ? "Choose an OpenAPI file in YAML or JSON."
            : "Choose an Engagement JSON file.";
      error.hidden = false;
      return;
    }
    const name = document.getElementById("upload-name").value.trim();
    if (!name) return;
    const namespaces = {
      source: "sources",
      api: "openapi",
      engagement: "engagement",
      docs: "docs",
      architecture: "likec4",
      diffs: "diffs",
      other: "files",
    };
    const item = {
      id: `added-${Date.now()}`,
      name,
      kind: state.uploadKind,
      namespace: namespaces[state.uploadKind],
      format: file.name.toLowerCase().endsWith(".zip")
        ? "ZIP"
        : state.uploadKind === "api"
          ? "OpenAPI"
          : "Text",
      revision: "demo-r1",
      size: `${Math.max(1, Math.ceil(file.size / 1024))} KiB`,
      updated: "Just now",
      description: "Added in this preview",
      content:
        file.size < 131072 && !file.name.endsWith(".zip")
          ? await file.text()
          : "Preview of the added material",
      files: ["uploaded-material"],
    };
    examples.unshift(item);
    document.getElementById("upload").close();
    if (state.uploadFromPicker) {
      state[state.pickerSlot] = item.id;
      document.getElementById("picker").close();
    } else {
      state.selected = item.id;
      state.category = "all";
      state.search = "";
    }
    render();
    toast(
      state.uploadFromPicker ? `${name} added and attached` : `${name} added`,
    );
  });
  render();
  if (
    screen === "check" &&
    ["source", "api", "engagement"].includes(params.get("picker"))
  )
    openPicker(params.get("picker"));
})();
