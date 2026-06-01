/* Contractor Explorer — single-file vanilla SPA.
 * No build step, no external deps: markdown rendering, version diffing and the
 * pipeline DAG are all implemented here so the UI works fully offline. */
'use strict';

// ───────────────────────── tiny helpers ─────────────────────────
const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') n.className = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k === 'text') n.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid == null) continue;
    n.appendChild(typeof kid === 'string' ? document.createTextNode(kid) : kid);
  }
  return n;
};
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const cache = new Map();
async function api(path, { fresh = false } = {}) {
  if (!fresh && cache.has(path)) return cache.get(path);
  const r = await fetch('/api/' + path);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  const data = await r.json();
  cache.set(path, data);
  return data;
}

let STATE = { overview: null, crossrefs: null };

// ───────────────────────── markdown ─────────────────────────
function mdInline(s) {
  s = esc(s);
  // inline code first — stash behind an ASCII sentinel so later passes
  // (bold/italic/links) can't touch its contents and prose digits don't collide.
  const codes = [];
  s = s.replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return `@@C${codes.length - 1}@@`; });
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  s = s.replace(/\b_([^_\n]+)_\b/g, '<em>$1</em>');
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  s = s.replace(/(^|[\s(])((https?:\/\/)[^\s<)]+)/g, '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
  s = s.replace(/@@C(\d+)@@/g, (_, i) => `<code>${codes[+i]}</code>`);
  return s;
}

function mdToHtml(src) {
  if (!src) return '<p class="md-empty" style="color:var(--faint)">— empty —</p>';
  const lines = src.replace(/\r\n/g, '\n').split('\n');
  let out = '';
  let i = 0;
  const listStack = []; // {type, indent}
  const closeLists = (toIndent = -1) => {
    while (listStack.length && listStack[listStack.length - 1].indent >= toIndent) {
      out += listStack.pop().type === 'ol' ? '</ol>' : '</ul>';
    }
  };
  while (i < lines.length) {
    let line = lines[i];

    // fenced code
    const fence = line.match(/^\s*```+(.*)$/);
    if (fence) {
      closeLists();
      const buf = [];
      i++;
      while (i < lines.length && !/^\s*```+\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++;
      out += `<pre><code>${esc(buf.join('\n'))}</code></pre>`;
      continue;
    }
    // table (GFM): header line + separator
    if (/\|/.test(line) && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[i + 1])) {
      closeLists();
      const parseRow = (l) => l.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map((c) => c.trim());
      const head = parseRow(line);
      i += 2;
      let t = '<table><thead><tr>' + head.map((h) => `<th>${mdInline(h)}</th>`).join('') + '</tr></thead><tbody>';
      while (i < lines.length && /\|/.test(lines[i]) && lines[i].trim()) {
        const cells = parseRow(lines[i]);
        t += '<tr>' + cells.map((c) => `<td>${mdInline(c)}</td>`).join('') + '</tr>';
        i++;
      }
      out += t + '</tbody></table>';
      continue;
    }
    // headings
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { closeLists(); out += `<h${h[1].length}>${mdInline(h[2])}</h${h[1].length}>`; i++; continue; }
    // hr
    if (/^\s*([-*_])\1\1+\s*$/.test(line)) { closeLists(); out += '<hr/>'; i++; continue; }
    // blockquote
    if (/^\s*>\s?/.test(line)) {
      closeLists();
      const buf = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) { buf.push(lines[i].replace(/^\s*>\s?/, '')); i++; }
      out += `<blockquote>${mdInline(buf.join(' '))}</blockquote>`;
      continue;
    }
    // list item
    const li = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
    if (li) {
      const indent = li[1].length;
      const type = /\d/.test(li[2]) ? 'ol' : 'ul';
      const top = listStack[listStack.length - 1];
      if (!top || indent > top.indent) {
        listStack.push({ type, indent });
        out += type === 'ol' ? '<ol>' : '<ul>';
      } else if (indent < top.indent) {
        closeLists(indent + 1);
        if (!listStack.length || listStack[listStack.length - 1].indent < indent) {
          listStack.push({ type, indent }); out += type === 'ol' ? '<ol>' : '<ul>';
        }
      }
      out += `<li>${mdInline(li[3])}</li>`;
      i++;
      continue;
    }
    // blank
    if (!line.trim()) { closeLists(); i++; continue; }
    // paragraph (merge following non-empty, non-special lines)
    closeLists();
    const buf = [line];
    i++;
    while (i < lines.length && lines[i].trim() &&
           !/^(\s*```|#{1,6}\s|\s*>|\s*([-*+]|\d+[.)])\s)/.test(lines[i]) &&
           !/^\s*([-*_])\1\1+\s*$/.test(lines[i])) {
      buf.push(lines[i]); i++;
    }
    out += `<p>${mdInline(buf.join(' '))}</p>`;
  }
  closeLists();
  return out;
}

// ───────────────────────── line diff (LCS) ─────────────────────────
function lineDiff(a, b) {
  const A = a.split('\n'), B = b.split('\n');
  const n = A.length, m = B.length;
  // guard pathological sizes
  const dp = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let x = n - 1; x >= 0; x--)
    for (let y = m - 1; y >= 0; y--)
      dp[x][y] = A[x] === B[y] ? dp[x + 1][y + 1] + 1 : Math.max(dp[x + 1][y], dp[x][y + 1]);
  const rows = [];
  let x = 0, y = 0;
  while (x < n && y < m) {
    if (A[x] === B[y]) { rows.push({ t: 'eq', s: A[x] }); x++; y++; }
    else if (dp[x + 1][y] >= dp[x][y + 1]) { rows.push({ t: 'del', s: A[x] }); x++; }
    else { rows.push({ t: 'add', s: B[y] }); y++; }
  }
  while (x < n) rows.push({ t: 'del', s: A[x++] });
  while (y < m) rows.push({ t: 'add', s: B[y++] });
  return rows;
}

// ───────────────────────── comments + source view ─────────────────────────
async function fetchComments(t) {
  const q = new URLSearchParams({ kind: t.kind, id: t.id, version: t.version });
  const r = await fetch('/api/comments?' + q);
  return r.ok ? r.json() : [];
}
async function createComment(t, lineStart, lineEnd, body) {
  const r = await fetch('/api/comments', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind: t.kind, id: t.id, version: t.version, line_start: lineStart, line_end: lineEnd, body }),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`);
  return r.json();
}
async function editComment(id, body) {
  const r = await fetch('/api/comments/' + id, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ body }),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`);
  return r.json();
}
async function deleteComment(id) {
  const r = await fetch('/api/comments/' + id, { method: 'DELETE' });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
}

function fmtTime(iso) {
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}
function lineRangeLabel(c) { return c.line_start === c.line_end ? `L${c.line_start}` : `L${c.line_start}–${c.line_end}`; }

function updateSourceBadge(toolbarEl, n) {
  const btn = [...toolbarEl.querySelectorAll('.mode-btn')].find((b) => b.textContent.replace(/\d+$/, '').trim() === 'Source');
  if (!btn) return;
  let b = btn.querySelector('.mode-badge');
  if (n > 0) { if (!b) { b = el('span', { class: 'mode-badge' }); btn.appendChild(b); } b.textContent = String(n); }
  else if (b) b.remove();
}

function modeBar(modes, current, onPick) {
  const bar = el('div', { class: 'mode-bar' });
  for (const m of modes) {
    bar.appendChild(el('span', { class: 'mode-btn' + (m.key === current ? ' on' : ''), onclick: () => onPick(m.key) },
      m.label, (m.badge != null && m.badge > 0) ? el('span', { class: 'mode-badge', text: String(m.badge) }) : null));
  }
  return bar;
}

/* A GitHub-style line-numbered source view with inline comment threads.
 * Owns its comment state; calls onCount(n) whenever the total changes so the
 * caller can refresh the "Source 💬n" badge. */
function makeSourceView(target, text, initialComments, onCount) {
  const wrap = el('div', { class: 'src-wrap' });
  let cmts = initialComments || [];
  let sel = null; // {start, end}

  const refresh = async () => { cmts = await fetchComments(target); onCount && onCount(cmts.length); render(); };

  function composer() {
    const label = `Line ${sel.start}${sel.end !== sel.start ? '–' + sel.end : ''}`;
    const ta = el('textarea', { class: 'cmt-ta', placeholder: `Comment on ${label}…  (markdown · Ctrl+Enter to save)` });
    const submit = async () => {
      const v = ta.value.trim();
      if (!v) return;
      try { await createComment(target, sel.start, sel.end, v); sel = null; await refresh(); }
      catch (e) { alert('Could not save: ' + e.message); }
    };
    ta.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); submit(); }
      if (e.key === 'Escape') { sel = null; render(); }
    });
    setTimeout(() => ta.focus(), 0);
    return el('div', { class: 'cmt-card cmt-composer' },
      el('div', { class: 'cmt-head' }, el('span', { class: 'cmt-anchor', text: label })),
      ta,
      el('div', { class: 'cmt-actions' },
        el('span', { class: 'toggle-btn on', text: 'Comment', onclick: submit }),
        el('span', { class: 'toggle-btn', text: 'Cancel', onclick: () => { sel = null; render(); } })));
  }

  function thread(c) {
    const card = el('div', { class: 'cmt-card' });
    const bodyWrap = el('div', { class: 'md cmt-body', html: mdToHtml(c.body) });
    const head = el('div', { class: 'cmt-head' },
      el('span', { class: 'cmt-anchor', text: lineRangeLabel(c) }),
      el('span', { class: 'cmt-time', text: fmtTime(c.updated_at) + (c.updated_at !== c.created_at ? ' (edited)' : '') }),
      el('span', { class: 'cmt-tools' },
        el('span', { class: 'cmt-link', text: 'edit', onclick: () => startEdit() }),
        el('span', { class: 'cmt-link danger', text: 'delete', onclick: async () => {
          if (!confirm('Delete this comment?')) return;
          try { await deleteComment(c.id); await refresh(); } catch (e) { alert(e.message); }
        } })));
    function startEdit() {
      const ta = el('textarea', { class: 'cmt-ta' }); ta.value = c.body;
      const save = async () => { const v = ta.value.trim(); if (!v) return; try { await editComment(c.id, v); await refresh(); } catch (e) { alert(e.message); } };
      ta.addEventListener('keydown', (e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); save(); } if (e.key === 'Escape') render(); });
      bodyWrap.replaceWith(el('div', { class: 'cmt-edit' }, ta,
        el('div', { class: 'cmt-actions' },
          el('span', { class: 'toggle-btn on', text: 'Save', onclick: save }),
          el('span', { class: 'toggle-btn', text: 'Cancel', onclick: () => render() }))));
      setTimeout(() => ta.focus(), 0);
    }
    card.appendChild(head);
    card.appendChild(bodyWrap);
    return card;
  }

  function render() {
    wrap.innerHTML = '';
    const lines = text.split('\n');
    const anchored = new Map(); // line_end -> [comments]
    for (const c of cmts) { if (!anchored.has(c.line_end)) anchored.set(c.line_end, []); anchored.get(c.line_end).push(c); }
    const commented = new Set();
    for (const c of cmts) for (let i = c.line_start; i <= c.line_end; i++) commented.add(i);

    const code = el('div', { class: 'src-code' });
    const pad = String(lines.length).length;
    for (let i = 1; i <= lines.length; i++) {
      const inSel = sel && i >= sel.start && i <= sel.end;
      const row = el('div', { class: 'src-line' + (inSel ? ' sel' : '') + (commented.has(i) ? ' has-cmt' : '') });
      row.appendChild(el('span', { class: 'src-ln', text: String(i).padStart(pad, ' '),
        title: 'click to comment · shift-click to extend range',
        onclick: (e) => { if (e.shiftKey && sel) sel = { start: Math.min(sel.start, i), end: Math.max(sel.start, i) }; else sel = { start: i, end: i }; render(); } }));
      row.appendChild(el('span', { class: 'src-plus', text: '＋', onclick: () => { sel = { start: i, end: i }; render(); } }));
      const t = lines[i - 1];
      row.appendChild(el('span', { class: 'src-text', text: t.length ? t : ' ' }));
      code.appendChild(row);
      if (sel && i === sel.end) code.appendChild(composer());
      for (const c of (anchored.get(i) || [])) code.appendChild(thread(c));
    }
    wrap.appendChild(code);
  }

  render();
  return wrap;
}

// ───────────────────────── sidebar ─────────────────────────
const COLLAPSED = JSON.parse(localStorage.getItem('cx_collapsed') || '{}');
const SECTIONS = [
  { key: 'agents', label: 'Agents', cls: 'acc-agents', route: 'agents' },
  { key: 'tasks', label: 'Tasks', cls: 'acc-tasks', route: 'tasks' },
  { key: 'workflows', label: 'Pipelines', cls: 'acc-workflows', route: 'workflows' },
  { key: 'skills', label: 'Skills', cls: 'acc-skills', route: 'skills' },
  { key: 'evals', label: 'Evals', cls: 'acc-evals', route: 'evals' },
];

function evalHeadline(x) {
  if (x.kind === 'vuln') return `F1 ${pct(x.headline.f1)} · P ${pct(x.headline.precision)} · R ${pct(x.headline.recall)}`;
  return `pass ${pct(x.headline.pass_rate)} (${x.headline.passed}/${x.headline.total})`;
}

function itemMeta(sec, x) {
  if (sec.key === 'agents') return { id: x.name, title: x.name, sub: x.summary, badge: x.active };
  if (sec.key === 'tasks') return { id: x.name, title: x.name, sub: x.summary, badge: x.active };
  if (sec.key === 'workflows') return { id: x.key, title: x.key, sub: x.summary, badge: null };
  if (sec.key === 'evals') return { id: x.id, title: x.name, sub: evalHeadline(x), badge: x.kind };
  return { id: x.name, title: x.name, sub: x.description, badge: x.references?.length ? `${x.references.length} ref` : null };
}

function renderSidebar(filter = '') {
  const nav = $('#sidebar');
  nav.innerHTML = '';
  const ov = STATE.overview;
  if (!ov) return;
  const f = filter.trim().toLowerCase();
  const [path] = currentRoute();

  for (const sec of SECTIONS) {
    const all = ov[sec.key] || [];
    const items = all.filter((x) => {
      if (!f) return true;
      const m = itemMeta(sec, x);
      return (m.title + ' ' + (m.sub || '')).toLowerCase().includes(f);
    });
    if (f && !items.length) continue;

    const collapsed = !f && COLLAPSED[sec.key];
    const group = el('div', { class: `nav-group ${sec.cls}${collapsed ? ' collapsed' : ''}` });
    const head = el('div', { class: 'nav-group-head', onclick: () => {
      COLLAPSED[sec.key] = !COLLAPSED[sec.key];
      localStorage.setItem('cx_collapsed', JSON.stringify(COLLAPSED));
      renderSidebar($('#search').value);
    } },
      el('span', { class: 'dot' }),
      el('span', { text: sec.label }),
      el('span', { class: 'count', text: String(items.length) }),
      el('span', { class: 'chev', text: '▾' }),
    );
    group.appendChild(head);

    const list = el('div', { class: 'nav-list' });
    if (!items.length) list.appendChild(el('div', { class: 'nav-empty', text: 'none' }));
    for (const x of items) {
      const m = itemMeta(sec, x);
      const active = path === sec.route && currentRoute()[1] === m.id;
      const item = el('div', { class: 'nav-item' + (active ? ' active' : ''),
        onclick: () => navigate(`#/${sec.route}/${encodeURIComponent(m.id)}`) },
        el('div', { class: 'ni-title' }, m.title,
          m.badge ? el('span', { class: 'mini-badge', style: 'background:var(--acc);color:#0c0e14', text: m.badge }) : null),
        m.sub ? el('div', { class: 'ni-sub', text: m.sub }) : null,
      );
      list.appendChild(item);
    }
    group.appendChild(list);
    nav.appendChild(group);
  }
}

// ───────────────────────── routing ─────────────────────────
function currentRoute() {
  const h = location.hash.replace(/^#\/?/, '');
  return h.split('/').map(decodeURIComponent);
}
function navigate(hash) { location.hash = hash; }

async function route() {
  const main = $('#main');
  main.classList.remove('wide');
  const parts = currentRoute();
  const [section, a, b, c] = parts;
  try {
    if (!section || section === 'overview') return renderOverview();
    if (section === 'agents' && a) return renderAgent(a);
    if (section === 'tasks' && a) return renderTask(a);
    if (section === 'workflows' && a) return renderWorkflow(a);
    if (section === 'skills' && a && b === 'ref' && c) return renderSkill(a, c);
    if (section === 'skills' && a) return renderSkill(a);
    if (section === 'evals' && a) return renderEval(a);
    renderOverview();
  } catch (err) {
    main.innerHTML = '';
    main.appendChild(el('div', { class: 'empty-state', text: 'Error: ' + err.message }));
  } finally {
    renderSidebar($('#search').value);
  }
}

// ───────────────────────── views ─────────────────────────
function pageHead(crumb, title, extras = [], sub = '') {
  return el('div', { class: 'page-head' },
    el('div', { class: 'crumb', text: crumb }),
    el('h1', { class: 'page-title' }, title, ...extras),
    sub ? el('div', { class: 'page-sub', text: sub }) : null,
  );
}

function renderOverview() {
  const ov = STATE.overview;
  const main = $('#main');
  main.innerHTML = '';
  main.appendChild(pageHead('Workspace', 'Contractor Explorer', [],
    'A live, read-only window into the agent prompts, task templates, pipelines and skills that drive this CLI. Edit a file on disk, hit Refresh, and it shows up here.'));

  const grid = el('div', { class: 'grid-2' });
  const meta = [
    ['agents', 'Agents', 'acc-agents', '🤖', 'agents', (x) => x.name],
    ['tasks', 'Tasks', 'acc-tasks', '📋', 'tasks', (x) => x.name],
    ['workflows', 'Pipelines', 'acc-workflows', '🔀', 'workflows', (x) => x.key],
    ['skills', 'Skills', 'acc-skills', '📚', 'skills', (x) => x.name],
    ['evals', 'Evals', 'acc-evals', '📊', 'evals', (x) => x.id],
  ];
  for (const [key, label, cls, icon, route, idOf] of meta) {
    const first = (ov[key] || [])[0];
    grid.appendChild(el('div', { class: `stat-card ${cls}`,
      onclick: () => first && navigate(`#/${route}/${encodeURIComponent(idOf(first))}`) },
      el('div', { class: 'num', text: String(ov.counts[key]) }),
      el('div', { class: 'lbl' }, `${icon} ${label}`),
    ));
  }
  main.appendChild(grid);

  // quick pipeline directory
  const wfCard = el('div', { class: 'card acc-workflows' },
    el('h3', { class: 'card-h', text: 'Pipelines' }));
  const ll = el('div', { class: 'linklist' });
  for (const w of ov.workflows) {
    ll.appendChild(el('div', { class: 'chip', style: '--acc:var(--violet)',
      onclick: () => navigate(`#/workflows/${encodeURIComponent(w.key)}`) },
      el('span', { class: 'chip-dot' }), w.key));
  }
  wfCard.appendChild(ll);
  main.appendChild(wfCard);

  if (ov.evals && ov.evals.length) {
    const evCard = el('div', { class: 'card acc-evals', style: '--acc:var(--evals)' },
      el('h3', { class: 'card-h', text: 'Eval runs' }));
    const er = el('div', { class: 'linklist' });
    for (const e of ov.evals) {
      er.appendChild(el('div', { class: 'chip', style: '--acc:var(--evals)',
        onclick: () => navigate(`#/evals/${encodeURIComponent(e.id)}`) },
        el('span', { class: 'chip-dot' }), `${e.name}`,
        el('span', { class: 'chip-metric', text: e.kind === 'vuln' ? `F1 ${pct(e.headline.f1)}` : pct(e.headline.pass_rate) })));
    }
    evCard.appendChild(er);
    main.appendChild(evCard);
  }
}

function usedByCard(title, keys, route, accent) {
  if (!keys || !keys.length) return null;
  const card = el('div', { class: 'card' }, el('h3', { class: 'card-h', text: title }));
  const ll = el('div', { class: 'linklist' });
  for (const k of keys) {
    ll.appendChild(el('div', { class: 'chip', style: `--acc:${accent}`,
      onclick: () => navigate(`#/${route}/${encodeURIComponent(k)}`) },
      el('span', { class: 'chip-dot' }), k));
  }
  card.appendChild(ll);
  return card;
}

function agentToolsCard(tools) {
  if (!tools || !tools.groups || !tools.groups.length) return null;
  const card = el('div', { class: 'card acc-agents', style: '--acc:var(--accent)' });
  const collapsed = { v: false };
  const head = el('h3', { class: 'card-h tools-head', style: 'cursor:pointer; display:flex; align-items:center; gap:8px',
    onclick: () => { collapsed.v = !collapsed.v; body.style.display = collapsed.v ? 'none' : ''; chev.textContent = collapsed.v ? '▸' : '▾'; } });
  const chev = el('span', { class: 'chev', text: '▾', style: 'color:var(--faint)' });
  head.appendChild(chev);
  head.appendChild(document.createTextNode('Tools'));
  head.appendChild(el('span', { class: 'tag accent', style: 'margin-left:4px', text: String(tools.count) }));
  card.appendChild(head);

  const body = el('div', { class: 'tool-groups' });
  for (const g of tools.groups) {
    const grp = el('div', { class: 'tool-group' });
    const gl = el('div', { class: 'tool-group-label' },
      el('span', { class: 'tool-group-name', text: g.label }),
      el('span', { class: 'tool-group-count', text: `${g.tools.length}` }),
      g.gate ? el('span', { class: 'tool-flag', text: `needs ${g.gate}` }) : null);
    grp.appendChild(gl);
    const rows = el('div', { class: 'tool-rows' });
    for (const t of g.tools) {
      rows.appendChild(el('div', { class: 'tool-row' + (t.conditional ? ' conditional' : '') },
        el('code', { class: 'tool-name', text: t.name }),
        t.doc ? el('span', { class: 'tool-doc', text: t.doc }) : null,
        (t.conditional && t.flag && !g.gate) ? el('span', { class: 'tool-flag', text: t.flag }) : null));
    }
    grp.appendChild(rows);
    body.appendChild(grp);
  }
  card.appendChild(body);
  return card;
}

async function renderAgent(name) {
  const main = $('#main');
  main.innerHTML = '<div class="loading">Loading agent…</div>';
  const [info, xr] = await Promise.all([api(`agents/${encodeURIComponent(name)}`), crossrefs()]);
  if (!info) { main.innerHTML = '<div class="empty-state">Agent not found.</div>'; return; }

  const state = { version: info.active, mode: 'rendered', base: info.versions[1] || info.active };
  main.innerHTML = '';
  main.appendChild(pageHead('Agent', name,
    [el('span', { class: 'tag accent', text: `${info.versions.length} version${info.versions.length > 1 ? 's' : ''}` })],
    info.summary));

  const usedBy = usedByCard('Used by pipelines', xr.agent_to_workflows?.[name], 'workflows', 'var(--violet)');
  if (usedBy) main.appendChild(usedBy);
  const toolsCard = agentToolsCard(info.tools);
  if (toolsCard) main.appendChild(toolsCard);

  const body = el('div');
  main.appendChild(body);

  async function draw() {
    body.innerHTML = '';
    const target = { kind: 'agent', id: name, version: state.version };
    const cmts = state.mode === 'diff' ? [] : await fetchComments(target);

    // version pills (hidden in diff mode, which has its own selectors)
    const head = el('div', { class: 'toolbar' });
    if (state.mode !== 'diff') {
      const pills = el('div', { class: 'pills' });
      for (const v of info.versions)
        pills.appendChild(el('span', {
          class: 'pill' + (v === info.active ? ' active-badge' : '') + (v === state.version ? ' active' : ''),
          text: v, onclick: () => { state.version = v; draw(); } }));
      head.appendChild(pills);
    }
    head.appendChild(modeBar([
      { key: 'rendered', label: 'Rendered' },
      { key: 'source', label: 'Source', badge: cmts.length },
      { key: 'diff', label: '⇄ Diff' },
    ], state.mode, (m) => { state.mode = m; draw(); }));
    body.appendChild(head);

    if (state.mode === 'rendered') {
      const ver = await api(`agents/${encodeURIComponent(name)}/${encodeURIComponent(state.version)}`);
      body.appendChild(el('div', { class: 'card' }, el('div', { class: 'md', html: mdToHtml(ver.content) })));
    } else if (state.mode === 'source') {
      const ver = await api(`agents/${encodeURIComponent(name)}/${encodeURIComponent(state.version)}`);
      body.appendChild(el('div', { class: 'card src-card' },
        makeSourceView(target, ver.content, cmts, (n) => updateSourceBadge(head, n))));
    } else {
      body.appendChild(el('div', { class: 'toolbar' },
        el('span', { class: 'seg' }, 'base', selectVersions(info.versions, state.base, (v) => { state.base = v; draw(); })),
        el('span', { class: 'seg' }, 'compare', selectVersions(info.versions, state.version, (v) => { state.version = v; draw(); }))));
      const [bA, bB] = await Promise.all([
        api(`agents/${encodeURIComponent(name)}/${encodeURIComponent(state.base)}`),
        api(`agents/${encodeURIComponent(name)}/${encodeURIComponent(state.version)}`),
      ]);
      body.appendChild(diffView(bA.content, bB.content, state.base, state.version));
    }
  }
  draw();
}

function selectVersions(versions, current, onChange) {
  const sel = el('select', { onchange: (e) => onChange(e.target.value) });
  for (const v of versions) sel.appendChild(el('option', { value: v, text: v, ...(v === current ? { selected: '' } : {}) }));
  return sel;
}

function diffView(a, b, la, lb) {
  if (a === b) return el('div', { class: 'diff' }, el('div', { class: 'diff-empty', text: `${la} and ${lb} are identical.` }));
  const rows = lineDiff(a, b);
  const wrap = el('div', { class: 'diff' });
  let nAdd = 0, nDel = 0;
  for (const r of rows) {
    if (r.t === 'add') nAdd++; if (r.t === 'del') nDel++;
    wrap.appendChild(el('div', { class: 'row ' + r.t },
      el('span', { class: 'gutter', text: r.t === 'add' ? '+' : r.t === 'del' ? '−' : '' }),
      el('span', { text: r.s || ' ' })));
  }
  return el('div', {},
    el('div', { class: 'toolbar' },
      el('span', { class: 'tag green', text: `+${nAdd}` }),
      el('span', { class: 'tag', style: 'color:var(--red);border-color:rgba(240,113,110,.4)', text: `−${nDel}` }),
      el('span', { class: 'seg', text: `${la} → ${lb}` })),
    wrap);
}

async function renderTask(name) {
  const main = $('#main');
  main.innerHTML = '<div class="loading">Loading task…</div>';
  const [info, xr] = await Promise.all([api(`tasks/${encodeURIComponent(name)}`), crossrefs()]);
  if (!info) { main.innerHTML = '<div class="empty-state">Task not found.</div>'; return; }

  const state = { version: info.version, mode: 'rendered' };
  main.innerHTML = '';
  main.appendChild(pageHead('Task template', name,
    [el('span', { class: 'tag', style: 'color:var(--teal);border-color:rgba(52,210,196,.4)', text: `active ${info.active}` })]));

  const body = el('div');
  main.appendChild(body);

  const usedBy = usedByCard('Used by pipelines', xr.task_to_workflows?.[name], 'workflows', 'var(--violet)');
  if (usedBy) main.appendChild(usedBy);

  async function draw() {
    const data = state.version === info.version ? info : await api(`tasks/${encodeURIComponent(name)}/${encodeURIComponent(state.version)}`);
    body.innerHTML = '';
    const target = { kind: 'task', id: name, version: state.version };
    const cmts = await fetchComments(target);

    const head = el('div', { class: 'toolbar' });
    if (info.versions.length > 1) {
      const pills = el('div', { class: 'pills acc-tasks', style: '--acc:var(--teal)' });
      for (const v of info.versions)
        pills.appendChild(el('span', { class: 'pill' + (v === state.version ? ' active' : '') + (v === info.active ? ' active-badge' : ''),
          text: v, onclick: () => { state.version = v; draw(); } }));
      head.appendChild(pills);
    }
    head.appendChild(modeBar([
      { key: 'rendered', label: 'Rendered' },
      { key: 'source', label: 'Source', badge: cmts.length },
    ], state.mode, (m) => { state.mode = m; draw(); }));
    body.appendChild(head);

    if (state.mode === 'source') {
      body.appendChild(el('div', { class: 'card src-card' },
        makeSourceView(target, data.raw || '', cmts, (n) => updateSourceBadge(head, n))));
      return;
    }

    // meta chips
    const f = data.fields || {};
    const metaRow = el('div', { class: 'chips-row', style: 'margin-bottom:14px' });
    if (f.iterations != null) metaRow.appendChild(el('span', { class: 'tag', text: `iterations: ${f.iterations}` }));
    if (f.format) metaRow.appendChild(el('span', { class: 'tag', text: `format: ${f.format}` }));
    if (f.max_steps != null) metaRow.appendChild(el('span', { class: 'tag', text: `max_steps: ${f.max_steps}` }));
    if (metaRow.children.length) body.appendChild(metaRow);

    // skills
    if (data.skills?.length) {
      const sc = el('div', { class: 'card acc-skills', style: '--acc:var(--green)' },
        el('h3', { class: 'card-h', text: 'Injected skills' }));
      const row = el('div', { class: 'chips-row' });
      for (const s of data.skills)
        row.appendChild(el('div', { class: 'chip', style: '--acc:var(--green)',
          onclick: () => navigate(`#/skills/${encodeURIComponent(s)}`) },
          el('span', { class: 'chip-dot' }), s));
      sc.appendChild(row);
      body.appendChild(sc);
    }

    const labels = { objective: 'Objective', instructions: 'Instructions', output_format: 'Output format',
      context: 'Context', artifacts: 'Artifacts' };
    const card = el('div', { class: 'card fields-list' });
    for (const key of ['objective', 'context', 'instructions', 'output_format', 'artifacts']) {
      if (f[key] == null) continue;
      let content = f[key];
      if (Array.isArray(content)) content = content.map((x) => '- `' + x + '`').join('\n');
      card.appendChild(el('div', { class: 'field-block' },
        el('div', { class: 'field-label', text: labels[key] || key }),
        el('div', { class: 'md', html: mdToHtml(String(content)) })));
    }
    body.appendChild(card);
  }
  draw();
}

async function renderSkill(name, ref) {
  const main = $('#main');
  main.innerHTML = '<div class="loading">Loading skill…</div>';
  const [info, xr] = await Promise.all([api(`skills/${encodeURIComponent(name)}`), crossrefs()]);
  if (!info) { main.innerHTML = '<div class="empty-state">Skill not found.</div>'; return; }

  main.innerHTML = '';
  main.appendChild(pageHead('Skill', name,
    [info.references.length ? el('span', { class: 'tag green', text: `${info.references.length} references` }) : null],
    info.description));

  // cross refs
  const tasks = usedByCard('Declared by tasks', xr.skill_to_tasks?.[name], 'tasks', 'var(--teal)');
  const wfs = usedByCard('Injected in pipelines', xr.skill_to_workflows?.[name], 'workflows', 'var(--violet)');
  if (tasks) main.appendChild(tasks);
  if (wfs) main.appendChild(wfs);

  // references
  if (info.references.length) {
    const refCard = el('div', { class: 'card acc-skills', style: '--acc:var(--green)' },
      el('h3', { class: 'card-h', text: 'References' }));
    const row = el('div', { class: 'chips-row' });
    row.appendChild(el('div', { class: 'chip' + (!ref ? '' : ''), style: '--acc:var(--green)' + (!ref ? ';background:var(--green);color:#0c0e14' : ''),
      onclick: () => navigate(`#/skills/${encodeURIComponent(name)}`) }, 'index'));
    for (const r of info.references)
      row.appendChild(el('div', { class: 'chip', style: '--acc:var(--green)' + (r === ref ? ';background:var(--green);color:#0c0e14' : ''),
        onclick: () => navigate(`#/skills/${encodeURIComponent(name)}/ref/${encodeURIComponent(r)}`) },
        el('span', { class: 'chip-dot' }), r));
    refCard.appendChild(row);
    main.appendChild(refCard);
  }

  // body (rendered | source w/ line comments)
  let renderedMd, sourceText, target;
  if (ref) {
    const rd = await api(`skills/${encodeURIComponent(name)}/ref/${encodeURIComponent(ref)}`);
    renderedMd = rd ? rd.content : '_missing_';
    sourceText = rd ? rd.content : '';
    target = { kind: 'skill', id: name, version: 'ref/' + ref };
  } else {
    renderedMd = info.content;
    sourceText = info.raw || info.content;
    target = { kind: 'skill', id: name, version: 'index' };
  }

  const mount = el('div');
  main.appendChild(mount);
  const sstate = { mode: 'rendered' };
  async function drawBody() {
    mount.innerHTML = '';
    const cmts = await fetchComments(target);
    const head = el('div', { class: 'toolbar' },
      ref ? el('span', { class: 'crumb', text: `references / ${ref}.md` }) : null,
      modeBar([
        { key: 'rendered', label: 'Rendered' },
        { key: 'source', label: 'Source', badge: cmts.length },
      ], sstate.mode, (m) => { sstate.mode = m; drawBody(); }));
    mount.appendChild(head);
    if (sstate.mode === 'source') {
      mount.appendChild(el('div', { class: 'card src-card' },
        makeSourceView(target, sourceText, cmts, (n) => updateSourceBadge(head, n))));
    } else {
      mount.appendChild(el('div', { class: 'card' }, el('div', { class: 'md', html: mdToHtml(renderedMd) })));
    }
  }
  drawBody();
}

// ───────────────────────── eval analytics ─────────────────────────
function statCard(num, label, accent) {
  return el('div', { class: 'stat-card', style: `--acc:${accent || 'var(--evals)'}; cursor:default` },
    el('div', { class: 'num', text: num }), el('div', { class: 'lbl', text: label }));
}

async function renderEval(id) {
  const main = $('#main');
  main.classList.add('wide');
  main.innerHTML = '<div class="loading">Loading eval run…</div>';
  const run = await api(`evals/${encodeURIComponent(id)}`);
  if (!run) { main.innerHTML = '<div class="empty-state">Eval run not found.</div>'; return; }
  const s = run.summary, t = s.totals;
  main.innerHTML = '';
  main.classList.remove('wide');

  const kindTag = el('span', { class: 'tag', style: `color:var(--evals);border-color:rgba(95,200,247,.4)`, text: run.kind === 'vuln' ? 'vuln detection' : 'exploitability' });
  const metaBits = [run.model, run.prompt_version ? `prompt ${run.prompt_version}` : null, run.timestamp ? new Date(run.timestamp).toLocaleString() : null].filter(Boolean).join('  ·  ');
  main.appendChild(pageHead('Eval run', run.name, [kindTag], metaBits));

  // ── headline stat cards ──
  const grid = el('div', { class: 'grid-2' });
  if (run.kind === 'vuln') {
    grid.appendChild(statCard(pct(s.micro.f1), 'F1 (micro)', 'var(--evals)'));
    grid.appendChild(statCard(pct(s.micro.precision), `Precision · ${t.tp}TP / ${t.fp}FP`, 'var(--green)'));
    grid.appendChild(statCard(pct(s.micro.recall), `Recall · ${t.fn} missed`, 'var(--amber)'));
    grid.appendChild(statCard(String(t.fixtures), `Fixtures · ${fmtTokens(t.input_tokens + t.output_tokens)} tok`, 'var(--accent)'));
  } else {
    grid.appendChild(statCard(pct(s.headline.pass_rate), `Pass rate · ${s.headline.passed}/${s.headline.total}`, 'var(--green)'));
    grid.appendChild(statCard(pct(s.headline.evidence_rate), 'Has evidence', 'var(--accent)'));
    grid.appendChild(statCard(String(t.http_requests), 'HTTP requests', 'var(--amber)'));
    grid.appendChild(statCard(String(t.fixtures), `Fixtures · ${fmtTokens(t.input_tokens + t.output_tokens)} tok`, 'var(--violet)'));
  }
  main.appendChild(grid);

  // ── classification / verdict overview ──
  if (run.kind === 'vuln') {
    const card = el('div', { class: 'card' }, el('h3', { class: 'card-h', text: 'Finding classification' }));
    card.appendChild(stackBar([
      { value: t.tp, color: 'var(--green)', label: 'true positive' },
      { value: t.fp, color: 'var(--red)', label: 'false positive' },
      { value: t.fn, color: 'var(--amber)', label: 'false negative (missed)' },
      { value: t.tn, color: 'var(--faint)', label: 'true negative' },
    ]));
    card.appendChild(el('div', { class: 'legend', style: 'margin-top:12px' },
      legendDot('var(--green)', `TP ${t.tp}`), legendDot('var(--red)', `FP ${t.fp}`),
      legendDot('var(--amber)', `FN ${t.fn}`), legendDot('var(--faint)', `TN ${t.tn}`)));
    main.appendChild(card);
    if (s.per_cwe.some((c) => c.tp + c.fp + c.fn > 0)) main.appendChild(cweCard(s.per_cwe));
  } else {
    main.appendChild(verdictMatrixCard(s.verdict_matrix));
  }

  // ── per-fixture breakdown ──
  main.appendChild(run.kind === 'vuln' ? vulnFixtureCard(run) : exploitFixtureCard(run));

  // ── tool usage ──
  if (s.tools.length) main.appendChild(toolUsageCard(s.tools));
}

function legendDot(color, label) {
  return el('span', {}, el('span', { class: 'sw', style: `background:${color};width:12px;height:12px;border-radius:3px` }), label);
}

function cweCard(rows) {
  const card = el('div', { class: 'card' }, el('h3', { class: 'card-h', text: 'Per-CWE recall' }));
  const tbl = el('div', { class: 'mini-table' });
  for (const r of rows.filter((x) => x.tp + x.fp + x.fn > 0)) {
    tbl.appendChild(el('div', { class: 'mt-row' },
      el('code', { class: 'mt-key', text: r.cwe }),
      el('div', { class: 'mt-bar' }, bar(r.recall, { color: 'var(--green)', label: `${r.tp}/${r.tp + r.fn}` })),
      el('span', { class: 'mt-meta', text: `P ${pct(r.precision)}` })));
  }
  card.appendChild(tbl);
  return card;
}

function verdictMatrixCard(matrix) {
  const card = el('div', { class: 'card' }, el('h3', { class: 'card-h', text: 'Verdict matrix · expected → actual' }));
  const cols = ['exploitable', 'not_exploitable', 'inconclusive', 'other'];
  const tbl = el('table', { class: 'vmatrix' });
  const head = el('tr', {}, el('th', { text: 'expected ↓' }));
  for (const c of cols) head.appendChild(el('th', { text: c.replace('_', ' ') }));
  tbl.appendChild(head);
  for (const exp of ['exploitable', 'not_exploitable', 'inconclusive']) {
    const row = matrix[exp] || {};
    const tr = el('tr', {}, el('th', { text: exp.replace('_', ' ') }));
    for (const c of cols) {
      const v = row[c] || 0;
      const correct = c === exp && v > 0;
      tr.appendChild(el('td', { class: v ? (correct ? 'vm-ok' : 'vm-bad') : 'vm-zero', text: String(v) }));
    }
    tbl.appendChild(tr);
  }
  card.appendChild(tbl);
  return card;
}

function vulnFixtureCard(run) {
  const card = el('div', { class: 'card' }, el('h3', { class: 'card-h', text: 'Per-fixture' }));
  for (const f of run.summary.fixtures) {
    const det = run.detail[f.slug];
    const header = el('div', { class: 'fx-head', onclick: (e) => e.currentTarget.parentElement.classList.toggle('open') },
      el('span', { class: 'fx-chev', text: '▸' }),
      el('span', { class: 'fx-slug', text: f.slug }),
      f.prompt_version ? el('span', { class: 'tag', text: f.prompt_version }) : null,
      el('span', { class: 'fx-metrics' },
        el('span', { class: 'fx-m', html: `F1 <b>${pct(f.f1)}</b>` }),
        el('span', { class: 'fx-m', html: `P <b>${pct(f.precision)}</b>` }),
        el('span', { class: 'fx-m', html: `R <b>${pct(f.recall)}</b>` }),
        el('span', { class: 'fx-m muted', text: `${f.tp}·${f.fp}·${f.fn}` }),
        el('span', { class: 'fx-m muted', text: `${fmtTokens(f.total_tokens)} tok` }),
        el('span', { class: 'fx-m muted', text: fmtDur(f.duration_s) })));
    const body = el('div', { class: 'fx-body' });
    if (det) {
      const mk = det.matches || [];
      if (mk.length) {
        body.appendChild(el('div', { class: 'fx-sub', text: 'Findings vs ground truth' }));
        for (const m of mk) {
          const cls = m.classification;
          body.appendChild(el('div', { class: 'finding-row' },
            el('span', { class: 'fc-tag fc-' + cls, text: cls }),
            el('code', { class: 'fc-file', text: (m.finding_file || m.ground_truth_id || '') }),
            m.finding_cwe ? el('span', { class: 'fc-cwe', text: m.finding_cwe }) : null));
        }
      }
      if (det.skills_loaded?.length) body.appendChild(el('div', { class: 'fx-note', text: 'skills: ' + det.skills_loaded.join(', ') }));
    }
    card.appendChild(el('div', { class: 'fx' }, header, body));
  }
  return card;
}

function exploitFixtureCard(run) {
  const card = el('div', { class: 'card' }, el('h3', { class: 'card-h', text: 'Per-fixture' }));
  for (const f of run.summary.fixtures) {
    const det = run.detail[f.slug];
    const header = el('div', { class: 'fx-head', onclick: (e) => e.currentTarget.parentElement.classList.toggle('open') },
      el('span', { class: 'fx-chev', text: '▸' }),
      el('span', { class: 'fx-slug', text: f.slug }),
      el('div', { class: 'fx-passbar' }, stackBar([
        { value: f.cases_passed, color: 'var(--green)', label: 'passed' },
        { value: f.cases_total - f.cases_passed, color: 'var(--red)', label: 'failed' },
      ], f.cases_total)),
      el('span', { class: 'fx-metrics' },
        el('span', { class: 'fx-m', html: `<b>${f.cases_passed}/${f.cases_total}</b>` }),
        el('span', { class: 'fx-m muted', text: `${f.http_requests} http` }),
        el('span', { class: 'fx-m muted', text: `${fmtTokens(f.total_tokens)} tok` }),
        el('span', { class: 'fx-m muted', text: fmtDur(f.duration_s) })));
    const body = el('div', { class: 'fx-body' });
    for (const c of (det?.cases || [])) {
      body.appendChild(el('div', { class: 'case-row' },
        el('span', { class: 'case-mark ' + (c.passed ? 'ok' : 'bad'), text: c.passed ? '✓' : '✗' }),
        el('code', { class: 'case-name', text: c.finding_name || c.id }),
        el('span', { class: 'case-verdict', html: `${c.expected_verdict} <span class="muted">→</span> <b class="${c.passed ? 'good' : 'bad'}">${c.actual_verdict || '—'}</b>` }),
        c.has_evidence ? el('span', { class: 'case-ev', text: '🔬 evidence' }) : null,
        el('span', { class: 'case-meta', text: `${c.http_requests ?? 0} http · ${fmtTokens(c.total_tokens)} tok · ${fmtDur(c.duration_s)}` })));
    }
    card.appendChild(el('div', { class: 'fx' }, header, body));
  }
  return card;
}

function toolUsageCard(tools) {
  const card = el('div', { class: 'card' }, el('h3', { class: 'card-h', text: 'Tool usage (aggregate)' }));
  const max = Math.max(...tools.map((t) => t.count), 1);
  const tbl = el('div', { class: 'mini-table' });
  for (const t of tools) {
    tbl.appendChild(el('div', { class: 'mt-row' },
      el('code', { class: 'mt-key', text: t.name }),
      el('div', { class: 'mt-bar' }, bar(t.count / max, { color: 'var(--accent)' })),
      el('span', { class: 'mt-meta', text: String(t.count) })));
  }
  card.appendChild(tbl);
  return card;
}

// ───────────────────────── workflow / DAG ─────────────────────────
async function renderWorkflow(key) {
  const main = $('#main');
  main.classList.add('wide');
  main.innerHTML = '<div class="loading">Loading pipeline…</div>';
  const wf = await api(`workflows/${encodeURIComponent(key)}`);
  if (!wf) { main.innerHTML = '<div class="empty-state">Pipeline not found.</div>'; return; }

  main.innerHTML = '';
  main.appendChild(pageHead('Pipeline', key,
    [el('span', { class: 'tag', style: 'color:var(--violet);border-color:rgba(169,139,255,.4)', text: wf.class_name })]));
  if (wf.doc) main.appendChild(el('div', { class: 'card' }, el('div', { class: 'md', html: mdToHtml(wf.doc) })));

  if (wf.warnings?.length) {
    main.appendChild(el('div', { class: 'warn-banner' }, '⚠️',
      el('div', {}, el('strong', { text: 'Statically derived — read with care.' }),
        el('ul', {}, ...wf.warnings.map((w) => el('li', { text: w }))))));
  }

  if (!wf.nodes.length) {
    main.appendChild(el('div', { class: 'empty-state', text: 'No task graph to draw for this pipeline.' }));
  } else {
    const detail = el('div', { class: 'dag-detail empty', text: 'Click a node to inspect its task, agent, skills and budget.' });
    main.appendChild(drawDAG(wf, detail));
    main.appendChild(dagLegend());
    main.appendChild(detail);
  }

  if (wf.budgets && Object.keys(wf.budgets).length) {
    const bc = el('div', { class: 'card' }, el('h3', { class: 'card-h', text: 'Token budgets (config.yaml)' }));
    const kv = el('div', { class: 'kv' });
    for (const [k, v] of Object.entries(wf.budgets)) { kv.appendChild(el('div', { class: 'k', text: k })); kv.appendChild(el('div', { text: String(v) })); }
    bc.appendChild(kv);
    main.appendChild(bc);
  }
}

function dagLegend() {
  return el('div', { class: 'legend' },
    el('span', {}, el('span', { class: 'sw', style: 'background:var(--panel);border:1px solid var(--border)' }), 'task'),
    el('span', {}, el('span', { class: 'sw', style: 'background:rgba(169,139,255,.25);border:1px solid var(--violet)' }), 'sub-workflow'),
    el('span', {}, el('span', { class: 'sw', style: 'border:1px dashed var(--muted);background:transparent' }), 'conditional'),
    el('span', {}, el('span', { class: 'sw', style: 'background:transparent;border-top:2px dashed #4a3d6e;border-radius:0' }), 'implied stage order'),
  );
}

function drawDAG(wf, detail) {
  // layering by longest path
  const nodes = wf.nodes.map((n) => ({ ...n }));
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const incoming = new Map(nodes.map((n) => [n.id, []]));
  const outgoing = new Map(nodes.map((n) => [n.id, []]));
  for (const e of wf.edges) {
    if (byId.has(e.source) && byId.has(e.target)) {
      outgoing.get(e.source).push(e.target);
      incoming.get(e.target).push(e.source);
    }
  }
  // task levels
  const level = new Map();
  const taskNodes = nodes.filter((n) => n.kind === 'task');
  const visiting = new Set();
  const computeLevel = (id) => {
    if (level.has(id)) return level.get(id);
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const ins = incoming.get(id) || [];
    const lv = ins.length ? Math.max(...ins.map(computeLevel)) + 1 : 0;
    level.set(id, lv);
    visiting.delete(id);
    return lv;
  };
  taskNodes.forEach((n) => computeLevel(n.id));
  let maxTaskLevel = 0;
  for (const n of taskNodes) maxTaskLevel = Math.max(maxTaskLevel, level.get(n.id));

  // sub-workflows: chained columns after the tasks
  const subNodes = nodes.filter((n) => n.kind === 'subworkflow').sort((a, b) => a.order - b.order);
  subNodes.forEach((n, i) => level.set(n.id, maxTaskLevel + 1 + i));

  // implied edges: last task -> first sub, sub_i -> sub_{i+1}
  const impliedEdges = [];
  if (subNodes.length && taskNodes.length) {
    const lastTask = taskNodes.reduce((a, b) => (level.get(a.id) > level.get(b.id) || (level.get(a.id) === level.get(b.id) && a.order > b.order) ? a : b));
    impliedEdges.push({ source: lastTask.id, target: subNodes[0].id, implied: true });
  }
  for (let i = 0; i < subNodes.length - 1; i++)
    impliedEdges.push({ source: subNodes[i].id, target: subNodes[i + 1].id, implied: true });

  // group by level → assign rows
  const cols = new Map();
  for (const n of nodes) {
    const lv = level.get(n.id) ?? 0;
    if (!cols.has(lv)) cols.set(lv, []);
    cols.get(lv).push(n);
  }
  const NW = 196, NH = 66, GAPX = 78, GAPY = 26, PAD = 28;
  let maxRows = 0;
  const pos = new Map();
  [...cols.keys()].sort((a, b) => a - b).forEach((lv) => {
    const list = cols.get(lv).sort((a, b) => a.order - b.order);
    maxRows = Math.max(maxRows, list.length);
    list.forEach((n, row) => pos.set(n.id, { x: PAD + lv * (NW + GAPX), y: PAD + row * (NH + GAPY), lv, row }));
  });
  const nLevels = cols.size ? Math.max(...cols.keys()) + 1 : 1;
  const width = PAD * 2 + nLevels * NW + (nLevels - 1) * GAPX;
  const height = PAD * 2 + maxRows * NH + (maxRows - 1) * GAPY;

  const SVGNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(SVGNS, 'svg');
  svg.setAttribute('class', 'dag');
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  const g = document.createElementNS(SVGNS, 'g');
  svg.appendChild(g);

  const mk = (tag, attrs) => { const e = document.createElementNS(SVGNS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; };
  const edgePath = (s, t) => {
    const a = pos.get(s), b = pos.get(t);
    if (!a || !b) return null;
    const x1 = a.x + NW, y1 = a.y + NH / 2, x2 = b.x, y2 = b.y + NH / 2;
    const mx = (x1 + x2) / 2;
    return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
  };

  const edgeEls = [];
  for (const e of [...wf.edges, ...impliedEdges]) {
    const d = edgePath(e.source, e.target);
    if (!d) continue;
    const p = mk('path', { d, class: 'edge' + (e.implied ? ' implied' : '') });
    if (e.label) { const tl = mk('title', {}); tl.textContent = e.label; p.appendChild(tl); }
    g.appendChild(p);
    edgeEls.push({ p, src: e.source, tgt: e.target });
  }

  let selected = null;
  for (const n of nodes) {
    const p = pos.get(n.id);
    const ng = mk('g', { class: `node node-${n.kind === 'task' ? 'task' : 'sub'}${n.conditional ? ' conditional' : ''}`, transform: `translate(${p.x},${p.y})` });
    ng.appendChild(mk('rect', { class: 'node-rect', width: NW, height: NH, rx: 11 }));
    const title = n.kind === 'task' ? n.task : n.class_name.replace(/Workflow$/, '');
    const t1 = mk('text', { class: 'node-title', x: 14, y: 26 }); t1.textContent = trunc(title, 24);
    ng.appendChild(t1);
    const subLabel = n.kind === 'task' ? (n.agent || '—') : 'sub-workflow';
    const t2 = mk('text', { class: 'node-sub-label', x: 14, y: 46 }); t2.textContent = trunc(subLabel, 26);
    ng.appendChild(t2);
    if (n.conditional) {
      const badge = mk('text', { class: 'node-badge', x: NW - 16, y: 20, fill: 'var(--amber)' }); badge.textContent = '?';
      ng.appendChild(badge);
    }
    ng.style.cursor = 'pointer';
    ng.addEventListener('click', () => {
      if (selected) selected.classList.remove('selected');
      ng.classList.add('selected'); selected = ng;
      for (const ee of edgeEls) ee.p.classList.toggle('hot', ee.src === n.id || ee.tgt === n.id);
      fillDetail(detail, n, wf);
    });
    g.appendChild(ng);
  }

  // pan + zoom
  let scale = 1, tx = 0, ty = 0;
  const apply = () => g.setAttribute('transform', `translate(${tx},${ty}) scale(${scale})`);
  const scroll = el('div', { class: 'dag-scroll' });
  scroll.appendChild(svg);
  scroll.addEventListener('wheel', (ev) => {
    if (!ev.ctrlKey && !ev.metaKey) return;
    ev.preventDefault();
    const f = ev.deltaY < 0 ? 1.1 : 0.9;
    scale = Math.min(2.2, Math.max(0.4, scale * f));
    apply();
  }, { passive: false });
  let dragging = false, sx = 0, sy = 0;
  scroll.addEventListener('mousedown', (ev) => {
    if (ev.target.closest('.node')) return;
    dragging = true; sx = ev.clientX - tx; sy = ev.clientY - ty; scroll.classList.add('grabbing');
  });
  window.addEventListener('mousemove', (ev) => { if (!dragging) return; tx = ev.clientX - sx; ty = ev.clientY - sy; apply(); });
  window.addEventListener('mouseup', () => { dragging = false; scroll.classList.remove('grabbing'); });

  const zoom = (f) => { scale = Math.min(2.2, Math.max(0.4, scale * f)); apply(); };
  const wrap = el('div', { class: 'dag-wrap' },
    el('div', { class: 'dag-toolbar' },
      el('button', { text: '+', title: 'Zoom in (or Ctrl+scroll)', onclick: () => zoom(1.15) }),
      el('button', { text: '−', title: 'Zoom out', onclick: () => zoom(0.87) }),
      el('button', { text: '⊙', title: 'Reset', onclick: () => { scale = 1; tx = 0; ty = 0; apply(); } })),
    scroll);
  return wrap;
}

function fillDetail(detail, n, wf) {
  detail.className = 'dag-detail';
  detail.innerHTML = '';
  if (n.kind === 'subworkflow') {
    detail.appendChild(el('h3', { class: 'card-h', text: 'Sub-workflow' }));
    detail.appendChild(el('div', { class: 'page-title', style: 'font-size:19px' }, n.class_name));
    if (n.workflow_key) {
      detail.appendChild(el('div', { style: 'margin-top:12px' },
        el('div', { class: 'chip', style: '--acc:var(--violet)',
          onclick: () => navigate(`#/workflows/${encodeURIComponent(n.workflow_key)}`) },
          el('span', { class: 'chip-dot' }), `open ${n.workflow_key} →`)));
    }
    return;
  }
  detail.appendChild(el('h3', { class: 'card-h', text: 'Task node' }));
  detail.appendChild(el('div', { class: 'page-title', style: 'font-size:19px' }, n.task,
    n.conditional ? el('span', { class: 'tag', style: 'color:var(--amber);border-color:rgba(232,181,82,.4)', text: 'conditional' }) : null));

  const kv = el('div', { class: 'kv', style: 'margin-top:12px' });
  const add = (k, v) => { kv.appendChild(el('div', { class: 'k', text: k })); kv.appendChild(v instanceof Node ? v : el('div', { text: v })); };
  if (n.agent) add('agent', el('span', { class: 'chip', style: '--acc:var(--accent)', onclick: () => navigate(`#/agents/${encodeURIComponent(n.agent)}`) }, el('span', { class: 'chip-dot' }), n.agent));
  if (n.namespace) add('namespace', n.namespace);
  if (n.budget) add('budget', `iterations ${n.budget.iterations ?? '–'} · max_attempts ${n.budget.max_attempts ?? '–'} · max_steps ${n.budget.max_steps ?? '–'}`);
  if (n.artifacts?.length) {
    const box = el('div', {});
    n.artifacts.forEach((a) => box.appendChild(el('div', { style: 'font-family:var(--mono);font-size:12px;color:var(--muted)', text: a })));
    add('consumes', box);
  }
  if (n.external_inputs?.length) {
    const box = el('div', {});
    n.external_inputs.forEach((a) => box.appendChild(el('div', { style: 'font-family:var(--mono);font-size:12px;color:var(--faint)', text: a })));
    add('external input', box);
  }
  if (n.skills?.length) {
    const box = el('div', { class: 'chips-row' });
    n.skills.forEach((s) => box.appendChild(el('div', { class: 'chip', style: '--acc:var(--green)', onclick: () => navigate(`#/skills/${encodeURIComponent(s)}`) }, el('span', { class: 'chip-dot' }), s)));
    add('skills', box);
  }
  add('', el('div', { class: 'chip', onclick: () => navigate(`#/tasks/${encodeURIComponent(n.task)}`) }, `open task template ${n.task} →`));
  detail.appendChild(kv);
}

const trunc = (s, n) => (s && s.length > n ? s.slice(0, n - 1) + '…' : s || '');
const pct = (x) => (x == null ? '–' : Math.round(x * 100) + '%');
const fmtTokens = (n) => (n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k' : String(n || 0));
const fmtDur = (s) => (s == null ? '–' : s >= 60 ? Math.floor(s / 60) + 'm' + Math.round(s % 60) + 's' : (+s).toFixed(0) + 's');

// horizontal bar (0..1) with optional label
function bar(value, opts = {}) {
  const v = Math.max(0, Math.min(1, value || 0));
  const color = opts.color || 'var(--accent)';
  return el('div', { class: 'hbar', title: opts.title || '' },
    el('div', { class: 'hbar-fill', style: `width:${(v * 100).toFixed(1)}%;background:${color}` }),
    opts.label != null ? el('span', { class: 'hbar-label', text: opts.label }) : null);
}

// stacked segments: [{value, color, label}]
function stackBar(segments, total) {
  const sum = total || segments.reduce((a, s) => a + s.value, 0) || 1;
  const wrap = el('div', { class: 'stack-bar' });
  for (const s of segments) {
    if (!s.value) continue;
    wrap.appendChild(el('div', { class: 'stack-seg', title: `${s.label}: ${s.value}`,
      style: `width:${(s.value / sum * 100).toFixed(1)}%;background:${s.color}` },
      s.value / sum > 0.08 ? el('span', { text: String(s.value) }) : null));
  }
  return wrap;
}

async function crossrefs() {
  if (!STATE.crossrefs) STATE.crossrefs = await api('crossrefs');
  return STATE.crossrefs;
}

// ───────────────────────── boot ─────────────────────────
async function boot() {
  try {
    STATE.overview = await api('overview', { fresh: true });
  } catch (err) {
    $('#main').innerHTML = `<div class="empty-state">Could not reach the server.<br>${esc(err.message)}</div>`;
    return;
  }
  renderSidebar();
  route();
}

window.addEventListener('hashchange', route);
$('#search').addEventListener('input', (e) => renderSidebar(e.target.value));
document.addEventListener('keydown', (e) => {
  if (e.key === '/' && document.activeElement !== $('#search')) { e.preventDefault(); $('#search').focus(); }
  if (e.key === 'Escape') { $('#search').blur(); }
});
$('#reload-btn').addEventListener('click', async () => {
  cache.clear(); STATE.crossrefs = null;
  STATE.overview = await api('overview', { fresh: true });
  renderSidebar($('#search').value); route();
});
document.querySelectorAll('[data-link]').forEach((n) => n.addEventListener('click', () => navigate(n.dataset.link)));

boot();
