// Lays out the ERD with ELK and writes the page data.
// Usage: node layout.cjs <schema.json> <meta.json> <usage.json> <out.json> <schemaVersion>
const fs = require('fs');
const ELK = require('elkjs/lib/elk.bundled.js');

const [schemaPath, metaPath, usagePath, outPath, schemaVersion] = process.argv.slice(2);
if (!schemaVersion) {
  console.error('usage: node layout.cjs <schema.json> <meta.json> <usage.json> <out.json> <schemaVersion>');
  process.exit(2);
}
const elk = new ELK();
const schema = JSON.parse(fs.readFileSync(schemaPath, 'utf8'));
const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
const usage = JSON.parse(fs.readFileSync(usagePath, 'utf8'));

const TYPE = {
  'timestamp with time zone': 'timestamptz', 'boolean': 'bool', 'integer': 'int',
  'bigint': 'bigint', 'smallint': 'smallint', 'double precision': 'float8',
  'text[]': 'text[]', 'character varying': 'varchar', 'jsonb': 'jsonb', 'bytea': 'bytea',
};
const shortType = (t) => TYPE[t] || t.replace('timestamp with time zone', 'timestamptz');

// Geometry, shared with the page renderer.
const G = { head: 26, row: 16, padY: 6, charCol: 6.6, charHead: 7.3, padX: 10, mark: 36, gap: 14, more: 16 };

const fkCols = {};
for (const f of schema.fks) {
  for (const c of f.fc) (fkCols[f.from] ||= new Set()).add(c);
}

// meta.json must describe exactly the migrated tables.
const names = new Set(schema.tables.map((t) => t.name));
const problems = [
  ['has no description for', [...names].filter((n) => !meta.tables[n])],
  ['describes tables that no longer exist', Object.keys(meta.tables).filter((n) => !names.has(n))],
  ['flags tables that no longer exist', Object.keys(meta.flags).filter((n) => !names.has(n))],
  ['lists as removed tables that still exist', Object.keys(meta.removed || {}).filter((n) => names.has(n))],
].filter(([, list]) => list.length);
if (problems.length) {
  for (const [what, list] of problems) console.error(`scripts/erd/meta.json ${what}: ${list.join(', ')}`);
  process.exit(1);
}
const tables = schema.tables.map((t) => {
  const [group, desc, why, procs] = meta.tables[t.name];
  const pk = new Set(t.pk || []);
  const fk = fkCols[t.name] || new Set();
  const cols = t.cols.map((c) => ({ n: c.n, t: shortType(c.t), nn: c.nn, pk: pk.has(c.n), fk: fk.has(c.n), gen: c.gen }));
  return {
    name: t.name, group, desc, why: why || '', procs: procs || [], cols, pk: t.pk || [], uniq: t.uniq || [],
    triggers: t.triggers || [], usage: usage[t.name] || [],
  };
});

function size(t, mode) {
  const shown = mode === 'keys' ? t.cols.filter((c) => c.pk || c.fk) : t.cols;
  const hidden = t.cols.length - shown.length;
  const nameW = Math.max(...shown.map((c) => c.n.length), 4);
  const typeW = Math.max(...shown.map((c) => c.t.length), 4);
  const w = Math.ceil(Math.max(
    G.padX * 2 + t.name.length * G.charHead + 34,
    G.padX * 2 + G.mark + nameW * G.charCol + G.gap + typeW * G.charCol,
  ));
  const h = G.head + G.padY * 2 + shown.length * G.row + (hidden > 0 ? G.more : 0);
  return { w, h, shown: shown.map((c) => c.n), hidden };
}

async function layout(mode) {
  const groups = meta.groups.map((g) => ({
    id: 'g:' + g.id,
    layoutOptions: { 'elk.padding': '[top=44,left=20,bottom=20,right=20]' },
    children: tables.filter((t) => t.group === g.id).map((t) => {
      const s = size(t, mode);
      return { id: t.name, width: s.w, height: s.h };
    }),
  }));
  // Edges run parent -> child so core entities sit left of their dependants.
  const edges = schema.fks.map((f, i) => ({ id: 'e' + i, sources: [f.to], targets: [f.from] }));
  const graph = {
    id: 'root',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'RIGHT',
      'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
      'elk.edgeRouting': 'ORTHOGONAL',
      'elk.layered.spacing.nodeNodeBetweenLayers': '56',
      'elk.spacing.nodeNode': '28',
      'elk.spacing.edgeNode': '16',
      'elk.spacing.edgeEdge': '8',
      'elk.layered.spacing.edgeNodeBetweenLayers': '16',
      'elk.layered.spacing.edgeEdgeBetweenLayers': '8',
      'elk.spacing.componentComponent': '60',
      'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
      'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
      'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
      'elk.aspectRatio': '1.6',
    },
    children: groups,
    edges,
  };
  const out = await elk.layout(graph);
  const pos = {};
  const gpos = {};
  for (const g of out.children) {
    gpos[g.id.slice(2)] = { x: g.x, y: g.y, w: g.width, h: g.height };
    for (const n of g.children) {
      const { shown, hidden } = size(tables.find((t) => t.name === n.id), mode);
      pos[n.id] = { x: g.x + n.x, y: g.y + n.y, w: n.width, h: n.height, shown, hidden };
    }
  }
  // Edge sections are relative to the edge's container (root or a group).
  const routes = out.edges.map((e) => {
    const s = e.sections[0];
    const o = e.container && e.container !== 'root' ? gpos[e.container.slice(2)] : { x: 0, y: 0 };
    return [s.startPoint, ...(s.bendPoints || []), s.endPoint].map((p) => [Math.round((o.x + p.x) * 10) / 10, Math.round((o.y + p.y) * 10) / 10]);
  });
  return { w: Math.ceil(out.width), h: Math.ceil(out.height), groups: gpos, nodes: pos, routes };
}

(async () => {
  const keys = await layout('keys');
  const all = await layout('all');
  const data = {
    geom: G,
    schemaVersion,
    groups: meta.groups.map((g) => ({ ...g, count: tables.filter((t) => t.group === g.id).length })),
    tables, fks: schema.fks.map((f) => ({ from: f.from, to: f.to, fc: f.fc, tc: f.tc, del: f.del })),
    flags: meta.flags, tags: meta.tags, removed: meta.removed || {},
    layouts: { keys, all },
  };
  fs.writeFileSync(outPath, JSON.stringify(data));
})();
