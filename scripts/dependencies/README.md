# Audit and Workflow dependency graph

[`docs/spec/audit-workflow-dependencies.html`](../../docs/spec/audit-workflow-dependencies.html)
is a standalone HTML visualization. Open it directly in a browser; all styles,
graph data and rendering code are embedded. Source links are relative to its
location in `docs/spec/`.

```sh
make docs-dependencies
make docs-dependencies-check
```

The architecture view summarizes the ownership boundaries documented in the
specification. Its manually curated nodes and sourced relationships live in
`architecture.json`; the specification remains authoritative. The catalog and
Stage views are extracted from the default `configs/` catalog, including
Workflow escalation variants, exact template bindings, Skills, standard
packages, completion contracts, toolsets, instructions and execution-selection
references. Examples and process-only catalogs are outside this snapshot.

An arrow in the catalog points from the consumer to its dependency. Purple
data edges follow that direction too: the consuming Workflow points to its
output producer. These edges retain their AuditProfile context and are shown
in that profile's graph or the full catalog. Stage `next` arrows instead show
execution order; dashed `escalate` arrows identify conditional Planner
selections. Architecture arrows are labeled with the relationship they express.

Use the root selector, dependency direction and recursive checkbox to explore
either direct or transitive relationships. Incoming and outgoing traversals
are computed separately when both directions are selected. Type filters hide
instructions and credential IDs initially; search and the selector include
every node, including hidden types. Clicking a node or line shows its sources.
SVG export preserves the chosen theme and exports the entire filtered graph.

`build_graph.py` uses the Runtime's locked PyYAML dependency. Missing catalog
references and missing source files fail generation. `--check` compares the
complete deterministic output to the committed HTML without writing files.
The displayed fingerprint covers the data inputs, including cited implementation
and specification sources. Regenerate after relevant catalog or source changes;
the page describes checked-in configuration rather than live Run/Audit records.
