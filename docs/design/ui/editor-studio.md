# Workflow, Audit and Agent editor ("studio")

[UI redesign explorations](README.md) · Mockups: [`mockups/studio/`](mockups/studio/)

A visualiser and editor for Workflow definitions, AuditProfiles with a live
Audit overlay, and Agent templates.

**Status (2026-10-07):** the first working **B node studio** is implemented at
`/catalog/studio`, linked from Library, definition details and each Check.
It uses V3B vocabulary and the light, dark and black [themes](themes.md).
Direction B was chosen on 2026-10-04. The mockups below predate that palette
decision and use the former lime accent and Inter.

## Implemented authoring and live view

- Import one authored Workflow, AuditProfile or AgentTemplate by file or paste.
  YAML comments, opaque fields, workspace settings and multiline values remain
  in the syntax tree. Aliases must be expanded before import to prevent one
  visual edit from also changing another anchored block. Imports are bounded
  to 1 MiB, 20,000 syntax entries and 256 blocks.
- A native canvas supports block drag/drop, pan, zoom, reset, keyboard movement,
  transition connections, artifact wires and a shared failure terminal.
  Layout stays separate from the authored YAML. Phones switch between Graph,
  Blocks, Properties and Console rather than squeezing all panes together.
- The inspector edits stage objectives, planners, sessions, agents, incoming
  files, results, output mappings and outcome/retry/escalation transitions;
  Check type inputs, role bindings, inventory, execution limits and review
  policies; and Agent model, instructions, sandbox, summarizer, tools and skills.
  Advanced settings remain editable through block YAML and whole-document YAML.
  Renaming mapped blocks updates their explicit structural references.
- Problems, unapplied YAML and Diff since import have distinct states. Local
  checks cover graph cycles/reachability, outcome contracts, required output
  flow at joins, role dependency cycles, selectors and common structural rules.
  Reads of a known later artifact are warnings: runtime/project namespaces can
  supply files that are absent from authored stage declarations. They cannot
  truthfully be treated as definite read-before-write errors without that context.
  Installed selectors, instruction files and runtime compatibility still require
  `contractor server config validate --root <bundle>`; the UI says so explicitly.
- Undo/redo is bounded in memory. Navigation, replacement and removal use the
  shared Dialog; reload warns about unsaved work. Drafts and imported YAML never
  go into localStorage or sessionStorage. Export can retain structurally incomplete
  drafts, with local errors visible, and applies pending syntactically valid YAML.
- Live mode reads the published definition at the Check's pinned digest and
  shows Check / Role / Review lanes, item states and acceptance, paginated reviews
  and the durable event stream. It fences a multi-request snapshot by Check
  revision and round, retains settled pages on errors and labels stale snapshots.
  If a historical Check's pinned profile differs from the catalog or cannot be
  loaded, its execution state remains visible with a separate definition warning;
  the current catalog roles are not attached to that historical execution.
  Item cells cover all loaded rounds and name their round; counts do not imply
  that an unloaded page was read. Design edits are kept separate from execution.
  Studio makes no mutation requests and introduces no API or publishing flow.

Useful follow-ups are catalog-backed selector palettes, graph auto-layout and
dedicated forms for the advanced workspace and execution policies currently
edited in YAML. These do not block authored import/edit/export or the live view.

## Scope decisions

- **Draft locally, export YAML.** The editor shows a graph and keeps a local
  draft in the browser; the user exports YAML. There is no server publish and
  no new API.
- **Audits: definition and live state.** Audits show both the AuditProfile
  definition and a live overlay of a running Audit.
- **Authored YAML is the source.** The public `WorkflowResource` is a lossy
  projection: it has no workspace section and returns resolved execution
  configuration. The editor therefore loads authored YAML, by file or paste.
  "Open from published" is possible only with a visible warning that the
  result is incomplete. Agent templates round-trip almost losslessly.
- **Validation follows the real rules.** The mockups show:
  - no cycles through `next`;
  - every Stage reachable from `entryStage`;
  - no Artifact read before a Stage writes it.

The mockups use real configurations (`openapi-from-workspace@7`,
`openapi-operation-trace@1`, `workspace_openapi_builder@3`). The live Audit
state on `crapi-workshop` (24 operations, their states and times) is
illustrative.

## A · Graph tab inside the Catalog

The cheapest direction and the closest to today's UI; it needs no new
dependencies. The graph is read-only, and structure changes go through the
form.

![A · Workflow draft](screenshots/studio-a-workflow.webp)

**Workflow:** a vertical Stage graph. Retry → fail branches run on the right,
Artifact hand-offs are labelled on the edges, and an edge that forms a cycle
is marked red. A form panel on the right edits the selected Stage.

![A · Audit](screenshots/studio-a-audit.webp)

**Audit:** a new *Flow* tab on the Audit page. It shows the profile pipeline
(inputs → inventory → round → role → results → review → report) with live
counters, and the items table below.

![A · Agent](screenshots/studio-a-agent.webp)

**Agent:** a sectioned form, with toolsets shown as tool chips. A right column
lists checks, the agent's capabilities, where the template is used, and a diff.

## B · Full-screen node studio (chosen)

The most powerful direction for editing and for building a Workflow from
scratch. It is also the most expensive: it needs its own canvas engine, graph
layout, drag and drop, and a story for small screens.

![B · Workflow draft](screenshots/studio-b-workflow.webp)

**Workflow:** a block palette (Stage, Workflow input and output, Parameter,
Agent templates, planners) feeds a canvas of nodes with Artifact "wires" and a
shared failure bus. A draft Stage with no incoming transition is flagged as
unreachable. An inspector edits the selected Stage: planner, session, agent,
outcome transitions and wires in and out. A bottom console has *Problems*,
*YAML* and *Diff vs @7* tabs. *Validate* and *Export YAML* sit in the header,
next to a Design / Live switch.

![B · Audit live](screenshots/studio-b-audit.webp)

**Audit:** Live mode with Audit / Role / Review swimlanes, a matrix of the 24
items coloured by state (not-accepted items ringed), and an event stream.

![B · Agent assembly](screenshots/studio-b-agent.webp)

**Agent:** the template in the centre, with slots around it for the model
policy, instructions, sandbox profile, skills and toolsets.

## C · Matrix and flows

The densest and most precise direction, and good for review. It is weaker at
showing execution order and transition logic than a graph.

![C · Workflow artifact matrix](screenshots/studio-c-workflow.webp)

**Workflow:** an Artifact × Stage matrix (W, R, r, RW; an error cell shows
"R !") plus a transitions table. Data-flow errors that a graph hides become
obvious here.

![C · Audit Sankey](screenshots/studio-c-audit.webp)

**Audit:** the profile as a text outline on the left, a Sankey on the right
(operations → state → outcome → coverage), and a *Needs attention* list.

![C · Agent capability matrix](screenshots/studio-c-agent.webp)

**Agent:** a tool-by-tool comparison with neighbouring templates and an
"only differences" filter. The draft is edited in its own column, and a
toolset can be copied whole from another template.

## Possible combinations

The direction was chosen as a whole, but individual screens can be mixed. For
example, C's Artifact matrix could be a second tab of the B Workflow studio,
and C's agent comparison could sit next to B's assembly view.
