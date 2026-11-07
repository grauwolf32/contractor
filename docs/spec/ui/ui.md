# Web UI

Requirement prefix: `UI` · [Specification index](../README.md)

## Purpose and scope

This document owns the Contractor Web UI: its deployment boundary and frontend
architecture, the shared cache of Server state and data freshness, browser
session recovery, the information architecture (the navigation rail, routes,
the Library, the consolidated Runs surface, Project and Audit sections), UI
vocabulary, accessibility, and the presentation rules of the Catalog, Run
setup, Project, Audit and Operations surfaces.

Other documents own the adjacent concepts:

- The user journeys the UI must support, including the capabilities of the
  Workflow user surface and the primary Project journey:
  [UI journeys](journeys.md).
- Public API conventions (authentication, sessions, CSRF, pagination,
  idempotency, the error envelope, the live event WebSocket), and the rule that
  UI-oriented query endpoints may aggregate existing durable or read-model
  state while mutations call the same domain application services as non-UI
  API clients: [Public API](../api/public-api.md).
- Trust boundaries and the shared security model:
  [Security](../platform/security.md).
- Projects, ProjectScope, the Project artifact shortcuts, Workflow
  recommendations, Evals Projects and the Queue read API:
  [Projects](../artifacts/projects.md).
- The WorkflowRun resource, its lifecycle, resume and metadata labels:
  [Runs](../execution/runs.md).
- Explicit terminal Run deletion and recoverable Project deletion:
  [Retention and deletion](../artifacts/retention-and-deletion.md); the
  durable queue admission gate (owner Queue Pause):
  [Scheduler](../execution/scheduler.md).
- RuntimeConfig versions, label bindings, runtime credentials and Agent labels:
  [RuntimeConfig](../configuration/runtime-config.md).
- The Operations Overview, Runtime Agent cards, the Scheduler settings
  resource, the performance page and allocation history:
  [Operations](../operations/operations.md).
- Git import and the owner Git SSH key: [Git import](../artifacts/git-import.md).
- Agent Skill packages and the SkillCatalog: [Skills](../tools/skills.md).
- Managed Evals: [Managed Evals](../evals/managed-evals.md).
- Audits: [Audits](../audits/audits.md),
  [Audit execution](../audits/execution.md) and
  [Review and reports](../audits/review-and-reports.md).
- AgentTemplates: [AgentTemplates](../configuration/agent-templates.md).
- Artifact list filters and their pagination semantics:
  [ArtifactStore](../artifacts/artifact-store.md).

## Concepts

Each term is defined once in the [glossary](../glossary.md).

- **Web UI** — the separately deployable Node.js service and the
  client-rendered single-page application it serves; it uses the Server's
  public API and is not a second Scheduler, configuration resolver or source of
  lifecycle truth.
- **Runtime config document** — the non-secret `/runtime-config.json` that
  Node serves, holding exactly the UI version, the supported API versions and
  one absolute `apiBaseUrl`.
- **Transport adapter** — the small handwritten adapter around the generated
  TypeScript API client that alone handles the base URL, session cookie, CSRF
  token, idempotency and precondition headers and error mapping.
- **Navigation rail** — the top-level navigation, in three groups: main,
  secondary and bottom.
- **Inbox** — the destination at `/` that lists decisions first, then blocked
  work, finished results and running checks.
- **Library** — the user-facing name of the Catalog, which groups reusable
  execution definitions.
- **Check type** — the user-facing name of an AuditProfile.
- **Runs surface** — the single **Runs** destination with its Queue and
  Completed views.
- **Run draft** — the in-memory, unsubmitted Run request for one exact Workflow
  version and scope, with its slot confirmations.
- **Format-only suggestion** — an automatically prepopulated Artifact candidate
  for a Workflow input slot whose exact revision the user has not confirmed.
- **Run setup drawer** — the Configure Run surface: a drawer on desktop and a
  full-width panel on mobile.
- **Workflow discovery card** — the card that Project and Catalog share for one
  logical Workflow name, with its version selector.
- **Project Overview** — the bounded root page of a Project.
- **Audit workspace snapshot** — the owner-scoped SQL snapshot that
  `GET /v1/audits/{auditId}/workspace` returns for the Audit pages.

## Model

### Deployment structure

```mermaid
flowchart LR
    B["Browser: client-rendered React application"]
    N["Node.js Web UI service"]
    S["Go Server: versioned authenticated public API"]
    B -->|"static assets and /runtime-config.json"| N
    B -->|"API calls, Artifact upload and download, Operations mutations"| S
    N -.- X["Node: no SSR, BFF, API proxying, session storage or domain operation; no Server credential; no database, Control Plane or Runtime Agent access"]
```

### Navigation

The navigation rail has three groups: main (Inbox to Reports), secondary (Runs,
Library, Evals) and bottom (Operations and the Account menu).

```text
Inbox (/)             decisions, blocked work, finished results, running checks
Projects (/projects)
Checks (/checks)      Audits across Projects
Issues (/issues)      findings across Projects
Reports (/reports)    Audit reports across Projects

Runs (/runs)          Queue (default), Completed, Run details
Library (/catalog)    Check types, Workflows, Agents, Skills, Files (/artifacts)
Evals (/evals)

Operations            Runtime Agents, Allocations, LLM configurations,
(/operations)         Credentials, Settings
Account menu          signed-in user, Settings, Theme, Sign out, UI version
```

### Runs surface

```text
Runs
  Queue       default; initializing | pending | running | waiting | cancelling
  Completed   succeeded | failed | cancelled
```

### Run input review

Each Artifact input slot of a Run draft is reviewed separately. A format-only
suggestion becomes a confirmed slot only through the user's action for that
one slot.

```mermaid
stateDiagram-v2
    [*] --> Unset
    Unset --> Suggested: automatic prepopulation of a candidate
    Suggested --> Confirmed: user confirms the exact revision for this slot
    Unset --> Confirmed: explicit picker selection or completed upload
    Suggested --> Confirmed: explicit picker selection or completed upload
```

Form readiness reports three separate conditions: missing fields, pending input
review, and a request ready for submission.

## Requirements

### Deployment boundary

**UI-R1.** Contractor deployment includes one separately deployable Node.js
Web UI for security Workflows, Project-bound Audits and single-VM operational
visibility.

**UI-R2.** The UI uses the Server's public API; it does not become a second
Scheduler, configuration resolver or source of lifecycle truth.

**UI-R3.** The Web UI is its own Node.js build artifact and runtime service.
The Go Server embeds and serves no HTML, JavaScript, CSS or other frontend
asset.

**UI-R4.** Rebuilding, releasing, restarting or rolling back the Web UI does
not rebuild or restart the Server, and Server startup and health never depend
on UI availability. Both services may run on the same VM.

**UI-R5.** The only product contract between the Web UI and the Server is the
versioned authenticated public Server API.

**UI-R6.** The UI build records its own version and its supported Server API
versions. It must show an explicit incompatible-version error rather than
guessing around a missing or changed field.

**UI-R7.** Additive API changes within a supported version do not require a
coordinated release. Until the first release an incompatible `/v1` change is
allowed only together with the UI change in one commit
([API-R3](../api/public-api.md)).

**UI-R8.** The UI is a client-rendered frontend. Node serves its static
assets and a non-secret `/runtime-config.json`; it performs no SSR, BFF, API
proxying, session storage or domain operation.

**UI-R9.** The runtime config contains exactly the UI version, the supported
API versions and one absolute `apiBaseUrl` (LIM-UI-1), so the same UI build
can target another Server without rebuilding.

**UI-R10.** `apiBaseUrl` has no userinfo, query or fragment and uses HTTPS;
HTTP is accepted only for a loopback origin in local development.

**UI-R11.** Browser code, not Node, calls the configured Go Server URL
directly through the authenticated public Server APIs, including for Artifact
upload and download and Operations mutations.

**UI-R12.** Node receives no Server credential and has no database, Control
Plane or Runtime Agent access.

**UI-R13.** Existing API clients can perform the same domain operations
without using the Web UI.

### Frontend implementation

**UI-R14.** The UI is a React single-page application written in TypeScript
with strict type checking. Vite builds the browser bundle.

**UI-R15.** React Router is used in Data Mode with browser history.

**UI-R16.** The Node static server returns `index.html` for known
client-route fallbacks, but never uses that fallback for assets,
`/runtime-config.json` or unknown API-looking paths. A route segment that names
a configuration accepts exactly the grammar of its public schema
(ConfigurationName and ConfigVersion, or RuntimeInfrastructureId and
RuntimeConfigVersion for Runtime configurations), so a direct link to any valid
version, including one with `+`, loads.

**UI-R17.** TanStack Query is the only shared cache of Server state. It owns
API queries, mutations, invalidation, refetch and the selected live-update
policy.

**UI-R18.** Router, form and transient presentation state stay local to the
narrowest route or component that needs them. The UI has no Redux, Zustand or
second client-side copy of the Workflow/Run domain model.

**UI-R19.** The UI build pins the exact public contract and a deterministic
TypeScript client generator. Generated DTO and client files are not
hand-edited, and CI fails when regeneration produces an uncommitted diff.

**UI-R20.** A small handwritten transport adapter around the generated client
is the only place that:

- resolves `apiBaseUrl` from the runtime config;
- opts into the session cookie and attaches the current CSRF token to unsafe
  requests;
- supplies the operation-specific `Idempotency-Key` and `If-Match` values
  provided by the caller;
- maps the specified success and error envelopes into one bounded client error
  type without logging response bodies or secrets.

**UI-R21.** The transport adapter does not implement domain transitions, does
not retry unsafe mutations with a new idempotency key and does not persist
credentials or session material (FAIL-UI-2).

**UI-R22.** Server contract tests exercise request and response examples
against the same OpenAPI document that the UI build pins.

**UI-R196.** Every JSON success response the UI reads, through the generated
client or a direct request, must conform to the response schema the pinned
OpenAPI document declares for its operation and status, and every Server event
frame must conform to `api/events/contractor-events-v1.schema.json`. The
validators are generated from those two documents and interpreted at run time
without `eval`; handwritten code must not repeat a shape check that a schema
expresses, and request-side form rules (identifier grammars, limits and enums)
are read from the same generated schemas. A response that breaks its schema,
carries an undeclared status or cannot be read fails with the one bounded
invalid-response client error (`invalid_api_response`) that names the
operation and the first failing location, never the body; the UI accepts none
of it. An event frame that breaks the schema is a protocol error: live
continuity ends and the view returns to its REST snapshot (UI-R29). Only
invariants between members that a schema cannot express are checked by hand,
next to the operation that relies on them: a resource that echoes the
requested identity, an ETag equal to the body's revision, ordering and
uniqueness across items, matching cursor generations and sequences, and
consistent Planner plans.

**UI-R197.** A view that reads every page of a list follows the Server's
cursors through one shared reader that stops after 20 pages (LIM-UI-15). When
pages remain, the view says that the list is incomplete instead of presenting
a partial list as whole, and a picker or check that needs the whole list
refuses to proceed rather than choose from part of it.

**UI-R23.** Vite's transform is not the type-checking gate: CI separately
runs strict TypeScript checking.

**UI-R24.** Vitest and React Testing Library cover units and routed
components. Playwright covers browser behaviour against the real Node static
service and Go Server, including login and session recovery, CORS and CSRF
failure, one Run lifecycle, Artifact transfer and an authorized Operations
mutation.

**UI-R25.** The Node major version, the package-manager version, the
generator and all frontend dependencies are lockfile-pinned. Their upgrades
and the UI release remain independent of the Go Server release while the
supported API version overlaps.

### Server state and data freshness

**UI-R26.** The UI never predicts an authoritative Run, StageExecution,
allocation, configuration-publication or credential-operation transition.

**UI-R27.** A successful mutation causes the affected query keys to be
invalidated and refetched; only a subsequent Server response establishes the
displayed domain state (FAIL-UI-2).

**UI-R28.** Optimistic updates are allowed for purely local drafts, but not
for execution lifecycle, published configuration or credential state.

**UI-R29.** WebSocket Operations notifications remain hints: after a
process-generation or revision gap the UI refetches the authoritative REST
snapshot (FAIL-UI-8).

**UI-R30.** No mutation is sent over WebSocket.

### Browser session

```mermaid
flowchart TD
    R["Domain API response: typed JSON or direct Artifact request"] --> A{"Status"}
    A -->|"401 from a request started with the current CSRF token"| C["Clear in-memory CSRF token, session and protected query cache; return to sign-in"]
    A -->|"late 401 from a request started with an older CSRF token"| K["Keep the newer login"]
    A -->|"403"| N["Session is not invalidated"]
    N --> U{"Unsafe request?"}
    U -->|"no"| D["Stays 403: no retry, no session invalidation"]
    U -->|"yes"| RS["Re-read the session once"]
    RS --> T{"Another tab replaced the session and the CSRF token changed?"}
    T -->|"no"| D
    T -->|"yes"| G{"Guarded by an idempotency key or conditional header?"}
    G -->|"yes"| Y["Retry once"]
    G -->|"no"| Q["Ask the user to try again with the refreshed token"]
```

**UI-R31.** An authoritative `401` from a domain API request clears the SPA's
in-memory CSRF token, authenticated session and protected query cache,
returning the user to sign-in (LIFE-UI-1). This applies to both typed JSON and
direct Artifact requests.

**UI-R32.** A `403` and failed login attempts do not invalidate a session.

**UI-R33.** A late `401` from a request started with an older CSRF token must
not invalidate a newer login.

**UI-R34.** After an unsafe request returns `403`, the SPA re-reads the
session once. If another tab replaced the session and the CSRF token changed,
it retries a request guarded by an idempotency key or conditional header once
(LIM-UI-2); other mutations ask the user to try again with the refreshed token
(FAIL-UI-1).

**UI-R35.** An unrelated origin or capability denial stays `403`, without a
retry loop or session invalidation.

**UI-R36.** This session handling does not require background session polling
that would extend idle expiry.

### Navigation and information architecture

**UI-R37.** The navigation is a rail of three groups (main, secondary,
bottom) with the destinations and routes shown in [Navigation](#navigation).

**UI-R38.** The rail is the main navigation; the user journeys in
[UI journeys](journeys.md) add no navigation items.

**UI-R39.** The Inbox at `/` puts optional Project Finding annotations first,
then blocked work, finished results and running checks. Its rail badge counts
canonical unassessed Project Findings through the owner-scoped Finding list.
Annotations do not block execution or completion. There are no item approval or
report acceptance requests (ACT-R9).

After successful initial reads, desktop entry selects the first listed item
once. Explicit item links, partial reads and returning to the queue keep the
user's selection; mobile entry starts with the queue. Unselected content is a
compact continuation, loading, partial-read or empty state rather than duplicate
section counters. Selected Findings reuse the complete Issues reader and
Project annotation owner; selected reports reuse the complete Reports reader,
status, safe Markdown and downloads. A completed check without an aggregate
report links to its ordinary task results. Inbox retains its queue navigation
and return context, without a duplicate Run/capacity summary.

**UI-R40.** An Audit page under `/projects/:projectId/audits/:auditId` marks
Checks as current in the rail.

**UI-R41.** Operations appears only with the `operations` capability;
Settings (`/operations/settings`, with the Git SSH key) does not need it.

**UI-R42.** Every existing destination stays reachable and every existing URL
loads directly.

**UI-R43.** The Catalog's canonical routes are `/catalog/audit-presets`
(Check types), `/catalog/workflows`, `/catalog/agents` and `/catalog/skills`;
`/catalog` opens Workflows. Exact Workflow details use
`/catalog/workflows/:name/:version`.

**UI-R44.** Runtime Agent processes stay in Operations, not in the Catalog.

**UI-R45.** The UI supports an extensible catalog of security checks and
supporting analyses.

**UI-R46.** Project artifact tiles may provide familiar OpenAPI, Engagement, LikeC4,
source, documentation and diff icons, but they are shortcuts over arbitrary
ArtifactRefs rather than hard-coded Server modes. The shortcut set and its
presentation-only classification are owned by
[Projects](../artifacts/projects.md) (PRJ-R13, PRJ-R15, PRJ-R16).

**UI-R47.** Project Workflows offer Recommended and All Workflow launch
surfaces; their contents follow
[Projects](../artifacts/projects.md) (PRJ-R45, PRJ-R47, PRJ-R49).

**UI-R48.** Evals renders evaluation Projects and eval-specific label
grouping. The [experiment journey](journeys.md) (UIJ-R100–103) makes
experiments the Evals entry point. Current evaluation workspaces have one
management surface at `/evals/workspaces` and `/evals/workspaces/:projectId`,
reachable from experiments, setup and datasets. Creating a workspace is named
separately from creating an experiment; both workspace entry points reuse the
same creation form. Ordinary Run and Audit execution and recorded materials
remain inspectable, with shared Project editing and confirmed deletion.

Browser Evals setup, launch and comparison, their chart projections and
evidence navigation, and the support for both Workflow and Audit experiments
without Playground are owned by [Managed Evals](../evals/managed-evals.md).

**UI-R49.** Skills is a global owner-level view over `skills/*` UserScope
artifacts and the existing SkillCatalog rules ([Skills](../tools/skills.md)).
A Project may show the Skills required by a candidate Workflow (PRJ-R21) only
read-only; it cannot own, copy or override them.

### Library

**UI-R50.** Agents, in the authenticated Library, lists exact published
AgentTemplate versions, with description and links to a read-only detail at
`/catalog/agents/:name/:version` that exposes the loaded base instruction
text.

**UI-R51.** Agent detail emphasizes the base instructions with Markdown
Preview, literal Source and Copy, alongside Skills, selected tools, Runtime,
sandbox and ModelPolicy.

**UI-R52.** Agent detail is read-only: it does not expose invocation context
or introduce a prompt editor.

**UI-R53.** The displayed instruction text is not an effective invocation
prompt: task context, conversation and expanded Skills are not included.

**UI-R54.** Markdown Preview executes no HTML and loads no remote images.

**UI-R202.** Markdown Preview renders a large body in sections, the first at
once and each further one in a later browser task, and the result equals a
whole-body render. A section starts only before an unindented line that
follows a blank line and does not start a list item, outside top-level
fenced code and HTML blocks; a body with a link reference or footnote
definition, or with a fence or HTML block whose end cannot be told line by
line, renders as one section. A section longer than LIM-UI-16 is shown as
text with a notice instead of being parsed; Source shows the whole body.

**UI-R213.** OpenAPI Preview renders a document of at most LIM-UI-17 at once.
A larger document that declares OpenAPI opens on Source and is neither parsed
nor rendered until the user asks for it; the Rendered view names the limit
and warns that rendering can stop the page from responding for a few
seconds. A document that turns out not to be OpenAPI says so and stays in
Source.

Rationale: the renderer builds every operation of an untagged document at
once, and none of its settings bounds that work: at 1 MiB it held the page
for over a second, and the YAML parse alone for about 0.3 s. Reopen the limit
when the renderer can render operations on demand.

**UI-R236.** An exact material with media type
`application/vnd.contractor.project-sitemap+json` uses a read-only sitemap
Preview beside Source, within the common LIM-UI-12 byte ceiling. The generated
public `SitemapDocument` schema owns its shape and versions; malformed content,
repeated JSON members and ambiguous selection identities show an invalid or
unsupported notice, never a valid empty map. Source and the existing exact
download remain available.

The browser presents saved origins, routes and methods with search and filters,
then concrete members, recorded outcomes/statuses/parameter names/timestamps,
declarations, provenance, notes, tags and available exact evidence references.
Saved suppression and omission facts stay visible. Observations and declarations
remain distinct and establish no current reachability, authentication identity,
physical request count, complete coverage or security verdict (SMT-R14).
No referenced bytes, source files or target URLs are fetched automatically;
existing owner-authorized readers handle an explicitly followed evidence link.

Local navigation and larger detail collections mount at most LIM-UI-22 rows
per page over the already-loaded bounded document; search reaches all loaded
routes. Domain grouping and identities remain unchanged. Selection follows
UI-R231; phone layouts offer a Browse routes disclosure and use the same
typography, theme tokens and shared controls as other material previews.

Rationale: this is a presentation over ordinary exact Artifact bytes, with no
new sitemap read API, mutable editor, retention hold or background collection.
Larger documents retain the ordinary download path. Reopen the byte ceiling
only with a separately bounded reader requirement.

**UI-R237.** Add material includes an Engagement shortcut (PRJ-R15), opening the
ordinary Artifact upload form with a shared icon, `engagement` namespace and
`application/vnd.contractor.engagement+json` media type. Suggestions remain
editable and the existing upload, cancellation, validation and exact-result
navigation own publication; the shortcut creates no implicit policy or check.

That exact media type has a read-only Preview beside Source in the common
Material reader. The existing shared `engagement.schema.json` owns the closed
document shape, enums and collection bounds. Malformed JSON, repeated members,
unsupported versions and structurally invalid content show an invalid/unsupported
notice while retaining Source and download. This display does not replace
Server-owned canonical network, identity cross-field or credential/admission
validation (AUD-R241–243).

Preview distinguishes allowed egress from exclusions and shows saved methods,
path prefixes and address ranges. Identities show their actual anonymous, Basic,
Bearer or cookies kind, origins, declared role/tenant and non-secret credential
references. No credential values or target URLs are fetched. Optional HTTP
limits are shown individually as configured or Not set; no default is invented
and Run/Audit budgets remain separate (AUD-R242).

Route navigation and nested lists mount at most LIM-UI-23 rows per page, with
search over the bounded loaded document, stable keyboard selection and a phone
Browse disclosure. Shared typography, theme tokens and 390px layouts follow the
other Material readers. The existing LIM-UI-12 ceiling and exact Artifact
authorization, revisions, download and retention remain unchanged.

**UI-R55.** Workflow logical Worker bindings link to the exact AgentTemplate
version.

**UI-R56.** Agent discovery and exact Agent usage keep predecessor cursors for
the same route and filters in page navigation state. Item links and explicit
return links preserve that state, so Previous opens the preceding result page
independently of browser history.

**UI-R57.** Browser Back and Forward restore each visited page's cursor
context.

**UI-R187.** Changing the search or the Agent version resets pagination.

**UI-R58.** A directly opened later page with no known predecessor disables
Previous and offers First page for the same query.

**UI-R59.** Workflow family discovery shows the complete inventory
(UI-R62); its detail links preserve search and return context.

**UI-R60.** Catalog and Audit read projections filter before pagination.

#### Workflow discovery cards

**UI-R61.** Project and Catalog use the same Workflow identity, version
selector and card structure. Each logical name appears once (LIM-UI-3).

**UI-R62.** The Catalog reads all bounded pages before offering a default, so
a later page cannot hide a newer version.

**UI-R63.** Numeric versions, including dot-separated integer components, sort
by components without floating-point conversion; the highest is initially
selected and marked Latest. Other published labels use deterministic natural
ordering without a Latest claim.

**UI-R64.** An explicit older selection survives background refresh.

**UI-R65.** Changing a version changes its displayed inputs, outputs, matching
and exact-version actions together.

**UI-R188.** An already open Run setup remains pinned to its selected version.

**UI-R66.** Catalog cards display input and output contracts, primary-output
flags, formats and parameter counts.

**UI-R182.** Project cards display only the selected version's artifact
matches, ambiguous choices, missing requirements and primary outputs.

**UI-R183.** Matching media types is not proof of semantic suitability or
Runtime capacity.

**UI-R184.** Existing outputs suppress recommendations, but the suppressed
Workflows remain available through **All workflows**.

**UI-R185.** The UI never silently selects an older compatible version.

The Workflow recommendation model and the suppression rule for Workflows whose
primary outputs exist are defined in [Projects](../artifacts/projects.md)
(PRJ-R49).

**UI-R67.** Workflow quick starts use the complete Workflow inventory and the
same numeric version policy as Project and Catalog cards.

**UI-R186.** Project Workflow matching loads every Artifact page before
considering a unique MIME match; incomplete or cyclic pagination displays an
error with explicit retry and supplies no partial match (FAIL-PRJ-3).

#### Workflow detail

**UI-R68.** An exact Workflow detail route starts with an overview: one
identity heading, the version selector, input and output contracts and the
published stage objectives. It uses the Catalog card formats and version
ordering.

**UI-R69.** Loading the version inventory never replaces the version
specified by the URL. Latest and Earlier labels require a complete numeric
family containing that exact version.

**UI-R70.** An inventory failure leaves the exact contract available with an
explicit retry (FAIL-UI-3).

**UI-R71.** Authored descriptions and primary-output flags are shown only
when published.

**UI-R72.** Entry is identified independently of list order; multi-stage
summaries do not imply a linear execution graph.

**UI-R73.** Exact Agents, resolved execution settings and transitions remain
accessible in a disclosure.

### Run setup

The capabilities the Workflow user surface offers (Workflow and version
listing, parameter and input rendering, uploads, Runtime
label selection through `runtimeLabels` such as `debug` or `caido`, metadata
labels, creating, cancelling, continuing and
repeating Runs, Stage and Planner progress, artifact previews and bounded
metrics) are listed in [UI journeys](journeys.md).

**UI-R74.** Configure Run opens a standalone UserScope drawer on desktop and a
full-width panel on mobile.

**UI-R75.** The Workflow detail view itself does not acquire a Run draft or
fetch input inventories until setup opens.

**UI-R76.** Before execution settings, the Run form presents declared
parameters and exact Artifact inputs: Artifact inputs and required parameters
precede collapsed optional parameters and advanced settings.

**UI-R77.** The drawer reuses the shared modal stack, inert background,
keyboard focus containment and focus restoration; closed disclosure contents
are excluded from the focus loop. Nested upload, Git import and discard dialogs
close independently.

**UI-R78.** The `#workflow-run-setup` destination opens the drawer directly,
including for Run-repeat drafts.

**UI-R79.** Closing setup preserves the session-memory draft for its exact
Workflow version and scope (LIFE-UI-2). Included optional values reopen
visibly, including an explicit empty string.

**UI-R80.** The drawer leaves required input review, repeat review,
ambiguous-response recovery and exact-request idempotency unchanged
(FAIL-UI-4).

**UI-R81.** Close and Escape cannot dismiss setup while its submission is
pending.

**UI-R82.** Automatically prepopulated Artifact candidates remain visibly
marked as format-only suggestions until the user confirms their exact revision
for that one slot. Explicit picker selection and a completed upload count as
that slot's confirmation.

**UI-R83.** Readiness separately reports missing fields, pending input review
and a request ready for submission.

**UI-R84.** The retained in-memory draft stores the slot-to-exact-ref
confirmation and enough bounded metadata to keep a confirmed revision
inspectable across ordinary inventory refetches; it stores neither upload bytes
nor secrets (LIFE-UI-2).

**UI-R181.** A media-type recommendation does not establish semantic input
suitability: the retained Run draft requires explicit per-slot review of each
populated exact suggestion without changing Server media-type validation.

The media-type recommendation itself is defined in
[Projects](../artifacts/projects.md).

#### Repeating and continuing Runs

**UI-R85.** A repeat Run uses authorized exact source refs rather than
treating RunScope refs as UserScope or ProjectScope inputs.

**UI-R86.** A draft for another Run opened from a terminal ordinary Run seeds
a separate in-memory draft without replacing a mounted or edited draft for the
same Workflow and scope (LIFE-UI-2).

**UI-R87.** All retained values of a repeat draft require explicit review,
and every exact Artifact selection still requires the ordinary per-slot
confirmation.

**UI-R88.** Submission of a repeat draft uses the normal create endpoint and
a fresh draft idempotency identity; lost-response replay of the unchanged new
request uses that same identity (FAIL-UI-4).

**UI-R89.** A configured Scheduler retry is shown as an attempt within the
existing Run, while a terminal failure has no implied retry.

**UI-R216.** Run detail leads with a Workflow-name heading, copyable exact
Run/Workflow identity, Created and available Finished timestamps and duration.
Started and Updated timestamps and a Timestamps disclosure do not appear in
the header. Successful Runs proceed directly to outputs: their header status
is sufficient, without a repeated success summary, next-step guidance or result
shortcuts. Other states show authoritative outcome and guidance in a responsive
surface before outputs and Technical details; causes, recovery controls and
focused-attempt links preserve their behavior. Declared primary outputs receive
visual emphasis for display order only. Preview remains explicit; opening the
page fetches no output payloads.

The heading omits the version already shown in Workflow. Run/Workflow copy and
output download controls show only their icon, retaining accessible names and
hover tooltips. The exact artifact-details link sits below the preview/download
controls on a separate line at every width.

**UI-R90.** Known placement and Gateway states link only to diagnostics the
current user is authorized to read; unknown causes recommend inspecting the
durable attempt without inventing a capacity or recovery claim.

The **Continue from failed stage** action, its eligibility from
`RunStatus.resumeStageExecutionId`, its confirmation, the resume request
(`POST /v1/runs/{runId}/resume` with the expected `stageExecutionId`), its
idempotency identity and response, the post-request refresh of Run, Queue and
Project queries, and the execution and cleanup rules are specified in
[Runs](../execution/runs.md) (RUN-R55, RUN-R56, RUN-R57).

#### Labels on the Run form

That Run-create, list and detail expose the separate immutable `labels` map,
that its key/value editor and exact filters never share controls with
RuntimeConfig selection, and that conventional `eval.*` shortcuts are
presentation over the generic label query without an eval-only Server
execution path are specified in [Runs](../execution/runs.md) (RUN-R74,
RUN-R76).

**UI-R203.** The Audit start form offers ordinary `labels` as editable
key/value chips with quick cloud/local additions and a visible bound, using
the shared metadata grammar and
reserved cloud/local validation. `cloud` selects any cloud, `cloud:aws`
selects cloud/aws, and `local:lab` selects local/lab. The UI uses this label
field for placement and has no separate location selector or fleet-availability
gate. It retains typed labels through failed reads and draft/start conflicts,
explains that start fixes the map and all child Runs inherit it, and shows the
Server-reported labels in Audit details. Run forms use their existing metadata
label editor and repeat copies those labels. RuntimeConfig controls remain
separate and are titled Runtime configuration.


Checks exposes **Web scan** as one Audit check type through
`/checks/new?type=web-scan`, with ordinary Project selection. Its bundled form
asks for a reviewed HTTP(S) URL, prefilled from the Project, and a rule source:
installed Nuclei rules or custom Nuclei YAML/ZIP. Custom rules are selected from
paged Project materials or uploaded in the form. Filters and request rate are
optional scan settings. The form authors exact bounded target and settings
artifacts and retains uploaded rules in ProjectScope using create-only,
content-addressed bindings. Ambiguous writes reconcile the exact retained
content before reuse. It then uses the ordinary Audit create/start keys,
revision fence, draft recovery and check detail/result lifecycle. All inputs
are fixed by the Audit baseline. No source-code or OpenAPI upload is needed.
The Checks page has one primary **Start a check** action; it has no separate
Web scan Run launcher or cross-Project sweep control.

**UI-R95.** The Run-create UI presents Runtime labels separately, as
infrastructure configuration.

**UI-R96.** Metadata labels use editable badges: `key:value` creates a pair,
and `key` creates a label with an empty string value. The UI does not offer
them as RuntimeConfig checkboxes.

**UI-R97.** Run list and detail show metadata labels independently from
pinned Runtime configuration provenance.

**UI-R98.** The label filter supports exact conjunctions and may provide
shortcuts for the conventional `purpose=eval`, `eval.name`, `eval.id` and
`eval.leg` keys without changing the generic API.

**UI-R100.** The UI must explain that metadata labels are immutable and
telemetry-visible.

**UI-R101.** The UI must not infer permissions, execution success or A/B
comparability from a label.

How the reserved default Runtime label binding, the default, Run and Agent
layers and the selected Agent labels are shown on the Run page, that a label
without a binding cannot be selected, and that the owner Run surface shows
labels and exact refs and digests rather than resolved credentials or physical
Runtime Agent configuration are owned by
[RuntimeConfig](../configuration/runtime-config.md).

### Runs surface

**UI-R102.** The navigation rail has one **Runs** destination, in its secondary
group, with two views: Queue (default) and Completed, holding the Run states
shown in [Runs surface](#runs-surface).

**UI-R103.** Queue renders the owner-scoped `/v1/queue` read projection of
[Projects](../artifacts/projects.md) (PRJ-R89 to PRJ-R94), including its
optional Project display context.

**UI-R104.** Completed uses the ordinary newest-first `/v1/runs` collection
with an additive `lifecycle=terminal` filter. The Server applies the filter
before keyset pagination; filtering one browser page is incorrect.

**UI-R105.** Exact `state` and metadata label filters may be combined with the
lifecycle filter by intersection, and all filters are bound into the cursor
kind.

**UI-R106.** The Queue and Completed views need not use the same endpoint.

Rationale: their ordering and display projections are different even though
both read `workflow_runs`.

**UI-R107.** The Queue view opens at `/runs`; Queue is not a separate rail
item.

### Projects

**UI-R108.** Project navigation leads to Overview, Artifacts, Workflows, Runs,
Audits, Findings and Settings at independent URLs under `/projects/:projectId`.

**UI-R109.** Add artifact is a contextual Artifacts action; Workflow cards open
exact Run setup.

**UI-R110.** The Project header exposes metadata refresh and the existing
confirmed deletion action directly
([Retention and deletion](../artifacts/retention-and-deletion.md)).

**UI-R189.** The Project header carries no duplicate Audits or Run analysis
shortcuts.

**UI-R111.** Cards are content-sized and do not stretch merely to match a
larger neighbouring contract.

**UI-R112.** The Project root is a bounded Overview rather than a concatenation
of every workspace collection. It reads:

- the Project's newest page of Audits, up to 50 (LIM-UI-4), the read the
  cross-project lists share;
- the first page of proposed findings of each of those Audits with the Server's
  total, keyed by the Audit revision, so an unchanged Audit is not read again
  and a changed one stays counted while its new revision loads;
- three Audits waiting for review (LIM-UI-5);
- five recent Runs (LIM-UI-6);
- three successful Runs (LIM-UI-7);
- a sample of current Artifacts: five recent ones and up to three each in
  `sources` and `openapi` (LIM-UI-8);
- the published Audit profiles, for check types whose input formats match.

**UI-R113.** Counts over Audits say when they cover only the 50 newest
(LIM-UI-4); the Run and Artifact samples are not total counts.

**UI-R114.** The Projects list counts each Project's possible issues to review
from the same per-Audit reads, so the list and the Overview agree.

**UI-R115.** Recent results resolve exact successful Run outputs and published
Workflow output roles; missing or unavailable primary outputs are not replaced
with guessed results.

**UI-R217.** Project Overview uses the reviewed canvas/card contrast: compact
composer, small section labels outside individual result/history surfaces,
review attention rows, and a row-based At a glance card before Materials in
the supporting column. Timeline is a collapsed disclosure by default; its
native checkboxes select any combination of four event kinds and retain the
bounded selection in the URL, with explicit All/none behavior. Expanding it
keeps the shared connector, status icons and time column. Filtered direct URLs
open the disclosure. Existing per-check reads supply at most three possible-issue
attention links, without a new read or a guessed aggregate count.
Project Overview leads with the objective composer, then decisions
waiting for review, recent results, recent Runs and the existing Timeline with
its connectors, status icons and filters. Materials and qualified summary counts
form supporting context. The composer uses one input/action surface and opens
ordinary check setup, preserving the objective, format-match suggestions and
keyboard submission; it does not start execution. Result cards lead with the
actual output name, known role and declared format hints. Preview and icon-only
download sit at the top right; the exact output reference remains visible and
`details →` follows the divider and producing Run identity/time on its own line.
Metadata and bounded content previews load only after explicit action, through
the same exact-revision mechanism as Run results. Recent Runs retain Server
order, a copyable full identity behind the shortened ID, reported state and
accessible full timestamps. Empty, partial and failed reads retain their truthful
outcomes. This hierarchy works at 320/390 px and in light, dark and black themes.

**UI-R190.** Result links retain their RunScope and exact revision.

**UI-R116.** The Overview does not load the complete input or Workflow
inventories.

**UI-R117.** Artifacts, Workflows, Runs, Audits, Findings and Settings mount
only when selected. They share the Project header, actions and section
navigation. All seven destinations remain on one horizontally scrollable row,
with the selected destination revealed on section changes and resize. Section
actions remain outside that scroll area.

**UI-R118.** Project Settings contains metadata and application access.

**UI-R119.** Adding an Artifact opens a chooser with the existing upload and
Git import flows.

**UI-R120.** Project Workflow setup uses the shared drawer in ProjectScope,
retains its ProjectScope form, the selected exact version and exact version
semantics, and keeps its draft across close and section navigation in the
current tab (LIFE-UI-2). A Workflow input declaring `sourceScope: user`, such
as the optional Nuclei rulepack, is selected, uploaded and previewed in the
owner's Library even when the Run belongs to a Project (SCN-R238).
Each input uses its own source scope for choices, pagination, uploads, previews
and retained exact metadata. A Workflow may combine ordinary Project inputs
with the owner rulepack; metadata for equal ArtifactRef keys in different
scopes must not replace one another.

**UI-R121.** Project Runs use pages of 25 (LIM-UI-9) and Server-backed
lifecycle and state filters. Workflow, state, Audit origin and update time
remain visible; full identity and metadata labels are disclosed on demand.

**UI-R122.** No page-local text filter is presented as a complete history
search.

**UI-R123.** Section URLs preserve filters and cursor stacks through reload
and detail return. Navigation between sections retains each section's query in
browser history state; filters and explicit Workflow versions remain in that
section's URL.

**UI-R124.** Links to Project sections use their explicit paths.

**UI-R125.** Eval workspaces retain their existing layout and scope, including
their in-page anchors.

**UI-R126.** The static server accepts direct requests for every section URL.

**UI-R127.** Run and Artifact drill-down links retain the originating URL and
nested return context.

**UI-R128.** Global and Project Artifact namespace and cursor filters are
URL-backed; Artifact revision selection retains the origin.

**UI-R129.** Direct detail links continue to have a deterministic parent
destination.

### Artifacts and uploads

**UI-R130.** The global **Artifacts** route is the owner's general UserScope
library. Its paginated list requests `excludeNamespace=skills`; exclusion is
evaluated by the Server before keyset pagination, so a page cannot become short
or incorrectly claim completion after browser-side filtering. An exact
`namespace` filter and the exclusion are part of the cursor identity.

The Artifact list API and its filter parameters are defined in
[ArtifactStore](../artifacts/artifact-store.md).

**UI-R131.** The **Skills** route requests exactly `namespace=skills` and
remains the single UI surface for creating and listing Skill packages. The
generic Artifacts upload form directs an attempted `skills` namespace to that
route instead of duplicating it.

**UI-R132.** UserScope and ProjectScope uploads share one accessible file-drop
component. Drop and file-picker input select exactly one local file
(LIM-UI-10), show its name and size, enforce the 64 MiB payload limit
(LIM-ART-4) before mutation, derive a safe editable name from the file stem and
infer known media types by browser value or extension.

**UI-R133.** Scope, CAS and mutation endpoints remain owned by the forms that
contain the file-drop component; sharing the drop interaction grants no
cross-scope authority.

What the browser may do with an Agent Skill ZIP
(`application/vnd.contractor.agent-skill+zip`) and what it must not do,
authoritative Skill package validation when a referencing Run initializes and
again in Runtime, and the bundled `configs/skills` sources that only create
missing owner artifacts at startup are specified in [Skills](../tools/skills.md)
(SKL-R63).

**UI-R136.** The personal **Repository access** editor lives directly at
`/operations/settings`, linked from the account menu. It remains available
without the Operations capability; in that case the page omits the server-wide
Scheduler settings and every other Operations section.

**UI-R137.** There is no separate `/settings` route and no compatibility
redirect.

**UI-R138.** Closing the Git-import dialog restores focus.

The shared Git-import dialog, the one owner-scoped encrypted SSH key, exact
revision review before replacement, request cancellation on close without
closing the parent Run form, assignment
of the imported ref to the requested input only and the display of Git
provenance are owned by [Git import](../artifacts/git-import.md) (GIT-R1,
GIT-R3, GIT-R6, GIT-R10, GIT-R11, GIT-R12, GIT-R37, GIT-R88).

### Selected HTTP evidence

**UI-R192.** The material preview recognizes only the current selected HTTP
exchange schema and media type owned by [Findings](../findings/findings.md)
(FND-R49, FND-R160 to FND-R162). It validates the closed structure, required
request, optional response, source identities and byte bounds before rendering.
Malformed or unsupported content gets a bounded error and keeps Download.
No obsolete body-only or Caido artifact format is interpreted as this exchange.

**UI-R193.** One selected exchange shows method and URL, ordered headers and
body for each attempt, and separate Request and Response views. Empty body,
absent response and incomplete response are distinct visible outcomes; 4xx/5xx
is received response data. JSON may be formatted; HTML stays source text and
never executes or loads embedded resources. Binary data is labelled and can
be downloaded. The original saved exchange remains downloadable at its exact
revision, including when it is too large for preview.

**UI-R194.** Preview reads admit at most 1 MiB of text payload
(LIM-UI-12). Display of each body is at most 8192 Unicode characters
(LIM-UI-13); longer content is explicitly shortened with a download action.
At most 50 structured headers are displayed at once (LIM-UI-14), with all
header pages reachable. Raw Caido header previews are bounded and the raw
message download preserves all saved bytes. Switching message or attempt is
keyboard-accessible at 390 px in all three themes.

**UI-R195.** Run material browsing reads metadata pages and loads a payload
only for the opened preview or download; there is no client traffic scan or
128-artifact retention assumption. The possible-issues/evidence shortcut uses
the Server namespace filter before pagination. HTTP exchanges are identified by
media type. A relationship to a possible issue is shown from its Server-owned
association or exact receipt evidence; matching names or URLs alone grant no
relationship. An Audit finding links its retained exact Project evidence when
available, otherwise its live exact source-Run evidence. Deleted evidence with
no retained copy has no fabricated link.

### Audits

**UI-R139.** The existing Checks list keeps its filters, compact selected-check
summary, progress and keyboard navigation. A check opens Overview / Work /
Hypotheses / Possible issues / Materials / Activity, with a link to its exact
report. The contextual Project check list and return navigation retain the
source list context. Work preserves the list/detail pattern over accepted Tasks;
it does not restore the removed Round/item executor.

**UI-R140.** Audit pages display technical Run outcome separately from semantic
assessment, and proposed findings separately from confirmed findings.

**UI-R215.** A selected Task separates its saved scoped result, tested conditions
and limitations from technical Run attempts. The display validates the common
result document and selects exactly one member by trusted Task identity, Audit
and exact hypothesis revision; another member's verdict never fans out to it.
Exploration may have no hypothesis or verdict. The original full batch remains
downloadable. A deleted source Run is marked as provenance while accepted exact
results remain readable. Possible issues open the canonical Project issue,
including its existing independent sources and analyst decision owner.

**UI-R141.** The generic Runs and Queue UI remains authoritative for execution
and links back to the Audit; an Audit does not create a second Runtime queue.

**UI-R142.** The check heading combines profile, Project and recorded date.
Exact identity and copy, profile digest and immutable baseline remain under
Technical details. The objective is prominent; scope remains available beside it.

**UI-R143.** `GET /v1/audits/{auditId}/workspace` returns one owner-scoped SQL
snapshot (`auditRevision`, `asOf`): current iteration, whole-Audit `totalTasks`,
`completedTasks`, `gaps`, `pendingTasks`, and separate `findings` and
`unreviewedFindings`. It carries no Round, issue bucket or approval count.

**UI-R144.** Completed Tasks (settled without a technical gap), gaps (settled
with a gap), and pending Tasks (ready or selected) partition all accepted Tasks.
The aggregate is bounded by the Audit's Task ceiling. A completed Task can
retain inconclusive or refuted conclusions; technical progress does not imply
an analyst assessment, a vulnerability or a successful security evaluation.
These counts describe Tasks, including preparation and generation; progress
labels never call them endpoints, requirements or scenarios.

**UI-R145.** Audit Overview begins with Task progress, retained results and
optional Project Finding annotation links.

**UI-R146.** Overview progress links to common Tasks, and Finding counts to
the corresponding canonical Finding filters. Task and Finding totals are not
added together.

**UI-R147.** Unavailable counts remain unknown; refresh errors identify
retained snapshot counts (FAIL-UI-6).

**UI-R148.** Execution state, coverage assessments and optional current Project
Finding annotations remain separate.

**UI-R149.** Profile and scope remain expanded for drafts; immutable baseline
details are available through disclosure.

**UI-R150.** Audit Overview adds no Evals experiment API or comparison
metrics.

Finding pages use bounded candidate windows, revision pins and signed continuation
cursors. A concurrent change conflicts, filter changes clear incompatible cursors,
and refresh starts a new context. The fixed checklist denominator and pending
fixed admission count come from the hypothesis owner. Generated/manual assertions and additional checks cannot change
that denominator. Loaded Task or hypothesis pages do not imply a whole-check
semantic coverage numerator; summaries label their page scope.

**UI-R153.** All activity reads 50 events at a time (LIM-UI-11), exposes Load
older activity, keeps settled events visible on continuation errors and
refreshes the complete loaded cursor chain while the Audit is nonterminal or
after a mutation or reconnect (FAIL-UI-6).

**UI-R233.** Hypotheses show authored grounds and retained exact sources separately
from system origin, with explicit fixed/generated/manual labels. Manual editing
and checking reuse the existing revision precondition and mutation receipts;
conflicts preserve unsaved text and require explicit reload. Save alone does not
schedule checking work. History reads one indexed hypothesis page at a time and
keeps the exact checked snapshot, including older revisions and negative results.

**UI-R234.** Materials use accepted Audit receipt pages and exact revision reads,
shared bounded format viewers, icon-only download and a concise `details →` below
the divider. Closing a preview cancels its read. A ground outside the loaded
receipt page remains explicitly unlocated, with its exact descriptor retained;
no latest revision or source-Run fallback is invented. Explicit Project publication
is create-only and retains its original body/key across uncertain acknowledgement.
A definitive refusal permits correction without overwriting another material.

**UI-R235.** Add work offers only aliases from the exact matching Program definition
and retained material selectors with explicit page continuation. A changed/missing
catalog definition is unavailable rather than substituted for the stored pin.
Checking hypotheses uses Server's pinned checking aliases independently. Work
acceptance retains its body, key and Audit precondition across ambiguous failure.
Technical progress renders a bounded number of weighted segments, not one DOM node
per Task. The common font/theme tokens apply at 390 px and in all three themes.

**UI-R154.** The current-state sentence is separate from recorded history.
Per-item activity summarizes its retained attempts and current results, without
claiming an immutable per-item ledger.

**UI-R155.** The UI uses bounded polling of authoritative Audit and profile
reads while an Audit is nonterminal or deleting, keyed by Audit revision or
ETag, and stops when the route is inactive or the Audit is terminal.
Paused Audits keep polling while `outstandingRunCount` is positive or
`collectionPending` is true. After both clear, dependent projections receive
one final fresh read before polling stops.

**UI-R156.** The UI refetches Audit reads immediately after a mutation and
after browser reconnect.

**UI-R157.** The existing Run event socket is not treated as an Audit
invalidation contract.

### Operations and configuration

The Operations Overview, Runtime Agent cards and their diagnostics disclosure,
the Edit labels dialog, offline identity removal, the `Workflow scheduling`
card of Settings, the performance page and completed allocation history are
owned by [Operations](../operations/operations.md).

Effect copy for new Run snapshots versus future allocations, publishing a
version without changing a binding, the display of the reserved default
binding, replacing a Runtime Agent's complete label set, the absence of a
disable toggle and the non-disclosure of secret payloads are owned by
[RuntimeConfig](../configuration/runtime-config.md).

**UI-R158.** The configuration workspace lists RuntimeConfig versions and
credentials. Publication and credential creation open separate accessible
dialogs through named actions; a version's binding action opens a dialog with
that exact RuntimeConfig preselected. Existing bindings are indicated on the
version row.

Open question: the navigation location and route of the configuration
workspace are not decided. The Runs surface has only the Queue and Completed
views (UI-R102), and the navigation lists LLM configurations and Credentials
under Operations ([Navigation](#navigation)); the client route table gains the
workspace's route once its location is decided.

**UI-R159.** Publishing a version and binding it remain separate actions.

**UI-R160.** Binding and Agent-label dialogs show the current exact version and
digest beside the proposal. A binding dialog lists current bindings
independently of the version used to open it.

**UI-R161.** Binding and Agent-label mutations use independent idempotency keys
plus revision preconditions.

**UI-R162.** A stale revision blocks another write until the authoritative
binding is explicitly reloaded. Reload preserves the proposal for review and
retry, even when another operator moved the binding to a different version,
and never silently submits it (FAIL-UI-7, LIFE-UI-4).

**UI-R163.** Publication errors likewise retain the unpublished proposal
(FAIL-UI-7, LIFE-UI-4).

**UI-R164.** Closing the credential dialog clears its unsaved secret fields.
Secret fields are also cleared on credential submission and kind change,
including rejected submissions (LIFE-UI-3).

**UI-R165.** A committed change of the Scheduler settings emits a typed
Operations invalidation with resource `schedulerSettings` and no resource ID.
The event remains a process-local hint; the UI refetches the authoritative
singleton.

**UI-R166.** A revision conflict on saving the Scheduler settings refetches and
shows the newly saved value instead of silently overwriting another operator's
change (FAIL-UI-5).

### Language

**UI-R167.** UI labels must preserve the execution vocabulary: a Runtime Agent
is the long-running single-slot process, while Worker is its temporary
allocation-scoped role. The UI must not present them as two independently
deployed services.

**UI-R168.** The UI calls the Catalog **Library** and AuditProfiles **Check
types**.

**UI-R169.** Missing authored presentation is not replaced with an invented
purpose or a placeholder paragraph.

**UI-R170.** Audit-specific pause wording remains exactly “Pause new Audit
Runs,” not “Pause execution.”

The Audit pause semantics are specified in
[Audit execution](../audits/execution.md) and the owner Queue Pause gate in
[Scheduler](../execution/scheduler.md).

### Accessibility

**UI-R171.** Section navigation shares control sizing and active and focus
states across Catalog, Runs, Operations, Projects and Audits.

**UI-R191.** Local Project and Audit navigation remains sticky.

**UI-R172.** Workspace overflow menus support Escape and outside dismissal.

**UI-R173.** Run removal, Project HTTP target configuration and Skill upload
use the shared modal focus containment, inert background and return-focus
behaviour.

**UI-R174.** A keyboard skip link targets the main content.

**UI-R175.** Icon actions retain accessible names and tooltips, and keyboard
focus follows the shared Dialog contract.

**UI-R176.** Preview tabs support arrows, Home and End, and a single tab stop
for the selected tab.

### Secrets and browser state

**UI-R177.** Public read APIs never return raw key material.

**UI-R178.** LiteLLM virtual keys and its master key never appear in durable
execution state, metrics, logs or browser persistence; only generated virtual
keys are encrypted in the LLM credential store.

**UI-R179.** Adapter-specific runtime credentials may be submitted once through
the authenticated write-only Operations boundary defined by
[RuntimeConfig](../configuration/runtime-config.md); they are then likewise
encrypted and never readable.

**UI-R180.** Browser passwords, session cookies and CSRF tokens never reach
Node or durable execution state or storage; the Go Server derives the sole
local principal from its authenticated in-memory session.

## Interfaces

The public API schema is
[contractor-public-v1.yaml](../../../api/openapi/contractor-public-v1.yaml);
public API conventions are in [Public API](../api/public-api.md).

| Interface | Use | Requirements |
| --- | --- | --- |
| `/runtime-config.json` (Node) | UI version, supported API versions, one absolute `apiBaseUrl` | UI-R8, UI-R9, UI-R10 |
| `/v1/queue` | Queue view | UI-R103 |
| `/v1/runs` with `lifecycle=terminal` | Completed view | UI-R104, UI-R105 |
| `GET /v1/audits/{auditId}/workspace` | Audit workspace snapshot | UI-R143, UI-R144 |
| `POST /v1/runs/{runId}/resume` | Continue from failed stage | [Runs](../execution/runs.md) RUN-R55 to RUN-R57 |

Client routes owned here:

| Route | Destination | Requirements |
| --- | --- | --- |
| `/` | Inbox | UI-R39 |
| `/projects`, `/projects/:projectId` | Projects and Project sections | UI-R108 |
| `/projects/:projectId/audits/:auditId` | Audit page | UI-R40, UI-R139 |
| `/checks`, `/issues`, `/reports` | Audits, findings and Audit reports across Projects | UI-R37 |
| `/runs` | Runs surface, Queue view | UI-R107 |
| `/catalog`, `/catalog/audit-presets`, `/catalog/workflows`, `/catalog/agents`, `/catalog/skills` | Library | UI-R43 |
| `/catalog/workflows/:name/:version` | Exact Workflow detail | UI-R43, UI-R68 |
| `/catalog/agents/:name/:version` | Read-only Agent detail | UI-R50 |
| `/artifacts` | Owner UserScope Artifact library | UI-R130 |
| `/evals` | Evals | UI-R48 |
| `/operations`, `/operations/settings` | Operations; Settings with Repository access | UI-R41, UI-R136 |
| `#workflow-run-setup` | Run setup drawer | UI-R78 |

### Retained duplicate observations

**UI-R198.** Existing finding lists and full summaries display Server-owned
automatic correlation separately from the human state, verdict and severity.
Every member with peers uses the same **Automatic match** label; the internal
representative never changes the visible priority. The existing finding card
contains a collapsed **Duplicates** disclosure;
there is no separate Audit comparison destination. Each row opens its own
Audit-owned finding and exact retained Evidence tab (`?tab=evidence`). Source/scope
controls stay under Search options. The presentation representative
has no priority or correctness meaning and is not shown as a winning finding.

**UI-R199.** Opening Duplicates reads bounded Project member pages on demand.
Every continuation remains reachable; displayed observation counts are partial
until traversal ends and never stand for unique or confirmed issue counts.
Repeated import of the same first receipt is distinguished from a different
source proposal without claiming independence from a different receipt alone.
Per-observation check context exposes profile, date and baseline/configuration
on demand and explicitly states that coverage has not been compared.

**UI-R200.** A separate **Possible matches** disclosure uses Server candidate
reasons and never promotes a weak site/subject/family match to a duplicate.
Original and contributing source proposals are selectable through all retained
provenance pages. Empty work-bound pages with a continuation are not the end
of search. Unavailable keys and truncated reasons are explained; absent matches
never imply fixed/new vulnerabilities or a clean check.

**UI-R201.** Switching findings/source proposals/scope and closing disclosures
abort dependent reads and discard their bound cursors. Invalid or stale
correlation cursors stop traversal and require a restart from the current head,
with old rows removed before the new snapshot is displayed. Retryable reads
use finite exponential backoff and Server Retry-After; no GET emits a duplicate
or analyst mutation. Machine keys and parser/configuration versions remain in
Technical details. These live projections do not rewrite frozen reports.

## Limits, failures and errors

Limits ([registry](../platform/limits.md)):

- LIM-UI-1 — runtime config: exactly one absolute `apiBaseUrl` (UI-R9).
- LIM-UI-2 — after a `403` on an unsafe request: one session re-read and one
  retry of a guarded request (UI-R34).
- LIM-UI-3 — Workflow discovery: one card per logical Workflow name
  (UI-R61).
- LIM-UI-4 — Project Overview and cross-project lists: newest page of up to 50
  Audits (UI-R112, UI-R113).
- LIM-UI-5 — Project Overview: three Audits waiting for review (UI-R112).
- LIM-UI-6 — Project Overview: five recent Runs (UI-R112).
- LIM-UI-7 — Project Overview: three successful Runs (UI-R112).
- LIM-UI-8 — Project Overview Artifact sample: five recent, up to three each
  in `sources` and `openapi` (UI-R112).
- LIM-UI-9 — Project Runs page: 25 Runs (UI-R121).
- LIM-UI-10 — file-drop component: exactly one local file per selection
  (UI-R132).
- LIM-UI-11 — Audit activity page: 50 events (UI-R153).

The upload component enforces the generic 64 MiB Artifact payload limit
LIM-ART-4 ([ArtifactStore](../artifacts/artifact-store.md)). The Scheduler
settings range and the performance polling interval are owned by
[Operations](../operations/operations.md).

Failure outcomes ([failure model](../platform/failure-model.md)):

- FAIL-UI-1 — domain API request answered with `401` or `403` (UI-R31 to
  UI-R35).
- FAIL-UI-2 — unsafe mutation through the transport adapter (UI-R21,
  UI-R27).
- FAIL-UI-3 — Workflow version inventory load (UI-R70).
- FAIL-UI-4 — Run creation from a Run draft (UI-R80, UI-R88).
- FAIL-UI-5 — Scheduler settings save (UI-R166).
- FAIL-UI-6 — Audit workspace, coverage and activity reads (UI-R147,
  UI-R153; mixed coverage revisions: REV-R172).
- FAIL-UI-7 — label binding, Agent-label and RuntimeConfig publication
  mutations (UI-R161, UI-R162, UI-R163).
- FAIL-UI-8 — WebSocket Operations notifications (UI-R29).

Project Workflow matching over the Artifact inventory (UI-R186) is registered
as FAIL-PRJ-3 by [Projects](../artifacts/projects.md): incomplete or cyclic
pagination shows an error with explicit retry and supplies no partial match.

Lifecycle ([registry](../platform/lifecycle.md)):

- LIFE-UI-1 — in-memory CSRF token, authenticated session and protected query
  cache (UI-R31).
- LIFE-UI-2 — in-memory Run draft (UI-R79, UI-R84, UI-R86, UI-R120).
- LIFE-UI-3 — unsaved secret fields of the credential dialog (UI-R164).
- LIFE-UI-4 — unsubmitted binding, Agent-label or RuntimeConfig proposal
  (UI-R162, UI-R163).

This document owns no error codes. The meaning of `401`, `403` and `409`
responses is defined in [Public API](../api/public-api.md) and
[Review and reports](../audits/review-and-reports.md); the UI's
incompatible-version error (UI-R6) has no wire code.

## Security

Security-relevant requirements: the Node boundary and the direct browser to
Server path (UI-R8 to UI-R12); the transport adapter's handling of session
cookie, CSRF token and error bodies (UI-R20, UI-R21); session invalidation and
CSRF recovery (UI-R31 to UI-R35); Markdown Preview without HTML execution or
remote images (UI-R54); diagnostics links limited to what the user may read
(UI-R90); Run drafts that never hold upload bytes or secrets (UI-R84);
upload authority (UI-R133); Operations visibility gated by the `operations`
capability (UI-R41); secret field clearing (UI-R164); and key material and
browser credentials (UI-R177 to UI-R180). Trust boundaries, the browser
session and CORS are in [Security](../platform/security.md) and
[Public API](../api/public-api.md).

## Observability

The transport adapter never logs response bodies or secrets (UI-R20).
Metadata labels are telemetry-visible, and the UI says so (UI-R100).
WebSocket Operations notifications are hints; after a gap the UI refetches the
authoritative REST snapshot (UI-R29). Shared rules are in
[observability](../platform/observability.md).

## Invariants

1. The UI can request domain operations but cannot manufacture execution state
   (UI-R2, UI-R26).
2. Run forms select authored Workflows; their execution policies contain exact
   published refs
   ([AgentTemplates](../configuration/agent-templates.md),
   [Runs](../execution/runs.md)).
3. Immutable base and escalation-variant ResolvedExecutionConfigs are
   authoritative after Run creation; no execution selection is resolved
   during an attempt. The default, Run and Agent infrastructure layers may
   complete or override only a Worker's physical connection settings under
   their separately pinned provenance
   ([AgentTemplates](../configuration/agent-templates.md),
   [RuntimeConfig](../configuration/runtime-config.md)).
4. The Planner and every logical Worker resolve their model policy, Gateway and
   optional matching credential independently
   ([AgentTemplates](../configuration/agent-templates.md)).
5. Published configuration versions and their digests never change in place
   ([Configuration catalog](../configuration/catalog.md)).
6. Public read APIs never return raw key material; LiteLLM keys never reach
   durable execution state, metrics, logs or browser persistence; runtime
   credentials are write-only (UI-R177 to UI-R179).
7. Operations reflects both observed Runtime state and authoritative Control
   Plane state without conflating them
   ([Operations](../operations/operations.md)).
8. LiteLLM, not Contractor, is authoritative for Gateway-wide model access,
   spend and rate limits; ModelPolicy call and token ceilings remain
   authoritative only for bounded execution of their selected Planner or Worker
   ([RuntimeConfig](../configuration/runtime-config.md)).
9. Existing API clients can perform the same domain operations without using
   the Web UI (UI-R13).
10. The Web UI is an independently releasable Node.js service; the Go Server
    never embeds frontend assets or requires UI availability (UI-R3,
    UI-R4).
11. Agent Skill packages use ordinary owner Artifact CAS and history; Runtime
    sees only exact RunScope forks and no owner Artifact authority
    ([Skills](../tools/skills.md)).
12. Node serves only client assets and public runtime configuration; browser
    code, not Node, directly invokes the configured Go Server API (UI-R8,
    UI-R11).
13. Browser passwords, session cookies and CSRF tokens never reach Node or
    durable execution state or storage (UI-R180).

## Non-goals

- React Router Framework Mode, SSR and pre-rendering are out of scope.
- Text search across all Project Runs is deferred pending an explicit Server
  query contract.
- Audit WebSocket subscriptions, their authorization, cursor and sequence
  replay, and resync are deferred as one complete later transport feature.

**UI-R204.** Bundled check-specific forms are selected by an explicit exact
AuditProfile version and compatible required input/media contracts. The Web
scan form owns target, Nuclei rule-source and selection controls; its host
owns ordinary Project selection and navigation, and it reuses shared Audit
create/start keys, revision fences, reconciliation and check result pages.
An incompatible contract remains on the generic material-based form. No
dynamic plugin loader or Studio editing is required.

The common Audit start model owns ordinary labels, Runtime configuration,
deadline, request identity and recovery for every bundled form. A specialized
form supplies its validated exact inputs to that model; it does not create a
second launch lifecycle. One pending guard covers input preparation through
create/start. After navigation or a change of Project, type or version,
in-flight preparation settles its current call but cannot publish further
inputs, create/start a stale check or navigate from a late response.

**UI-R205.** A custom Nuclei rule check is the custom-rule mode of Web scan,
inside one selected Project. The form explains its scope, permits one YAML
rule or ZIP rulepack, retains exact rule revisions and displays actual Audit
states and evidence through the shared check pages. File size and selected
Project material metadata enforce SCN-R239's YAML/ZIP byte limits before file
reads, hashing, retained-input writes or create/start requests. Both save and
start explain an oversized selection and remain unavailable; boundary-sized
metadata remains selectable, including materials from subsequent list pages.
The UI validates the shared `nuclei-input-cases.json` rulepack metadata cases;
the Server remains authoritative. It never interprets scan
completion as a clean security verdict. The separate CLI/API for cross-Project
scan batches retains the scanner contract; it is not a basic check type or an
alternate Web UI launch path.

**UI-R206.** Specialized scanner result and Audit input views are bundled
widgets too. Result dispatch uses the report's tool and supported schema,
independently of its originating Workflow; shared error outcomes remain one
definition. ffuf completeness and Katana discovery presentation belong to their
registered adapters. Audit Nuclei/SQLMap widgets own prepared-target labels and
coverage kind identity; hosts retain authorized evidence links and receipts.
Unknown reports keep raw preview; unsupported scanner input views remain
explicit and preserve evidence access. Generic lifecycle and format-based
OpenAPI/LikeC4 previews remain shared.

**UI-R207.** The shared User/Project/Run material Technical details disclosure
shows every ART-R130 storage timestamp with absolute time and timezone. Labels
identify storage scope creation, shared blob registration, storage version
creation and current binding change. Explain that shared dates may predate the
material or target scope, blob registration is not upload/file creation time,
and locking/unlocking can change the current binding date without changing
content, including on a historical revision. Keep ordinary Created/revision
information intact. Metadata loading, failures and selection changes must not
show another material's dates; no missing date is guessed. Use shared Facts and
TechnicalDetails, keyboard disclosure, theme tokens and a layout that fits
390 px in light, dark and black themes.

An uploaded/imported input replacement immediately revokes the prior confirmed
selection. Every Run submission path remains blocked while its exact metadata
is unresolved, including optional slots and failed reads. A successful read or
an explicit new library selection, confirmation or cancellation resolves the
intent; superseded completions cannot restore an older choice. A retained draft
keeps the new exact reference unconfirmed when a pending read is cancelled.

### Runtime adapter diagnostic disclosure

**UI-R208.** One Stage attempt's Technical details has an on-demand retained
Runtime adapter diagnostic view (OBS-R113 to OBS-R115). It displays exact
lossless decimal counters, logical attribution, flush results, local discard
reasons and available duration/stop facts. Missing, partial, unavailable and
shortened data remain visibly distinct. Every page is reachable through Load
more, a generation conflict requires refresh, navigation cancels obsolete
reads and terminal data is not polled indefinitely. Use shared components and
tokens at 390 px and in light, dark and black themes.
Each actual Run page entry marks its previously cached attempt facts stale
and refreshes active disclosures once; closed disclosures wait until opened.
Cold initial reads are not duplicated, unrelated Run invalidation and
same-page live projection remounts do not refetch, and previously loaded facts
stay visibly identified while a background refresh is pending or has failed.
Entry and state-change invalidation use bounded cache passes, with work linear
in retained attempts and cached queries rather than a cache scan per attempt.

### Retained tool-call disclosure

**UI-R209.** An attempt's Technical details has an owner-only Tool calls panel
(OBS-R116 to OBS-R119). It selects the Planner or one logical Worker, renders
exact decimal measurements and sanitized argument JSON as text, expands
arguments on demand and loads every later page explicitly. It explains that
retained records are not a complete event history and do not have recorded
wall-clock start times. Missing measurements, unavailable or pending reports,
incomplete reports, empty lists, report/field truncation and failed reads stay
distinct. A generation conflict requires refresh; failed refresh keeps prior
facts visibly stale. Unrelated Run updates and projection remounts do not
refetch loaded settled pages. Navigation cancels obsolete reads. Use shared
components and tokens at 390 px and in light, dark and black themes.
The same actual-page-entry, closed-disclosure and cached-fact refresh rule as
UI-R208 applies, including after navigation cancelled a terminal-state read.

### Check-local SARIF export

**UI-R210.** A check's Possible issues list offers an on-demand **Export
SARIF** panel. Selection is explicit, binds each finding ID to the observed
revision, can span all list pages and filters, and is bounded to the public
export request's 256 selections. Select this page selects only that visible
page and refuses a page that would exceed the bound. Navigating to another
check resets selection, option and publication state; no Project-wide combined
export is offered. Empty selection cannot publish through the UI.

**UI-R211.** **Include duplicates** is a checkbox, initially unchecked, that
sends `includeDuplicates`. The UI explains that the default emits one selected
representative per duplicate group and that the saved snapshot retains every
selected identity and independent review (FND-R199). It displays only the
Server's returned selected/result/warning counts, without predicting equality
or copying ratings. Changing selection or option starts an explicit new
publication intent; retrying an unchanged ambiguous or retryable publication
uses the original exact body and client key, with no automatic mutation retry.

**UI-R212.** A changed page or selected revision and a publication conflict
require explicit clear/refresh; refetching cannot silently replace selected
revisions. Publication controls remain disabled in flight. After publication,
**Download SARIF** reads the saved export ID; failed downloads can be repeated
without another publication. Snapshot and diagnostics downloads and the saved
time remain under Technical details. Each part is streamed under its public
byte bound and checked against its exact size, revision and SHA-256 before
saving. Missing, inaccessible, changed and corrupt downloads show an error,
never a successful file. Leaving the mounted check cancels pending transport
and prevents a late download. Use shared components and tokens at 390 px and
in light, dark and black themes.


**UI-R214.** After session recovery or sign-in, protected routes obtain
API-R113 shared configuration before rendering their content. The API client
validates its schema and canonical digest and caches one immutable snapshot
for its lifetime. Navigation does not poll or refetch a successful snapshot.
A failed read blocks protected content with an error and an explicit retry;
retryable failures use the existing bounded session-query retry policy.
Unknown versions, fields or invalid digests have no local policy fallback.
Logout retains no protected view or session query state; a new application
loads configuration again. Artifact upload controls and explanatory limits,
ordinary and Audit Artifact downloads and archive admission use the received
payload ceiling; specialized smaller budgets stay independent. Requests for
operations declaring the common public `Limit` parameter cap their preferred
size by `limits.publicPageMaxItems`; absent sizes use that ceiling. Continuation
cursors and filters are preserved. Domain-specific list limits remain separate. The initial
Node-served runtime-config document still supplies the API base URL, UI version
and supported API versions; it contains no shared policy numbers.

The CTL-R171 Git input byte ceiling controls repository import form and client
validation before mutation. Run label editors, draft validation and metadata
filters use the received label count and UTF-8 key/value bounds. Audit authoring
uses the received derived count that leaves space for child Run provenance.
Reusable pure validators receive policy explicitly; they do not consult global
mutable settings or compiled numeric defaults. Runtime placement labels retain
their separate grammar and bounds.

**UI-R218.** The common UI font tokens use the system sans-serif and monospace
stacks actually rendered by the reviewed standalone OpenDesign prototypes.
They apply globally, including Run, Project, shell, controls and code; routes
must not introduce their own font family or load a competing bundled face.
Run detail uses the same canvas and independent-card composition, with compact
header/facts, a single vertical outcome/next-step surface for non-success states,
and result actions at the top right beside their title. Shared icons represent
check type selection, forward actions, output documents, copy/download and
navigation; existing state/action eligibility and explicit exact previews remain.

**UI-R219.** A Run opened from a supported Run list retains that source list in
an adjacent shared list/detail pane. Queue/Completed filters, metadata labels,
Project context, cursor and Overview sample bounds belong to the source and
survive sibling Run selection and return. The pane has no separate view
switches and reads only the same bounded page through existing APIs/query keys,
with the source polling policy and explicit read-error/empty/absent-selection
states. Navigation carries a validated read descriptor, never a browser-storage
snapshot of business records. Direct or unsupported origins show ordinary Run
detail without an invented list. Mobile uses the existing back-to-source and
focus behavior, retaining router return state.

**UI-R220.** Available Run outputs use a concise visible `details` link while
retaining their exact Artifact destination and an accessible name identifying
the output revision. On narrow cards Preview and download form a separate row
below metadata badges; desktop actions remain beside the output title.


### Issue review workspace

**UI-R221.** The Issues list/detail workspace shares the global font/theme tokens.
The detail header keeps title, subject, human state, a link to independent verification results and the separate
AI suggestion or analyst severity above Summary/Evidence/History. File subjects
use a compact FILE badge and file icon before the path. The Summary
presents the original proposal and retained evidence in the main column, with
Impact before Verification and Limitations in the supporting column when width
allows. A recorded-decision surface uses only the actual analyst decision and
keeps its actor, recorded time and rationale readable. The shared FindingDecision
owner retains eligibility, required inputs, stale checks, receipts and focus rules;
Mark as duplicate is an explicit action. The compact composer expands after a
verdict is selected. Why sits above a full-width rationale field, with Record
decision on its own row (full width on phones) and the existing required-input
help. At phone widths the header and full composer scroll with the content; the
four review verdicts use two columns, and the redundant Next item footer action
is omitted. Previous/count/Next share one header row and View in Project (or an explicitly selected check context)
occupies the next; J/K navigation remains.

**UI-R222.** Issues offers severity, newest, oldest, state and title sorting plus
text search over loaded title/subject previews and Project. Severity sorting uses an
analyst rating when set, otherwise the separate AI suggestion, with unset last;
ties are deterministic by creation time and identity. State and analyst severity
filters read the canonical Project state; an absent assessment is shown explicitly. Coloured severity pills in the source list
retain explicit analyst/AI labels and never imply an analyst rating from a
suggestion. Sort/search are URL-backed projections of
the existing bounded loaded list, preserve the selected finding and its draft, and
supply the same order to neighbour/keyboard navigation. Incomplete and failed
reads remain explicit; the UI never claims to sort unloaded rows.

**UI-R223.** Issue evidence offers explicit Preview, copy, download and concise
`details →` below the evidence divider, at the selected source's exact retained Project revision. Neither metadata
nor payload is read automatically except for the bounded saved source snippets:
at most the first two evidence files may display text/plain snippets, each up to
64 KiB after authoritative metadata admission. Larger or structured/HTTP evidence
remains explicit. Preview reuses the shared Artifact size/format bounds; failure
is visible and retryable, without another-scope fallback. Absent
scope has no fabricated action or bytes; source locations do not imply readable
source snippets. Captured HTTP credential values keep independent show/hide
controls. Sorting and presentation introduce no new business persistence or
verification/analyst state.

### Material reader

**UI-R224.** Material/file detail in User, Project, Run and Eval contexts uses
one content-first composition: a compact exact identity and selected revision,
authoritative current/historical and locked chips, icon-only copy/download with
accessible labels, and Versions/Refresh above the main reader. Facts, Git source,
eligible new-version upload, storage details and paged history/lineage form a
quiet supporting column; on phones they follow the content in ordinary flow.
Copy/download retain the exact selected identity/bytes, return state survives
version selection, and existing upload eligibility/conflict owners remain.
Preview/archive reads stay explicit and retain their scope and size policies.
The existing Markdown, diff, HTTP, Scalar OpenAPI and LikeC4 engines remain the
rendering owners, styled with the shared font/theme tokens. Source wrapping is
a local control; line-number DOM expansion is bounded by UI-R225. After explicit archive Browse,
a readable SKILL.md is selected first, otherwise a readable root README.md;
other archives ask for a selection. The archive tree can collapse on phones and
selecting a file returns the reading space to its content. No file is executed,
no API request is sent by a renderer and no synthetic preview data ships in UI.

**UI-R225.** Material source previews number up to 4,000 lines (LIM-UI-18),
  including blank and wrapped lines. Number labels are presentation only; the
  literal text, CRLF/LF endings and downloads remain unchanged. Beyond that
  display bound the complete source stays in a single text node, with an
  explanation that numbering is hidden.

**UI-R226.** OpenAPI material reference navigation selects one saved operation
  or schema. The existing Scalar renderer owns parameter/schema/reference
  interpretation, including Swagger 2 and OpenAPI 3. A selected direct operation
  retains all path-level data and root component definitions. Referenced path
  items are passed intact to Scalar. Documents with references to other paths
  stay complete in Scalar, with operation visibility controlled by its public
  slug generator. API information remains available explicitly. On narrow
  screens the API browser collapses above the selection. Navigation renders at most 50 rows
  (LIM-UI-19); search and local paging reach the entire bounded document.
  Display selection does not execute requests or allow remote references.

### New check workspace

**UI-R227.** New check uses a single workspace: objective above a searchable
check-type chooser, setup and an ordinary-flow summary/action card. On desktop
these form three columns; at medium widths the summary follows setup; on phones
the chooser folds above setup and the summary follows it. No pinned mobile action
area covers fields. Shared font, theme, icon, field and button primitives apply.
Existing generic, preparation and Web scan controls, catalog/material pagination,
read failures and keyboard navigation remain available. Ctrl/Command+Enter starts
from setup, but not from a link or the type chooser. Exact published profiles remain selectable without presenting task limits as source Scope.

**UI-R228.** A common program whose `seed_source.input` selects a JSON-capable
assertion input permits inline checklist authoring using
`contractor.audit.assertions.v1`. The composer starts with one empty check;
Add/remove preserves each surviving item's key. A shared required rationale and
optional conditions, evidence requirements and explicit version use the current
assertion schema. Conditions/evidence/version are disclosed on demand. Blank
checks, absent rationale or invalid schema prevent saving. An open checklist draft
blocks Start/Save draft until explicitly saved and attached or closed; an older
selected material is never silently used in place of that draft. Bounds come from the
common assertion source schema; this is not the retired inventory checklist codec.

**UI-R229.** Explicit Save and attach retains the exact assertion document as an
ordinary Project Artifact through the existing content-derived create-only CAS
owner, then attaches the returned exact revision. An uncertain write is reconciled
only by matching metadata and complete bytes/digest. A different binding is never
overwritten. Failed retention preserves the editor draft; opening or editing it
creates no material or Audit. Existing context fences stop abandoned forms from
continuing to attach or launch. Start/Save draft keeps the ordinary Audit request
keys and stale-revision handling. Material deletion/retention stays Project-owned.

**UI-R230.** New check Scope describes selected exact API, target or explicitly
typed engagement inputs. Source-only checks have no separate Scope card or summary
row. Task limits never claim endpoint counts, coverage or authorization. Until
preparation records the worklist, the UI explains that the accepted endpoint set
is established at start. Selected exact materials link to their ordinary reader;
this descriptive presentation adds no automatic content reads. Known shipped
check types use concise names and distinct shared glyphs; unknown programs keep
a neutral label without inferred capabilities. Exact profiles remain individually
selectable; presentation does not reconstruct retired inventory/variant semantics.

**UI-R231.** Context-list and API operation selection keeps its navigation and
outer page stable. Shared scroll containers reserve their scrollbar gutter;
keyboard row focus avoids scrolling ancestors before revealing the row. The
embedded API renderer updates its configuration without remounting for each
operation. Intentional anchors, local references, mobile list/detail transitions
and the existing keyboard focus contract remain available.
