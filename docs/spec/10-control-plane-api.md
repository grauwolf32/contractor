# 10 — Control Plane API

Status: **Draft**  
Depends on: [02](02-domain-contracts.md), [05](05-artifacts-and-run-state.md), [06](06-planner-and-workflow.md), [07](07-agent-runtime-and-a2a.md)

## API surface

The initial HTTP API is versioned under `/v1`:

- `POST /v1/runs` — submit a run;
- `POST /v1/artifacts` — stream and publish one checksummed ordinary input blob;
- `POST /v1/project-snapshots` — validate and publish a canonical source
  manifest over previously uploaded blobs;
- `GET /v1/runs/{run_id}` — current status projection;
- `GET /v1/runs` — paginated filtered list;
- `POST /v1/runs/{run_id}/cancel` — request cancellation;
- `GET /v1/runs/{run_id}/artifacts` — list authorized output refs;
- `GET /v1/artifacts/{artifact_id}/versions/{version}` — fetch metadata/content;
- `GET /v1/runs/{run_id}/events` — paginated diagnostic timeline;
- `GET /v1/workflows` — versioned public workflow/profile catalog;
- `GET /v1/agents` — fleet status for operators;
- `/health/live` and `/health/ready` — process probes.

The exact web framework is an adapter choice. Public responses expose v2 DTOs,
not ADK Session/Event, SQLAlchemy or A2A SDK models.

`POST /v1/runs` accepts a versioned objective, exact input ArtifactRefs,
expected-output contracts, an authorized workflow/execution profile and an
authorized Planner-profile selection. Bounded caller overrides are part of the
profile schema. Server resolves these inputs to an immutable `RunSpec` and
exact `PlannerStrategyRef` and commits both before returning success. Unknown
objective fields, artifact kinds, policy overrides and output contracts fail
closed. The request never accepts a caller-constructed `TaskSpec`, `WorkerJob`,
Attempt/fencing identity, implementation module or unvalidated Agent endpoint.

The first release has one canonical source-ingestion path. A caller uploads
content as bounded streaming multipart to `POST /v1/artifacts`, declaring its
ordinary allow-listed blob kind, media type, byte count and SHA-256 digest. That
route rejects reserved compound/system kinds, including `project-snapshot`. The
caller then publishes a `ProjectSnapshotManifest` through
`POST /v1/project-snapshots`; only this operation may mint the reserved
`project-snapshot` kind. It applies spec 02's exact manifest/path versions,
canonical ordering and `snapshot_sha256` preimage and atomically records
immutable blob dependencies. Both operations require caller-scoped idempotency
keys and return exact ArtifactRefs only after digest, quota, authorization and
atomic-publication checks. Runs reference the resulting `project-snapshot`;
neither endpoint accepts a local path. Future Git or object-store importers must
produce and validate the same manifest contract.

The private `/internal/v1/agent-registrations` fleet-control API is specified in
[07](07-agent-runtime-and-a2a.md). It may share the Server process and HTTP
runtime, but uses a separate router, mTLS authorization policy and DTO surface;
it is not part of the public Control Plane `/v1` contract.

## Requirements

- **API-001** — Run submission MUST require an idempotency key scoped to the
  authenticated caller. Repetition returns the original Run.
- **API-002** — Submission becomes successful only after the initial run row
  and `PlannerRunState` version plus required submission audit facts are durably
  committed in the same RunState Unit of Work.
- **API-003** — `GET /runs/{id}` reads the run status projection and current
  RunState reference; telemetry availability MUST NOT affect it.
- **API-004** — Cancellation is idempotent and asynchronous. The response
  distinguishes `requested`, `already_terminal` and `not_found`. `requested` is
  returned only after the cancellation RunState transition, execute fencing,
  required cancel outbox records and required audit facts commit in one Unit of
  Work; Control Plane MUST NOT append the required fact separately.
- **API-005** — List endpoints MUST use stable cursor pagination and bounded
  page sizes.
- **API-006** — Errors MUST use `application/problem+json` plus the stable
  `ErrorEnvelope` code/correlation fields.
- **API-007** — Authorization MUST be checked at run and artifact scope. Agent
  registration credentials MUST not authorize Control Plane operations.
- **API-008** — Readiness is false during incompatible migrations, a missing
  dependency required by an enabled Planner/Worker strategy or shutdown drain.
  Proxy or ADK is not universally required. A temporary lack of healthy Agents
  MAY leave readiness true but MUST be visible in status.
- **API-009** — Request deadlines and disconnect cancellation MUST propagate to
  orchestration only where doing so cannot abandon a committed operation.
- **API-010** — API schemas MUST be generated and checked for backward
  compatibility in CI.
- **API-011** — A status response MUST carry `run_state_version`; if the stored
  projection version disagrees with its RunState pointer, the API MUST return a
  stable `state_repairing` status/error and trigger repair rather than present
  the stale projection as current.
- **API-012** — Fleet-control credentials MUST authorize only private Agent
  registration operations and MUST NOT authorize public run, artifact,
  telemetry or operator endpoints.
- **API-013** — Run submission MUST validate the objective against its declared
  contract and resolve the requested/default execution and strategy profiles to
  an immutable authorized `RunSpec` and `PlannerStrategyRef`. Recovery MUST NOT
  re-resolve either from current defaults.
- **API-014** — Run status MUST expose the selected strategy identity and, when
  authorized, the exact RunSpec identity. Event queries MUST distinguish
  authoritative lifecycle entries from optional/sampled diagnostic detail.
- **API-015** — Input publication MUST stream with configured per-object,
  snapshot and caller quotas; verify kind, media type, size and digest before
  visibility; and remove or quarantine an incomplete upload after a bounded
  retention period. Generic blob publication MUST reject every reserved
  compound/system artifact kind, including `project-snapshot`.
- **API-016** — A project snapshot MUST contain only authorized exact blob refs
  and pass the exact manifest version, Unicode/path canonicalization, ordering
  and `snapshot_sha256` algorithm in spec 02. Only the snapshot-publication route
  may assign its reserved kind and media type. Publication and later Run
  submission MUST recheck caller/tenant scope without leaking whether an
  unauthorized ArtifactRef exists.
- **API-017** — Authentication and run/artifact authorization are part of the
  first externally reachable API slice. A deployment MUST NOT expose mutating
  or content-reading `/v1` routes in an unauthenticated bootstrap mode.
- **API-018** — Workflow catalog entries MUST expose stable public key,
  objective/input/output contract versions, allowed Planner profiles and
  bounded override schema. They MUST NOT expose implementation modules,
  credentials or mutable Agent endpoints. Run submission MUST reject an unknown
  key or input/output contract mismatch before creating RunState.
- **API-019** — Fleet status, audit facts and ADK/session diagnostic detail
  require an explicit operator permission in addition to tenant/run scope. A
  normal run owner receives only its normalized, redacted lifecycle/detail and
  cannot enumerate Agent endpoints, other principals or audit targets.

## Acceptance

1. Twenty concurrent submissions with one idempotency key create one Run.
2. Status remains queryable while the telemetry sink is disabled.
3. Repeated cancellation converges to one terminal outcome without new
   Attempts after the first request.
4. An unauthorized caller cannot enumerate run or artifact existence.
5. Cursor pagination is stable while newer runs and events are inserted.
6. Corrupting the projection version yields `state_repairing`; after repair the
   response version matches the verified RunState without changing its plan.
7. Agent mTLS credentials can use fleet-control endpoints but receive a
   forbidden response from every public Control Plane operation.
8. Submitting the same objective with decomposing and passthrough profiles pins
   the requested exact RunSpec and strategy in two runs; later default changes
   affect neither run.
9. A caller-supplied TaskSpec/WorkerJob, implementation module or unauthorized
   exact Agent target is rejected before RunState creation.
10. A streamed upload with a wrong size/digest or interrupted body never
    becomes visible; idempotent replay of a completed upload returns the same
    ArtifactRef without storing a second blob. Declaring `project-snapshot` on
    this generic route is rejected.
11. A valid project snapshot can be submitted as a Run input and materialized
    by an Agent; wrong Unicode normalization/order/`snapshot_sha256` preimage,
    traversal, path or case-fold collision and cross-principal blob references
    are rejected without revealing artifact existence.
12. Anonymous requests cannot submit/cancel a Run, upload/read artifacts or
    query events in the first executable deployment.
13. The catalog fixture covers every enabled migration-ledger workflow key;
    selecting a key with a wrong objective/input contract fails before a Run is
    created, and disabling a profile removes it without changing old RunSpecs.
14. A run owner can read its normalized timeline but cannot query fleet/audit or
    raw session detail; an authorized operator receives only paginated redacted
    records within the selected tenant/run scope.
15. Fault injection at every submission/cancellation Unit-of-Work write leaves
    either RunState/projection/applicable outbox/required audit facts all
    committed or none; an API success is never returned for a missing required
    audit fact.
