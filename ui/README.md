# Contractor UI

This directory is an independently built React single-page application. The
Node process serves only the generated static files, `/runtime-config.json`,
and `/healthz`; it never proxies API requests or stores browser/model
credentials. The browser calls the configured Go Server origin directly.

The pinned toolchain is Node 24.20, Corepack 0.36 and pnpm 11.24. Install and
verify it from the repository root:

```shell
npm install --global corepack@0.36.0
make ui-verify
make ui-browser-install
```

For frontend development, Vite serves source files on loopback. Supply a
matching runtime config through the production static service when testing the
built application:

```shell
corepack pnpm --dir ui build
CONTRACTOR_UI_API_BASE_URL=http://127.0.0.1:8080 \
  corepack pnpm --dir ui start
```

`CONTRACTOR_UI_API_BASE_URL` is required. It must be an origin-only HTTPS URL;
HTTP is accepted only for an IP-literal loopback address. Contractor Server
must list the UI origin (for example `http://127.0.0.1:4173`) in its exact
browser-origin allowlist. Production UI and API origins must also be HTTPS and
same-site so the Server's `SameSite=Lax` session cookie is eligible for direct
browser requests.

Optional static-service settings are:

- `CONTRACTOR_UI_HOST` (default `127.0.0.1`);
- `CONTRACTOR_UI_PORT` (default `4173`);
- `CONTRACTOR_UI_DIST_DIR` (default `ui/dist`).

The generated file at `src/api/generated/public.ts` comes only from
`api/openapi/contractor-public-v1.yaml`. Run `make ui-generate`; never edit the
file by hand. `make ui-generate-check` regenerates it and fails on a tracked
diff.

The `/artifacts` route manages UserScope Workflow inputs. Uploads are limited
to 64 MiB and use explicit create/CAS preconditions; the UI never retries a
PUT after a conflict or lost response. Version and lineage views retain exact
revisions. Inline preview is opt-in, capped at 256 KiB, restricted to a small
text media-type allowlist, and rendered as escaped text. Other payloads remain
available only through exact-revision download.

The `/workflows` route lists exact published Workflow versions and renders the
selected parameter, input, output, Stage and escalation contract. Run drafts
accept only declared strings, loaded exact Artifact revisions, published
ModelPolicy/LLMGatewayConfig selectors and active credential IDs. One in-memory
idempotency key remains bound to the canonical submitted draft after response
loss; an unchanged explicit retry reuses it, while changed submitted content
gets a new key. The UI navigates to the Server-returned Run ID only after a 202
response and never inserts a speculative Run into its cache.

The `/runs` route lists owner-scoped Run snapshots and renders ordered Stage
attempts, Scheduler decisions, resolved execution configuration, safe aggregate
metrics, exact RunScope Artifacts, and frozen outputs. Cancellation always
refetches the authoritative aggregate, including when it races a terminal
transition; the browser never predicts a lifecycle state.

One `contractor.events.v1` WebSocket is multiplexed across open Run views and
the Operations workspace.
Lifecycle frames only invalidate REST queries. Closed, typed Planner frames may
advance the nested subtask projection in exact cursor order, while duplicate
frames are ignored and any sequence gap, generation change, explicit resync, or
unknown frame discards live continuity and requires a REST snapshot. Prompt and
model text, tool payloads, provider bodies, credentials, and physical Runtime
placement are not accepted by the browser event DTO.

The `/operations` workspace begins from one coherent REST snapshot and keeps
Runtime Agent observations separate from authoritative single-slot allocation
state. Its WebSocket frames are invalidation hints only; every change, cursor
gap, or generation change refetches REST. Execution state has no force-idle,
reassign, finish, abort, or release controls.

Configuration pages inspect all versioned kinds, while only ModelPolicy and
LLMGatewayConfig support clone-to-draft, typed validation, and create-only
publication. Credential forms select an exact manager-enabled Gateway and
exact ModelPolicies plus LiteLLM-enforced spend/rate/concurrency policy. No
token field or token response exists in frontend types or state. Every listed
credential is active; replacement uses a new ID, and deletion remains blocked
with safe Run links while a non-terminal Run pins the old ID. Configuration and
credential mutations retain one in-memory idempotency key for an unchanged
canonical draft and always reconcile through Server reads.

The real browser gate is orchestrated from the repository root because it also
starts Go Server, PostgreSQL-backed Scheduler and Python Runtime Agent:

```shell
CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
  make test-ui-stack
```

`corepack pnpm --dir ui test:e2e` alone intentionally skips the real-stack
scenario unless the harness supplies its isolated URLs, credentials, source
fixture and evidence paths. The pinned baseline is Chromium; broader browser
and visual-regression matrices are deferred.
