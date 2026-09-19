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
HTTP is accepted for IP-literal loopback and RFC 1918 development addresses. Contractor Server
must list the UI origin (for example `http://127.0.0.1:4173`) in its exact
browser-origin allowlist. Production UI and API origins must also be HTTPS and
same-site so the Server's `SameSite=Lax` session cookie is eligible for direct
browser requests.

Optional static-service settings are:

- `CONTRACTOR_UI_HOST` (default `127.0.0.1`);
- `CONTRACTOR_UI_PORT` (default `4173`);
- `CONTRACTOR_UI_DIST_DIR` (default `ui/dist`).

The sidebar, `/runtime-config.json` and `/healthz` use the UI version from
`package.json`. Increment it for each UI release: patch for fixes, minor for new
features, major for breaking changes. For a patch release, run
`corepack pnpm --dir ui version patch --no-git-tag-version` from the repository
root before building. Rebuilding alone does not increment the version. Deploy
the matching `package.json` with the new bundle and restart the UI service so
the browser and runtime configuration report the same version.

The generated file at `src/api/generated/public.ts` comes only from
`api/openapi/contractor-public-v1.yaml`. Run `make ui-generate`; never edit the
file by hand. `make ui-generate-check` regenerates it and fails on a tracked
diff.

The `/artifacts` route manages UserScope Workflow inputs. Uploads are limited
to 64 MiB and use explicit create/CAS preconditions; the UI never retries a
PUT after a conflict or lost response. Version and lineage views retain exact
revisions. Inline preview is opt-in, capped at 256 KiB, restricted to a small
text media-type allowlist, with local document rendering and an escaped source
view. ZIP and Skill packages also offer **Browse files**: a collapsible folder
tree and on-demand UTF-8 file previews, including `SKILL.md`. The same viewer is
available for User, Project and Run artifacts. Each request pins the exact
artifact revision and rechecks its scope on Go Server. Unsupported or oversized
files remain available through the original archive download.

Archive inspection never extracts entries to disk. Directory validation happens
before ZIP reader allocation; limits are 64 MiB stored, 4096 entries including
implicit directories, 4 MiB central directory, 256 MiB declared expanded size,
1024 UTF-8 bytes per path and 32 path components. A selected file is read through
a 256 KiB output limit and a 1 MiB compressed-input limit, with size and CRC
checks. Unsafe paths, duplicates, file/directory collisions, links, special
files, encryption and unsupported compression are rejected. ZIP64 directories
and split archives are download-only. Nested archives are not expanded.
Markdown uses the existing renderer with HTML and images disabled; HTML, SVG
and scripts are shown as escaped source. Responses are JSON with `nosniff` and
`no-store`; the client bounds response bytes and does not retain inactive file
previews in the query cache.

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

Project workspaces include their Audit history and draft creation directly.
Coverage shows every page of the current round, with result filters and search
across task text, conclusions, gaps and evidence. The API reads the exact retained
task and accepted result packages, so this works with custom checklists and
existing Audits without a profile-specific title catalog or a new model call.
Task documents are available on demand; package evidence is limited to the
selected check. Findings navigation stays within the selected Audit, with a
separate link to all Project findings. Audit deletion is visible on cards and
detail pages, with confirmation and server-authoritative reconciliation.

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
