# Operating configurable toolsets and MCP

`mcp@1` is a code-owned configurable ToolsetFactory using the official MCP Python
SDK 1.30.0 and **Streamable HTTP**. It is not a Runtime adapter. Its descriptor
advertises `toolDiscovery: allocation`, configuration schema
`mcp-streamable-http@1`, required configuration, and compatibility with `adk@1`.
Existing static toolsets keep their static tool lists and unchanged wire shape.

## Supported boundary

- HTTP(S) Streamable HTTP only. SSE events inside that transport are supported;
  legacy HTTP+SSE endpoints, stdio and transport fallback are not.
- Optional static authentication headers through encrypted `mcp-headers@1`
  credentials. There is no OAuth negotiation, browser secret readback or mTLS
  client identity configuration for the external MCP server.
- Endpoint up to 2048 UTF-8 bytes, no userinfo/query/fragment/whitespace. Redirects
  are rejected. TLS verification stays enabled; optional CA bundles contain 1–8
  X.509 PEM certificates, up to 64 KiB, with no private keys.
- 1–32 authentication headers; ASCII token names up to 64 bytes, values up to
  4096 UTF-8 bytes, 16 KiB total. Case duplicates, CR/LF, hop-by-hop, Host,
  Content-Length, proxy authentication and MCP/HTTP transport-owned headers are
  rejected. Credential kind must match the selected configuration.
- One frozen inventory per allocation, exact selected names only, at most 64
  portable tool names. Discovery is bounded to 16 pages, 1024 tools, 8 MiB total;
  JSON/SSE messages are limited to 2 MiB. List-change notifications do not expand
  or replace the active declarations.
- Selected input/output schemas use the closed JSON Schema subset in
  [spec 02](../spec/02-runtime-and-a2a.md). Object roots, local validation without
  coercion, nested typed properties/arrays, required fields, enums and documented
  constraints are supported. External refs, unsupported composition/keywords,
  excessive depth/size and incompatible declarations fail preparation. Limits:
  64 KiB per schema, 256 KiB selected declarations, depth 16, 2048 schema nodes.
- Arguments are at most 64 KiB. Results are at most 256 KiB/64 blocks and contain
  text blocks plus optional structured object content. `isError` is preserved and
  counted as a tool failure. Images/audio/resource blocks, automatic URL fetching,
  sampling, roots, elicitation and prompts are not supported. There is no silent
  truncation or coercion of an incompatible result.

The client reuses the configured `tool-http` proxy route and private-host guards.
Do not point MCP at private Contractor control/artifact/A2A endpoints. Allocation
MCP settings are projected before credential lookup, so a native-only consumer
receives no MCP config, header material or MCP credential provenance.

## Lifecycle and diagnostics

Preparation owns all tools before network I/O. One owner task enters and exits
SDK contexts, owns the HTTP transport, freezes declarations and closes the session
on finalization, abort, cancellation, preparation failure or lease loss. Cleanup
is bounded and idempotent. Unconfirmed cleanup retains the owner, fences the slot
and can force Runtime process exit; it never reports reusable capacity falsely.
A lost reply after an accepted `tools/call` is `outcome_unknown` and is not replayed
or reconnected automatically. A later independent model decision is a new tool
call and consumes the normal tool budget.

Runtime errors contain safe codes such as `missing_tool`, `unsupported_schema`,
`invalid_arguments`, `unsupported_content`, `result_too_large`, `private_content`
or `outcome_unknown`, according to the failed boundary. Preparation wraps these
as `mcp_<code>` on the private API; the ordinary Scheduler stage termination may
expose its existing `allocation_preparation_failed` category. Inspect Operations
capabilities, pinned config refs, coverage and stage/allocation reports together.
Endpoints, headers, server session IDs and remote error bodies must not be copied
into public errors, metrics, model results or Runtime logs.

If no compatible Runtime is available, placement waits under its existing finite
Stage deadline. An old Runtime advertising a static MCP-like list cannot satisfy
allocation discovery. If MCP dependencies cannot be imported locally, only the
MCP capability is suppressed; inspect the published capability observation and
install the locked Runtime environment. Native toolsets and `tool@1` stay usable.
Generic external tools have unknown effects; Audit admission rejects them under
all current effect policies, regardless of remote `readOnlyHint` or tool names.

## Rollout and rollback

Upgrade Server and database first. Migration `000056` extends the closed encrypted
credential kind checks; RuntimeConfig, requirements and safe coverage use existing
resources. Then update Runtime dependencies/code and confirm the allocation
MCP capability before publishing MCP templates/launching Runs. Finally update UI.
Existing native configurations omit the new fields and retain their canonical
digests. No immutable published version is rewritten during rollout.

Keep older Runtime Agents only for compatible native workloads. Do not roll back
Server/schema while MCP credential/config/Run/Audit references are retained: an old
binary cannot interpret the new closed kind/settings. Stop new MCP admission,
finish or cancel/release affected allocations, and preserve required retained
history/credential holds before considering a schema downgrade. Returning a label
to an earlier version affects new Runs only; it does not repair an active Run.
Never delete a credential to force a running allocation to switch settings.

## Reproducible release check

Prepare the locked Runtime environment and UI dependencies/browser, and provide
`CONTRACTOR_TEST_DATABASE_URL` pointing to disposable PostgreSQL. Run:

```sh
make test-toolset-runtime-configuration-release
```

The gate checks dependencies and local Chromium, validates wire/public contracts,
runs named PostgreSQL/race and real TCP MCP lifecycle cases, builds/tests the UI,
and starts real Server/Scheduler/Runtime/ADK and browser processes. A local
scripted model chooses exact tools; no paid provider or external MCP is involved.
The browser uses production APIs. Network fault injection discards a response only
after the real mutation commits, then verifies exact idempotent recovery.

`scripts/toolset-runtime-configuration-matrix.json` lists mandatory case identities,
including parametrized fault cases. Missing cases, absent prerequisites, failures,
skips and browser retries fail the gate. Reports go into a unique ignored
`.local/toolset-release/run-*` directory. Only a fully passing run writes
`tasks/evidence/v61-009.json`, recording commands, executed cases, dependency
versions and limits. It is fixture evidence, not a live-model quality benchmark or
external interoperability certification.
