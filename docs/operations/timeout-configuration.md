# Operational timeout configuration

Server process settings belong to `ServerConfig.spec`, not AgentTemplates,
ModelPolicies, RuntimeLabels or AuditProfiles. The sample is
[`configs/server.local.yaml`](../../configs/server.local.yaml). Changes require a
process restart. The startup `effective operational settings` log includes only
resolved durations and the Audit claim batch; it excludes URLs, paths and secrets.

## Settings and ownership

| ServerConfig path | Default | Scope |
| --- | --- | --- |
| `runtimeRequestTimeout` | `30s` | One private Runtime or A2A HTTP request |
| `scheduler.operationTimeout` | `30s` | One Scheduler persistence, prepare or reconciliation operation |
| `scheduler.finalizationTimeout` | `10s` | Absolute finalization deadline recorded with the Stage candidate |
| `scheduler.abortTimeout` | `10s` | Absolute abort deadline recorded with the termination transition |
| `runtimeLifecycle.cleanupTimeout` | `30s` | One release batch or failed-prepare cleanup, including abort then release |
| `projectLifecycle.operationTimeout` | `30s` | A Project deletion controller operation |
| `projectLifecycle.claimDuration` | `1m` | Project deletion claim lease; at least 1us and must exceed its operation timeout |
| `auditController.pollInterval` | `1s` | Delay between Audit claim attempts |
| `auditController.claimLease` | `30s` | Audit controller ownership lease, 1s..5m and at least twice its operation timeout |
| `auditController.operationTimeout` | `10s` | Claim, reconcile or release operation; includes work inside reconciliation |
| `auditController.claimBatch` | `8` | Concurrently claimed Audits, 1..100; separate from child Run batches in AuditProfile |
| `database.connectTimeout` | `5s` | Establish one PostgreSQL connection |
| `database.acquireTimeout` | `2s` | Wait for a pooled connection |
| `database.queryTimeout` | `20s` | Client database operation budget |
| `database.statementTimeout` | `15s` | PostgreSQL statement execution |
| `database.lockTimeout` | `2s` | PostgreSQL lock acquisition |
| `database.idleTransactionTimeout` | `30s` | PostgreSQL idle transaction lifetime |
| `a2a.pollInterval` | `100ms` | Delay before polling a non-terminal A2A task |
| `credentialManagement.connectTimeout` | `3s` | LiteLLM virtual-key API connect/TLS timeout, at most 1m |
| `credentialManagement.requestTimeout` | `15s` | One LiteLLM virtual-key management request, at most 2m |

Explicit durations must be positive and parse as Go durations (for example
`250ms`, `10s`, `2m`). Database operation budgets must be 1ms..24h and satisfy
`lockTimeout < statementTimeout < queryTimeout`. Duration overflow is rejected.
A larger inner timeout is permitted where an outer deadline already bounds it:
for example, an HTTP request with a 30s timeout inside 10s finalization still ends
by the 10s deadline. Credential management budgets do not control LLM inference.

## Input precedence and PostgreSQL

For each new grouped setting: defaults < YAML < environment < flags.
`spec.scheduler.finalizationTimeout` maps to
`CONTRACTOR_SCHEDULER_FINALIZATION_TIMEOUT` and
`--scheduler-finalization-timeout`. Other group/field names follow the same
conversion to uppercase underscores or lowercase hyphens; for example,
`a2a.pollInterval` maps to `CONTRACTOR_A2A_POLL_INTERVAL` and
`--a2a-poll-interval`. `auditController.claimBatch` follows this rule too.

The connection URL and credentials remain outside ServerConfig. The production
pool rejects `connect_timeout`, `statement_timeout`, `lock_timeout` and
`idle_in_transaction_session_timeout` in PostgreSQL connection settings,
including the relevant pgx environment inputs and `options` overrides. Move these
values into the `database` group instead of configuring the same timeout twice.
Connection-count, TLS and other unrelated DSN settings retain pgx semantics.
Maintenance and diagnostic pools retain their existing explicit policies.

## Deadlines and migration

Before V49-001, changing `runtimeRequestTimeout` also changed Scheduler and
Project lifecycle operations and Runtime cleanup. It now controls only HTTP.
Existing files without the new blocks receive the defaults above. If a deployment
intentionally depended on the coupling, copy its old value into
`scheduler.operationTimeout`, `projectLifecycle.operationTimeout` and
`runtimeLifecycle.cleanupTimeout`; increase the Project claim duration if needed.
Set finalization and abort budgets explicitly when tuning those phases.

Finalize/abort fan-out uses the earlier of the caller's deadline and the stored
terminal deadline. A cleanup setting no longer silently caps those phases.
Recovery reuses the stored deadline and operation ID; a process restart or config
change cannot grant another full finalization/abort budget. Each release batch
shares one cleanup deadline. Failed-prepare cleanup shares one deadline across
its abort and release steps, and a failed release retains the allocation grant.
Transport timeouts can end individual requests earlier without releasing
ownership or extending an enclosing operation.

The existing `plannerTimeout`, `workerRequestTimeout` and `shutdownTimeout` keep
their separate meanings. This change does not alter domain payload limits,
heartbeat leases, model policies or inference retry rules.

## CLI and local Runtime workspace

Both `contractor check` and `contractor context check [name]` use the global
`--timeout` / `CONTRACTOR_TIMEOUT` setting, whose default remains 30s.

Runtime uses `WorkspaceSettings.operation_timeout_seconds`, configured with
`--workspace-operation-timeout-seconds` or
`CONTRACTOR_WORKSPACE_OPERATION_TIMEOUT_SECONDS` (default `30.0`; seconds as a
finite positive number). CLI overrides environment. An explicit override requires
`--workspace-storage local`; setting it for memory or disabled workspaces fails.
The value applies to local direct filesystem operations and close waits, including
serialization and initialization. It is process-local and is not advertised as a
Worker capability or placed into Allocation settings.

Initialization and close use the earlier of this operation budget and the
supplied enclosing deadline. Allocation release, rollback and terminal cleanup
forward their remaining deadline into close. An expired wait fences the workspace
but keeps its physical I/O and disposal tasks owned until they actually finish.
A release retry can resume idempotent cleanup and join the same disposal task;
it cannot reuse the slot or remove storage while the original I/O is running.
Already saved finalize/abort deadlines remain unchanged on retry. A completed,
failed release attempt retains the existing ability to retry physical cleanup
under a new release attempt budget.
