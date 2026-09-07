# V39-002: trusted Audit completion pinning

The trusted Run Service selects `workerCompletion` from the claim-locked,
immutable AuditProfile check binding. It validates the resolved Workflow,
manifest membership and exact task packages before creating model work. Input
fork receipts determine the RunScope refs; the selected Stage result mapping
determines the versionless output binding.

Migration 000054 stores the Run contract and the allocation projection.
Deferred ownership checks make child Run creation, binding, input forks and
completion pinning one transaction. The contract cannot be removed or rewritten;
ordinary Runs cannot acquire it. Existing rows retain omission semantics.
Allocation inserts cannot omit or change their selected Run contract.

Scheduler selects only the named Stage/Worker and resolves contract inputs by
pinned revision, including on later Stage attempts. Escalation keeps the same
completion authority. Registry and production placement require both
`audit-results@2` and the explicit completion capability. Prepare carries the
same contract and independently rejects unsupported selections. Selection of
`audit-results@2` without a trusted contract is an error.

The test catalog lives in `testdata/configs`, independent of local changes to
`configs`. Existing small config-loader fixtures remain in
`internal/config/testdata/valid`. No operator configuration or deployed catalog
version is part of this change.

Verification used an isolated disposable PostgreSQL 17 container. All commands
below passed with `CONTRACTOR_TEST_DATABASE_URL` set; new database tests were
executed, not skipped:

```sh
go test ./internal/auditservice ./internal/auditstore ./internal/runservice ./internal/runstore ./internal/scheduler ./internal/controlplane ./internal/contracts ./internal/auditcontroller
go test -race ./internal/scheduler ./internal/controlplane
go test -count=1 ./internal/runservice ./internal/runstore ./internal/scheduler ./internal/controlplane -run AuditCompletion
```

Regression coverage includes ordinary omission and spoofed labels, invalid
Workflow/task ownership, the mixed Runtime fleet, exact prepare transport,
reservation clone/replay, input alias replacement, transaction rollback,
service/store recreation with an unrelated catalog, immutable Run/allocation
records, escalation, same-Run Stage retries and a fresh child Run after an
Audit-level retry. The latter does not inherit a prior failed Run's output.

Runtime still does not advertise the new completion capability. Collector,
completion-gate activation, diagnostics and rollout remain V39-003/005–007.
This task does not run model evaluations or activate a deployed profile.
