# Contractor v2 research catalog

This directory carries research hypotheses forward from the pre-v2 Contractor
repository and records hypotheses that are specific to the v2 execution model.
It is documentation and an evaluation control-plane input, not production
configuration. Production packages must not import hypothesis IDs or make
Scheduler decisions from these records.

## Completed experiments and implementation reviews

- [V60 deep review — 2026-09-20](2026-09-20-v60-deep-review-results.md) —
  execution, Runtime, storage, Audit, operations and configuration checks;
  scoped corrections, actual gates and evidence boundaries.
- [Run deletion and Audit revisions — V60-012](2026-09-20-run-deletion-audit-revisions.md) —
  atomic invalidation, import/purge lock ordering and immutable report retry.
- [Project review — first pass, 2026-09-20](2026-09-20-project-review-first-pass.md) —
  reproduced defects, accepted decisions and explicit limits; [detailed V60 plan](../plans/2026-09-20-project-review.md).

- [Public OpenAPI review — 2026-09-19](2026-09-19-public-openapi-review.md) —
  verified schema, handler and generated-client mismatches.
- [Public OpenAPI corrections — 2026-09-20](2026-09-20-public-openapi-corrections-results.md) —
  all 12 findings corrected in V59, client regressions and isolated PostgreSQL verification.

- [V57-004: Worker result assembly and size limits](2026-09-19-worker-result-assembly.md) —
  typed Audit handoff, removal of its synthetic JSON bound and history of remaining limits.
- [V57-005: A2A invocation connection reuse](2026-09-19-a2a-connection-reuse-decision.md) —
  local TLS measurements and fault matrix; decision: keep production transport unchanged.

## Drafts and evaluation design

- [Annotation participation index](annotation-index-proposal.md) — proposed
  attribution and snapshot reconciliation. The accepted evidence boundary is
  in [specification 13](../spec/13-taint-annotations.md#annotation-artifacts-and-evidence).
- [`stateflow@1` Planner draft](stateflow-1.md) — proposed
  explicit Planner state and deterministic Worker context; not a registered
  production Planner.
- [Portable evaluation specification](../spec/26-portable-evaluation-format.md)
  owns the accepted format.
- [Instruction-evaluation fixtures and workflow](../../tests/eval/agent_instructions/README.md)
- [Live-model evaluation commands](../testing/live-models.md)

Return to the [documentation index](../README.md) for implementation guides.

## Files

- `hypotheses/legacy.yaml` is a normalized transcription of all 228 hypotheses
  from the legacy research memo. It preserves the original claim, mechanism,
  proposed test, decision rule and metric as plain text.
- `v2-mechanism-map.yaml` maps the legacy research programs to implemented v2
  mechanisms, near-term evaluation seams and missing capabilities.
- `hypotheses/v2-native.yaml` contains new hypotheses arising from v2-specific
  boundaries: immutable Run snapshots, Planner/Worker separation, typed live
  observations, Worker session modes, terminal summarization and exact artifact
  provenance.

The source memo described 51 directions (`A` through `AY`) plus eleven `PV`
Planner variants. Its inline `DONE` and `TESTED` markers describe evidence from
the old implementation. They are retained as historical evidence only. Every
imported hypothesis has `v2_status: proposed`; it must be re-bound to v2 arms,
fixtures and scorers before it can become `ready` or receive a v2 decision.

The old machine registry had one structured hypothesis, `AW1`, and one draft
experiment, `EXP-2026-001`. Their metadata is retained under AW1 in the legacy
catalog, but the experiment is not activated or treated as frozen for v2.

## Record conventions

`legacy.yaml` uses this shape:

```yaml
- id: AW1
  direction: AW
  claim: ...
  legacy_design:
    mechanism: ...
    test: ...
    decision_rule: ...
    metrics: [...]
  legacy_evidence:
    status: not_recorded
  source_anchor: AW1
  v2_status: proposed
```

HTML presentation tags from the source memo are deliberately removed. Wording,
numbers and named legacy components remain unchanged so that a later v2 rewrite
cannot silently replace the original claim. `legacy_design` is therefore not a
v2 implementation prescription.

New records use explicit control and treatment descriptions, a primary metric,
guardrails, related legacy IDs and prerequisites. Minimum useful effects and
non-inferiority margins may remain unset while a record is `proposed`; they must
be frozen before the first model call.

## Evaluation boundary

An experiment should resolve an arm into ordinary immutable v2 configuration:

- an exact Workflow version and its frozen Run snapshot;
- exact ModelPolicy, LLMGatewayConfig and ExecutionConfig refs;
- exact AgentTemplate, instruction and Agent Skill revisions;
- exact input ArtifactRefs;
- Runtime labels only when infrastructure is intentionally part of the arm.

Research metadata identifies Runs but must never alter execution semantics. One
`leg × fixture × case × sample` is one fresh WorkflowRun. Scheduler retries and
escalations remain attempts inside that sample. Raw Run evidence is immutable;
scorer output is a separate versioned derivative that can be recomputed.

Before a record advances from `proposed` to `ready`, its experiment must freeze:

1. the single intended arm difference and a snapshot-diff assertion;
2. fixture and ground-truth revisions;
3. scorer identity, parameters and digest;
4. sample count, requested seeds and arm ordering;
5. primary metric, guardrails, effect threshold and stopping rule;
6. maximum cost and the handling of incomplete telemetry;
7. the confirmatory held-out slice required before promotion.
