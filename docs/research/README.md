# Contractor v2 research catalog

This directory carries research hypotheses forward from the pre-v2 Contractor
repository and records hypotheses that are specific to the v2 execution model.
It is documentation and an evaluation control-plane input, not production
configuration. Production packages must not import hypothesis IDs or make
Scheduler decisions from these records.

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
