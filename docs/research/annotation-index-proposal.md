# Annotations as an optional Workflow artifact

Status: proposal. The automatic annotation index is not implemented and is not a
prerequisite for extending the general [finding creation and reading tools](../spec/27-findings-tools-and-collections.md).

Annotations are one possible investigation output, alongside a report, graph,
diff or validation log. Each Workflow chooses its outputs. The general Audit,
Finding, hypothesis and completion mechanisms use artifact references and do
not require `annotations.json`, `trace_id`, a function or a sink.

## First version

For an analysis variant that uses annotations, exporting its artifacts through
ordinary Workflow outputs is sufficient. The
[taint-trace Workflow](../../configs/workflows/taint_trace_from_workspace_v4_memory.yaml)
already exports workspace state/diff. The current
[OpenAPI graph Worker](../../configs/workflows/audit_openapi_operation_trace_v3_memory.yaml)
does not enable annotation tools or this export. These are added to the selected
analysis variant, rather than to every scenario that produces findings.

An artifact can exist without findings or hypotheses. It becomes evidence for
a specific conclusion through an explicit reference and the requirements of
the evidence contract. Annotated source alone does not establish a vulnerability,
the correctness of a flow or the completeness of an investigation.

All evidence references for a finding must already exist when it is published.
Before the final export, a finding can reference a previously published report
or source evidence. References to future revisions are invalid.

## When a structured index is needed

An index is useful when a consumer needs to distinguish newly created annotations
from reused ones, associate them with assignments, or aggregate results from
parallel Workers. It is a separate artifact format owned by the annotation tools
and can be validated by a specialized handler. Its schema does not become fields
in the general Audit result; a generic handler registry is not needed in advance.

The current [taint-annotations@1](../../runtime/src/contractor_runtime/toolsets/taint_annotations/tools.py)
returns path, symbol, kind, line numbers and changed. `annotate_trace` accepts a
model-supplied `target`, defaulting to `unknown`; this is not a trusted assignment
identity. `annotate_validate` and `annotate_sink` have no such argument. These
tools do not themselves track operation participation.

If an index is adopted, the following rules should guide its implementation:

- Execution provenance comes from the existing trusted context. An Audit adapter
  can associate a record with the assigned item; an ordinary Workflow can attach
  its pinned target. The annotation tool does not access the Audit database, and
  a specific OpenAPI operation ID does not become a required argument.
- A code annotation and its participation in an investigation are separate
  records. A shared sink can be associated with several assignments; data and
  control information remains specific to each path/call site. Even a single
  operation can call the same function twice with different arguments.
- A stable annotation ID accounts for the baseline, an unambiguous code location
  and normalized content. A line number or temporary graph symbol ID alone is
  insufficient. Stability across arbitrary refactoring is not required in the
  first version.
- Before export, accepted tool records are reconciled with the same snapshot
  used by the workspace exporter. Rollback or deletion must not leave an
  annotation active; records are not reconstructed from model text or diagnostic
  metrics.
- Isolated overlays are preserved. Identical records can be merged, while
  contexts and semantic conflicts remain distinguishable. Concatenating textual
  diffs does not replace this rule; when needed, a combined source tree is built
  from the baseline and accepted records.

## Limits of diffs and shared functions

In the [workspace overlay](../../runtime/src/contractor_runtime/projectfs/overlay.py),
state describes the accumulated changes relative to the source, while diff
describes changes since the checkpoint. This is sufficient to represent changes.
A diff is insufficient for investigation attribution: a repeated annotation call
can return `changed=false`, and an existing comment does not prove that the
current Worker checked it. If attribution is needed, it must be recorded
explicitly in the selected format.

For example, GET calls a shared function after validating its input, while POST
calls it directly. The function/sink description can be reused, but conclusions
about controls differ. A matching function or `file + CWE` is not sufficient to
merge findings automatically. This distinction applies regardless of whether
annotations exist.

A possible export extension should align with the
[common completion contract / V39](../spec/25-audit-worker-finalization.md)
if its references need to be included in the deterministic result. This requires
neither a new independent finalizer nor a mandatory annotation stage.
