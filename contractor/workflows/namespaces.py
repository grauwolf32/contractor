"""Shared memory-namespace prefixes for the trace-annotation workflow family.

Every trace producer writes its per-path memories and vulnerability reports
under ``{prefix}:{namespace}:{path_key}`` (so the report artifact lands at
``user:vulnerability-reports/{prefix}:{namespace}:{path_key}``). Consumers —
``trace-verify``, ``vuln-assess`` — must read with the *same* prefix the
producer wrote with. These constants exist so the read and write keys cannot
drift apart (audit precedent: vuln_assess once read ``trace-annotation:``
while the trace stage wrote ``trace-graph-pathpar:`` and silently saw nothing;
see tests/units/contractor_tests/workflows/test_vuln_assess_namespace.py).
"""

from __future__ import annotations

# ``trace`` (TraceAnnotationWorkflow) and ``trace-direct``
# (TraceAnnotationDirectWorkflow) share one prefix.
TRACE_ANNOTATION_NAMESPACE_PREFIX: str = "trace-annotation"

# ``trace-graph`` (TraceGraphWorkflow) — the production default.
TRACE_GRAPH_NAMESPACE_PREFIX: str = "trace-graph"

# ``trace-graph-pathpar`` (TraceGraphPathParWorkflow).
TRACE_GRAPH_PATHPAR_NAMESPACE_PREFIX: str = "trace-graph-pathpar"

# ``trace-postdiff`` (TracePostDiffWorkflow) — annotate-only trace stage
# followed by a post-diff analytics stage; the analytics agent is the
# finding producer for this prefix.
TRACE_POSTDIFF_NAMESPACE_PREFIX: str = "trace-postdiff"

# Every prefix a trace producer may have written findings under, in the
# order consumers should probe them.
TRACE_NAMESPACE_PREFIXES: tuple[str, ...] = (
    TRACE_ANNOTATION_NAMESPACE_PREFIX,
    TRACE_GRAPH_NAMESPACE_PREFIX,
    TRACE_GRAPH_PATHPAR_NAMESPACE_PREFIX,
    TRACE_POSTDIFF_NAMESPACE_PREFIX,
)
