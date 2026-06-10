"""Regression tests for the trace-verify namespace mismatch: trace-verify only
probed the 'trace-annotation:' prefix while trace-graph (the production
default) writes 'trace-graph:' and pathpar writes 'trace-graph-pathpar:', so
verify silently found zero findings after a trace-graph run. All producers and
consumers now share the prefixes from contractor.workflows.namespaces.
"""

from __future__ import annotations

import contractor.workflows.trace_annotation.workflow as ta
import contractor.workflows.trace_annotation_direct.workflow as tad
import contractor.workflows.trace_graph.workflow as tg
import contractor.workflows.trace_graph_pathpar.workflow as tgp
from contractor.workflows.namespaces import (
    TRACE_ANNOTATION_NAMESPACE_PREFIX,
    TRACE_GRAPH_NAMESPACE_PREFIX,
    TRACE_GRAPH_PATHPAR_NAMESPACE_PREFIX,
    TRACE_NAMESPACE_PREFIXES,
)


def test_prefix_values():
    assert TRACE_ANNOTATION_NAMESPACE_PREFIX == "trace-annotation"
    assert TRACE_GRAPH_NAMESPACE_PREFIX == "trace-graph"
    assert TRACE_GRAPH_PATHPAR_NAMESPACE_PREFIX == "trace-graph-pathpar"


def test_probe_tuple_covers_every_producer_prefix():
    assert TRACE_NAMESPACE_PREFIXES == (
        TRACE_ANNOTATION_NAMESPACE_PREFIX,
        TRACE_GRAPH_NAMESPACE_PREFIX,
        TRACE_GRAPH_PATHPAR_NAMESPACE_PREFIX,
    )


def test_producers_reference_the_shared_constants():
    # Each producer module must build its per-path namespace from the *same*
    # constant trace-verify probes with, so write and read keys cannot drift.
    assert ta.TRACE_ANNOTATION_NAMESPACE_PREFIX is TRACE_ANNOTATION_NAMESPACE_PREFIX
    assert tad.TRACE_ANNOTATION_NAMESPACE_PREFIX is TRACE_ANNOTATION_NAMESPACE_PREFIX
    assert tg.TRACE_GRAPH_NAMESPACE_PREFIX is TRACE_GRAPH_NAMESPACE_PREFIX
    assert tgp.PATH_NAMESPACE_PREFIX is TRACE_GRAPH_PATHPAR_NAMESPACE_PREFIX
