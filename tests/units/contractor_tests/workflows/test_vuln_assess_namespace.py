"""Regression test for the audit HIGH finding: vuln_assess read the vuln-report
namespace under a 'trace-annotation:' prefix while the trace stage
(TraceGraphPathParWorkflow) wrote it under 'trace-graph-pathpar:', so the
exploit stage never saw any findings. Both now share PATH_NAMESPACE_PREFIX.
"""

from __future__ import annotations

import contractor.workflows.vuln_assess.workflow as va
from contractor.workflows.trace_graph_pathpar.workflow import PATH_NAMESPACE_PREFIX


def test_prefix_value():
    assert PATH_NAMESPACE_PREFIX == "trace-graph-pathpar"


def test_collector_and_trace_stage_share_one_prefix():
    # vuln_assess must reference the *same* constant the trace stage writes with,
    # so the read key and write key cannot drift apart again.
    assert va.PATH_NAMESPACE_PREFIX is PATH_NAMESPACE_PREFIX


def test_write_read_keys_match():
    ns, path_key = "openapi", "api_users_user_id"
    write_key = f"user:vulnerability-reports/{PATH_NAMESPACE_PREFIX}:{ns}:{path_key}"
    read_key = f"user:vulnerability-reports/{va.PATH_NAMESPACE_PREFIX}:{ns}:{path_key}"
    assert write_key == read_key
