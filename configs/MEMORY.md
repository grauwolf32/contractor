# Working catalog with shared Memory

Use the active selectors below for new Runs and Audits. These successors select
all six `memory-tools@1` operations explicitly. Exact legacy selectors remain
available for compatibility; they do not select Memory. Existing pinned Runs are
unchanged. Custom templates continue to opt in explicitly.

The machine-readable inventory is [memory-catalog.json](memory-catalog.json).
Successor Workflows select successor templates, and successor AuditProfiles select
those Workflows. `http_explorer` remains a reusable standalone Catalog template.
Version numbers reserved by the frozen instruction evaluation are skipped.

## AgentTemplate

| Legacy selector | Active selector |
| --- | --- |
| `artifact_builder@1` | [`artifact_builder@2`](agent-templates/artifact_builder_v2_memory.yaml) |
| `audit_asvs_source_verifier@1` | [`audit_asvs_source_verifier@3`](agent-templates/audit_asvs_source_verifier_v3_memory.yaml) |
| `audit_openapi_operation_tracer@1` | [`audit_openapi_operation_tracer@3`](agent-templates/audit_openapi_operation_tracer_v3_memory.yaml) |
| `audit_openapi_operation_tracer@2` | [`audit_openapi_operation_tracer@4`](agent-templates/audit_openapi_operation_tracer_v4_memory.yaml) |
| `audit_risk_source_checker@1` | [`audit_risk_source_checker@3`](agent-templates/audit_risk_source_checker_v3_memory.yaml) |
| `audit_source_checker@1` | [`audit_source_checker@3`](agent-templates/audit_source_checker_v3_memory.yaml) |
| `caido_analyst@1` | [`caido_analyst@3`](agent-templates/caido_analyst_v3_memory.yaml) |
| `findings_analyst@1` | [`findings_analyst@2`](agent-templates/findings_analyst_v2_memory.yaml) |
| `http_explorer@1` | [`http_explorer@3`](agent-templates/http_explorer_v3_memory.yaml) |
| `likec4_builder@2` | [`likec4_builder@4`](agent-templates/likec4_builder_v4_memory.yaml) |
| `likec4_validator@2` | [`likec4_validator@4`](agent-templates/likec4_validator_v4_memory.yaml) |
| `openapi_builder@1` | [`openapi_builder@3`](agent-templates/openapi_builder_v3_memory.yaml) |
| `openapi_validator@1` | [`openapi_validator@3`](agent-templates/openapi_validator_v3_memory.yaml) |
| `podman_python_fixer@1` | [`podman_python_fixer@2`](agent-templates/podman_python_fixer_v2_memory.yaml) |
| `workspace_likec4_builder@1` | [`workspace_likec4_builder@3`](agent-templates/workspace_likec4_builder_v3_memory.yaml) |
| `workspace_likec4_validator@1` | [`workspace_likec4_validator@3`](agent-templates/workspace_likec4_validator_v3_memory.yaml) |
| `workspace_openapi_builder@1` | [`workspace_openapi_builder@3`](agent-templates/workspace_openapi_builder_v3_memory.yaml) |
| `workspace_openapi_validator@1` | [`workspace_openapi_validator@3`](agent-templates/workspace_openapi_validator_v3_memory.yaml) |
| `workspace_source_graph_analyst@1` | [`workspace_source_graph_analyst@3`](agent-templates/workspace_source_graph_analyst_v3_memory.yaml) |
| `workspace_taint_analyst@1` | [`workspace_taint_analyst@3`](agent-templates/workspace_taint_analyst_v3_memory.yaml) |

## Workflow

| Legacy selector | Active selector |
| --- | --- |
| `artifact-copy@1` | [`artifact-copy@2`](workflows/artifact_copy_v2_memory.yaml) |
| `audit-asvs-source-verification@1` | [`audit-asvs-source-verification@3`](workflows/audit_asvs_source_verification_v3_memory.yaml) |
| `audit-openapi-operation-trace@1` | [`audit-openapi-operation-trace@3`](workflows/audit_openapi_operation_trace_v3_memory.yaml) |
| `audit-openapi-operation-trace@2` | [`audit-openapi-operation-trace@4`](workflows/audit_openapi_operation_trace_v4_memory.yaml) |
| `audit-source-check@1` | [`audit-source-check@3`](workflows/audit_source_check_v3_memory.yaml) |
| `audit-top10-source-risk@1` | [`audit-top10-source-risk@3`](workflows/audit_top10_source_risk_v3_memory.yaml) |
| `findings-review@1` | [`findings-review@2`](workflows/findings_review_v2_memory.yaml) |
| `likec4-from-analysis@3` | [`likec4-from-analysis@5`](workflows/likec4_from_analysis_v5_memory.yaml) |
| `likec4-from-workspace@5` | [`likec4-from-workspace@7`](workflows/likec4_from_workspace_v7_memory.yaml) |
| `likec4-from-workspace-streamline@2` | [`likec4-from-workspace-streamline@4`](workflows/likec4_from_workspace_streamline_v4_memory.yaml) |
| `openapi-from-analysis@2` | [`openapi-from-analysis@4`](workflows/openapi_from_analysis_v4_memory.yaml) |
| `openapi-from-workspace@5` | [`openapi-from-workspace@7`](workflows/openapi_from_workspace_v7_memory.yaml) |
| `openapi-from-workspace-streamline@1` | [`openapi-from-workspace-streamline@2`](workflows/openapi_from_workspace_streamline_v2_memory.yaml) |
| `podman-python-check@1` | [`podman-python-check@2`](workflows/podman_python_check_v2_memory.yaml) |
| `security-analysis@2` | [`security-analysis@4`](workflows/security_analysis_v4_memory.yaml) |
| `taint-trace-from-workspace@2` | [`taint-trace-from-workspace@4`](workflows/taint_trace_from_workspace_v4_memory.yaml) |

## AuditProfile

| Legacy selector | Active selector |
| --- | --- |
| `openapi-operation-trace@2` | [`openapi-operation-trace@4`](audit-profiles/openapi_operation_trace_v4_memory.yaml) |
| `openapi-operation-trace@3` | [`openapi-operation-trace@5`](audit-profiles/openapi_operation_trace_v5_memory.yaml) |
| `owasp-asvs-5-0-l1-source-review@1` | [`owasp-asvs-5-0-l1-source-review@2`](audit-profiles/owasp_asvs_5_0_l1_source_review_v2_memory.yaml) |
| `owasp-top10-2025-source-risk@1` | [`owasp-top10-2025-source-risk@2`](audit-profiles/owasp_top10_2025_source_risk_v2_memory.yaml) |
| `source-checklist@1` | [`source-checklist@2`](audit-profiles/source_checklist_v2_memory.yaml) |

## Lifetime and completion

Notes are shared only within a Run and resolved Agent Namespace, including retries
and later Stages. A new Run starts empty. Namespaces remain isolated unless their
existing bindings intentionally share one. Instructions explain discovery, useful
coordination writes, untrusted note content, quotas and rereading after conflicts.
Memory uses ordinary budgets and never substitutes for artifacts, findings, or
Audit completion. Streamline and Router mirror selected operations; passthrough
retains its direct dispatch without a synthetic Planner loop.

Copyable examples: [Router](examples/router_openapi_workflow_memory.yaml) and
[Streamline](examples/streamline_review_workflow_memory.yaml).
