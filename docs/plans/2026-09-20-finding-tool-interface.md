# Finding authoring interfaces

Status: implemented in the working tree. On 2026-09-20 the user explicitly
rejected supporting old finding schemas while the application is in development.
The current contract is [specification 27](../spec/27-findings-tools-and-collections.md#finding-proposal-and-selected-authoring-interfaces).
No deployment or model campaign is part of this change.

## Current design

One proposal schema, `contractor.audit.finding-proposal.v1`, evolves in place.
There are no alternate schema readers, old authoring implementations, conversion
adapters or migration paths. The schema name is an identifier, not a promise to
support previous development shapes. Subject is a nullable typed field.

One publisher serves three explicitly selected interfaces. The model sees the
same tool name, `finding`; AgentTemplate selection fixes its schema before the
worker starts. Selecting conflicting creation interfaces is rejected.

| Selection | Required arguments | Optional location arguments |
| --- | --- | --- |
| `security-findings@1` | title, description | typed source/web locations |
| `security-findings-code@1` | title, description, file | line or inclusive range |
| `security-findings-http@1` | title, description, url, method | request_id |

All accept optional cwe, exact evidence_refs and typed standard_refs. Reading
through list_findings is independently selectable from security-findings@1.
The current source/HTTP templates select their matching facade in place.

```python
finding(title="Missing ownership check", description="Observed source behavior",
        file="shop/views.py", range={"start_line": 115, "end_line": 118}, cwe="CWE-639")

finding(title="Order disclosure", description="Observed behavior and auth context",
        url="https://app.example/orders/123", method="GET", request_id=42)
```

The general facade permits mixed or absent locations. No interface creates final
review decisions, infers exploitability or merges independently reported issues.

## Implementation choices

- Runtime hashes the ADK function-call identity into client_key; the intake's
  invocation namespace remains authoritative. Missing identity fails before
  submission. The returned key links a proposal into Audit check results.
- HTTP keeps its latest 128 outgoing exchanges in memory, with actual headers,
  body bytes and redirect/retry attempts. Lookup requires the same invocation.
  The model sees short history summaries; expired IDs give a repairable error.
- A selected exchange is copied into the finding proposal. Existing final
  response-body references remain exact evidence links. No outgoing artifact is
  created for each request. Authored URL/method remain separate from evidence.
- Optional CWE uses the bundled MITRE 4.20 catalog with source URL and SHA-256.
  The crAPI public-API adapter uses the same declared mapping.
- Location validation, request capture, retained HTTP evidence, submission,
  publisher and facades have separate modules. Named limits reference the owning
  specification and existing transport bounds.

The facade split follows existing source/HTTP responsibilities. Improved LLM
reliability is not yet measured. SARIF export and multiple selected HTTP exchange
IDs remain separate work.

## Historical comparison

The inspected branch was v.1.0 at
9c76b56cf7b83377fb1dd5e4a17440fa27b723f3. Its report_vulnerability tool required
eight arguments and stored a single unstructured file/URL place in mutable YAML
maps. Separate verification tools saved optional string request tags; a later
Caido collector resolved exchanges with heuristic fallbacks. These historical
implementations are comparison material only and are not supported by this code.
The old read_file option used disabled-by-default absolute `N | text` prefixes;
that display behavior is implemented in the current workspace reader.
