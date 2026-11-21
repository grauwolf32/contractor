---
name: caido-stored-evidence
description: "Use visible bounded Caido and HTTP operations for authorized replay, fuzzing, workflow, discovery, and evidence-review procedures."
compatibility: "Contractor adk@1 native Agent Skill disclosure; requires explicit authorized-target operations in the current Worker invocation"
metadata:
  source-revision: 9c76b56cf7b83377fb1dd5e4a17440fa27b723f3
---

# Caido analysis

Use this guidance only through the exact operations visible in the current
Worker invocation. The skill does not add an operation, establish network
authority, or imply that a particular Caido workflow is installed or enabled.

## Safety and scope

- Confirm explicit authorization, the exact target boundary, permitted test
  intensity, and stop conditions before sending traffic.
- Treat `caido_scope` as an organizational filter, not an access-control
  boundary. Never broaden authorization from hosts found in history or sitemap.
- Prefer the smallest non-destructive probe, controlled identities and data,
  and a baseline/probe comparison. Stop after the predicted signal is proven.
- A timeout or lost response has an unknown remote outcome. Inspect state before
  making another explicit mutation call; do not blindly repeat it.
- Passive findings are hypotheses. Corroborate each relevant item against an
  authorized exchange before reporting it.

## HTTP traffic and Caido traffic

Use `http_request` for a bounded direct probe and `http_read_body` for a bounded
slice of its retained body. `http_history` summarizes requests made by that
allocation. Deployment routing determines whether such traffic traverses
Caido, so do not assume it will appear in `caido_history`.

Use `caido_replay` when traffic must definitely pass through Caido for passive
analysis or when starting from a captured request. Every replay has a fresh
`request_tag`. Correlate that tag in `caido_history` when needed.

## Available Caido operations

- `caido_scope(action="list", limit=100, offset=0)` reads one page of scopes.
  Read `total_scopes` and `truncated`; when truncated, list again with
  `offset=next_offset`. `caido_scope(action="create", name=..., allowlist=...,
  denylist=...)` creates one scope with at most 256 allow/deny terms in total.
- `caido_history(filter="", limit=20, offset=0)` reads one newest-first page
  using bounded HTTPQL.
- `caido_request_detail(request_id)` returns metadata and bounded raw previews,
  retaining the exchange in this invocation's recent buffer for finding selection.
- `caido_replay(...)` accepts exactly one source: `request_id`, or
  `raw_request` plus `host` and optional port/TLS. `wait=false` returns after
  start; a `timeout` result ends only local observation and does not mean Caido
  stopped the task.
- `caido_automate_run(request_id, targets, payloads, strategy="ALL",
  workers=5, delay_ms=0)` starts one fuzz task. Use 1..32 unique non-overlapping
  target strings present in the captured raw bytes, at most 1000 payloads and
  at most 1 MiB of payload text in total. Placeholder locations are calculated
  on the exact tagged UTF-8 request sent to Caido.
- `caido_automate_results(session_id, entry_id="", limit=50, offset=0, ...)`
  reads a bounded result page. Omit `entry_id` to select the session's newest
  entry; an `entry_id` from another session is `not_found`.
- `caido_sitemap(parent_id="", scope_id="", depth="DIRECT", cursor="", limit=100)`
  starts a root or descendant selection; depth is `DIRECT` or `ALL`. Read
  `total_entries`, `offset` and `truncated`. When truncated, continue with
  `caido_sitemap(cursor=next_cursor)` alone; pages contain at most 100 entries
  and may be shorter for large labels. A new selection or refresh invalidates
  earlier cursors. If the whole provider response is too large, narrow the
  parent or scope.
- `caido_workflow_list(kind="", limit=100, offset=0)` reads one page of
  installed workflows. When `truncated`, call again with the same `kind` and
  `offset=next_offset` before concluding a workflow is missing. IDs are
  instance-specific: resolve by name each time rather than assuming an ID.
- `caido_workflow_run(workflow_id, input="...")` runs a convert workflow;
  `caido_workflow_run(workflow_id, request_id=42)` starts an active
  workflow. Supply exactly one input form. Large or binary convert output is an
  exact `caido.output.*` artifact plus a bounded preview.
- `caido_workflow_findings(limit=20, offset=0)` reads newest findings raised by
  workflows. A finding's `request_id` links back to its captured exchange.

## Practical sequence

1. Use `caido_scope` and `caido_history` to establish the authorized baseline.
2. Inspect only relevant candidates with `caido_request_detail`.
3. Use `caido_replay` for a single confirming change. Preserve its
   `request_tag`, request ID and response status.
4. Use `caido_automate_run` only for a genuine multi-payload question, then
   inspect bounded pages with `caido_automate_results`.
5. Resolve optional recipes with `caido_workflow_list`. Convert workflows
   transform text; active workflows operate on one captured request; passive
   workflows analyze traffic already seen by Caido.
6. Read `caido_workflow_findings`, correlate relevant request IDs, and report
   evidence plus limitations. Record relevant observations through `finding`,
   selecting the local integer `request_id` from an authorized history or request
   result. Runtime reads and saves the stored request and available response and
   returns exact evidence refs. Calling detail first is optional for analysis.
   Ordinary tool calls do not save HTTP artifacts. The recent shared buffer holds
   at most 128 exchanges and 64 MiB. If an ID expired, select a retained source
   from history; never substitute a provider ID. Do not create
   a finding solely to preserve routine traffic; that traffic remains in Caido
   when routed through it. An empty finding list is not proof of absence because a
   workflow may be missing, disabled, or may not have observed the traffic.

Useful recipes commonly include Copy As Python Requests, GraphQL Introspection
Query, Clean HTTP Request and CORS Checker. Names and availability vary. Prefer
ordinary reasoning for simple encoding and decoding, and do not delegate the
assessment to an external AI workflow.

Request selectors use local integer request_id values returned by the tools. HTTP
and Caido share this local space; Caido actions require a Caido-backed ID. Provider
request IDs are internal. History discovery alone does not capture evidence: inspect
the exchange in the current invocation before publishing it.
