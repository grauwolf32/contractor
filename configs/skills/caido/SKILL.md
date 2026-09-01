---
name: caido
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

- `caido_scope(action="list")` lists bounded scopes.
  `caido_scope(action="create", name=..., allowlist=..., denylist=...)` creates
  one scope with at most 256 allow/deny terms in total.
- `caido_history(filter="", limit=20, offset=0)` reads one newest-first page
  using bounded HTTPQL.
- `caido_request_detail(request_id)` returns metadata, bounded raw previews and
  an exact exchange artifact when raw bytes exist.
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
  reads a bounded result page. Omit `entry_id` to select the newest entry.
- `caido_sitemap(parent_id="", scope_id="", depth="DIRECT")` browses a
  bounded root or descendant set; depth is `DIRECT` or `ALL`.
- `caido_workflow_list(kind="")` lists installed workflows. IDs are
  instance-specific: resolve by name each time rather than assuming an ID.
- `caido_workflow_run(workflow_id, input="...")` runs a convert workflow;
  `caido_workflow_run(workflow_id, request_id="...")` starts an active
  workflow. Supply exactly one input form. Large or binary convert output is an
  exact `caido.output.*` artifact plus a bounded preview.
- `caido_workflow_findings(limit=20, offset=0)` reads newest findings raised by
  workflows. A finding's `request_id` links back to its captured exchange.

## Practical sequence

1. Use `caido_scope` and `caido_history` to establish the authorized baseline.
2. Inspect only relevant candidates with `caido_request_detail`.
3. Use `caido_replay` for a single confirming change. Preserve its
   `request_tag`, response status and exact exchange artifact.
4. Use `caido_automate_run` only for a genuine multi-payload question, then
   inspect bounded pages with `caido_automate_results`.
5. Resolve optional recipes with `caido_workflow_list`. Convert workflows
   transform text; active workflows operate on one captured request; passive
   workflows analyze traffic already seen by Caido.
6. Read `caido_workflow_findings`, correlate relevant request IDs, and report
   evidence plus limitations. An empty list is not proof of absence because a
   workflow may be missing, disabled, or may not have observed the traffic.

Useful recipes commonly include Copy As Python Requests, GraphQL Introspection
Query, Clean HTTP Request and CORS Checker. Names and availability vary. Prefer
ordinary reasoning for simple encoding and decoding, and do not delegate the
assessment to an external AI workflow.
