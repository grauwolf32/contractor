---
description: How the exploitability agent uses Caido workflows — convert/active/passive workflow recipes, the curated set worth using, and how to harvest passive findings. Use alongside the caido_workflow_* tools.
---

# Caido Workflows Skill

Caido **workflows** are reusable automations exposed to you through three
tools. They complement `caido_replay` / `caido_automate_run` — workflows are
canned transforms and checks, not a fuzzer.

## Caido vs http_request

Use `http_request` for one-off confirming probes and traffic you'll cite
directly. Route traffic through Caido (`caido_replay` / `caido_automate_run`)
when you want passive workflows to score it, when you need a convert recipe (PoC
generation, GraphQL introspection), or an active check (CORS). **IMPORTANT:**
passive findings only fire on traffic Caido actually proxied — `http_request`
traffic is invisible to Caido, so probes sent there will **not** appear in
`caido_workflow_findings`. Don't treat an empty findings list as "no issues" if
you never routed traffic through Caido.

## The three tools

- **`caido_workflow_list`** — discover what's installed. Returns each
  workflow's `id`, `name`, `kind`, `enabled`. **Ids are instance-specific**
  (e.g. `g:2` for built-ins, a uuid for installed ones), so always resolve a
  workflow by **name** via this tool before running it. The installed set
  grows over time — list, don't assume.
- **`caido_workflow_run`** — run a workflow:
  - `caido_workflow_run(workflow_id, input="<text>")` → **convert**: returns
    transformed `output`.
  - `caido_workflow_run(workflow_id, request_id="<id>")` → **active**: runs a
    check against a captured request (id from `caido_history`); returns a task
    id, and any findings surface via `caido_workflow_findings`.
- **`caido_workflow_findings`** — harvest findings raised by **passive**
  workflows on the traffic you've already sent. The `reporter` field says which
  workflow raised each one.

## When to use which kind

| Kind | How it runs | You use it to… |
|------|-------------|----------------|
| convert | on demand, on text | build an artifact (PoC, introspection query) or reshape a request payload |
| active | on demand, vs a request | run a one-shot check against a specific endpoint |
| passive | automatically, on all proxied traffic | get free signal — read it back with `caido_workflow_findings` |

## Curated set worth using

Resolve these by name with `caido_workflow_list`. If a workflow you want isn't
installed, note it in your summary and fall back to doing the step manually.

### Convert (run with `input=`)
- **Copy As Python Requests** — turn a raw HTTP request into a runnable
  `requests` PoC. Use when you've confirmed a vuln and want a reproduction
  script for the evidence.
- **GraphQL Introspection Query** — emits a full introspection query. Use when
  the target exposes a GraphQL endpoint and you need its schema.
- **nowafpls** — pads a request body to bypass naive WAFs. Use only when a
  probe is being blocked (403/406 from a WAF) and you suspect the payload, not
  the logic, is filtered.
- **JSON to XML** — reshape a JSON body to XML. Use for content-type confusion
  / XXE pivots on endpoints that accept both.
- **Clean HTTP Request** — strip noisy headers before replay/fuzzing.

### Active (run with `request_id=`)
- **CORS Checker** — actively probes a request for CORS misconfigurations.
  Use when assessing a CORS / cross-origin finding.

### Passive (enable once, then read via `caido_workflow_findings`)
These run on their own. Your job is to **call `caido_workflow_findings`** after
sending probe traffic and fold relevant findings into your assessment:
- **secret-sniffer**, **leakz** — secrets / API keys / PII in responses.
  Corroborate or expand a secret-disclosure finding against live responses.
- **url-in-param**, **redirect-to-param** — SSRF / open-redirect candidates.
- **findsso** — OAuth / OIDC / SAML flow detection, for auth-related findings.
- **content-length-mismatch** — request-smuggling / desync signal.

> Passive workflows must be **enabled** in the Caido instance to fire (one-time
> setup, not something you do per assessment). If `caido_workflow_findings`
> returns nothing, they may simply be disabled — don't block on it.

## Don't bother

- Plain encoders — **Base64/URL/HTML/Unicode encode-decode**, **JSON
  escape/unescape**. Do these inline yourself; round-tripping through Caido
  wastes a tool call.
- **AI Mentor** — it calls an external LLM to analyze a request. You are the
  analyst; skip it.

## Tying findings to evidence

A passive finding (e.g. a leaked key flagged by `secret-sniffer`) points at a
`request_id`. Pull the full exchange with `caido_request_detail(request_id)`,
confirm the signal is real (not a false positive), and cite the `request_tag`
of the proving exchange in your verdict's `request_ids` — same evidence rule as
the `exploit` skill.
