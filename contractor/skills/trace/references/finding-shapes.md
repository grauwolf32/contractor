---
description: Finding shape A/B/C taxonomy, required fields, severity/confidence heuristics, and `report_vulnerability` field mapping for the trace skill. Each finding must match exactly one shape. Reaching a sink alone is NOT a vulnerability.
---

# Finding Shapes (A / B / C)

A finding must match exactly ONE shape. Shapes B and C are first-class —
do NOT force a taint→sink narrative when the bug is a missing control or
an at-rest/in-transit defect.

## Shape A — Tainted flow into a sink

Mechanic: tainted/derived input → optional transformations → sink, AND
the input controls structure / path / command (not just a value), AND a
control that would block exploitation is missing or weak.

Examples:
- request path → filesystem.read with no path confinement
- request field → shell.exec via string concatenation
- request URL → http.request with no outbound allowlist
- request column name → db.query.raw via concatenation

Reaching a sink alone is NOT Shape A. The structural-control mechanic
must be visible in the code. Driver-bound placeholders are NOT Shape A.

## Shape B — Sensitive operation reachable without required control

Mechanic: a sensitive operation (admin action, write to another user's
data, token issuance, credential check) is reachable without a control
that would normally block it.

Source rows: `absent` / `weak` entries in the per-handler control
checklist (see `trace/references/controls`).

Possible missing controls: auth, authz, signature_verify, expiry_check,
role_check, ownership_check, csrf, rate_limit.

## Shape C — Sensitive value at rest / in transit / in response without protection

Mechanic: a sensitive value is exposed without the protection it
requires.

Examples:
- plaintext credentials in storage
- hardcoded secrets in source
- cookie missing HttpOnly / Secure / SameSite
- sensitive fields returned unfiltered (password, apiKey, token,
  paymentMemo, internal flags)
- secrets written to logs
- weak randomness used for security-relevant values
- TLS verification disabled on outbound calls

Defects on opened files outside the traced flow are Shape C — report
once with a deterministic slug (file path + symbol).

## Required fields per finding (§7 report)

  shape:                A | B | C
  control_missing:      one of {auth, authz, signature_verify,
      expiry_check, role_check, ownership_check, csrf,
      path_confinement, input_validation, output_filter,
      secret_storage, cookie_flags, tls, rate_limit, other}
  evidence_lines:       file:line, file:line, ...
  exploit_precondition: e.g. unauthenticated | any authenticated user |
                        knows victim id | can set header X
  severity:             info | low | medium | high | critical
  confidence:           low | medium | high
  title:                short title
  summary:              one-sentence summary

Optional (Shape A):
  source / path / sink — the source, transformation chain, and sink

## Severity heuristics

Pick the lowest tier that matches the worst plausible outcome on this
handler, given the exploit precondition you recorded.

  critical — unauth RCE, unauth auth bypass, raw SQLi on auth/admin path,
             arbitrary file read of secrets, account takeover by anyone
  high     — authenticated RCE, IDOR exposing other users' PII or
             secrets, SSRF reaching internal/metadata, privileged action
             without authz, password/token reset bypass
  medium   — IDOR exposing non-PII business data, open redirect, stored
             XSS in an authenticated-only surface, weak crypto on
             session/token, missing CSRF on state-changing endpoint
  low      — reflected XSS gated behind auth, log of low-sensitivity
             secret, cookie missing one flag where transport already
             mitigates, rate-limit gap on a non-sensitive operation
  info     — defense-in-depth gap with no current impact path; finding
             the user explicitly asked you to record without judgment

## Confidence heuristics

  high   — every line in `evidence_lines` is on the actually-traced
           path; sink is labeled exactly per `trace/references/sinks`;
           the missing-control row is observable in code (not inferred
           from the absence of a search hit)
  medium — main mechanic is supported but one link is inferred (e.g.,
           framework default behavior, decorator semantics not opened)
  low    — relies on a project-wide convention not directly verified on
           this handler, or a reachability claim you could not confirm

If you would be embarrassed to defend the finding given only
`evidence_lines`, downgrade or do not report.

## Mapping §7 fields → `report_vulnerability` tool

The `report_vulnerability` tool stores: `name`, `place_type`, `place`,
`title`, `summary`, `severity`, `confidence`, `details`. Map as follows.

  name        — deterministic slug, stable across re-runs.
                Shape A: `<sink_kind>-<entrypoint_symbol>` (e.g.
                  `db.query.raw-getOrder`)
                Shape B: `<control_missing>-<handler_symbol>` (e.g.
                  `authz-deleteUser`)
                Shape C: `<defect_class>-<file_basename>-<symbol>`
                  (e.g. `hardcoded-secret-config-API_KEY`)
                Lowercase; `[a-z0-9._-]` only; no spaces.
  place_type  — `file` for code findings; `url` only when reporting
                against a deployed endpoint URL.
  place       — primary file path (entrypoint for A/B; defect site for C).
                Use the same path style as the source tree (relative).
  title       — short, ≤ 80 chars, no trailing period.
  summary     — one sentence; what + where + impact, no remediation.
  severity    — see heuristics above.
  confidence  — see heuristics above.
  details     — markdown; pack the §7 structured fields here:
                  - shape, control_missing, exploit_precondition
                  - evidence_lines (every cited file:line)
                  - source / path / sink (Shape A only)
                  - short reproduction or reasoning paragraph

`details` is the only freeform field — it must remain self-contained so
the finding can be triaged without re-reading source.

## Reporting template (§7)

```
- shape:                <A|B|C>
  control_missing:      <closed-set tag>
  evidence_lines:       <file:line, ...>
  exploit_precondition: <preconditions>
  severity:             <info|low|medium|high|critical>
  confidence:           <low|medium|high>
  title:                <short title>
  summary:              <one sentence>
  # Shape A only:
  source:               <where input enters>
  path:                 <fn A → fn B → fn C>
  sink:                 <kind + arg>
```

## When NOT to report

- the code path you traced does not actually carry user-controlled data
  to the sink (e.g., the "tainted" arg is a server-issued id)
- the control is present and effective on this handler (verified, not
  assumed from middleware presence elsewhere)
- the suspicion is based on naming alone (`exec_query` is not by itself
  a sink label)
- a "missing" control is supplied by a decorator/middleware you did not
  open; open it before reporting `absent`
- the structural-control mechanic (Shape A) requires inferring framework
  behavior you have not seen — downgrade to Shape B (input_validation
  absent) or omit
- a sibling-handler comparison would have shown the control IS applied
  consistently — avoid duplicate reports on the same defect class

If no vulnerability is confidently supported by the code, do not report
one. An empty finding list is a valid result.
