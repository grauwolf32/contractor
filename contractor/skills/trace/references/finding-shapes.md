---
description: Finding shape A/B/C taxonomy and required fields for the trace skill. Each finding must match exactly one shape. Reaching a sink alone is NOT a vulnerability.
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
must be visible in the code.

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
once with a deterministic slug (e.g., file path + symbol).

## Required fields per finding

Every finding (in the §7 report or via `report_vulnerability`):

  shape:                A | B | C
  control_missing:      one of {auth, authz, signature_verify,
      expiry_check, role_check, ownership_check, csrf,
      path_confinement, input_validation, output_filter,
      secret_storage, cookie_flags, tls, rate_limit, other}
  evidence_lines:       file:line, file:line, ...
  exploit_precondition: e.g. unauthenticated | any authenticated user |
                        knows victim id | can set header X
  severity:             low | medium | high | critical
  confidence:           low | medium | high
  title:                short title
  summary:              one-sentence summary

Optional (Shape A):
  source / path / sink — the source, transformation chain, and sink

## Reporting template

```
- shape:                <A|B|C>
  control_missing:      <closed-set tag>
  evidence_lines:       <file:line, ...>
  exploit_precondition: <preconditions>
  severity:             <low|medium|high|critical>
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
  to the sink
- the control is present and effective on this handler
- the suspicion is based on naming alone, not on visible code behavior

If no vulnerability is confidently supported by the code, do not report
one.
