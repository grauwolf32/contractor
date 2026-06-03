---
description: Finding shape A/B/C/D taxonomy (taint→sink, missing control, at-rest/response exposure, business-logic invariant break), required fields, severity/confidence heuristics, and `report_vulnerability` field mapping for the trace skill. Each finding must match exactly one shape. Reaching a sink alone is NOT a vulnerability.
---

# Finding Shapes (A / B / C / D)

A finding must match exactly ONE shape. Shapes B, C, and D are
first-class — do NOT force a taint→sink narrative when the bug is a
missing control, an at-rest/in-transit defect, or a violated
application invariant.

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

**Reportable from structure alone.** Shape B does not need a tainted-data
flow OR a demonstrated exploit. A sensitive operation reachable without a
dominating control IS the finding — report it on the visible code. Common
miss: a **self-service registration / profile-update handler that assigns
or can set a role / capability / admin flag from request input without an
authorization check** → privilege escalation (CWE-269), even though no
"input → sink" taint exists. Do not stay silent because you couldn't run it.

## Shape C — Sensitive value at rest / in transit / in response without protection

Mechanic: a sensitive value is exposed without the protection it
requires.

Examples:
- plaintext credentials in storage (passwords stored without hashing)
- hardcoded secrets in source (signing keys, API keys)
- hardcoded user credentials in database seed/init functions
  (e.g. `User(username="admin", password="admin123")` in seed data)
- cookie missing HttpOnly / Secure / SameSite
- sensitive fields returned unfiltered (password, apiKey, token,
  paymentMemo, internal flags)
- validation/schema error messages returned verbatim to clients
  (leaks field names, types, and internal structure)
- secrets written to logs
- weak randomness used for security-relevant values
- TLS verification disabled on outbound calls
- CORS reflects the request Origin into `Access-Control-Allow-Origin`, or
  uses `*` together with `Allow-Credentials: true` → Origin Validation
  Error (CWE-346)

Hardcoded-secret fingerprints (any string literal matching → Shape C,
CWE-798, even without a suggestive variable name):

| Pattern                          | Issuer / kind        |
|----------------------------------|----------------------|
| `AKIA…` / `ASIA…`                | AWS access key       |
| `ghp_…` / `github_pat_…`         | GitHub token         |
| `glpat-…`                        | GitLab token         |
| `AIza…`                          | Google API key       |
| `sk-ant-api03-…`                 | Anthropic key        |
| `sk-…T3BlbkFJ…`                  | OpenAI key           |
| `xoxb-` / `xoxp-`                | Slack token          |
| `sk_live_` / `rk_live_`          | Stripe key           |
| `SG.`                            | SendGrid key         |
| `npm_…`                          | npm token            |
| `-----BEGIN … PRIVATE KEY-----`  | private key (PEM)    |
| `ey….ey….`                       | JWT                  |

FP guard: a `*_test_*` Stripe key or an obvious placeholder/example value
(e.g. `sk_test_…`, `XXXX`, `your-key-here`) is low/info, not high.

Defects on opened files outside the traced flow are Shape C — report
once with a deterministic slug (file path + symbol).

### Shape C detection sweep (do this on every handler that returns or stores data)

Shape C is the most-missed shape because nothing "flows to a sink" — you
must actively look. Two passes:

1. **Response sweep.** For each value the handler returns, enumerate the
   *fields actually serialized* (follow the ORM model / struct / dict, not
   just the variable name). Returning a whole domain object (`return user`,
   `jsonify(row)`, `Model.objects...` serialized whole) leaks every field
   it has. Ask of each field: is it a credential, token, hash, secret,
   internal flag, audit/owner id, full PAN, email/phone, or other PII? Any
   sensitive field with no explicit projection/DTO excluding it → Shape C
   (`output_filter`). A sibling handler that DOES project the same object
   confirms the gap.
2. **At-rest / storage sweep.** For each value the handler persists, ask:
   is a credential stored without a slow KDF+salt; is a card number/SSN/
   token stored in cleartext or recoverable form; is a secret written to a
   log/metric/trace; is a security-relevant value (token, OTP, reset code,
   session id) generated with non-CSPRNG randomness? Each → Shape C
   (`secret_storage` / `secret.log` / weak-randomness).

Record these even when every access-control row is `present` — a perfectly
authorized response can still over-expose.

## Shape D — Business-logic / abuse of functionality

Mechanic: a state-changing operation violates an **application invariant**
even though the per-handler controls (auth, authz, validation) are all
present. The defect is in WHAT the operation permits, not in a tainted
sink or a missing checklist row. No taint flow and no `absent` control row
is required — report on the visible code by naming the invariant and the
sequence that breaks it.

Look for Shape D whenever the handler touches a resource whose correctness
depends on rules a framework cannot enforce: money, balances, inventory/
quantity, credits/points/votes, quotas/limits, ownership transfer, or a
multi-step workflow.

Detection heuristics (each is a Shape D candidate):

- **Missing idempotency / replay** — a charge / transfer / redeem / submit
  applies cumulatively when the same request is sent twice (no idempotency
  key, nonce, or once-only guard) → double-spend / replay.
  → `control_missing: idempotency`
- **Check-then-act without atomicity (TOCTOU / race)** — balance / quota /
  stock is read, validated, then mutated in separate steps with no lock,
  transaction, `SELECT … FOR UPDATE`, or atomic decrement → concurrent
  requests pass the check together (negative balance, over-redeem,
  oversell). → `control_missing: atomicity`
- **Unbounded / unsigned amount** — quantity/amount/price not constrained
  for sign and range → a negative amount reverses the transfer; a huge one
  overflows or drains. → `control_missing: input_validation` (Shape D when
  the impact is an invariant break, not a sink)
- **Client-trusted value** — price, total, discount, role, status, balance,
  or another user's id taken from the request body and used as truth
  instead of being recomputed / looked up server-side → tampering.
  → `control_missing: business_logic`
- **Workflow / state-machine bypass** — a step that must follow a prior one
  (pay before ship, verify before activate, 2FA before sensitive action) is
  independently reachable or can be reordered → `control_missing: state_machine`
- **Quota / limit not enforced server-side** — a per-user cap (trials,
  invites, votes, withdrawals) is enforced optimistically/in the client but
  not on the server. → `control_missing: business_logic`

Boundary with other shapes: if the root cause is a missing cryptographic
signature/nonce on a *credential or token*, prefer Shape C/`signature_verify`;
if it is a missing application-level once-only/atomicity guard on an
*operation*, it is Shape D. If tainted input controls a sink, that is
Shape A — Shape D is for invariant breaks that need no taint.

## Required fields per finding (§7 report)

  shape:                A | B | C | D
  control_missing:      one of {auth, authz, signature_verify,
      expiry_check, role_check, ownership_check, csrf,
      path_confinement, input_validation, output_filter,
      secret_storage, cookie_flags, tls, rate_limit,
      idempotency, atomicity, state_machine, business_logic, other}
  evidence_lines:       file:line, file:line, ...
  exploit_precondition: e.g. unauthenticated | any authenticated user |
                        knows victim id | can set header X
  severity:             info | low | medium | high | critical
  confidence:           low | medium | high
  title:                short title
  summary:              one-sentence summary

Optional (Shape A):
  source / path / sink — the source, transformation chain, and sink

Optional (all shapes) — the witness, for the downstream verifier:
  exploit_hypothesis:   a single concrete witness a verifier could check
                        WITHOUT you firing it — the entry input and the
                        observable it should produce. State it as
                        `<entry: METHOD path + input> ⇒ <expected observable>`,
                        e.g. `POST /orders {"sort":"id;DROP--"} ⇒ 500 with SQL
                        parse error` or `GET /users/2 as user 1 ⇒ 200 returning
                        user 2's email`. Generate it from visible code; do not
                        execute it. This is the proof obligation the
                        trace_verifier / exploitability agent will discharge.

## Severity heuristics

Pick the lowest tier that matches the worst plausible outcome on this
handler, given the exploit precondition you recorded.

  critical — unauth RCE, unauth auth bypass, raw SQLi on auth/admin path,
             arbitrary file read of secrets, account takeover by anyone
  high     — authenticated RCE, IDOR exposing other users' PII or
             secrets, SSRF reaching internal/metadata, privileged action
             without authz, password/token reset bypass, financial
             invariant break by any user (double-spend, negative transfer,
             balance/quota race), cleartext storage of credentials/PAN
  medium   — IDOR exposing non-PII business data, open redirect, stored
             XSS in an authenticated-only surface, weak crypto on
             session/token, missing CSRF on state-changing endpoint,
             workflow-step bypass or per-user limit not enforced
             server-side, sensitive field over-exposed in an
             authenticated-only response
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
                Shape D: `<control_missing>-<handler_symbol>` (e.g.
                  `atomicity-transfer`, `idempotency-redeemCoupon`)
                Lowercase; `[a-z0-9._-]` only; no spaces.
  place_type  — `file` for code findings; `url` only when reporting
                against a deployed endpoint URL.
  place       — the source file containing the vulnerable function or
                defect (entrypoint handler file for A/B/D; defect site for
                C). NEVER report against spec files (OpenAPI YAML, proto,
                swagger), documentation, or configuration that merely
                describes the endpoint — find and cite the source function.
                Use the same path style as the source tree (relative).
  title       — short, ≤ 80 chars, no trailing period.
  summary     — one sentence; what + where + impact, no remediation.
  severity    — see heuristics above.
  confidence  — see heuristics above.
  details     — markdown; pack the §7 structured fields here:
                  - **CWE ID** (e.g. CWE-89 for SQL injection; load
                    `cwe-mapping` reference if unsure)
                  - shape, control_missing, exploit_precondition
                  - evidence_lines (every cited file:line)
                  - source / path / sink (Shape A only)
                  - violated invariant + violating sequence (Shape D only)
                  - short reproduction or reasoning paragraph

`details` is the only freeform field — it must remain self-contained so
the finding can be triaged without re-reading source.

## Reporting template (§7)

```
- shape:                <A|B|C|D>
  control_missing:      <closed-set tag>
  evidence_lines:       <file:line, ...>
  exploit_precondition: <preconditions>
  severity:             <info|low|medium|high|critical>
  confidence:           <low|medium|high>
  title:                <short title>
  summary:              <one sentence>
  exploit_hypothesis:   <entry input ⇒ expected observable>   # optional witness
  # Shape A only:
  source:               <where input enters>
  path:                 <fn A → fn B → fn C>
  sink:                 <kind + arg>
  # Shape D only:
  invariant:            <rule that must hold, e.g. "balance never negative">
  violating_sequence:   <e.g. two concurrent POST /transfer on one balance>
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
