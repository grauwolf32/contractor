---
description: STRIDE threat-modeling skill — per-category signal catalog, artifact cross-correlation heuristics, object-reference/IDOR reasoning, per-dependency-tag threat classes, and severity ranking. Knowledge DB for the threat_analysis task; read references on demand via skills_read.
---

# STRIDE Threat Modeling Skill

Use this skill when modeling threats from a system's artifacts (LikeC4 model,
OpenAPI spec, project-information inventory, dependency inventory) plus its
source code. The goal is **fewer, well-supported, code-anchored threats**, each
ranked by reachability and impact — not a mechanical STRIDE checklist.

## Operating Rules

1. **Artifacts are the system map.** Cross-correlate them; read code only to
   confirm or refute a specific hypothesis. Never rebuild an artifact.
2. **Anchor every threat** to a concrete reference: `file:line`, an OpenAPI
   method+path, a LikeC4 element id, a config key, or a dependency name — plus
   the upstream artifact that surfaced it.
3. **Reachability is part of severity.** A threat behind strong auth and no
   reachable path is at most low severity unless justified.
4. **De-duplicate.** Extend an existing report when it covers the same asset +
   STRIDE category; do not raise a near-duplicate.
5. **Model only — never verify exploitability.** That is a downstream agent's job.

## The Six Categories (what to look for)

- **S — Spoofing** — identity/authentication weakness: missing/weak authn,
  forgeable tokens, hardcoded or fallback secrets (`JWT_SECRET ||= "dev"`),
  trusting client-supplied identity headers (`X-Admin`, `X-User-Id`), alg=none.
- **T — Tampering** — integrity of inputs/config/artifacts: unsigned webhooks,
  mass-assignment / unfiltered object binding, mutable trust state, missing
  HMAC/signature checks, TOCTOU.
- **R — Repudiation** — audit gaps: security-relevant actions not logged,
  attacker-controllable timestamps/log fields, no tamper-evident audit trail.
- **I — Information Disclosure** — sensitive data crossing to the wrong zone:
  endpoints returning PII/PAN/secrets, `security: []` endpoints exposing data,
  verbose errors, enumeration oracles (login/promo/reset reveal validity).
- **D — Denial of Service** — resource exhaustion: unbounded input, missing
  rate limits on auth/expensive paths, amplification, unpaginated queries.
- **E — Elevation of Privilege** — authz gaps: missing ownership checks
  (**IDOR/BOLA**), role confusion, function-level access-control gaps, anonymous
  access to admin/internal routes.

## Artifact Cross-Correlation Heuristics

Build the threat surface by intersecting artifacts — each pairing yields a
specific category to check:

| Signal (from artifacts) | Candidate threat(s) |
|---|---|
| LikeC4 trust-boundary crossing × OpenAPI endpoint | S / T / I / E on that endpoint |
| OpenAPI endpoint with `security: []` × sensitive response | I (info disclosure), E (anon access) |
| **Object-reference path param (`/x/{id}`) reaching a data read/write** | **E (IDOR/BOLA), I** — see below |
| Project-info crypto/auth location × LikeC4 trust zone | S / E (weak control at a boundary) |
| Project-info secrets handling × dependency `secrets`/`cryptography` | I / T (exposure, weak primitive) |
| LikeC4 external dep × dependency tag | category-specific (see tag catalog) |
| Auth/login/reset/promo endpoint | D (no rate limit), I (enumeration oracle) |

## Object-Reference (IDOR/BOLA) Discipline — high-yield, easy to miss

Every endpoint that takes an **object reference** (path/query param like
`{id}`, `{accountId}`, `{cardId}`, `{orderId}`, or a body field naming a
resource) is an IDOR/BOLA candidate. For **each** such endpoint:

- Treat **item-level** routes (`/accounts/{id}`, `/transfers/{id}/cancel`)
  separately from **collection** routes (`/accounts`) — they have distinct
  authz and are easy to overlook when only the collection is modeled.
- Confirm an **ownership/tenancy check** binds the referenced object to the
  caller. If absent → raise **E (IDOR)**; if it also returns sensitive data
  (PII/PAN/tokens) → also **I**.
- Enumerate these explicitly rather than collapsing them into one
  "IDOR on resources" report — one report per distinct object-reference surface
  so coverage is auditable.

## Per-Dependency-Tag Threat Classes

Each dependency tag implies a class to check on the flows that use it:

- `database` → injection (SQL/NoSQL), mass assignment, unbounded query (D).
- `queue` → message tampering, replay, poison-message (T/D).
- `cryptography` → weak primitive, ECB, static IV/nonce, custom crypto (I/S).
- `secrets` → hardcoded/fallback secret, secret in logs/responses, no rotation (I).
- `http` (outbound) → **SSRF**, unvalidated redirect, TLS verification disabled (I/T/E).
- `s3`/object-store → public bucket, predictable keys, missing ACL (I).
- `authentication` → session fixation, weak token, missing expiry, spoofing (S).
- `grpc`/IPC → missing authn between services, trust-on-first-use (S/T).

## Severity Ranking

Rank each threat by the product of three axes — state them in the report:

- **reachability**: anonymous > authenticated > internal-only.
- **blast radius**: whole system / all tenants > single tenant > single user.
- **detectability**: silent > logged-only > alerted.

A purely internal, single-user, alerted threat is low; an anonymous,
system-wide, silent one is critical.

## Report Shape (mandatory)

One vulnerability report per threat. Title `[<S|T|R|I|D|E>] <short title>`.
`details` must carry, in order: `category: threat`, `stride: <letter>`,
`asset`, `entry_point` (handler + file:line), `trust_boundary` (from LikeC4),
`dispatch_path`, `existing_controls`, `missing_controls`, `reachability`,
`artifact_refs` (likec4 / openapi / project_info / dependency), `rationale`.

## Coverage Stop Condition

Stop when every OpenAPI endpoint, every dependency tag, and every LikeC4
trust-boundary crossing has been considered at least once (each recorded as
"threat raised" or "no credible threat: <reason>"), and all credible threats
are persisted. Then produce final output — do not open new subtasks.
