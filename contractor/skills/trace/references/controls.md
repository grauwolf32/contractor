---
description: Per-handler control checklist for the trace skill. Walk this list before reporting; each absent/weak row on a sensitive operation is a Shape B candidate.
---

# Per-Handler Control Checklist

Walk this list for the assigned handler/route/operation/job/consumer.
A control on a sibling handler, nearby route, or similarly named
function does not count unless the current target reaches it.

## The checklist

| control            | what to look for                                         |
| ------------------ | -------------------------------------------------------- |
| authentication     | session/token validation before reaching handler logic   |
| authorization      | caller permitted on THIS resource (role/policy check)    |
| ownership/scoping  | resource id belongs to the caller (subject == owner)     |
| input_validation   | schema/type/format check on each request-derived arg     |
| output_filter      | domain object passed through a projection/sanitizer      |
| rate_limit / csrf  | applicable to write/state-changing operations            |
| cors               | response ACAO not reflected/`*`-with-credentials          |

## Status semantics

- `present (file:line)` — control is implemented; record where
- `absent`              — control is missing entirely on this handler
- `weak`                — control exists but is bypassable, partial, or
                          does not enforce the intended invariant
- `N/A`                 — control does not apply (e.g. CSRF on a
                          pure-read endpoint that uses bearer tokens)

When using `weak`, include a short reason in parentheses. Examples:

  weak (role checked, owner not)
  weak (JWT decoded but exp not checked)
  weak (schema validates type but not allowed field names)

## Evaluation order — where to look before marking a row `absent`

1. route registration
2. router group / blueprint / module
3. middleware chain
4. decorators / annotations on the handler
5. framework guards (e.g. policy attributes)
6. schema validators
7. service-layer policy helpers
8. repository scoping filters
9. response serializers / projections

A function named `authorize`, `secure`, `safe`, `sanitize`, or `verify`
counts only after its implementation is opened or otherwise visible.

## Per-control guidance

### authentication

`present` when visible code verifies caller identity before sensitive
logic — verified session lookup, JWT verification with signature+expiry,
API key lookup against trusted server-side storage, or framework guard
whose implementation/config is visible.

`weak` when:
- token is decoded but not cryptographically verified
- expiry/issuer/audience required by design but unchecked
- authentication is optional on a path that exposes private data
- user id is trusted from a request field/header without verification

### authorization

`present` when visible code proves the authenticated caller has
permission for this action and resource.

`weak` when:
- role checked against the wrong resource
- admin check guards a tenant-scoped operation but tenant permission is
  not checked
- policy helper is called with caller-controlled subject/resource
- permission check happens after the sensitive operation
- authorization is performed on a parent object but the child object is
  independently addressable

### ownership/scoping

`present` when the target resource is scoped to the authenticated
subject/tenant/organization (e.g. query includes `id` AND
`owner_id = request.user.id`; route ignores caller-supplied owner id
and uses session identity).

`weak` when:
- resource is fetched by id only
- ownership is inferred from a request field
- role checked but owner/tenant not
- query scopes only one branch (e.g. list view scoped, detail view not)

### input_validation

`present` when request-derived values are constrained in a way that
matters for their later use — schema rejects invalid type/format,
anchored regex with fail-closed branch, enum/allowlist constrains
sort/filename/redirect/template, numeric range rejects out-of-range.

`weak` when:
- type cast supplies a default instead of rejecting
- schema drops unknown fields but does not constrain the field that
  drives the sink
- validation covers presence but not allowed values
- regex is partial/unanchored when exact format matters
- validation occurs after the sink, or its result is ignored

Validation for type ≠ validation for structure. A string can be
well-formed and still unsafe as a path, SQL fragment, redirect URL, or
command token.

### output_filter

`present` when response data is explicitly projected or sanitized
before returning — DTO/serializer excludes sensitive fields, projection
omits password hashes/tokens/API keys/internal flags/payment memos,
error/debug branch uses the same projection or a safe error object.

`weak` when:
- one branch filters but another returns the raw domain object
- projection excludes `password` but not other sensitive fields
- debug/error response leaks the unfiltered object
- nested objects are not filtered
- sibling handler filters but this handler does not

### rate_limit / csrf

`present` when visible code enforces rate limits on
login/password-reset/token-issuance/invite/expensive-search/etc., CSRF
token validation for browser-authenticated state-changing requests, or
replay protection for signed webhooks. Also check for the presence of
rate-limiting middleware/library project-wide (e.g. Flask-Limiter,
django-ratelimit, slowapi, express-rate-limit). If no rate-limiting
library is imported anywhere in the project, all auth-sensitive
endpoints have `absent` rate limiting.

`weak` when:
- rate limit keyed only on IP behind an untrusted proxy
- limit applied after the expensive/sensitive operation
- CSRF token generated but not validated
- CSRF check skipped for content types the endpoint still accepts
- webhook signature checked but timestamp/replay window absent

`N/A` when the endpoint is pure read with no enumeration/brute-force
concern, or uses non-browser bearer tokens and is not CSRF-reachable.

### cors

`present` when the response `Access-Control-Allow-Origin` is a fixed
allowlist (exact origins), or credentials are not enabled for a wildcard.

`weak` when:
- the request `Origin` header is reflected verbatim into `ACAO`
- `ACAO: *` is sent together with `Access-Control-Allow-Credentials: true`
- the origin allowlist matches by substring/prefix/suffix (e.g.
  `endsWith("example.com")` lets `evil-example.com` through)

→ Origin Validation Error (CWE-346).

## Dominance rule (a control only counts if it dominates the operation)

A control is `present` only when it **dominates** the sensitive operation:
it must run on EVERY path that reaches the operation, BEFORE the operation,
and its failure must STOP the operation (fail-closed). A control that exists
but does not dominate is `weak` or `absent`, not `present`.

Ask: *is there ANY path to the sensitive operation that skips this control,
or reaches it after the effect, or continues past its failure?* If yes, the
control does not dominate.

Non-dominating patterns (mark `weak`, with the reason in parentheses):

- check runs AFTER the side effect (write/query/response already happened)
  → `weak (authz after effect)`
- check on one branch but another branch reaches the same sink unchecked
  → `weak (path B skips check)`
- check failure is caught / logged / returns a warning but execution
  continues → `weak (not fail-closed)`
- check on a parent route/object but a child route/object is independently
  addressable and reaches the operation directly
  → `weak (parent guarded, child reachable)`
- check depends on a value the attacker also controls (subject/resource id
  taken from the same request) → `weak (guard input attacker-controlled)`

A control that dominates on the traced path but that you could not confirm
covers sibling paths is `present` for THIS path — note the unverified
siblings under Uncertainties rather than over-claiming project-wide.

## Common confusions (do NOT mark `present` for these)

| Confusion                              | Correct interpretation                                          |
| -------------------------------------- | --------------------------------------------------------------- |
| Request schema exists                  | `input_validation`, not authorization                           |
| User is authenticated                  | authentication only; still evaluate authorization and ownership |
| Route has `@authorize` decorator       | open decorator before counting it                               |
| Repository fetches by `id`             | not ownership unless scoped to subject/tenant                   |
| Response serializer exists elsewhere   | not output filtering here unless this handler uses it           |
| Type cast succeeds                     | not validation unless invalid values are rejected               |
| CSRF token is set in template          | not present unless server validates it                          |
| Policy helper is imported              | not present unless called on this path                          |

## Sensitivity gate (when does an absent/weak row become Shape B?)

Sensitive operations:

- any write/state-changing operation
- any read of another user's/tenant's/organization's resources
- user listing / search endpoints that return emails, usernames, or PII
- access to PII, secrets, credentials, audit data, billing, or internal
  security data
- any admin / privileged action
- database init / seed / reset / migration endpoints
- debug / diagnostic endpoints
- token issuance or token verification
- login, password reset, credential change, invite creation
- self-service registration / signup, or profile update, that assigns or
  can be manipulated to set a role, capability, group, or admin status
  (privilege escalation, CWE-269)
- expensive operation where abuse has security or availability impact

If the row is `absent` or `weak` on a sensitive operation, raise a
Shape B finding with `control_missing` set to the appropriate tag.

## Shape B candidate mapping

| Checklist row      | Typical `control_missing`                |
| ------------------ | ---------------------------------------- |
| authentication     | `auth`                                   |
| authorization      | `authz` or `role_check`                  |
| ownership/scoping  | `ownership_check`                        |
| input_validation   | `input_validation`                       |
| output_filter      | `output_filter`                          |
| rate_limit / csrf  | `rate_limit` or `csrf`                   |
| cors               | `cors` (CORS misconfig, Shape C)         |

Token verification defects may use `signature_verify` or
`expiry_check`. Use the more specific tag when available.

## Handling uncertainty

If relevant code is not visible after opening the likely control
locations (routing, middleware, decorators, helpers):

- mark the row `absent` in the checklist
- note the uncertainty in the §7 Uncertainties block
- lower confidence when the finding depends on framework behavior or
  unavailable code
- do not claim "no control exists anywhere"; claim "no control is
  visible on this traced path"

High confidence requires visible evidence, not just failure to find a
control.

## Output table format

When reporting in the §7 block, use one status per row (do NOT keep
the `a | b | c` placeholder syntax — pipes break the markdown table).
Show `file:line` for `present`; add a short reason for `weak`.

```
| control            | status                              |
|--------------------|-------------------------------------|
| authentication     | present (auth/middleware.py:42)     |
| authorization      | absent                              |
| ownership/scoping  | weak (role checked, owner not)      |
| input_validation   | present (schemas/order.py:18)       |
| output_filter      | absent                              |
| rate_limit / csrf  | N/A                                 |
```
