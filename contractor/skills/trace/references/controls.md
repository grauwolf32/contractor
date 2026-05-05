---
description: Per-handler control checklist for the trace skill. Walk this list before reporting; each absent/weak row on a sensitive operation is a Shape B candidate.
---

# Per-Handler Control Checklist

Walk this list for the target handler before reporting. Mark each row as
`present (file:line)`, `absent`, `weak`, or `N/A`.

## The checklist

| control            | what to look for                                         |
| ------------------ | -------------------------------------------------------- |
| authentication     | session/token validation before reaching handler logic   |
| authorization      | caller permitted on THIS resource (role/policy check)    |
| ownership/scoping  | resource id belongs to the caller (subject == owner)     |
| input_validation   | schema/type/format check on each request-derived arg     |
| output_filter      | domain object passed through a projection/sanitizer      |
| rate_limit / csrf  | applicable to write/state-changing operations            |

## Status semantics

- `present (file:line)` — control is implemented; record where
- `absent`              — control is missing entirely on this handler
- `weak`                — control exists but is bypassable, partial, or
                          does not enforce the intended invariant
- `N/A`                 — control does not apply (e.g. CSRF on a pure-read
                          endpoint that uses bearer tokens)

## Sensitivity gate (when does an absent/weak row become Shape B?)

Operations classed as "sensitive" for this purpose:

- any write/state-changing operation
- any read of another user's resources
- any admin / privileged action
- token issuance, credential change, password reset
- access to PII, secrets, or audit data

If the row is `absent` or `weak` on a sensitive operation, raise a
Shape B finding with `control_missing` set to the row name.

## Tracing tips

- The control may be applied in middleware, a decorator, or a router
  wrapper rather than the handler itself. Inspect the routing layer
  before declaring `absent`.
- A control on a sibling handler is not a substitute for the target.
- Schema-level validation that does not enforce the invariant the
  handler relies on is `weak`, not `present`.
- If you cannot determine status from visible code, mark the row
  `absent` conservatively and capture the uncertainty in the §7
  Uncertainties block.

## Output table format

When reporting in the §7 block, use one status per row (do NOT keep the
`a | b | c` placeholder syntax — pipes break the markdown table). Show
`file:line` for `present`; add a short reason for `weak`.

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
