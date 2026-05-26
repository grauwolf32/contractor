---
description: Per-endpoint security control checklist. Walk this for EVERY endpoint before finishing the scan. An absent row on a sensitive endpoint is a finding.
---

# Per-Endpoint Control Checklist

For EVERY endpoint discovered (HTTP handler, CLI command, async consumer, background job, database seed function), fill this table:

| Control | Status | Notes |
|---------|--------|-------|
| authentication | present / absent / weak / N/A | Who can reach this? |
| authorization | present / absent / weak / N/A | Is the caller permitted? |
| ownership | present / absent / weak / N/A | Is the resource scoped to caller? |
| input_validation | present / absent / weak / N/A | Are inputs constrained? |
| output_filter | present / absent / weak / N/A | Are sensitive fields excluded from response? |
| rate_limit | present / absent / weak / N/A | Is abuse/brute-force prevented? |

## Where to look for controls (check all before marking absent)

1. Route registration / blueprint / module group
2. Middleware chain
3. Decorators / annotations on the handler
4. Framework guards (e.g. `@login_required`, `@requires_auth`)
5. Schema validators (e.g. Pydantic, jsonschema, marshmallow)
6. Service-layer policy helpers
7. Repository scoping filters (e.g. `.filter(owner_id=user.id)`)

A function NAMED `authorize` or `secure` counts ONLY after you read its implementation.

## When absent/weak = vulnerability

An absent or weak control on a sensitive endpoint IS a finding. Sensitive endpoints:
- Any write / state-changing operation
- Any read of another user's data
- Login, registration, password reset, credential change
- Token issuance / verification
- Admin / privileged actions
- Database reset / seed / migration endpoints
- Debug / diagnostic endpoints
- Expensive operations (search, export, bulk)
