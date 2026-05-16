---
description: Taint sources and argument-state taxonomy for the trace skill. Use to decide whether an argument is tainted/validated/clean/derived in @trace annotations.
---

# Taint Sources & Argument States

Argument state describes the value at the current call site. It does
not describe whether the user caused the code to run. A value is not
tainted merely because it appears in a request-handling path.

## Argument states

  tainted   — derived from external, untrusted, or user-controlled input
  validated — explicit validation or sanitization is visible in code
  clean     — trusted constant, internal-only value, or non-user-controlled
  derived   — computed, transformed, parsed, mapped, filtered, decoded,
              encoded, or otherwise computed from other values

Important: `derived` does not mean safe. If a value is derived from
tainted input, it remains attacker-influenced unless effective
validation is applied.

Be conservative:

- mark `tainted` only with code evidence the value originates from an
  external source (request body/query/path/header/cookie, file content,
  environment populated at runtime, message payload, third-party API
  response)
- mark `validated` only when validation is explicit AND fail-closed
  (schema check, type cast with reject, allowlist match, anchored
  regex match, length/range bound, signature/expiry verification)
- mark `derived` when the value depends on other values; if any input
  is tainted, the derived value remains tainted unless validation is
  applied
- mark `clean` when the value is hardcoded, internal config, or
  otherwise not influenceable by external actors

## Common taint sources (project-agnostic)

Synchronous request entrypoints
  - request body, query string, path parameters, headers, cookies
  - multipart upload filename and content
  - WebSocket / Server-Sent Events frames
  - gRPC / GraphQL request fields

Asynchronous entrypoints
  - message-queue payloads (Kafka, SQS, RabbitMQ, NATS, etc.)
  - event-bus messages
  - webhook deliveries from third parties
  - scheduled job payloads sourced from user-writable storage
  - retry/dead-letter payloads that preserve user input

Storage and IPC
  - file content read from a path that came from request input
  - third-party API responses fetched on behalf of a user
  - shared cache entries written by another principal
  - IPC messages from lower-trust processes

Boundary cases
  - environment variables and config — usually `clean`, but mark
    `tainted` when populated from a tenant-controlled or user-uploaded
    source, dynamic DB-backed config, or untrusted CI/CD variables
  - timestamps and IDs generated server-side — `clean` or `derived`
  - values reflected from request to response (`Set-Cookie`,
    `Location`) — preserve the upstream taint state

Hardcoded secrets are not "tainted"; they are Shape C candidates.

## Decision table

| Origin                                              | Default state |
| --------------------------------------------------- | ------------- |
| Request body / query / path / header / cookie       | tainted       |
| Upload filename / content                           | tainted       |
| WebSocket / gRPC / GraphQL client field             | tainted       |
| Queue or webhook payload                            | tainted       |
| User-controlled database field                      | tainted       |
| External API response fetched for user-controlled request | tainted |
| Validated payload after a fail-closed schema/type guard | validated |
| Tainted value after formatting/encoding/decoding/parsing | derived  |
| Hardcoded literal or config baked at build time     | clean         |
| Server-generated id / timestamp / random nonce      | clean / derived |
| Authenticated subject after verified auth           | validated     |

## Validation rules

Mark a value `validated` only when the relevant check is visible AND
fail-closed.

Examples that CAN validate:

- schema rejects invalid payloads
- type guard rejects wrong type
- regex match is anchored AND rejects invalid format
- allowlist/enum rejects unknown values
- numeric range check rejects out-of-range values
- length bound rejects too-long or too-short values
- parser rejects malformed syntax before use
- sanitizer removes/escapes dangerous content for the specific output context
- signature verification proves message integrity
- expiry check rejects stale token or signed request

Examples that do NOT validate:

- type cast with fallback default (`int(x or 0)`, `parseInt(x) || 0`)
- trimming whitespace, lowercasing, URL encoding by itself
- base64 decoding or JSON parsing without schema/field constraints
- checking only that a value exists
- schema that validates one field while the sink is driven by another
- validation result computed but ignored
- validation that occurs after the sink
- regex that is partial or unanchored when exact format matters

Validation is context-specific. A value validated as an integer may
still be unsafe as an authorization subject. A value validated as a
string may still be unsafe as a path, SQL fragment, redirect URL,
command token, or template name.

## Taint propagation cheatsheet

| Operation                                              | Resulting state |
| ------------------------------------------------------ | --------------- |
| Concatenation/interpolation of tainted input           | derived (still tainted) |
| Formatting/slicing/trimming/case-conversion of tainted | derived         |
| Encoding/decoding/escaping/parsing of tainted          | derived unless context-appropriate sanitizer |
| Hash/HMAC/digest of tainted input                      | derived (digest not user-controlled, but input choice was) |
| Lookup against fail-closed allowlist                   | validated on matched branch only; other branch terminates |
| Schema validation with rejected invalid branch         | validated for checked fields after the guard |
| Schema drops unknown fields                            | kept fields validated; dropped fields out of scope |
| Type cast with rejection                               | validated for that type after the guard |
| Type cast with fallback/default                        | derived or still tainted; not validated |
| Combining clean and tainted values                     | derived         |
| ORM fetch by user-controlled id without ownership      | row identity is attacker-selected; evaluate ownership |
| ORM fetch scoped by authenticated subject              | identity scoped; user-content fields remain tainted |

## Branch sensitivity

State can differ by branch. Pick the state that applies at the call site
under analysis.

Example — fail-closed allowlist (validated branch):

```python
if sort in ALLOWED_SORT_FIELDS:
    query(sort)            # sort is validated here
else:
    raise BadRequest()
```

Example — fallback default (still derived/tainted; not validated):

```python
sort = request.args.get("sort", "created_at")
if sort not in ALLOWED_SORT_FIELDS:
    sort = "created_at"
query(sort)                 # after replacement, treat as clean/validated
```

Example — cast with default is NOT validation:

```python
limit = int(request.args.get("limit") or 100)   # derived; not validated
```

## Stored data nuance

A row read from storage is not uniformly clean or tainted. Evaluate
each field. For example:

  orders.id              clean / server-generated
  orders.owner_id        clean / server-generated or validated by auth
  orders.shipping_note   tainted (user content)
  orders.internal_status clean / internal
  users.display_name     tainted (user content)
  users.password_hash    sensitive — Shape C if returned/logged

When a full domain object is returned, logged, serialized, or rendered,
evaluate each sensitive/user-controlled field and the output filter.

## Authenticated identity

Subject identity attached by a verified authentication layer is usually
`validated` for downstream use (e.g. `request.user.id` after session
lookup, decoded JWT `sub` after signature+expiry verification, API key
owner after lookup in trusted storage).

The user controlled which account they signed in as, not the value at
the point of use. However, validated identity does not prove access to
every resource id in the request — still evaluate authorization and
ownership.

## Not tainted by default (common confusions)

These look user-adjacent but are usually `clean` or server-controlled.
Mark them `tainted` only with explicit code evidence the user reaches
the underlying value.

- Identifiers minted server-side (UUIDs from `uuid4()`, autoincrement
  PKs, request IDs from middleware) — `clean`/`derived` from clean.
- Subject identity attached by the auth layer after signature+expiry
  verification — `validated` for downstream use.
- Build-time configuration baked into the binary or read from a config
  file the user cannot influence — `clean`.
- Constants in source — `clean` (but if the constant is a hardcoded
  secret, that is a Shape C defect, independent of taint).
- Values returned by a server-side ORM query that filters by the
  authenticated subject — `clean` for the row identity; the row's
  user-supplied content fields remain `tainted`.

A "user touched the request that caused this code to run" argument is
not enough. The argument state describes the value at this call site,
not the chain of causation.

## Quick decision procedure

1. Where did this value originate?
2. Can an external actor choose or influence the value?
3. Has visible code constrained it with a fail-closed check?
4. Is the value transformed from another value? If so, what was that
   value's state?
5. Which state describes the value at this exact call site?

Use the least optimistic state supported by visible code.
