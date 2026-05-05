---
description: Taint sources and argument-state taxonomy for the trace skill. Use to decide whether an argument is tainted/validated/clean/derived in @trace annotations.
---

# Taint Sources & Argument States

## Argument states

  tainted   — derived from external, untrusted, or user-controlled input
  validated — explicit validation or sanitization is visible in code
  clean     — trusted constant, internal-only value, or non-user-controlled
  derived   — computed, transformed, parsed, mapped, filtered, decoded,
              encoded, or otherwise computed from other values

Be conservative:

- mark `tainted` only with code evidence the value originates from an
  external source (request body/query/path/header/cookie, file content,
  environment populated at runtime, message payload, third-party API
  response)
- mark `validated` only when validation is explicit (schema check, type
  cast with reject, allowlist match, regex match, length/range bound)
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

Storage and IPC
  - file content read from a path that came from request input
  - third-party API responses fetched on behalf of a user
  - shared cache entries written by another principal

Boundary cases
  - environment variables and config — usually `clean`, but mark
    `tainted` when populated from a tenant-controlled or user-uploaded
    source
  - timestamps and IDs generated server-side — `clean` or `derived`
  - values reflected from request to response (`Set-Cookie`,
    `Location`) — preserve the upstream taint state

## Decision table

| Origin                                              | Default state |
| --------------------------------------------------- | ------------- |
| Request body / query / path / header / cookie       | tainted       |
| Validated payload after a schema/type guard         | validated     |
| Result of pure transform on a tainted value         | derived       |
| Hardcoded literal or config baked at build time     | clean         |
| Internal-only value computed from clean inputs      | clean         |
| Output of an external API call initiated by user    | tainted       |
| Data read from a store that previously persisted    |               |
| user-controlled content                             | tainted       |

## Taint propagation cheatsheet

- Concatenation, formatting, slicing, encoding, decoding of a tainted
  value → `derived` (still tainted).
- Hashing or HMAC of a tainted value → `derived`; the digest itself is
  not user-controlled but the choice of input was.
- Lookup against an allowlist that fails closed → `validated` for the
  matched branch only; the other branch terminates.
- Schema validation that drops unknown fields → `validated` for kept
  fields; dropped fields are no longer in scope.
- Type cast without rejection (e.g., string → int with default) → still
  `tainted`/`derived`; not `validated`.
