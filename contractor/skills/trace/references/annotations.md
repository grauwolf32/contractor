---
description: Annotation discipline for the trace skill — exact comment forms, when to annotate, how to verify, and how to avoid duplication.
---

# Annotation Discipline

## Exact forms

Place comments directly above function definitions. Use only:

  # @trace target=<target_id_or_unknown> args=<arg:state,...> calls=<symbol,...>
  # @validate arg=<arg> kind=<kind>
  # @sink kind=<kind> arg=<arg_or_unknown>

No other annotation forms are recognized.

`<state>` ∈ { tainted, validated, clean, derived } — see
`trace/references/sources` for state semantics.

`<kind>` for `@validate` — short label of the validation performed
(e.g. `schema`, `type`, `regex`, `allowlist`, `length`).

`<kind>` for `@sink` — exact label from the catalogue in
`trace/references/sinks`.

## Annotate when at least one applies

- the function is the entrypoint for the assigned target
- it performs explicit validation/sanitization on tainted data
- it passes tainted/derived data into a sink
- it performs a key transformation needed to explain the path

## Do NOT annotate

- generic helpers (logging, formatting, type-conversion utilities)
  unless they perform validation, a sink, or a key transformation on
  target-relevant data
- functions only weakly suspected to be on the path
- functions whose annotation would duplicate one already nearby

## Before inserting

- check whether the function already has trace annotations
- do not insert duplicate lines
- do not create conflicting annotations
- if existing annotations are partial but clearly correct, preserve
  them; only add missing lines if they materially improve trace
  coverage and remain consistent with the code

## Verification

Before ending PHASE 1:

  changed_paths   → confirm only intended files changed
  diff            → confirm only intended lines were added

If verification shows unintended changes (whitespace edits, accidental
overwrites, extra annotations), use `restore` on the affected path and
re-insert only the intended lines.

## Examples

Entrypoint with one tainted arg passing into a SQL sink:

```
# @trace target=getOrder args=req:tainted calls=findOrderById
function getOrder(req) { ... }
```

Validation step:

```
# @validate arg=orderId kind=regex
function isValidOrderId(orderId) { ... }
```

Sink wrapper:

```
# @sink kind=db.query arg=sql
function executeOrderQuery(sql, params) { ... }
```

A function that is both validator and sink (rare; allowed):

```
# @trace target=getOrder args=raw:tainted calls=db.exec.raw
# @validate arg=raw kind=type
# @sink kind=db.exec.raw arg=raw
function dispatch(raw) { ... }
```
