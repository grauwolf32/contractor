---
description: Annotation discipline for the trace skill — exact comment forms, comment syntax per language, when to annotate, how to verify, and how to avoid duplication.
---

# Annotation Discipline

Annotations are machine-readable comments placed above relevant function
definitions. They describe visible code only — never invent calls, files,
arguments, middleware, or framework behavior. Insert them only through the
dedicated operations below; never synthesize comment edits manually.

## Exact forms

Use only these calls:

```text
annotate_trace(
  path,
  symbol,
  target="unknown",
  args="",
  calls="",
  definition_line=0
)
annotate_validate(path, symbol, arg, kind, definition_line=0)
annotate_sink(path, symbol, kind, arg="unknown", definition_line=0)
```

They insert these canonical forms directly above function definitions:

  @trace target=<target_id_or_unknown> args=<arg:state,...> calls=<symbol,...>
  @validate arg=<arg> kind=<kind>
  @sink kind=<kind> arg=<arg_or_unknown>

No other annotation forms are recognized. `path` is a normalized
workspace-relative path and `symbol` is an exact, case-sensitive unqualified
function-like name. If the same `(path, symbol)` identifies more than one
declaration, inspect the current source and repeat with that declaration's
positive 1-based `definition_line`; never guess or accept the first match.

`<state>` ∈ { tainted, validated, clean, derived } — see
`references/sources.md` for state semantics.

### Validation kinds

For `@validate`, use a short label that describes the visible
validation. Use the most specific accurate kind. Do NOT mark a value
validated because a helper name sounds safe — inspect the helper first.

  schema | type | regex | allowlist | enum | length | range
  parser | escape | sanitize | signature | expiry

### Sink kinds

For `@sink`, use an exact label from `references/sinks.md`. Common
examples:

  db.query | db.query.raw | filesystem.read | shell.exec |
  shell.exec.args | http.request | template.render.raw |
  auth.token.verify | cookie.set | secret.log

Do not invent sink labels.

## Comment syntax per language

The annotation body (`@trace ...`) is identical across languages; only
the comment marker changes. Match the surrounding file's convention.

| Language(s) | Marker |
| --- | --- |
| Python, Ruby, Bash, Elixir | `# @trace ...` |
| Haskell, Lua | `-- @trace ...` |
| JavaScript, TypeScript, TSX, Go, Rust, Java, Kotlin, C, C++, C#, PHP, Scala, Swift | `// @trace ...` |


Use the form that is a real comment in the file's language. Do not
introduce a `#` line into a JS or Go file.

## Placement

The annotation operations place annotations directly above the
function/method/handler/closure definition they describe and above any
syntax-owned decorator, attribute, export, or template prefix.

Good:

```python
# @trace target=getOrder args=req:tainted calls=findOrderById
def get_order(req): ...
```

Bad (annotation inside function body):

```python
def get_order(req):
    # @trace target=getOrder args=req:tainted calls=findOrderById
    ...
```

If the language uses decorators, place annotations above the decorator
block:

```python
# @trace target=getOrder args=req:tainted calls=findOrderById
@app.get("/orders/{id}")
def get_order(req): ...
```

For methods inside classes, annotate the method, not the class, unless
the class itself is the executable entrypoint.

## `target=` value

- `target=<target_id>` — the assigned operation/handler/route id
  (e.g. `target=getOrder`, `target=POST_/users/{id}`). Use the exact
  id from the assignment when you can.
- `target=unknown` — the function is on the path of multiple assigned
  targets, the target id is not yet pinned down, or the function is
  clearly on the path but no stable id is provided.

Do not invent a target id.

## `args=` value

Format: `args=<arg:state,...>`. Use argument names as written in the
signature when possible. For methods, omit `self` / `this` unless it
carries target-relevant data. Mark only arguments relevant to the
trace; do not annotate every parameter by default.

## `calls=` value

List direct, target-relevant callees that continue the trace. Include
symbols that matter for validation, sink, or key transformation. Skip
generic logging, formatting, metrics, or unrelated helpers. If a call
is dynamic and unresolved, use the visible dispatcher symbol rather
than guessing the runtime target.

## Multi-target functions

When a single function is on the path of two or more assigned targets, call
`annotate_trace` once per target, in assignment order:

```
# @trace target=getOrder args=req:tainted calls=findOrderById
# @trace target=listOrders args=req:tainted calls=findOrderById
def shared_handler(req): ...
```

Do NOT collapse multiple targets into one comma-separated `target=`
value — the annotation form accepts a single id per line.

## Annotate when at least one applies

- function is the entrypoint for the assigned target
- it receives or transforms target-relevant tainted/derived data
- it performs explicit validation/sanitization
- it passes tainted/derived data into a sink
- it directly performs or wraps a sink
- it enforces auth/authz/ownership/signature/expiry/CSRF/rate-limit/
  output-filter relevant to a finding
- it performs a key transformation needed to explain the path

## Do NOT annotate

- generic helpers (logging, formatting, type conversion) unless they
  validate, sink, or transform target-relevant data
- functions only weakly suspected to be on the path
- functions reached only through unverified dynamic dispatch
- duplicate wrappers where a nearby annotation already captures the
  same trace edge
- test files, examples, fixtures, or generated code unless the
  assigned target executes them
- unrelated hardcoded-secret sites unless reporting a Shape C defect there

## Validation annotations

Use `@validate` on the function or method that performs the validation,
not merely on the caller.

Good (validator function):

```python
# @validate arg=order_id kind=regex
def is_valid_order_id(order_id): ...
```

Also acceptable (inline validation in handler):

```python
# @trace target=getOrder args=order_id:tainted calls=findOrderById
# @validate arg=order_id kind=regex
def get_order(order_id):
    if not ORDER_ID_RE.match(order_id):
        raise BadRequest()
    ...
```

Do NOT use `@validate` when:

- invalid values are silently defaulted
- validation result is ignored
- validation occurs after the sink
- the check only proves presence but the later sink requires an
  allowlist or structural constraint

## Sink annotations

Use `@sink` only when the function directly performs the sink or
clearly wraps the runtime call that performs it.

Good (wrapper around the driver call):

```python
# @sink kind=db.query arg=sql
def query(sql, params):
    return conn.execute(sql, params)
```

Bad (handler is not the sink):

```python
# @sink kind=db.query.raw arg=order_id   # WRONG
def get_order(order_id):
    return repo.find(order_id)
```

The handler may call a sink later, but the handler itself is not the
sink unless it directly performs or wraps the sink.

## Before inserting

- inspect whether the function already has trace annotations
- preserve correct existing annotations
- add only missing annotations that materially improve the trace
- rely on exact replay returning `changed=false`; do not treat it as a failure
- if the same trace target already has a different body, stop on conflict and
  inspect or roll back instead of silently replacing it
- do not create conflicting states for the same argument at the same
  function entry
- do not change unrelated formatting or comments

## Verification

Before ending annotation work, call `changed_paths`, then inspect the bounded
result with `diff`. Confirm that only the intended files and lines changed. If
those operations are unavailable, return proposed annotations without claiming
that they were applied.

Confirm comment syntax matches the file language, annotations are
directly above the right function, and no unrelated whitespace edits
were introduced.

If verification shows unintended changes, call `rollback_changes` for the
affected path (or the complete current change set) and re-insert only the
intended lines.

## Examples

Entrypoint with one tainted arg passing into a SQL sink (Python):

```
# @trace target=getOrder args=req:tainted calls=findOrderById
def get_order(req): ...
```

Same shape in TypeScript:

```
// @trace target=getOrder args=req:tainted calls=findOrderById
function getOrder(req: Request) { ... }
```

Validation step:

```
# @validate arg=orderId kind=regex
def is_valid_order_id(order_id): ...
```

Parameterized sink wrapper:

```
# @sink kind=db.query arg=sql
def execute_order_query(sql, params): ...
```

Raw SQL sink wrapper:

```
# @sink kind=db.query.raw arg=sql
def execute_raw_query(sql): ...
```

Shell args form (no shell):

```
# @sink kind=shell.exec.args arg=argv
def run_tool(argv): return subprocess.run(argv, shell=False)
```

Shell string form (shell=True):

```
# @sink kind=shell.exec arg=cmd
def run_tool(cmd): return subprocess.run(cmd, shell=True)
```

A function that is both validator and sink (rare; allowed):

```
# @trace target=dispatch args=raw:tainted calls=db.exec.raw
# @validate arg=raw kind=allowlist
# @sink kind=db.exec.raw arg=raw
def dispatch(raw): ...
```
