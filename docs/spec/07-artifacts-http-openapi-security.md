# Artifacts, HTTP, OpenAPI, and security records

## 1. Scope

This file specifies durable artifact identities, task-result publication,
memory and cross-namespace retrieval, HTTP session/body persistence, OpenAPI
construction and linting, vulnerability and verification records, isolated
code execution, Caido proxy automation, and LikeC4 validation. It uses the
common tool envelope defined in file 06.

Logical artifact keys and tool names are stable product identifiers. Storage
drivers, database engines, HTTP libraries, schema validators, and container
runtimes are not prescribed.

## 2. Artifact identity and task outputs

### 2.1 Artifact service boundary

The logical artifact service is scoped by `(application, user, filename)`.
Workflow artifacts use no session ID, which makes all of one user's task
outputs, memories, HTTP state, and bespoke artifacts visible in one pool.
Artifacts carry either text or binary content; the contracts in this file
identify text payloads explicitly.

Artifact operations are asynchronous. An ordinary store failure reaches the
calling tool/runtime as an error unless a section explicitly declares
best-effort persistence.

### 2.2 Key validation

A task artifact key is trimmed of surrounding whitespace and leading/trailing
forward slashes. It MUST be non-empty and MUST NOT contain a path segment equal
to `..`. A portable reimplementation MUST apply the same rule after treating
both `/` and `\` as separators and MUST reject NUL and absolute/drive-qualified
paths.

Each task publication maps one validated key to:

```text
<key>/result
<key>/summary
<key>/records
```

All three are text. `result` and `summary` default to empty strings. A string
`records` value is stored verbatim; any other value is JSON-encoded with
Unicode preserved, with null/empty input represented as `[]`. Publication
returns the three kind-to-filename mappings.

The three writes occur in `result`, `summary`, `records` order and are not a
transaction in the compatibility implementation. Resume/checkpoint validation
therefore MUST require all expected artifacts before treating publication as
complete.

### 2.3 Collision-resistant derived segments

Fan-out identifiers are converted to one portable segment by this exact
domain-separated algorithm:

1. A raw value is left unchanged only when it is non-empty, at most 160
   characters, matches `[a-z0-9_-]+`, does not start with `h_`, and is not a
   lowercase reserved device basename (`aux`, `con`, `nul`, `prn`, `com1`..
   `com9`, `lpt1`..`lpt9`).
2. Otherwise, collapse every run outside ASCII letters, digits, `_`, and `-`
   to `_`, trim `_`, and use `item` if empty.
3. Append `_h` plus the full lowercase SHA-256 hex digest of the original raw
   UTF-8 bytes.
4. Prefix with `h_` and truncate only the readable portion so the total is at
   most 160 characters.

The `h_` reservation makes raw-safe and encoded domains disjoint. Including
the original bytes in the digest preserves case, whitespace, and punctuation
distinctions on case-insensitive filesystems.

## 3. Memory artifacts and skills/inbox views

### 3.1 Store and record schema

One memory namespace is one YAML text artifact:

```text
user:memory/<namespace>
```

Its top level maps note storage names to records:

```yaml
login_handler:
  name: login_handler
  memory: Full note body
  description: Short preview
  tags: [authentication]
  links: [jwt_validation]
  ordinal: 1
  created_at: "<UTC ISO timestamp>"
  updated_at: "<UTC ISO timestamp>"
```

Missing legacy fields default to empty values and a one-based fallback ordinal.
Dump order is `(ordinal, name)`. New ordinals are one greater than the current
maximum. JSON output is the native record; preview output omits `memory`.
Markdown, YAML, and XML renderings preserve all logical fields, and XML escapes
both element text and quoted name attributes. Optional type hints wrap only
string formats in a matching code fence.

### 3.2 Memory operations

The standard registry is:

| Tool | Behavior |
|---|---|
| `write_memory(name, memory, description, tags)` | Create or fully replace body/description/tags. At most the first three tags are stored. An overwrite preserves ordinal, creation time, and links; it updates `updated_at`. |
| `append_memory(name, text)` | Join non-empty old/new bodies with one newline; preserve description/tags/links. Missing name is an error. |
| `link_memories(name, links)` | Add symmetric graph edges, deduplicate, ignore self-links, and report absent/reserved targets under `unknown_targets`. Missing/reserved source is an error. |
| `read_memory(name)` | Full ordinary note or not-found error. |
| `search_memory(tags)` | Previews whose tags intersect the query, ordered by `(ordinal,name)`. |
| `list_tags()` | Unique sorted tags from all notes, including reserved notes. |
| `list_memories()` | Ordinary previews in insertion order. |
| `skills_list`, `skills_read` | Dedicated view of `skill`-tagged notes. |
| `inbox_list`, `inbox_read` | Dedicated view of `inbox`-tagged notes. |

`skill` and `inbox` are reserved tags. Any note carrying either is hidden from
generic list/read/search. Search also removes reserved tags from the query.
Generic writes MUST NOT overwrite a reserved note by name unless the new write
itself is a system-managed reserved write. Skill alias resolution and injection
are specified in file 06.

Every operation reloads the artifact before reading or mutation and saves the
whole map after mutation. Successful generic reads/writes and skill reads
append deduplicated canonical names to deterministic observation state.

### 3.3 External injection

System injection merges supplied notes into an existing YAML map. Existing
names retain ordinal and creation time but receive the supplied body/metadata
and a fresh update time. New notes receive monotonically increasing ordinals
and timestamps. This is used for skill bodies and other trusted outer-world
memory.

## 4. Cross-namespace artifact pool and retrieval

### 4.1 Key taxonomy and visibility

The pool parser classifies keys as:

| Raw key | Namespace | Kind |
|---|---|---|
| `user:memory/<namespace>` | `<namespace>` | `memory` |
| `<namespace>/result` | `<namespace>` | `result` |
| `<namespace>/summary` | `<namespace>` | `summary` |
| `<namespace>/records` | `<namespace>` | `records` |
| anything else | entire raw key | `raw` |

An allowlist of filename-style glob masks fences the pool. A key matches a mask
when either its parsed namespace or its raw key matches. A key outside the
factory allowlist MUST be invisible to list, read, document expansion, and
search; a direct load behaves as not found.

The agent tools are:

- `pool_namespaces()` -> sorted `{namespace,kinds,count}` groups;
- `pool_list(mask)` -> `{key,namespace,kind}` rows, capped using the configured
  read-line cap (2,000 by default);
- `pool_read(key,offset,limit)` -> a character window, using 50,000 characters
  when limit is zero/non-positive;
- `pool_read_memory(namespace,name)` -> one full note or all note previews,
  including a `reserved` flag;
- `pool_search(query,mask,k)` -> ranked hits, default `k=8` when zero.

A memory artifact expands to one search document per non-empty note body.
`skill`/`inbox` notes are excluded by default; other artifacts are one document
each. A document ID is the raw key, or `<key>#<note-name>` for memory. Search
hits carry `key`, `namespace`, `kind`, numeric `score`, `snippet`, and optional
`note_name`.

### 4.2 Keyword backend

The dependency-free backend lowercases and whitespace-splits the query. A
document's score is the sum of case-insensitive occurrence counts of all terms.
The snippet is a trimmed window from 160 characters before the earliest first
match through 160 characters after it. Positive-score hits sort descending;
ties retain document traversal order. Empty-term queries return no hits.

### 4.3 Dense retrieval backend

The optional dense backend chunks documents into 1,200-character windows with
200-character overlap and embeds them using a configured model. Its relational
schema is logically:

```text
artifact_chunks(
  id, app_name, user_id, doc_id, key, namespace, kind, note_name,
  chunk_idx, content_hash, body, embedding, updated_at,
  unique(app_name,user_id,doc_id,chunk_idx)
)
```

Embedding dimension defaults to 1,024 and MUST match the selected model.
Indexing skips a document when its stored SHA-256 content hash matches. A
changed document deletes all prior chunks and inserts the newly embedded
windows in one database transaction. Counters are `{indexed, skipped}`.

Search embeds the query, cosine-ranks at most `max(4*k,k)` rows for the selected
application/user, applies both pool allowlists and the per-call mask after
retrieval, deduplicates by document ID, and returns at most `k`. Score is one
minus cosine distance and snippets are the first 320 characters of the matched
chunk.

## 5. Persistent HTTP tools

### 5.1 Public registry and limits

The registry is `http_request`, `http_read_body`, `http_history`,
`http_session_set`, `http_session_get`, and `http_session_clear`.

Defaults are a 30-second total request timeout, 2,048-character text preview,
20 history entries, three total send attempts, 0.5-second initial backoff, and
8-second maximum backoff. History and preview capacities MUST be positive.
Retry statuses default to 408, 425, 429, 500, 502, 503, and 504.

Supported advertised methods are GET, POST, PUT, PATCH, DELETE, HEAD, and
OPTIONS. Body kinds are:

- `none`: send no body;
- `json`: send a JSON-compatible object/array/value;
- `form`: require a mapping and send form fields;
- `text`: require text or bytes and send verbatim.

Per-request headers override session defaults. Explicit `Authorization` in the
merged headers overrides configured auth. Auth is `none`, bearer with non-empty
token, or basic with non-empty username and an optional password. Redirects are
followed by default. Cookies received by a short-lived request client are
merged back into the persistent session jar.

### 5.2 Response record

`http_request` returns:

```json
{
  "result": {
    "request_id": 7,
    "request_tag": "case-h000007",
    "method": "GET",
    "final_url": "https://target/final",
    "status": 200,
    "content_type": "application/json",
    "content_length": 8123,
    "headers": {"content-type": "application/json"},
    "body_kind": "text",
    "body_preview": "...",
    "body_truncated": true,
    "body_artifact": "http/exploit/responses/00000007.json",
    "elapsed_ms": 42
  }
}
```

`content_length` is response bytes; preview truncation counts decoded
characters. MIME is textual for `text/*`, the explicit JSON/XML/JavaScript/
ECMAScript/XHTML/form/YAML types, or a `+json`, `+xml`, or `+yaml` suffix. A
textual body up to the preview cap is returned fully as the preview. A longer
one adds a total-character marker. Binary preview is null. Empty preview is
`""` and has no body artifact.

An optional request-tag prefix injects
`X-Request-Id: <prefix>-h<six-digit-id>` and returns it as `request_tag`;
without a prefix the tag is empty.

### 5.3 Persistence schemas

Session state is JSON text at:

```text
http/<name>/session.json
```

```json
{
  "cookies": {"sid": "..."},
  "default_headers": {"Accept": "application/json"},
  "auth": {"kind": "bearer", "token": "..."},
  "history": [
    {
      "request_id": 7,
      "method": "GET",
      "url": "https://target/final",
      "status": 200,
      "content_type": "application/json",
      "content_length": 8123,
      "elapsed_ms": 42
    }
  ],
  "next_request_id": 8
}
```

Stored auth and default headers contain real values. Tool-visible session get/
set returns cookies, headers with an `Authorization` value replaced by
`***redacted***`, and only `auth_kind`; it never returns bearer/basic secrets.
`http_session_clear` resets cookies, defaults, auth, history, and next ID to 1,
but leaves already stored body artifacts intact.

Every non-empty response body is JSON text at:

```text
http/<name>/responses/<eight-digit-request-id>.json
```

Text payload: `{"kind":"text","content_type":"...","text":"..."}`.
Binary payload: `{"kind":"binary","content_type":"...","data_b64":"..."}`.

`http_read_body(request_id,offset=0,length=4096)` requires a context, a
non-negative offset, and positive length. Text offsets/lengths are characters;
binary offsets/lengths are bytes and data is returned as base64. Results include
`request_id`, `kind`, `content_type`, `offset`, actual `length`, and
`total_length`.

Malformed/missing session or body JSON is an explicit error; a missing session
clears the in-memory state and reports not loaded.

### 5.4 Retry, identity, concurrency, and cancellation

Only timeout, network, remote-protocol failures, and configured response
statuses retry. Delay before attempt `n+1` is
`min(base_delay * 2^(n-1), max_delay)`. There is no jitter and no `Retry-After`
interpretation. Exhaustion returns the last failure as an error.

All operations touching one `http/<name>/session.json` namespace share one
asynchronous lock per event loop, even across separately constructed clients.
A request holds that lock across load, network traffic, body persistence,
history update, and final save; requests in one namespace are therefore fully
serialized.

The next request ID is incremented and session state is saved before traffic is
sent. This reservation is mandatory. Thus a network failure, cancellation after
send/body save, or failure of the final session save cannot cause a later
client to overwrite that request's body artifact. Cancellation is never
retried or enveloped and releases the asynchronous lock normally.

## 6. OpenAPI construction and resolution

### 6.1 Artifact and base document

One builder namespace stores YAML text at `user:oas-<name>`. A missing, empty,
whitespace, or YAML-null artifact loads this base document:

```yaml
openapi: 3.0.3
info: {title: "", description: "", version: 1.0.0}
paths: {}
components:
  schemas: {}
  parameters: {}
  responses: {}
  securitySchemes: {}
  examples: {}
  requestBodies: {}
  headers: {}
  links: {}
  callbacks: {}
```

Every mutation within one builder instance performs load -> deep copy ->
modify/deep-merge -> structural diff -> save under one asynchronous lock.
Mappings merge recursively; lists and scalar values replace. A returned diff
is:

```json
{"added":{},"removed":{},"changed":{}}
```

Nested changes recursively use the same shape; scalar changes are
`{"from":old,"to":new}`. The save metadata includes a monotonically assigned
artifact version when the storage driver supplies one.

JSON mode returns native objects, YAML mode returns YAML strings, and accepted
XML/Markdown selections fall back to native objects. The full-schema tool
always returns YAML text.

### 6.2 Evidence gate

Path and component upserts require at least one evidence file. Evidence ending
case-insensitively in `.json`, `.md`, `.yaml`, or `.yml` is prohibited, and
every path must exist in the confined source filesystem. The stored definition
adds `x-path-files` or `x-component-files` with the evidence list.

### 6.3 Path tools

`upsert_path(path,path_def,path_files)` validates a Path Item, strips outer
whitespace from the path, and deep-merges it under `paths`. A Path Item may
contain `$ref`, summary, description, shared parameters, and GET/PUT/POST/
DELETE/OPTIONS/HEAD/PATCH/TRACE operations. An operation may contain tags,
summary, description, external docs, optional `operationId`, parameters,
request body, responses, callbacks, deprecation, security, and extension fields.

`list_paths`, `get_path`, and `remove_path` are exact-key operations after
whitespace stripping. Missing/remove-twice is an error. Upsert is additive at
mapping levels rather than full operation replacement.

### 6.4 Component tools

Agent-facing component buckets are exactly `schemas`, `securitySchemes`,
`requestBodies`, `headers`, and `responses`. The tools are upsert, list, get,
and remove by bucket/name. A component definition MUST be an object, not a JSON
string.

- A response requires `description`; headers, content, links, and extensions
  are allowed.
- A request body requires `content`; `description`, `required=false`, and
  extensions are allowed.
- A security scheme requires type `apiKey|http|mutualTLS|oauth2|openIdConnect`;
  the model accepts the standard conditional fields and extensions.
- Schema and Header objects are accepted without deeper validation by this
  layer.

Validation errors return structured field path, message, type, component model
name, and original input encoded in the error string.

### 6.5 Info and server tools

`set_info(title,framework,code_language)` updates title and supplied
`x-framework`/`x-code-language` while preserving version, description, and
other existing fields. `get_info` returns the block.

`add_server(url,description)` strips URL whitespace, rejects an exact duplicate,
and appends `{url,description-or-empty}`. `remove_server` removes every exact
matching stripped URL and errors if absent. List preserves stored order.

### 6.6 Local reference resolution

Reference resolution deep-copies its inputs and supports JSON Pointer refs
beginning `#/`. Pointer tokens decode `~1` to `/` and `~0` to `~`; dictionaries
and numeric list indexes are traversed. Missing keys, invalid indexes,
unexpected scalar traversal, and depth above 100 are errors.

When a `$ref` resolves to an object, sibling fields overwrite resolved fields.
External/non-local refs are retained. A repeated ref on the active recursion
path becomes `{"$circular_ref":"<ref>"}`. Whole-schema resolution first
resolves each named schema against the original unmodified document with its
self-ref pre-marked, then installs all named results together and resolves the
remainder. This avoids iteration-order-dependent mutual recursion.

The circular marker is an analysis representation, not valid OpenAPI output;
resolved documents MUST NOT silently replace the persisted canonical schema.

### 6.7 Vacuum-compatible linting

The linter loads `user:oas-<name>`, passes its YAML/JSON bytes on standard input
to an external spectral-report command, and accepts process exit 0 or 1. Other
exit codes are execution errors with decoded stderr. Output MUST be a JSON
array; malformed or differently shaped output returns error plus parse detail
and raw output.

Issue severities use `0=error`, `1=warning`, `2=information`, `3=hint`.
Agent lint returns only 0 and 1, sorted ascending, with an optional result
limit. Each issue's source `range` is replaced by a `snippet` extracted using
one-based lines and zero-based characters; invalid coordinates produce an
empty snippet. The external command runs off the event loop.

## 7. Vulnerability and verification records

### 7.1 Vulnerability reports

Reports are YAML text at:

```text
user:vulnerability-reports/<namespace>
```

The top-level map is record name to:

```json
{
  "name": "sqli-login",
  "place_type": "file",
  "place": "/auth.py:42",
  "title": "SQL injection in login",
  "summary": "...",
  "severity": "high",
  "confidence": "medium",
  "details": "...",
  "ordinal": 1,
  "created_at": "<UTC ISO>",
  "updated_at": "<UTC ISO>"
}
```

`place_type` is `file|url`; severity is `info|low|medium|high|critical`; and
confidence is `low|medium|high`. Records are immutable values. Writing the same
name fully overwrites the report while preserving ordinal/creation time and
refreshing update time. New ordinals increase monotonically. List order is
`(ordinal,name)` and previews omit `details`. The public registry exposes
report/upsert, get, and list; backend deletion exists but is not agent-facing.

### 7.2 Verification findings

Verifications are stored separately at:

```text
user:vulnerability-verifications/<namespace>
```

```json
{
  "name": "sqli-login",
  "source_namespace": "vuln-sweep",
  "verdict": "exploitable",
  "summary": "...",
  "attacker_control_at_sink": "full",
  "sink_reached": true,
  "entry_point": "POST /login",
  "data_flow": ["route:10", "query:42"],
  "path_broken_at": null,
  "impact": "...",
  "notes": "...",
  "evidence_request_ids": ["case-h000007"],
  "verified_at": "<UTC ISO>"
}
```

Verdict is `exploitable|exploitable_unverified|not_exploitable|inconclusive`;
attacker control is `full|partial|none`. Findings sort by name. A write with the
same name replaces the prior verdict. The full tool accepts every field;
`submit_verdict` is a simplified form with empty data flow, no broken path,
source namespace equal to the tool namespace, notes from `evidence`, and impact
equal to summary only for the two exploitable verdicts. Public read/list tools
return full or preview records. Evidence IDs are opaque HTTP/Caido request tags,
not artifact keys themselves.

JSON, Markdown, YAML, and XML renderers preserve constrained values and escape
XML. Report preview omits details. Verification preview includes name, source,
verdict, summary, attacker control, sink reached, and verification time.

### 7.3 Lossless malformed-row handling and collisions

Both YAML stores apply these rules before any whole-map rewrite:

1. A missing artifact is an empty store.
2. A non-mapping top level is a hard error; no overwrite occurs.
3. A non-object row or row that fails current schema normalization is retained
   verbatim in an `unparsed` side map. Valid siblings remain usable.
4. Parsed records are keyed by their internal logical `name`, not necessarily
   the original YAML key.
5. Two parsed rows resolving to one logical name are a hard error.
6. A parsed logical name colliding with an unparsed storage key is a hard error.
7. Every successful CRUD save writes parsed rows plus all unparsed rows, so an
   unrelated edit cannot silently discard forward-incompatible data.

These collision errors are deliberate data-preservation barriers and MUST NOT
be downgraded to last-writer-wins behavior.

## 8. Isolated code execution

### 8.1 Sandbox lifecycle and boundary

`run_python` and `execute_bash` execute in an ephemeral external container,
never the host process. One lazily started container is reused for a logical
worker namespace so `/work` state persists across calls. The source root, when
local, is mounted read-only at `/project`; `/work` is writable container state.
The container receives no host environment wholesale.

Compatibility launch controls are:

- image `contractor-sandbox:latest`;
- host networking;
- 2 GiB memory, 2 CPUs, 512 process IDs;
- automatic removal and a two-hour sleeping-container TTL;
- deterministic container name `contractor-sbx-` plus the first 16 hex digits
  of SHA-1(namespace);
- teardown at root run completion, explicit teardown, and process exit.

The default command timeout is 120 seconds. An in-container hard-kill timeout
is backed by a host subprocess timeout 15 seconds longer. Timeout exit is 124
and stderr gains a marker. Standard output and standard error are each capped
at 60,000 characters.

### 8.2 Tool results and artifacts

Python prepends optional setup snippets, writes
`/work/script_<sequence>.py`, and executes it with Python 3. Shell commands run
through `sh -c` in the same container. Both return:

```json
{
  "stdout": "...",
  "stderr": "...",
  "exit_code": 0,
  "artifacts": ["code-exec/<safe-namespace>/script_001.py"]
}
```

The execution sequence is shared by both tools. Before execution, the sandbox
lists regular files no deeper than two levels below `/work`. It collects at
most 20 newly created files, each truncated to 1,000,000 bytes; pre-existing
files merely modified by the command are not collected. Scripts/commands and
new files are stored under `code-exec/<sanitized-namespace>/...`. Artifact
persistence is best effort: failure is logged but does not replace the
execution result.

## 9. Caido proxy tools

### 9.1 Client boundary

The integration sends GraphQL requests to `<configured-url>/graphql` with an
optional bearer access token and a 30-second default timeout. HTTP status or
GraphQL errors become a Caido error and then a normal tool error envelope.
Blob fields are base64-decoded to UTF-8 with replacement; undecodable base64 is
returned verbatim.

The registry contains:

| Tool | Contract |
|---|---|
| `caido_scope(action="list",name,allowlist,denylist)` | List scopes or create one; create requires name. |
| `caido_history(filter="",limit=20,offset=0)` | Newest-first HTTPQL history. Limit clamps to 1..100; offset to zero. |
| `caido_request_detail(request_id)` | Full decoded raw request and optional response. |
| `caido_replay(...)` | Replay an existing history ID or a supplied raw request plus host/port/TLS. |
| `caido_automate_run(request_id,targets,payloads,strategy="ALL",workers=5,delay_ms=0)` | Start fuzzing; strategy is `SEQUENTIAL|PARALLEL|MATRIX|ALL`, workers clamp 1..50, delay to zero. |
| `caido_automate_results(session_id,entry_id,limit=50,offset=0,sort_by,ascending)` | Poll one entry, selecting latest when absent; page clamps to 1..100. |
| `caido_sitemap(parent_id,scope_id,depth="DIRECT")` | Root or descendant sitemap entries. |
| `caido_workflow_list(kind="")` | Installed convert/active/passive workflows, optionally filtered. |
| `caido_workflow_run(workflow_id,input,request_id)` | `request_id` dispatches active; otherwise non-empty input dispatches convert. |
| `caido_workflow_findings(limit=20,offset=0)` | Newest-first structured findings, limit 1..100. |

### 9.2 Replay and automation details

Replay creates a session, starts a task, and optionally polls every 0.5 seconds
until response, entry error, or the default 15-second deadline. It returns
session/entry/task identity when asynchronous, or request/response metadata and
decoded raw response when complete.

An optional tag prefix injects
`X-Request-Id: <prefix>-c<six-digit-counter>` into the raw request.
Existing-request and raw request paths both use the modified bytes for the
started task. HTTP-tool tags use `h` and replay tags use `c`, preventing
cross-tool collisions for one prefix.

Automation finds each target in the raw request's decoded **bytes** and uses
only its first occurrence. Placeholder offsets are byte offsets, not Unicode
character positions. Any absent target is an error with a 200-character raw
preview. Settings update content length, never follows redirects, never retries
failed fuzz requests, and applies one simple payload list. Results include
payload values, request ID, error, method/host/path/query, status, response
length, and round-trip time.

## 10. LikeC4 validation

`validate_likec4(path="/architecture.c4")` reads the overlay-visible UTF-8
source on every call and validates a temporary one-file project off the event
loop. Missing source is a normal error.

Command resolution is lazy and cached: prefer a direct `likec4` executable,
then `bunx`, `pnpx`, or `npx`. Package runners receive noninteractive
auto-confirm flags where available. The fixed invocation requests JSON and no
layout, passes a temporary file/project path, discards standard input, captures
output, and defaults to a hard 120-second timeout.

A nonzero exit caused by model issues is accepted if JSON output exists.
Accepted output is either a bare issue array or an object with an `errors`
array. Package-runner banners before the JSON are tolerated by retrying parse
from candidate `{`/`[` positions. No output is an execution error. Invalid or
unexpected JSON returns `error`, optional `details`, and `raw_output`.

## 11. Security, concurrency, and cancellation requirements

### 11.1 Required trust boundaries

1. Artifact-derived filenames MUST remain below the configured artifact root
   after separator normalization and canonicalization.
2. Pool masks MUST be enforced on direct reads as well as discovery/search.
3. Secrets in HTTP auth/default headers MUST be redacted from agent-visible
   session inspection. At-rest secret policy MUST be explicit because the
   compatibility session artifact contains credentials and cookies in clear
   JSON text.
4. HTTP, Caido, and container-host-network traffic MUST be limited to targets
   the operator authorized. Agent prose saying “set a scope” is not an access
   control.
5. Source mounts in code execution MUST be read-only and host environment
   variables MUST NOT be inherited wholesale.
6. External linter/package-runner/container executables are trusted-code
   boundaries. Production deployments SHOULD pin versions/images and verify
   provenance.
7. YAML and JSON parsing MUST use non-executing safe parsers.
8. Malformed security records MUST be preserved or block rewrite; they MUST
   NOT disappear during unrelated CRUD.

### 11.2 Concurrency ownership

| Resource | Compatibility serialization |
|---|---|
| one HTTP artifact namespace, same event loop | shared lock across client instances; entire request serialized |
| one OpenAPI builder instance | one lock around load-modify-save |
| one memory/report/verification object | local load/save locks, but multi-step mutation is not one critical section |
| one overlay | re-entrant state lock |
| sandbox registry | process-wide thread lock; one container per namespace |
| Caido client | one reusable asynchronous HTTP client, no operation lock |
| LikeC4 command cache | no effective shared-operation lock |

Cancellation MUST propagate through all common frontend guards. HTTP ID
reservation remains durable before cancellation can leave a stored body.
Task/container/proxy operations that launch external work need an explicit
cancellation/cleanup policy; current compatibility behavior is described as a
gap below.

## 12. Compatibility gaps and reconstruction risks

The following are implementation-derived issues, not recommended new design:

1. Direct artifact-key validation checks only forward-slash `..` segments.
   Backslash traversal and platform-specific absolute forms are not rejected at
   this layer. Derived slugs are safe, but caller-supplied keys need portable
   validation.
2. Task-result triple publication is non-transactional. A middle write failure
   leaves a partial key. Memory, report, verification, and OpenAPI locks are
   object-local; separately constructed objects for one artifact namespace can
   lose concurrent updates.
3. Even within one memory/report/verification object, mutation calls perform a
   locked load, release the lock, mutate, then perform a separately locked
   save. Concurrent calls can overwrite a sibling update. Use a namespace-wide
   atomic load-modify-save or optimistic version check.
4. `pool_read` uses `returned` as the absolute end character offset rather
   than the page item count prescribed by the generic page envelope. Dense
   indexing never deletes rows for documents removed from the artifact pool,
   so search can return stale/deleted content. Post-filtering only `4*k`
   candidates may also miss allowed hits farther down the vector ranking.
5. The dense backend interpolates configurable table name and vector dimension
   into schema SQL. Dimension should be a validated positive integer and table
   name a quoted/allowlisted identifier; otherwise privileged configuration is
   an injection surface.
6. HTTP tools have no URL/host allowlist and can reach loopback, link-local, or
   cloud metadata addresses through direct or configured proxy routes. The
   per-loop namespace lock is not cross-process/cross-loop coordination. Retry
   parameters and method/URL schemes are not uniformly runtime-validated.
7. OpenAPI evidence checks require existence but not regular-file type. Paths
   need not begin `/`; server URLs and local evidence are not semantically
   validated. Security-scheme conditional requirements (`apiKey.name/in`,
   OAuth flows, OpenID URL) are documented but not enforced. Upsert mutates the
   caller's component/path object by adding evidence fields.
8. OpenAPI locking/version metadata does not provide cross-instance optimistic
   concurrency. A truthy non-mapping YAML artifact is not rejected immediately.
   The external Vacuum invocation has no timeout and its raw error output is
   unbounded; the linter factory resolves the executable eagerly, unlike the
   graceful lazy LikeC4 integration.
9. Verification storage documentation says multiple attempts can coexist, but
   the store is keyed only by finding name and overwrites a prior attempt.
   Semantic combinations are not validated: an `exploitable` verdict may say
   `sink_reached=false`, omit impact, or cite no evidence. `source_namespace`
   is not checked against an existing source report.
10. Code execution grants the container host networking with no egress
    allowlist, accepts arbitrarily large/negative timeout inputs, and assumes
    run-level cleanup is sequential. Cancelling the awaiting task does not stop
    a blocking command already running in a worker thread. Output collection
    misses modified pre-existing files, and namespace artifact sanitization is
    collision-prone (`a/b` and `a-b` can converge).
11. Caido disables TLS certificate verification even for non-loopback URLs,
    keeps a reusable client without an exposed close seam, does not persist or
    lock its request-tag counter, and injects a second tag header rather than
    replacing an existing one. Raw request/response and payload collections are
    unbounded inline. Scope creation is advice, not enforcement, so automation
    can fuzz any captured host and has no payload-count/byte quota.
12. LikeC4 fallback runners may fetch and execute an unpinned latest package.
    Issue arrays, banners, and `raw_output` are unbounded, and repeated parse
    attempts from every JSON-looking offset can be quadratic on hostile output.
13. Vulnerability/verification lossless row preservation is strong, but its
    whole-map save remains non-transactional with respect to other writers and
    has no artifact version compare-and-swap.

## 13. Reconstruction acceptance checks

A conforming implementation MUST test at least:

1. artifact key cleanup, both-separator traversal rejection, safe/encoded slug
   domain separation, device names, case variants, empty/long values, and
   partial triple publication detection;
2. memory ordinal/timestamp/link preservation, tag truncation, reserved-note
   isolation, skill/inbox reads, YAML round trips, and concurrent writers;
3. pool mask fencing on direct reads, reserved-document exclusion, keyword
   ranking, honest character windows, dense content-hash replacement, stale
   deletion, and post-filter retrieval;
4. HTTP MIME classification, short/long text and binary body artifacts,
   paged body reads, cookie/auth persistence and redaction, retry status versus
   permanent status, closure of per-request clients, distinct IDs under
   overlap/separate clients, final-save failure, and cancellation after body
   save;
5. OpenAPI empty artifact fallback, operationId omission, response Link
   objects, evidence rejection, concurrent upserts, info-version preservation,
   server duplicate/removal, local pointer escaping, self/mutual cycles, and
   linter severity/snippet/error handling with a hard timeout;
6. vulnerability and verification constrained vocabularies, overwrite
   semantics, preview/full formats, request evidence IDs, malformed sibling
   preservation, non-mapping refusal, duplicate logical names, and parsed/raw
   key collisions;
7. container launch flags, clean environment, read-only source, timeout/exit
   code, output caps, shared sandbox state, best-effort artifacts, cancellation,
   and teardown;
8. Caido HTTP/GraphQL errors, byte-accurate non-ASCII placeholders, scope and
   page bounds, replay tags/poll timeout, fuzzing bounds, raw-output limits, and
   client closure;
9. LikeC4 lazy executable resolution, noninteractive fixed invocation, timeout,
   clean/error/legacy output shapes, banner handling, unavailable command, and
   bounded rich errors.

## 14. Authoritative implementation inventory

This contract was reconstructed from `contractor/runners/artifacts.py`,
`contractor/runners/skills.py`, and the tool modules `artifact_pool`,
`artifact_rag`, `memory`, `http`, `openapi/*`, `vuln`, `podman`, `caido`, and
`likec4`, plus their runner/workflow call sites and unit/integration tests. The
inventory is informative; this file is the language-neutral behavioral
authority.
