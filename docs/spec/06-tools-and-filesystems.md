# Tools and filesystems

## 1. Scope and conformance

This file specifies the agent-facing tool protocol, virtual filesystems,
copy-on-write editing, source navigation, structured annotations, task-plan
tools, skill loading, and deterministic worker observations. Stable tool names,
wire fields, persistence kinds, and safety behavior are normative. A conforming
implementation may use any filesystem interface, parser, graph engine, or agent
runtime.

The terms **backend** and **frontend** are logical roles. A backend performs an
operation and may raise an ordinary error. A frontend exposes a typed callable
to an agent and converts the outcome to the common envelope below.

## 2. Common tool protocol

### 2.1 Result envelope

Every frontend tool MUST return exactly one of these shapes:

```json
{"result": "<any value>", "<optional success metadata>": "..."}
```

```json
{"error": "<human-readable message>", "<optional error metadata>": "..."}
```

`result` and `error` MUST NOT coexist. An already-enveloped backend result is
passed through; any other return value is placed under `result`. An ordinary
raised exception becomes `{"error": String(exception)}`. Cancellation,
process termination, and equivalent non-ordinary control-flow exceptions MUST
propagate and MUST NOT be converted into a tool error.

Callable signatures are part of the tool schema. Wrapping MUST preserve the
original parameter names, types, defaults, and documentation visible to the
agent runtime.

### 2.2 Honest pages

A bounded or paginated collection has this schema:

```json
{
  "result": [],
  "total_items": 123,
  "returned": 20,
  "truncated": true,
  "offset": 40,
  "limit": 20
}
```

- `total_items` is the count before page truncation.
- `returned` is the number of items in this response, even when `result` is a
  formatted string rather than an array.
- `truncated` is `total_items > returned` for an un-offset capped result. A
  windowed byte/character API SHOULD additionally expose an end offset because
  the generic rule alone cannot describe whether a later window exists.
- `offset`, `limit`, `kind`, `notice`, and similar fields are optional metadata.

### 2.3 Default bounds

Unless an agent factory supplies stricter values, the tool layer uses:

| Quantity | Default |
|---|---:|
| filesystem list page | 100 entries |
| text read/diff output | 50,000 UTF-8 bytes |
| lines in one file read | 2,000 |
| files scanned by one filesystem glob/grep | 100,000 |
| source-parser walk depth | 50 directories |
| files scanned by source-parser tools | 100,000 |
| graph caller/callee results | 200 |
| graph paths | 25 |
| graph path depth | 30 |

Call-specific positive values override defaults. Negative offsets are clamped
to zero where documented. A reconstruction SHOULD validate every externally
supplied limit as a non-negative bounded integer; compatibility exceptions are
listed in section 13.

## 3. File result formats

### 3.1 Entry and location schema

A filesystem entry has the logical fields:

```json
{
  "kind": "file",
  "name": "app.py",
  "path": "/src/app.py",
  "size": 321,
  "filetype": "<optional detector result>",
  "loc": {
    "line_start": 7,
    "line_end": 7,
    "content": "matched excerpt"
  }
}
```

`kind` is `file` or `dir`; directories report size zero. Location line indexes
are zero-based and inclusive. When byte locations are selected, `byte_start`
is zero-based and `byte_end` is exclusive. Byte offsets are computed over UTF-8
encoding. Grep excerpts include surrounding line context, are capped at 500
characters with an ellipsis, and are ordered by `(path, line_start)`.

File-type detection is best effort. Detection failure yields no `filetype`, not
a failed file operation. Results may be cached per filesystem identity and MUST
be invalidated for a changed path and its descendants.

### 3.2 Rendering

The file formatter supports native object, newline-delimited JSON-string, and
XML renderings. The accepted `yaml` and `markdown` selections intentionally
fall back to native JSON-compatible objects. Flags may independently suppress
basic file information and file-type information. XML MUST escape dynamic text.

### 3.3 Bounded text

File text is cut at the first of the byte budget or line budget. The preferred
cut is a complete-line boundary. A truncated result appends a footer reporting:

- the number of lines actually emitted;
- the number of lines remaining;
- `read_file offset=<next-zero-based-line>` when a safe resume point exists.

Footer fitting may remove additional complete lines; all footer counts and the
resume offset MUST reflect the lines ultimately emitted. If no full line plus
footer fits, return a UTF-8-safe byte prefix and a `truncated mid-line` footer.
The footer MUST NOT advertise the same offset again for a single overlong line;
if later lines exist it may advertise the following line. Empty input returns
empty output.

## 4. Filesystem abstraction and confinement

### 4.1 Required filesystem operations

The tool layer consumes an abstract filesystem with `exists`, `is_file`,
`is_directory`, `info`, `list`, `walk`, `glob`, byte/text read and open, and,
for writable use, create/write/copy/move/remove operations. Backends may be
local, in-memory, remote, or a merged overlay, but paths visible to an agent use
forward slashes and a virtual root `/`.

Unicode path input is normalized consistently before comparison. Invalid
Unicode is treated as a missing path. Relative tool paths are resolved against
the toolset's configured root; absolute virtual paths remain absolute.

### 4.2 Rooted local filesystem

The local source adapter maps virtual `/x` to a canonical host path under one
canonical project root. Construction fails unless the root is a directory.

It MUST enforce these rules:

1. A path whose resolved target is outside the root appears nonexistent.
2. `..` traversal in a glob returns no matches.
3. Public results use virtual paths and MUST NOT disclose the host root.
4. Walk and directory listing do not descend through symlink directories and
   do not enumerate symlink names.
5. A direct in-root symlink whose resolved target is also in the root is
   readable for compatibility; an out-of-root target is blocked.
6. Missing or blocked paths yield empty list/glob results or normal
   not-found behavior. An existing unreadable path raises a sanitized error;
   it MUST NOT masquerade as an empty directory.
7. Walk and glob never follow symlink directories.

The path mapper accepts virtual paths, `file://` paths, and already-absolute
host paths only when the latter resolve inside the root. It validates and
returns the same canonical target. Implementations that can use descriptor-
relative opens with no-follow flags SHOULD do so to close replacement races.

### 4.3 Glob semantics

Glob matching is path aware and applies to relative paths without a leading
slash:

- `*`, `?`, and bracket expressions never cross `/`;
- `**` spans zero or more path segments;
- `**/*.py` therefore includes a top-level `a.py`;
- a relative pattern is rooted under the explicit `path` argument;
- results are unique and lexicographically sorted.

The rooted local and overlay implementations count scanned files, stop at the
configured ceiling, and return a truncation flag. The frontend adds
`walk_truncated: true` and a notice asking the caller to narrow its path or
pattern. A generic third-party backend may lack the scanned-glob extension; its
fallback glob is currently unbounded and is a reconstruction risk.

### 4.4 Ignore policy

Default ignore patterns cover version-control metadata, Contractor state,
virtual environments and caches, dependency/build directories, common binary
media and archives, fonts, binary office documents, editor state, and OS
metadata. Caller patterns are appended, empty patterns are removed, and exact
duplicates preserve first occurrence. A pattern is matched against both the
normalized full path and basename. Reading or writing an ignored path returns
`path <normalized-path> is ignored`; directory listing itself may enter an
otherwise ignored parent but filters ignored children.

## 5. Read-only file tools

The standard registry always contains `ls`, `glob`, `read_file`, and `grep`.
It optionally contains the interaction tools below.

| Tool | Inputs | Result and behavior |
|---|---|---|
| `ls` | `path` | Immediate non-ignored children in the selected format. Missing path is an error. |
| `glob` | `pattern`, optional `path`, `offset=0` | Sorted entries, capped by the configured item limit, with honest page metadata and optional walk-truncation metadata. |
| `read_file` | `file`, optional zero-based line `offset`, optional line `limit`, `with_line_numbers=false` | UTF-8 text; undecodable bytes are ignored. Numbered output uses `N | text` with one-based display numbers. Offset past EOF returns `""`. |
| `grep` | regular expression `pattern`, optional `path`, `offset=0` | Match entries with location/excerpt metadata. Invalid expressions return a diagnostic containing the expression and parser error. Unreadable individual files are skipped. |
| `interaction_stats` | `path="/"`, `pattern="**/*"` | Counts total/touched/untouched files and a percentage; an empty universe is 100 percent explored. |
| `list_touched_files` | path/pattern/page | Files read or matched. |
| `list_untouched_files` | path/pattern/page | Files neither read nor matched. |
| `list_match_only_files` | path/pattern/page | Files with a grep match and no full read. |
| `reset_interaction_tracking` | none | Clears counters and returns `"ok"`. |

A grep over one file is paginated like a directory grep. A directory grep
counts every visited filename against the scan ceiling, including ignored or
unreadable files. Only a file with at least one match is recorded as matched.
A file read is recorded after the backend returns its content.

Interaction state per path is:

```json
{
  "path": "/a.py",
  "has_read": true,
  "has_match": true,
  "read_count": 2,
  "match_count": 1,
  "operations": {"read_file": 2, "grep": 1}
}
```

The state resets when a new non-null model-runtime invocation ID reaches the
same tool instance. With no invocation context, direct callers retain
cumulative behavior. Optional coverage-gap capture walks at most 20,000
candidates, retains the lexicographically first 2,000 non-ignored files, and
runs at most once per invocation. It is off by default.

## 6. Writable file tools

Writable tools wrap a non-overlay filesystem in a memory overlay by default.
Their registry contains all read tools plus the operations below. Successful
simple mutations return `{"result":{"ok":true,"op":"...",...}}`.

| Tool | Contract |
|---|---|
| `write_file(path, content, encoding="utf-8")` | Create or replace the entire file; return encoded byte size. |
| `append_file(path, content, encoding="utf-8")` | Append, creating when supported; return appended encoded size. |
| `mkdir(path, create_parents=true, exist_ok=true)` | Create a directory. |
| `rm(path, recursive=false)` | Path must exist; non-empty directory requires recursion. |
| `cp(src, dst, recursive=false)` | Source must exist; validate both paths. |
| `mv(src, dst, recursive=false)` | Copy then remove; recursive directory moves reject self/descendant targets. |
| `insert_line(path, content, anchor, where="before", occurrence=1)` | Find the one-based occurrence of an anchor substring in existing lines and insert before/after it. Empty anchor and occurrence below one are errors. Preserve CRLF/LF. Exact adjacent duplicate is a successful no-op. |
| `edit(path, old_string, new_string, replace_all=false, encoding="utf-8")` | Literal replacement. A missing file is created only when `old_string` is empty. An existing file rejects empty `old_string`. Without `replace_all`, exactly one occurrence is required. Preserve the file's CRLF style when the supplied LF fragment does not match verbatim. Decode errors fail rather than rewrite lossy text. |
| `replace_range(path, start_line, end_line, content, preserve_trailing_newline=true)` | One-based inclusive replacement. `end_line=start_line-1` inserts before `start_line`; `(line_count+1,line_count)` appends. Bounds are validated. Empty content deletes. |
| `restore(path, recursive=true)` | Overlay only. Drop overlay changes and reveal an underlay path; the underlay path must exist. |
| `changed_paths()` | Overlay only. Return sorted `added`, `modified`, and `deleted` path arrays. Empty overlay-only directories are omitted. |
| `diff(root="/", context_lines=3)` | Overlay only. Unified text diff, bounded by the text-output cap. |

Write content is not length-limited by this layer. A deployment that accepts
untrusted model output SHOULD impose a maximum mutation size and storage quota.

## 7. Copy-on-write memory overlay

### 7.1 Effective view and state

The overlay has a read-only underlay plus three in-memory collections:

- `files: path -> bytes` for materialized files;
- `dirs: set(path)`, always including `/`;
- `deleted: set(path)` tombstones.

A tombstone hides its path and every descendant. Reads prefer overlay bytes,
then the underlay. Writes never modify the underlay, create missing parents,
remove covering tombstones, and reject crossing a file in the parent chain.
Directory/file type conflicts use normal file-exists/not-a-directory errors.
All state operations on one overlay instance are serialized by a re-entrant
lock.

Recursive removal deletes overlay descendants and adds an underlay tombstone at
the subtree root. Removing `/` is prohibited through the directory-removal
operation. Recursive copy materializes an empty destination directory even
when the source is empty. A write handle buffers bytes/text and commits to the
overlay when closed; `x` requires absence, `a` starts at EOF, and `+` permits
read/write semantics.

### 7.2 Full snapshot format

Internal state export has this versioned schema:

```json
{
  "version": 1,
  "kind": "overlay_state",
  "state": {
    "files": {"/a.bin": "<base64>"},
    "dirs": ["/empty"],
    "deleted": ["/old"]
  }
}
```

Paths and map keys are sorted when exported. `/` is implicit in `dirs`.
Import rejects a non-`overlay_state` kind and replaces current overlay state.
An in-memory `snapshot()` may be restored explicitly or as the most recent
snapshot; absence of either is an error.

### 7.3 Base-relative patch format

`save(root)` produces deterministic patch version 1:

```json
{
  "version": 1,
  "kind": "overlay_patch",
  "root": "/",
  "patches": [
    {"op": "delete_path", "path": "/old", "type": "file", "base_hash": "<sha256>"},
    {"op": "create_dir", "path": "/empty"},
    {"op": "write_file", "path": "/new", "content_b64": "..."},
    {"op": "write_file", "path": "/changed", "base_hash": "<sha256>", "content_b64": "..."}
  ]
}
```

Generation first emits sorted deletions, then visits visible paths in sorted
order. New empty directories get `create_dir`; non-empty parents are implied by
child writes. New files carry bytes only. Modified or deleted underlay files
also carry the SHA-256 of the observed underlay bytes. A file/directory type
change emits `delete_path` followed by recreation.

`load` rejects another kind and unknown operations. It resets first unless
asked not to. Before overwriting or deleting an existing underlay file with a
`base_hash`, it recomputes SHA-256 and fails on a mismatch, including an
expected-file/actual-directory mismatch. File content is strict base64.

### 7.4 Diff

The diff visits the sorted union of underlay and effective paths. It emits
`diff --overlay a/path b/path` plus one of `new file`, `modified file`,
`deleted file`, `new directory`, `deleted directory`, or a type-change marker.
UTF-8 data without a NUL byte receives a unified diff; other data emits
`Binary files differ`. Text lines are normalized to newline-terminated diff
input, and the complete result ends in one newline. No changes returns `""`.

### 7.5 Fork and merge

A fork wraps the same read-only underlay and replays a saved patch. The merge
algorithm considers only materialized file bytes that differ from the captured
pre-fork `files` map:

1. A single writer wins.
2. Identical bytes from multiple writers are not a conflict.
3. Different bytes are a conflict; the longest byte sequence wins, with fork
   order breaking equal-length ties. The conflicted path is returned and
   logged.
4. Directory sets are unioned.
5. Only tombstones created after the fork baseline are unioned. If no explicit
   baseline is supplied, each standard fork's recorded tombstone baseline is
   used.

This is a compatibility algorithm specialized for additive trace annotations,
not a general three-way merge. Section 13 lists its unresolved cases.

## 8. GitLab-backed read-only filesystem

The remote adapter presents one repository/ref as an in-memory read-only tree.
Its settings and bounds are:

| Setting | Default / rule |
|---|---|
| base URL | `https://gitlab.com`, trailing slash removed |
| ref | `master` |
| page size | 100, constrained to 1..100 |
| total request timeout | 60 seconds, positive |
| concurrent downloads | 3, at least 1 |
| maximum file bytes | 50 MiB, non-negative |
| retries after first request | 5, non-negative |
| backoff | `5 * 2^attempt` seconds |
| retry statuses | 429, 500, 502, 503, 504 |

Authentication priority is private token, OAuth bearer token, then CI job
token. The project ID and file path are percent-encoded as single URL
components. The adapter fetches the recursive tree, publishes a tree-only
index, downloads blobs under a semaphore, then atomically replaces it with the
full index. Files above the maximum size remain listed but reading them fails;
per-file failures are available as load errors and do not fail the whole load.

Loading states are `not_started`, `loading_tree`, `loading_files`, `ready`, and
`failed`. Nonblocking construction starts a background thread. While incomplete:

- `glob` uses an available tree or fetches the tree synchronously;
- `grep` uses the remote blob-search API;
- a file read uses the single-file API;
- stat/list/exists wait up to 30 seconds for the tree and walk waits up to 60.

Retries apply only to configured statuses and transport/timeout failures.
Permanent HTTP statuses fail immediately. Every retry response is released.
Cancellation MUST propagate through async loading rather than be recorded as an
ordinary per-file failure.

Remote grep returns `path`, one-based `line_number`, `line`, and `match`.
Invalid regex falls back to literal matching for this backend. Once ready,
search operates only on successfully cached file data. All write/open-append
modes are rejected as read-only.

## 9. Source structure tools

### 9.1 Definition and symbol search

Supported language identifiers are `python`, `javascript`, `typescript`,
`tsx`, `go`, `rust`, `java`, `kotlin`, `c`, `cpp`, `c_sharp`, `ruby`, `php`,
`scala`, `swift`, `lua`, `elixir`, `haskell`, and `bash`, selected by common
source extensions.

`search_def(symbol, path="", language="")` scans candidate source text
case-insensitively for the bare final symbol segment, then parses only those
candidates. A structural hit returns up to 50 rows with `symbol`, `file`,
one-based `line`/`end_line`, zero-based `column`, parser `node_type`,
`language`, and at most 15 context lines; `kind` is `definition`. If no
structural definition exists, up to 20 line matches with two context lines are
returned as `kind: grep_fallback`. No hit is `kind: none`. Unknown languages
return an error listing supported identifiers.

`list_symbols(path="", language="", node_type="", offset=0, limit=null)`
returns sorted symbol rows. Its default page is 300. Parsing/read failures are
skipped. Parsed trees and symbol resolutions are content-hash cached; explicit
file invalidation removes both caches for that path.

The walk is bounded by depth and file count, deduplicates path strings, and
stops silently except for runtime logging. Results do not currently carry a
walk-truncation flag.

### 9.2 Call graph tools

Graph tools are attached only when the filesystem can be resolved to a local
host root. The graph is built lazily once and reads that host tree, not
uncommitted overlay bytes. Returned host paths may be rewritten to the agent's
virtual root; a resolver result of null preserves the original path.

The registry is:

- `graph_summary()`;
- `find_symbol(symbol)`, capped at 50;
- `find_callers(symbol, max_results)` and `find_callees(...)`, including edge
  confidence and unresolved symbolic callees;
- `paths_between(src, dst, max_paths)`;
- `entrypoint_paths_to(symbol, max_paths, max_depth)`;
- `attack_surface()`;
- `complexity_hotspots(threshold=10)`;
- `functions_that_raise(exception)`.

A slim node contains `id`, `name`, `kind`, `file`, `start_line`, and
`end_line`. Unknown caller/callee symbols return an empty page with a note.
Non-UTF-8 files are skipped rather than aborting graph construction.

### 9.3 Structured annotations

The tools `annotate_trace`, `annotate_validate`, and `annotate_sink` locate a
real structural function definition in the requested file; grep fallback is
never accepted. They insert one indentation-preserving comment immediately
above it, retain LF/CRLF style, and use `#` for Python, Ruby, shell, Elixir, and
PHP and `//` otherwise.

Canonical lines are:

```text
@trace target=<id> [args=name:state,...] [calls=symbol,...]
@validate arg=<name> kind=<validation-kind>
@sink kind=<sink-kind> arg=<name-or-unknown>
```

Trace argument state is one of `tainted`, `validated`, `clean`, or `derived`.
An entry is `name:state`; an empty name or unknown state is an error.
Validate requires both `arg` and `kind`; sink requires `kind`. A same-kind
annotation immediately above the function is rejected as a duplicate. Success
returns file/function/kind, inserted line, shifted function line, and comment
text.

## 10. Task-plan tools

### 10.1 Models and state machine

A subtask is:

```json
{
  "task_id": "1.2",
  "title": "<non-empty action title>",
  "description": "<non-empty scope and deliverable>",
  "status": "new"
}
```

IDs are zero-based dotted numeric paths. New root tasks use the next root
number; decomposition adds `.1` through `.3`. Valid transitions are:

```text
new        -> done | incomplete | malformed | skipped
incomplete -> decomposed | skipped
malformed  -> decomposed | skipped
done | decomposed | skipped -> no transition
```

A worker result contains exact `task_id`, status `done|incomplete`, factual
`output`, and `summary`. One malformed/unparseable result is represented by a
runtime-generated `malformed` record retaining capped raw output.

### 10.2 Registry and rules

| Tool | Rule |
|---|---|
| `add_subtask(title, description)` | Reject when the externally configured plan capacity is reached. |
| `get_current_subtask()` | Error when no tasks exist; otherwise return the indexed task, including an unresolved incomplete/malformed task. |
| `list_subtasks(view="remaining")` | `remaining` returns current and later plan entries; `all` returns history. |
| `get_records()` | Most recent records, default cap 20. |
| `execute_current_subtask()` | Only `new`; claim before the first await; default total attempt budget three. Empty, unparseable, and mismatched-ID responses consume attempts. |
| `decompose_subtask(task_id, decomposition)` | Current ID must match and status must be incomplete/malformed. Accept wrapper object or a bare 1..3 array. Capacity includes parent plus children. |
| `skip(task_id, reason)` | Optional registry entry. Nonblank reason; only new/incomplete/malformed. An incomplete non-final task must normally be decomposed while capacity remains. |
| `finish(status, result)` | `done` requires at least one done task and no new tasks; `failed` may finish without completed work. Terminates the invocation after persistence. |

An execution claim contains current task ID and a unique call ID in task-scoped
state. Duplicate same-turn calls are rejected. Completion rechecks both claim
and current task so a result cannot be applied after the plan advances. The
claim is released on success, validation error, ordinary exception, and
cancellation.

The default malformed/raw string field cap is 20,000 characters plus a
truncation marker. The optional final summarizer has no tools and receives only
the latest capped records. JSON, YAML, Markdown, and XML subtask/result formats
are supported. Parsing tolerates a matching fenced block and sanitized model
wrappers; the structured schema remains authoritative.

## 11. Skills and deterministic observations

### 11.1 Skill package loading

A skill is a directory under the configured skill root. Every recursive `.md`
file is loaded in lexical path order. `index.md` becomes note name `<skill>`;
another file becomes `<skill>/<relative-path-without-extension>`. A leading
YAML frontmatter mapping may supply `description`; malformed/non-mapping
frontmatter is treated as ordinary content. Defaults are `<skill> skill` for
the index and `<skill> skill / <relative-path>` for references.

Injection writes each file as a memory note tagged `skill` and the owning skill
name in `user:memory/<task-namespace>`. Unknown configured skill directories
fail before task execution. `skills_read` resolves, in order, exact canonical
name, unique suffix, then unique basename; a trailing `.md` is ignored. An
ambiguous fallback is not a match. Successful reads record the canonical name.

### 11.2 Observation projection

Raw worker facts use the logical state keys `worker_usage`, `skills_read`,
`memories_written`, `memories_read`, and `file_paths`. Projection is disabled by
default and supports switches for tool counts/errors, file counters, up to 25
read paths, up to 25 unvisited in-scope paths plus a remainder marker, skills,
memories, malformed-only mode, persisted-record inclusion, and immediate-result
inclusion. A configured tool allowlist filters tool counts.

The environment override `CONTRACTOR_EVAL_OBSERVATIONS` is a JSON object merged
field by field over workflow configuration. Invalid JSON, a non-object,
unknown fields, or non-string `tracked_tools` entries are configuration errors.
Success records omit an entirely empty projection; malformed records retain it
because absence of tool activity is diagnostic.

## 12. Error and cancellation matrix

| Condition | Required outcome |
|---|---|
| missing/invalid/ignored path | error envelope, except sandbox escape enumeration appears empty |
| unreadable existing directory | sanitized error, not empty list |
| malformed regex in generic grep | error with parser detail |
| remote-file partial load failure | file listed; read fails; failure available in loader diagnostics |
| stale overlay base hash | hard error before that operation is applied |
| unsupported patch/state kind or operation | hard error |
| parser failure on one source file | skip file and continue |
| missing local graph capability | omit graph tools |
| duplicate task execution | error; winning call retains claim |
| task-plan cancellation | release owned claim and propagate cancellation |
| frontend ordinary exception | error envelope |
| frontend cancellation | propagate unchanged |

## 13. Compatibility gaps and reconstruction risks

These behaviors exist in the inspected implementation but SHOULD NOT be treated
as desirable design without an explicit compatibility decision:

1. Root listing hides all symlink names while direct in-root symlinks are
   readable. The documented phrase “symlinks are never followed” is therefore
   not literally true. Canonical-path checking also does not fully prevent a
   hostile local process replacing the final path with a symlink between check
   and open; descriptor-relative no-follow access is stronger.
2. Generic filesystem backends without `glob_scanned` bypass the scan ceiling.
   Source-structure scans stop at their ceiling without returning truncation
   metadata, so an agent can mistake partial results for complete results.
3. Overlay patch hash checks are skipped if a formerly existing base file has
   disappeared; a stale modification can then recreate it. Version is emitted
   but not validated. Patch application is operation-by-operation, not
   transactional, so an error may leave a partially applied overlay.
4. Fork merge does not perform a three-way merge. A longest-byte winner can be
   semantically wrong; equal-size ties depend on fork order. Concurrent
   write/delete and file/directory conflicts are not resolved, tombstones may
   hide a selected write, and the merge does not explicitly invalidate file
   type caches.
5. Reloading the GitLab adapter does not cancel/join a previous background
   loader. A stale loader can race a new ref/settings load. The complete
   repository is retained in memory and tree/search fallback calls can be
   expensive despite per-file and connection bounds.
6. Structural annotation duplicate detection examines only the line directly
   above a definition. Interleaving different annotation kinds can hide an
   older same-kind annotation and permit a duplicate. Annotation fields accept
   newlines and are not escaped, and read-locate-write is not atomic, so
   concurrent edits to one file can be lost.
7. Local call-graph construction reads the host underlay and can be stale after
   overlay edits. Its UTF-8 hardening changes parser-library process-global
   state. Several graph tools pass through backend-sized collections without
   applying the advertised result cap.
8. Skill names are joined to the skill root without first requiring a single
   safe path segment and canonical containment. Configuration is trusted today;
   a hardened loader must reject traversal and symlink escapes.
9. Numeric task/file/graph limits are not uniformly validated. Zero/negative
   retry or page values can produce surprising slices or an immediate malformed
   result instead of a configuration error. File mutations have no byte quota.
10. Task claims rely on the runtime's single-threaded shared-state update
    contract. They are not a cross-process lock.

## 14. Reconstruction acceptance checks

A conforming implementation MUST at minimum test:

1. parent traversal, absolute-host escape, out-of-root symlink escape, hidden
   symlink enumeration, sanitized unreadable-directory errors, and Unicode
   normalization;
2. path-aware `*`/`**` parity across local, overlay, and GitLab-tree matching;
3. honest glob/grep/read truncation including a single giant UTF-8 line;
4. read/match interaction separation and per-invocation reset;
5. exact edit, CRLF adaptation, non-UTF-8 refusal, insertion, range insertion,
   deletion, and EOF append;
6. overlay snapshot and patch round trips for bytes, empty directories,
   deletions, type changes, stale hashes, recursive copy/move, and binary diff;
7. fork conflict reporting and pre-fork tombstone suppression;
8. GitLab retry classification, response release, max-file rejection, and glob
   parity before/after readiness;
9. structural definition fallback, parser-cache invalidation, graph path
   rewriting, non-UTF-8 graph skip, annotation validation, and duplicate
   rejection;
10. every task transition, capacity edge, malformed retry, duplicate claim,
    stale completion, cancellation release, and finish precondition;
11. skill index/reference naming, safe frontmatter, ambiguous alias rejection,
    and deterministic observation caps.

## 15. Authoritative implementation inventory

The behavior above was reconstructed from `cli/fs.py`, every module under
`contractor/tools/fs`, `contractor/tools/code`, `contractor/tools/tasks`,
`contractor/tools/observations.py`, `contractor/tools/result.py`, and
`contractor/runners/skills.py`, together with their unit tests under
`tests/units/contractor_tests/tools` and the rooted-filesystem tests in
`test_fs.py`. This inventory is informative; the contracts in this document
are the language-neutral authority.
