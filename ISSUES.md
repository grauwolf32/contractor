# Known Issues / Bug Log

Findings from a full-codebase bug hunt (clean working tree, no diff under review).
Severity reflects correctness/impact, not effort. HIGH items below were each
verified — by executable repro or by code+test inspection — and the verification
is recorded inline.

Status legend: ☐ open · ☑ fixed

---

## HIGH

### ☑ H1 — TPM rate limiter never accumulates tokens across requests — FIXED
**`contractor/callbacks/ratelimits.py:53-55`** (`TpmRatelimitCallback.__call__`)

```python
diff = token_count - (self.token_count or 0)   # 53
els  = current_time - self.timer_start         # 54
self.token_count = token_count                 # 55  <-- rebaselined on EVERY call
...
if diff > self.tpm_limit:                       # 57
    delay = 60 - els + 1
```

`self.token_count` is reset to the current cumulative count on *every* invocation,
so `diff` only ever measures one LLM round-trip — not tokens since `timer_start`.
The `timer_start`/`els` window is only reset inside the `if diff > tpm_limit`
branch, so it never spans a minute. Net effect: the per-minute token limit trips
only if a **single request** exceeds `tpm_limit`. (The sibling `RpmRatelimitCallback`
accumulates `request_count` correctly — only TPM is broken.)

**Verified (repro):** with `tpm_limit=10000`, ten 5k-token requests in the same
second (50k tokens) produced **zero** throttle sleeps; a single 15k-token request
was the only thing that tripped it.

**Fixed** (`contractor/callbacks/ratelimits.py`): `self.token_count` now stays the
window baseline; it is rolled (with `timer_start`) only when the window resets —
either after sleeping out an over-budget window, or naturally once `els >= 60`.
Re-verified: 50k tokens in 1s against a 10k limit now throttles (3 sleeps) instead
of 0.

---

### ☑ H2 — `RootedLocalFileSystem.glob` ignores subdirectories for non-recursive patterns — FIXED
**`cli/fs.py:158-165`** (and over-match at **`cli/fs.py:191-196`**)

```python
if recursive:
    walker = os.walk(self.root_path, followlinks=False)
else:
    top_entries = os.listdir(self.root_path)
    walker = iter([(self.root_path, [], top_entries)])   # root dir ONLY
```

Any pattern without `**` walks only the root directory, so a pattern that names a
subdirectory never matches files inside it. `glob` is a primary discovery tool for
the agents, so they wrongly conclude files don't exist. Separately, the recursive
fallback at lines 191-196 matches only the pattern's tail segment, ignoring the
leading directory part, so it over-matches top-level files.

**Verified (repro):** in a tree with `/top.py`, `/sub/b.py`, `/sub/deep/c.py`:
- `glob('sub/*.py')`   → `[]`            (expected `['/sub/b.py']`)
- `glob('/sub/*.py')`  → `[]`            (expected `['/sub/b.py']`)
- `glob('sub/**/*.py')`→ `['/sub/deep/c.py', '/top.py']`  (leaks `/top.py`, misses `/sub/b.py`)

**Fixed** (`cli/fs.py`): replaced the non-path-aware `fnmatch` matcher (whose `*`
crossed `/`) and the tail-pattern fallback with a path-aware `_glob_to_regex`
translator — `*`/`?`/`[...]` stay within a segment, `**` spans any number of
segments. `glob` now always walks the full tree, so non-recursive subdir patterns
work. Re-verified across 9 cases: `glob('sub/*.py')` → `['/sub/b.py']`,
`glob('sub/**/*.py')` → `['/sub/b.py','/sub/deep/c.py']` (no `/top.py` leak).

---

### ☑ H3 — docs said "in a row" but `iterations` is cumulative ("in total") — DOCS FIXED
**`contractor/runners/task_runner.py:784-831`** (`_run_task_with_retries`)

The success counter is **cumulative** (a failure does not reset it), so with
`iterations=2` the sequence `success → fail → success` is accepted as DONE. This is
the **intended** behavior — `iterations` means N successful runs *in total* across
the `max_attempts` budget, not a consecutive streak.

The bug was in the documentation, which said "in a row":
- `CLAUDE.md:85` and `docs/README.md:135` previously read "successful runs **in a row**".

**Verified (code + test):** the code (no `successful_runs = 0` reset on failure) and
`tests/units/contractor_tests/runners/test_task_runner.py:263`
(`test_failure_does_not_increment_successful_runs`) both implement cumulative
semantics correctly; only the two doc lines were wrong.

**Resolution:** corrected `CLAUDE.md` + `docs/README.md` to say "in total
(cumulative across attempts — a failure does not reset the count)". No code change.

---

## MEDIUM

### ☑ M1 — `_fmt_error` only ever inspects the `"error"` key — FIXED
**`cli/render.py:439`**

```python
for field in ("error", "error_message", "errors"):
    if result.get("error") not in (None, "", [], {}):   # bug: should be result.get(field)
        err_key = field
        break
```

The loop iterates three candidate field names but the predicate always reads
`result.get("error")`. A tool result whose error lives under `error_message` or
`errors` (no `error` key) renders nothing — the error is silently dropped from the
non-UI renderer. **Fixed** (`cli/render.py`): predicate now reads `result.get(field)`.
Re-verified: results keyed `error_message`/`errors` now render.

### ☑ M2 — `edit` silently corrupts files containing non-UTF-8 bytes — FIXED
**`contractor/tools/fs/write_tools.py:506`**

```python
current_content = self.fs.read_text(normalized_path, encoding=encoding, errors="ignore")
```

`edit` (the documented primary editing tool) reads with `errors="ignore"`, dropping
undecodable bytes, then writes the cleaned text back (`_commit_write` uses
`errors="strict"`). Verified by sub-agent: a file with `\xff\xfe` shrank 23→21 bytes
after editing an unrelated region, still reporting `ok/changed`. Sibling tools
(`replace_range`, `insert_line`) correctly abort on a decode error. **Fixed**
(`contractor/tools/fs/write_tools.py`): dropped `errors="ignore"` so `edit` fails
loudly (returns `{"error": ...}`) on non-decodable files like its siblings.
Regression test: `test_edit_does_not_corrupt_non_utf8_file`.

### ☑ M3 — OpenAPI `$ref` mutual-recursion inlines asymmetrically / blows up — FIXED
**`contractor/tools/openapi/ref_resolver.py:108-116`**

`resolve_local_refs` seeds each schema only with its own self-ref, so a cycle
`A→B→A` resolves to different depths depending on entry point, and N
mutually-recursive types inline each cycle ~N times (quadratic/exponential on dense
specs). Terminates, but corrupts output shape and can balloon memory. **Fixed**
(`contractor/tools/openapi/ref_resolver.py`): each named schema is now resolved
against the *original* `schema` into a separate `resolved_named` dict that is
written back only after all schemas are done, so a partially-inlined schema can no
longer feed into a later one's resolution — making the output independent of
iteration order.

### ☑ M4 — Caido fuzz offsets are character indices, not byte offsets — FIXED
**`contractor/tools/caido.py:395-405, 712-722`**

`_find_placeholder_offsets` uses `str.find` on decoded text and feeds char offsets
into Automate, which expects byte offsets against the raw base64 request. Any
multibyte UTF-8 before the fuzz target (hostnames, cookies, prior params)
misaligns the mutation window. **Fixed** (`contractor/tools/caido.py`):
`_find_placeholder_offsets` now takes `raw: bytes` and searches
`target.encode("utf-8")`; the call site decodes the base64 Blob with
`base64.b64decode` and passes the raw bytes, so offsets are byte-aligned.

### ☑ M5 — `run_python` drops the `exit_code` it documents — FIXED
**`contractor/tools/podman.py:341-353`**

`_exec` returns `rc`, bound as `rc, out, err = ...`, but the returned dict omits it
while the docstring promises an `exit_code` field. The model is told to read a field
that never exists, so it can't distinguish script success from failure. **Fixed**
(`contractor/tools/podman.py`): the `rc` was actually dropped one layer deeper —
ADK's `CodeExecutionResult` has no exit-code field. Added a thin `_ExecResult`
subclass carrying `exit_code`; `run_python`/`run_bash` now populate it and both
tools return `"exit_code"` (and `execute_bash`'s docstring lists it). Regression
tests: `test_run_python_propagates_nonzero_exit_code`,
`test_run_bash_propagates_nonzero_exit_code`, `test_execute_bash_surfaces_exit_code`.

---

## LOW

- **L1** — `assert` in production code (stripped under `-O`, against project rule):
  `contractor/callbacks/ratelimits.py:28`, `contractor/callbacks/guardrails.py:67,155,316`.
- **L2** — `SubtaskDecomposition` allows >3 children despite the "1-3" contract — no
  `max_length=3` on the model (`contractor/tools/tasks/models.py:135-148`).
- **L3** — Missing upstream artifact injected as an empty string with no warning
  (`contractor/runners/task_runner.py:465-480` + `_helpers.py:24-41`); a typo'd
  `template_key` gives the downstream task a blank inbox instead of failing.
- **L4** — CLI validators raise bare `ValueError` → traceback instead of a clean
  `click.BadParameter`/`UsageError` (`cli/main.py:317-318`, `cli/utils.py`).
- **L5** — `ref.lstrip("#/")` strips a char-set not a prefix
  (`contractor/tools/openapi/ref_resolver.py:38`); use `ref[2:]` after the
  `startswith("#/")` check.
- **L6** — Mutable default arg `ext=[...]` (`contractor/tools/openapi/openapi.py:206`).
- **L7** — `_send_with_retries` mutates the shared `self._client.timeout`
  (`contractor/tools/http.py:456-487`); latent race if requests ever overlap.

---

## Verified NOT bugs (ruled out during the hunt)

- **Sandbox escape** — symlink and `..` traversal are blocked on both reads and
  writes; `_is_within_sandbox` uses `root + os.sep` so sibling-prefix dirs don't
  leak. Could not break out.
- Subtask state machine (`SUBTASK_STATUS_TRANSITIONS`) is enforced on every
  transition; `incomplete`/`malformed` are non-executable; `finish` refuses while
  any subtask is `new`. No bypass.
- Artifact naming round-trips: `{key}/{result|summary|records}` matches consumption
  keys and the `artifact__<safe>` template-var mapping.
- `MetricsSink` writes under an `asyncio.Lock` via `asyncio.to_thread`, opens/closes
  per append (no leak), and falls back to `repr` for non-serializable objects.
- `TokenUsageCallback` per-invocation accumulation is correct (no double-count).
