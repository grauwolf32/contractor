# V410-030: native controlled HTTP

2026-11-04, macOS arm64, Python 3.13.16, mitmproxy 12.3.1.
Branch: `codex/sitemap-source-comparison` (not merged into main).
Implementation: `fb127850b6e73fe38deed027be45e33cd84ab62e`.
Task: [V410-030](../done/v410-030-mitmproxy-controlled-http.yml).

## Delivered boundary

HTTP tools and real pinned Nuclei execute through one native engine owned by the
ordinary allocation HTTPTransport. Programmatic requests enter native HTTP layers
as messages; native scanner HTTP/TLS ingress delegates each target request to
that same owner. The controlled httpcore dispatcher and handwritten h11/TLS
scanner parser are removed. Ordinary non-engagement HTTP retains its own existing
policy transport.

HTTP/1.1 supports keepalive. TLS ALPN offers HTTP/2 and HTTP/1.1; HTTP/2 supports
multiplexing and individual stream cancellation. HTTP/1.0 responses and scanner
requests are refused. HTTP/3 and WebSocket capability are not advertised.

Contractor retains route/method/identity admission, all-answer DNS policy,
numeric pinning and actual peer checks, lease/deadline checks, configured
engagement limits, redirects, safe pre-send retry, cookie epochs and bounded raw
results. Pools cannot cross credential identity or exact origin. Short session
locks protect snapshots and updates without serializing network requests.

The embedding uses native protocol parsers, HPACK, body framing, flow control,
TLS configuration and the connection dispatcher. Version-pinned adapters add
header bounds, queue cancellation/admission and I/O guards. Callback failures
stop execution, with content-free diagnostics. The allocation joins callbacks,
listeners and writers before erasing state and temporary certificates. Native
context, logging and TLS keylog configuration are saved/restored; TLS context
caches are cleared. Master.run is not called, so the Runtime's asyncio task
factory and exception handler are preserved.

## Real-path checks

- Independently observed HTTP/1.1 and HTTP/2 reuse on real private-interface sockets.
- Concurrent streams for one identity; cancellation resets one stream and leaves
  the connection usable. A remote MAX_CONCURRENT_STREAMS queue cannot send a
  cancelled request later. Native connection-permit waiters progress after
  earlier responses become idle.
- Fresh DNS rejection on an already reused connection; mismatched actual peer
  port and expiry after TCP/TLS setup send no forbidden HTTP application bytes.
- Credential/origin pool isolation observed through actual target headers and
  distinct connections. Header/body bounds cover HTTP/1.1 and HTTP/2.
- Real Nuclei and HTTP tools share the request allowance; scanner credentials,
  TLS verification, redirects, retry ownership, unsupported templates and
  cancellation/lease loss/timeout retain their conformance tests.
- Native frontend HTTP/1.1 and HTTP/2 are tested with real clients; HTTP/1.0,
  unauthenticated clients and oversized requests never reach the owner.
- Startup failure restores native process context and removes the private
  directory. Cleanup refusal retains the exact writer for retry. Injected
  callback content and TLS keylog sinks do not disclose secrets.
- Existing domain policy/session tests use an explicitly test-only engine double;
  they are not presented as native wire conformance.

## Memory observation

Reproduce from `runtime/`:

```sh
uv run --locked python ../scripts/experiments/mitmproxy_production_memory.py \
  --output /tmp/native-memory.json
```

[Three fresh-process samples](v410-030-native-memory.json) use the actual
production owner/engine, three rounds of four concurrent HTTP/1.1 responses of
8 MiB each, and retain the 32 MiB result set until each round is sampled.

| Observation | RSS |
| --- | --- |
| Runtime CLI and fixture imports | 125.89–126.03 MiB |
| Native engine ready plus one 8 MiB fixture payload | 176.55–176.66 MiB |
| Third round with four retained results | 216.23–216.44 MiB |
| After joined close and GC | 216.23–216.44 MiB |

The adapter retains no completed native flows or response futures; the regression
checks the lifetime of the first exchange kept by a native socket task. Four
idle sockets exist before release, all join on release, and the certificate
directory is removed. RSS is not a live-object count and does not immediately
fall when Python/native allocators retain pages. These measurements do not prove
that process RSS has returned to baseline. They also cannot be compared directly
with V410-028's discard-stream workload, which did not retain Contractor response
results. The fixture repeats a synthetic payload on macOS; these RSS values
are workload-specific observations, not a production capacity bound.

## Scope and remaining choices

The integration is pinned to mitmproxy internals; an upgrade requires repeating
native protocol/admission/lifecycle tests. A shared HTTP/2 connection failure can
fail other streams, with no automatic replay after an entered send. Scanner
frontend connections remain per invocation; outgoing target pools are allocation
resources. Optional HTTPX Brotli support installed by mitmproxy does not widen
bounded decoding: ordinary clients advertise only the supported gzip/deflate
codings, and other codings stay raw under HTP-R63.

Caido routing/removal, transparent Podman networking, new ffuf/Katana integration,
trusted sitemap observations and distributed quotas are outside this delivery.
Linux packaging and a live external Workflow were not exercised by this macOS
local-fixture gate.

## Validation

- `make test-runtime`: 5945 passed, 135 skipped; the Runtime report verifier passed.
  Skips are the existing environment-gated tests, not evidence of a pass for
  unavailable Podman/integration environments.
- `make lint`: passed, including Go/static/export, UI generation/gates, scripts,
  deployment fixture tests, release graph, task/spec/ERD and Python checks.
- After removal of the unused bridge-directory argument, the complete focused
  HTTP engine/bridge, real Nuclei and scanner package selection passed: 558 cases.
- `uv run ruff check .`, `uv run ruff format --check .` and
  `uv run lint-imports --no-cache --no-logo`: passed.
- The memory probe completed three fresh processes, 36 target responses total,
  with joined cleanup. Its Ruff check/format checks passed.
- `python3 scripts/check_spec.py` and `uv run python ../scripts/check_tasks.py`:
  passed. `git diff --check`: passed.

Go 1.26.9 and Node 24.20/Corepack from the existing local tool cache were placed
on PATH for the complete gates. No live model, external target scan, demo
migration, deployment or main-branch merge was performed.
