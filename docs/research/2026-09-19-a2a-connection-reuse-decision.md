# V57-005: A2A connections within one invocation

Date: 2026-09-19. Decision: **no-change for production**. The experiment completes
the hypothesis check but does not authorize enabling keepalive with a single flag.

Reuse does remove almost all repeated TLS handshakes during polling. However,
it changes behavior on certificate expiry and certain transport failures.
In the local environment, with the current 100 ms poll interval, the time saving
is much smaller than the reduction in connection count. There is no evidence
of a substantial real Run speedup or reduced production CPU usage.

Production code and settings were unchanged. V57-004 remains a separate task
for detailed discussion with the user.

## What was measured

Experiment baseline: `be33cebbf8b9b5fd833d238b2bf0547d5694f319`, Go 1.25.6,
A2A Go SDK 2.5.0, Linux amd64, Intel Core i7-7700K, 8 logical CPUs.
HTTP/1.1 runs over real loopback TLS 1.3 with a local CA, a Control Plane client
certificate and Runtime SPKI binding. HTTP/2 and the TLS session cache are
disabled; no network outside loopback or models are used.

The experiment uses real `Invoker.Invoke`, the SDK JSON-RPC client and production
`cloneBoundedHTTPClient`. The server is controlled: it sends valid SDK-encoded
responses without running Python Runtime. Counters observe TCP accepts, TLS peer
verification, decoded `SendMessage` and `GetTask`, rather than inferring them
from the number of HTTP requests.

The comparison covers current `NewMTLS`, a test copy of its transport with
`DisableKeepAlives=true`, and a test transport with reuse. Each experimental
transport is created within one `Invoke`, belongs to one endpoint and principal,
and explicitly closes idle connections in `Destroy`. The base TLS configuration
is loaded before the measured call, as in production; a new copy is allocated
for the invocation. Responses are not drained to enable keepalive.

There are two distinct workloads:

- Fixed state transitions: one `SendMessage`, followed by 8 `GetTask` calls in
  the benchmark. This compares equal RPC workloads; Task readiness depends on
  poll number. A separate 20-poll test checks parity with real `NewMTLS`.
- Fixed work duration: the server becomes ready 800 ms after receiving
  `SendMessage`, independently of poll count. The interval is 100 ms; the actual
  number of polls may differ. This test does not substitute client speed for
  Task work duration.

## Measurements

| Poll interval | Baseline, ms: median [min–max] | Reuse, ms: median [min–max] | TCP / TLS per invocation |
| --- | --- | --- | --- |
| 2ms | 38.58 [37.80–45.22] | 20.52 [20.41–23.53] | 9 → 1 |
| 100ms | 826.84 [824.44–833.75] | 809.21 [808.77–810.10] | 9 → 1 |

At 100 ms, the median difference is **17.63 ms, about 2.1%** in this synthetic
workload. This is not an estimate of real Run acceleration. At 2 ms, transport
overhead accounts for a much larger share; that interval is not the current setting.

In the separate fixed-800-ms-work experiment: baseline **822.55 ms**, 8 polls and
9 TCP/TLS; reuse **808.63 ms**, 8 polls and 1 TCP/TLS. This is one pair, without
an assessment of statistical significance. Under the race detector, baseline
made 7 polls and reuse 8: readiness time is preserved, while observation count
depends on client overhead. Race timings are not used for performance conclusions.

The parity test of real `NewMTLS` against the test baseline produced identical
**21 TCP, 21 TLS, 1 SendMessage, 20 GetTask**; reuse produced **1 TCP, 1 TLS**
with the same RPC count. Single-run parity-test timings are not benchmarks.

The benchmark runs without the race detector: 3 iterations per sample, 3 samples
per interval/reuse pair. PKI and server startup are excluded; SDK client
construction, invocation and owned-transport closure are included. This is a
small series on a shared workstation, not a statistically established production
benchmark. Raw values for every sample and individual test observations are
retained in the [evidence](../../tasks/evidence/v57-005.json).

## POST retries: what the experiment proves and what it does not

In SDK 2.5.0, both operations are HTTP **POST** requests created from `bytes.Buffer`,
so `Request.GetBody` is populated. The JSON-RPC SDK has no application retry loop.
Go 1.25.6 `Transport.shouldRetryRequest` can retry a request on a reused connection
after an error before any bytes were written, if the body can be recreated.
After a possible write, retry additionally depends on replayability; for POST,
`Idempotency-Key` and `X-Idempotency-Key` affect it.

The matrix runs through the SDK and bounded wrapper. A write fault is injected
above already verified TLS, immediately before/after one plaintext HTTP byte.
This path separately performs hostname/SPKI verification in `DialTLSContext`;
the benchmark uses the ordinary transport without this wrapper. The lost-response
fixture fully decodes the request and records its effect, then closes the socket
without an HTTP response.

| Case | Observation |
| --- | --- |
| First SendMessage, zero/partial write | Error, 0 decoded effects, no retry |
| Reused POST, zero write, GetBody retained | New TCP/TLS, successful retry, 1 target effect |
| Reused POST, zero write, GetBody=nil | Error, 0 effects, no retry |
| Reused POST, partial write of one byte, no idempotency header | Error, 0 decoded effects, no retry |
| Reused SendMessage, lost response, no header | Error, 1 effect |
| Reused SendMessage, lost response, either of the two headers | Success after retry, **2 effects**, identical RPC ID, body SHA256 and request ID |
| Same case, GetBody=nil | Error, 1 effect |
| Real SendMessage → GetTask order with poll failures | The original SendMessage remains the only one; GetTask itself may be retried |

The dangerous SendMessage retry was observed **after a deliberate read-only
warmup using the same SDK client**. In the current invocation, SendMessage is
first on a new transport; the experiment found no repeated dispatch in that
order. The header alone does not establish server idempotency either: the fixture
intentionally does not hide physical retries behind deduplication. Request ID
and JSON-RPC ID do not provide exactly-once effects.

Test-only `GetBody=nil` was sufficient to prohibit the observed hidden retries
of nonempty POST requests, including zero-write retry. This is a possible strict
“no automatic retry” policy, not an implemented fix. An alternative is to
explicitly permit proven zero-write retries for GetTask while prohibiting headers
that expand replayability. Policy selection belongs to a separate implementation.

## Certificates, allocation and connection termination

| Check | Result and boundary |
| --- | --- |
| Valid CA/SAN, wrong SPKI | Both strategies reject the peer before a protected HTTP request |
| New key at the same endpoint after closing the old connection | Old principal rejected; a newly and explicitly bound invocation passes a separate handshake |
| Certificate already expired at first connection | Both strategies reject the request before HTTP |
| Client verification clock passes NotAfter between SendMessage and poll | Baseline performs a new handshake and fails; reuse polls over the established session |
| Cancellation during poll delay, first SendMessage headers/body and active GetTask | `planner_cancelled`; 1 SendMessage, 0 polls before polling starts or 1 active poll; all connections closed |
| Deadline during first SendMessage headers/body and active GetTask | `worker_deadline_exceeded`, active request cancelled, all connections closed |
| Next allocation at the same endpoint | New owned transport, new handshake, exact tenant for each invocation |
| Simulated process retirement | Old sockets closed; a new request with the old tenant reaches HTTP, but the gate rejects it before Worker dispatch |
| SDK Destroy and CloseIdleConnections on the bounded HTTP client | Do not close the pooled socket; an explicit call on the original transport closes it |

Active GetTask was separately confirmed with `httptrace.GotConn`: baseline uses
a second fresh connection, while reuse uses the first again; on cancellation or
deadline, both sides close it.

Expiry is tested with a controlled client `tls.Config.Time`, without waiting for
calendar expiry or changing the system clock. Continuing an established TLS
session is normal keepalive behavior, not an SPKI-check bypass. It nevertheless
differs from the current client's behavior. **A transparent replacement preserving
certificate verification on every poll has not been demonstrated.**

A certificate identifies a principal, not a process or allocation. A same-key
restart does not make an old tenant valid. The retirement fixture models the
HTTP gate; it does not prove production watchdog, Registry or two-phase release
behavior. Closing the transport alone does not release an allocation. These
boundaries come from [Runtime/A2A](../spec/02-runtime-and-a2a.md),
[lifecycle](../spec/04-execution-lifecycle-and-metrics.md) and
[identity/configuration](../spec/07-runtime-labels-and-infrastructure-config.md).

## Bounded responses and reuse conditions

Each case sends two sequential SDK requests. The counter sits below the
production bounded wrapper and measures bytes the application reads from the
response body; `Close` on every body is checked separately.

| Response | Result | TCP: baseline / reuse |
| --- | --- | --- |
| Complete valid Content-Length | Decode succeeds | 2 / 1 |
| Complete valid chunked | Decode succeeds | 2 / 1 |
| Content-Length > 1 MiB | Rejected before body reading, 0 bytes | 2 / 2 |
| Chunked JSON requiring >1 MiB to complete | Decode rejected, exactly limit+1 bytes per response | 2 / 2 |
| Short malformed/truncated JSON | Decode rejected; a fully consumed entity may preserve the socket | 2 / 1 |
| Valid first JSON, then a deliberately unfinished response | First object accepted without EOF; body closed, server context cancelled | 2 / 2 |

The SDK decodes one JSON value and closes the body. This bound therefore does
not prove full HTTP entity-size validation, absence of trailing data, a wire-byte
limit or an RSS limit. The unfinished tail is a controlled partial-consumption
check, not a claim that every oversized response is rejected. This behavior
occurs in both strategies; the experiment changes neither the existing parser
nor validation.

## Decision and possible separate task

Retain `DisableKeepAlives=true` for now. Handshake count was reduced, but enabling
reuse still requires an agreed certificate-lifetime policy, explicit transport
ownership and a decision on hidden POST retries. One local series is insufficient
to justify that scope of changes through a Run-latency benefit.

If later measurements of real runs show substantial TLS costs, a separate
follow-up must include:

1. One transport per invocation/principal/endpoint; SendMessage first on a new
   connection, with no shared pool across allocations/invocations.
2. An explicit decision on GetBody and idempotency headers; reproduce the matrix
   when updating Go/SDK. Do not introduce blind retries of semantic operations.
3. An authenticated-session lifetime policy and tests covering expiry, key
   rotation and retirement; do not promise repeated TLS verification on each poll.
4. Cleanup of the original transport on success/error/cancel; active requests
   terminate through context cancellation. SDK Destroy alone is insufficient.
5. Preserved bounded reads without unbounded draining; separate TLS/CPU and
   Run-latency metrics under representative load.

V57-005 does not automatically create or implement that follow-up.

## Reproducibility

Commands, results and observations for all cases are in
[tasks/evidence/v57-005.json](../../tasks/evidence/v57-005.json). Main checks:

```sh
go test -race -count=1 ./internal/planner/a2a -run '^TestConnectionReuseExperiment' -v
go test ./internal/planner/a2a -run '^$' -bench '^BenchmarkConnectionReuseExperiment$' -benchtime=3x -count=3
go test -race -count=1 ./internal/planner/a2a ./internal/controlplane ./internal/mtls
make test-mtls
git diff --check
```

Experiment code: [fixture and benchmark](../../internal/planner/a2a/client_connection_experiment_test.go),
[retry matrix](../../internal/planner/a2a/client_connection_retry_experiment_test.go),
[lifecycle matrix](../../internal/planner/a2a/client_connection_lifecycle_experiment_test.go).
Verified production behavior: [A2A client](../../internal/planner/a2a/client.go),
[mTLS](../../internal/mtls/mtls.go); SDK `a2aclient/jsonrpc.go` (`newHTTPRequest`,
`sendRequest`, `Destroy`), Go `net/http/transport.go` (`shouldRetryRequest`) and
`net/http/request.go` (`isReplayable`) in the installed versions listed above.
