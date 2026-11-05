# Controlled HTTP conformance

`make test-engagement-http-process` verifies allocation-owned HTTP through real
Server/Runtime processes, real TCP/TLS peers and a disposable PostgreSQL store.
The locked Runtime environment and an active RFC1918 interface are required.
`CONTRACTOR_HTTP_TEST_BIND_ADDRESS` selects an assigned local test address when
several are available. Missing prerequisites fail setup. Target loopback is denied;
private Server/Runtime endpoints retain ordinary mTLS loopback connections.

The strict engagement document contains `schema`, `egress.routes/exclusions`,
`identities` and optional `limits` ([AUD-R241–248](../spec/audits/audits.md#AUD-R241)).
Workflow tools determine the required HTTP owner; they cannot widen its authority.
Offline Workers have no HTTP assignment. Audit owns material inputs, exact retained
Artifact/credential pins, execution budgets and Start/Resume deadlines. There is no
source mode or engagement-owned material registry. Ordinary projectless and
Project Runs select an explicitly typed policy input under
[RUN-R112–115](../spec/execution/runs.md#RUN-R112); its original input fork
provides policy authority without an Audit or duplicate policy snapshot.

The exact mandatory cases live in
[`scripts/engagement-http-matrix.json`](../../scripts/engagement-http-matrix.json).
Go JSON and Runtime JUnit reports must contain every registered parent, subcase
and parametrized identity. Skips and empty selections fail. Evidence lives in a
fresh `reports/engagement-http/run-*` directory; `result.json` is written only
when all cases pass and includes matrix/report digests.

| Boundary | Independent evidence |
| --- | --- |
| Ordinary Run | A public projectless Run uses its typed policy input, explicit bearer identity, route denial and local request cap through the production Server/Runtime preparation path. The credential is protected during execution and deletable after joined release while the policy Artifact remains. |
| Scope and destinations | Server-pinned allocation plus actual RFC1918 target listeners verify origins, path segments, methods, exclusions, canonical redirects, DNS rebinding and protected aliases. Forbidden requests emit no application bytes. |
| Peer and TLS | Production socket transport verifies numeric peer, Host, SNI and certificate trust. A forbidden DNS answer rejects the entire set; environment proxy traps see no traffic. |
| Optional allocation limits | Runtime tests use injected clocks to verify request count and rate, body bounds and fresh independent allocation counters. PostgreSQL tests verify the selected artifact's limits are pinned and projected without defaults. |
| Retry | GET/HEAD/OPTIONS retry once only after pre-send connect/TLS failure. Write methods, partial writes and received HTTP statuses never automatically retry. |
| Lifecycle | Stalled connections close and work joins on cancellation, lease loss and process replacement. Fresh allocations have fresh sessions and counters. There is no journal, acknowledgement or transport-stop proof. |
| Credentials | Real target bytes verify explicit Basic/Bearer/cookie selection, anonymous isolation and origin separation. Invalid selection fails before target traffic. Cancellation clears private state. |
| Framing | Exact and overflowing URL/header/body cases include informational headers, bounded streamed bodies, five redirects and truncation metadata. |

Connection reuse follows HTP-R138. Assertions check actual request paths, counts,
headers, bodies and forbidden-target bytes; they do not require one TCP connection
per request. Keep-alive and HTTP/2 retain per-attempt scope and destination checks.

The complete AuditService suite separately verifies ordinary material-input holds,
replay, purge and Start/Resume deadlines. Shared Go/Python/schema cases reject the
removed document wrappers and private policy fields. `make test-nuclei-engagement`
is the separate real-engine gate for Nuclei sharing the allocation HTTP owner,
credentials and optional request allowance.

The private process proxy rejects any use of the removed per-request Server API.
Its observations do not authorize requests or implement a replacement journal.
Interactive HTTP bodies remain private observations, without trusted proof status.
These cases do not claim enforcement for browsers, scanners or arbitrary egress;
scanner integration requires its own actual controlled execution path.

The focused gate runs all three observers. Consolidated release ownership remains
process shard B, the full Runtime report and the full Go unit report. Each report
uses this same manifest; the release graph and gate tests reject omitted cases.
Setup, case and teardown deadlines bound hangs; rate correctness uses injected
clocks. Teardown joins owned processes, sockets and tasks before cleanup.
