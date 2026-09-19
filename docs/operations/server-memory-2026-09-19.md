# Server memory investigation — 2026-09-19

[Operations](README.md) · [Go profiling](go-profiling.md) ·
[Runtime investigation](runtime-memory-2026-09-19.md)

The local stand was updated to `be33cebbf8b9b5fd833d238b2bf0547d5694f319`.
The Server uses about **55–60 MiB RSS** after warm-up. Its live Go heap is
approximately **6–8 MiB**. The largest optimization opportunity found is repeated
decoding of persisted allocation reports: the UI's 50-record history page
allocates about **15.3 MiB per request** to produce a roughly **33 KiB response**.
This is temporary allocation volume, not additional permanently retained memory.

## Deployment and scope

- Release: `.local/demo/releases/main-be33cebb-20260919T192159Z`.
- Source: `git archive` of the committed revision; unrelated working-tree edits
  were excluded.
- Go executables were rebuilt with Go 1.25.6. Runtime dependencies were installed
  with `uv sync --locked --offline --no-dev`; importing the new gateway client
  succeeded and the OpenAI SDK is absent from this environment.
- UI remains version `0.2.0`. Its source and dependency files did not change
  between the previous release, `dc4d6f3a`, and this revision, so the existing
  built assets and dependency directory were reused.
- Server, UI, Runtime A and Runtime B were restarted. Readiness and UI health
  return HTTP 200; both Runtime instances registered, sent heartbeats, and have
  confirmed leases. No service error-priority journal entries were observed
  after deployment verification.
- The queue remains paused at revision `29`; there were no active allocations.
  No new workflow or model call was submitted for this investigation.
- The Server source, configuration schema and database migrations are unchanged
  by this release. Its memory measurements are characterization, not a claimed
  Server memory improvement from the Runtime changes.

Runtime RSS after deployment was approximately **123.0 / 123.1 MiB**, compared
with approximately **148 MiB per process** before deployment. These are idle
process measurements; working Runtime results are documented separately.

## RSS accounting

The following rows are additive resident mapping categories from
`/proc/<pid>/smaps`, captured at `2026-09-19T19:31:37Z` after the read workload.
They describe one actual **59.54 MiB** Server observation.

| Resident mappings | MiB | Meaning |
| --- | ---: | --- |
| Executable pages of the Go binary | 12.20 | Application and linked library machine code |
| Other pages of the binary | 14.07 | Read-only constants, runtime/type tables, embedded data and writable globals |
| Go heap mappings | 25.31 | Heap objects, allocation slack, unreleased pages and stacks allocated in the heap arena |
| Other Go mappings | 5.56 | Runtime metadata, GC bookkeeping, profiler buckets and allocator indexes |
| System libraries and other mappings | 2.40 | libc, dynamic loader, resolver, native mappings and process stack |
| **Total RSS** | **59.54** | |

Before deployment, a corresponding observation was **55.20 MiB**:
12.45 MiB code, 14.29 MiB other binary pages, 20.92 MiB Go heap mappings,
5.14 MiB other Go mappings, and 2.39 MiB system/other mappings.

The 25.31 MiB heap mapping row does **not** mean there are 25.31 MiB of useful
application objects. The latest Operations sample near the final observation
reported **6.44 MiB live heap**, 23 goroutines and about 0.0024 CPU cores while
idle. Before deployment, with the in-memory metrics history already warmed for
an hour, live heap was **7.39 MiB**. Stack usage in post-load `MemStats` was about
**0.97 MiB**. These counters are taken at different instants and must not be
added to the mapping table as additional memory.

`HeapAlloc` also includes objects created since the previous GC, while
`HeapSys` includes reserved/addressable heap space. For example, a post-load
snapshot reported approximately 95 MiB `HeapSys` but already 80.7 MiB
`HeapReleased`. Neither `HeapSys` nor total virtual address space is RSS.
Go documents this distinction and the memory/CPU trade-off of GC in its
[GC guide](https://go.dev/doc/gc-guide).

## What remains in the live heap

The live heap contains configuration snapshots, compiled validators, JSON/YAML
type caches, connection and statement state, operational metrics history,
HTTP/TLS buffers and runtime objects. A sampled profile can identify owners but
does not provide an exact accounting of every small object.

Two owners were checked more closely:

1. **Compiled regular expressions: approximately 2.29 MiB in an isolated
   initialization probe.** The probe imports the same CLI package, runs two GCs,
   and writes a heap profile with `GODEBUG=memprofilerate=1`. This avoids using
   noisy 512 KiB sample estimates for individual validators. Its total sampled
   heap was 2.87 MiB; the production Server additionally creates configuration,
   HTTP and database state. Many application regexes implement bounded ASCII
   identifiers, digest strings and header names. Repeated patterns across
   packages and regex expansion for bounds such as `{0,255}` are visible in
   the profile. This probe is an initialization experiment, not a second live
   Server measurement.
2. **One hour of fine metrics: roughly 1.4–1.5 MiB of JSON frame data.** The
   collector retains at most 240 frames, with an 8 MiB byte cap. A captured
   representative frame is about 6.2 KiB; the old Server's retained allocation
   profile attributes about 1.51 MiB to `Collector.collect` JSON buffers.
   Each frame repeats about 1.9 KiB of GC histogram bounds, representing roughly
   0.45 MiB over 240 frames. The fresh Server has fewer retained frames, so its
   smaller live heap immediately after restart is expected.

The remaining owners are smaller and distributed. The working database pool
had six connections after the read workload, including one acquired connection,
against a maximum of eight. The goroutine dump contained 25 goroutines while
the diagnostic request itself was active. This investigation did not identify
an unbounded thread, goroutine or connection population.

## Largest working cost: allocation history

The confirmed workload issued **420 successful GET requests in 30 seconds**,
14 requests per second, using one reusable HTTP connection. It cycled through
performance, Runtime registrations, active allocations, allocation history,
queue, model policies and audit profiles. There were 60 history requests with
`limit=100`. All responses were HTTP 200. Server RSS sampled once per second
ranged from **54.31 to 58.52 MiB**, averaging **57.80 MiB**.

A 35-second allocation profile covering this workload recorded approximately
**2,223 MiB allocated cumulatively**:

| Profile owner | Allocation volume | Share |
| --- | ---: | ---: |
| Allocation history HTTP handler, including descendants | 2,180 MiB | 98.1% |
| `scanAllocationResourceSummaries`, including descendants | 2,170 MiB | 97.6% |
| Duplicate-key scanning, including descendants | 1,251 MiB | 56.3% |
| `AllocationFinalReport.Validate`, including descendants | 282 MiB | 12.7% |

These are **overlapping cumulative stacks**, not additive categories. The
allocated data is repeatedly reclaimed; the roughly 2.2 GiB total never
resides in memory simultaneously. This short read workload is not evidence
about long-running workflows, large uploads, or all possible memory leaks.

An independent sequence of 20 successful requests per page size confirmed the
cost without the mixed workload:

| History page | Temporary allocation/request | Mean response time | Response size/request |
| --- | ---: | ---: | ---: |
| 10 records | 2.41 MiB | 9.1 ms | 6.83 KiB |
| **50 records — current UI page size** | **15.31 MiB** | **53.1 ms** | **32.68 KiB** |
| 100 records | 36.07 MiB | 115.7 ms | 64.84 KiB |

Allocation estimates use the difference in `MemStats.TotalAlloc`, subtracting
a no-request profiling control of about 1.97 MiB, then dividing by 20. Normal
background activity remains included, so these are approximate observed
costs, not benchmark guarantees. Different page sizes select different sets
of historical reports and their cost is not exactly linear. The 50-record
sequence caused 58 GC cycles; the 100-record sequence caused 137.

The source path explains the result:

- [`allocation_resources.go`](../../internal/telemetry/allocation_resources.go)
  selects the **complete** persisted `effective.report` for every row.
- `scanAllocationResourceSummaries` decodes `AllocationFinalReport` and invokes
  `Validate`, although the resulting list mainly uses `Runtime.Resources` and
  allocation metadata.
- Nested `AllocationFinalReport` and `ExecutionReport` unmarshalling repeatedly
  invokes the token-based duplicate-key scanner in
  [`private.go`](../../internal/contracts/private.go).
- Report validation also serializes report content again to check size and
  other invariants in [`telemetry.go`](../../internal/contracts/telemetry.go).

The profile's `encoding/json.(*scanner).error` frames occur inside token
decoding; they do not imply that the HTTP requests failed. All confirmed
workload requests succeeded.

## Temporary 64 MiB password buffer

An early observation after deployment showed **113.39 MiB RSS** and roughly
78 MiB `HeapAlloc`. The cumulative profile attributes a **64 MiB** allocation
to `argon2.initBlocks`, called through `passwordHash.verify` and the login
handler. This was a password verification during the observation window,
not retained configuration or a growing application cache.

After the subsequent GC and scavenging, RSS was observed at **48.22 MiB** and
the live heap at about **5.46 MiB**. Explicit `heap?gc=1` snapshots were used
to separate retained objects from garbage during profiling. They did not
change the running Server's GC configuration.

The memory cost is intentional in
[`auth/password.go`](../../internal/auth/password.go). Lowering password-hash
parameters is not proposed as a memory optimization. Any deployment memory
budget must account for this transient work as well as ordinary traffic.

## Recommended changes, in order

1. **Build a small resource-summary read path.** Persist a validated projection
   when accepting a final report, or select only the resource fields and
   metadata needed by this endpoint. Preserve ownership, pagination, effective
   report ordering, report expiry, partial/legacy states and malformed-resource
   behavior. Keep complete report validation at ingestion. This also benefits
   Stage resource lists, which share the current scanner. Validate the change
   against existing reports and compare allocation volume, RSS, CPU and latency.
   The measurement shows about 15 MiB of allocation per current UI page to
   target; it does not mean 15 MiB can be removed from idle RSS.
2. **Reduce repeated JSON passes.** Where strict wire decoding still needs
   duplicate-key and unknown-field checks, perform them once per document and
   avoid serializing already encoded data just to check its original size.
   Preserve strict validation and optional-field compatibility. This is also
   relevant to telemetry ingestion and other users of the same contracts.
3. **Consolidate simple validators.** Share identical compiled patterns and
   consider small ASCII/length validators for bounded identifiers and hashes.
   The measured regex footprint is about 2.3 MiB, an upper bound for the entire
   category, not a promised saving. Verify accepted/rejected input equivalence
   before replacing any validator.
4. **Compact fine metrics storage if a further small saving is useful.** Keep
   invariant histogram bounds once and retain compact samples, while preserving
   the existing API and history window. The entire frame payload is only about
   1.5 MiB today; removing repeated bounds alone targets roughly 0.45 MiB before
   representation overhead. Replacing JSON round-trip copies and numeric
   validation can also reduce temporary allocations.

A dedicated `contractor-server` executable has about 1.70 MiB less loadable
code/data than the umbrella CLI in this build (`size` output). Its real RSS
benefit was not measured, and the isolated import probes differ by only about
19 KiB of heap. It is a secondary packaging option, not the main optimization.
Removing debug symbols mainly changes file size; it does not eliminate the
resident application code and data in the RSS table.

No Server algorithm, password parameter, `GOGC`, or `GOMEMLIMIT` setting was
changed during this investigation. GC tuning can trade CPU for memory but
does not eliminate the redundant report processing above; the
[Go GC guide](https://go.dev/doc/gc-guide) describes that trade-off.

## Evidence and reproduction

Raw evidence is local and ignored by Git under
`.local/server-memory-20260919/`. Main files:

- `deployment.json`: release and service verification.
- `before-deploy-smaps.txt`, `before-performance.json`: aged Server baseline.
- `final-observation.json`, `final-smaps.txt`: additive 59.54 MiB RSS breakdown.
- `after-deploy-heap.pb.gz`, `after-deploy-gc-heap.pb.gz`: early heap and
  password-buffer diagnosis.
- `read-load.json`, `read-load-allocs.pb.gz`: confirmed all-200 mixed workload.
- `after-read-load-gc-heap.pb.gz`, `after-read-load-goroutines.txt`: retained
  allocations and goroutines after that workload.
- `history-isolation.json`: 10/50/100-record comparisons and profiling control.
- `cli-init-full-heap.pb.gz`, `server-init-full-heap.pb.gz`: isolated import
  probes with every allocation sampled.
- `profile.py`: local capture/read-workload helper; it reads the existing
  local bearer credential without emitting it.

For example, from the repository root:

```sh
go tool pprof -top -alloc_space \
  .local/demo/current/bin/contractor \
  .local/server-memory-20260919/read-load-allocs.pb.gz

go tool pprof -top -cum -alloc_space \
  .local/demo/current/bin/contractor \
  .local/server-memory-20260919/read-load-allocs.pb.gz

go tool pprof -top \
  .local/demo/current/bin/contractor \
  .local/server-memory-20260919/after-read-load-gc-heap.pb.gz
```

The live process uses the default sampled memory profiler. Individual entries
near 512 KiB can be sampling artifacts, not exact package sizes. Memory
profiles can also lag reclamation by up to two GC cycles; an initial 30-second
idle allocation delta had no samples and was not interpreted as zero
allocation. See [`runtime.MemProfile`](https://pkg.go.dev/runtime#MemProfile)
and [`runtime/pprof`](https://pkg.go.dev/runtime/pprof). The explicit GC probes,
profiling overhead, 15-second Operations sampling cadence, limited workload,
and fresh versus full metrics history must be considered when comparing
snapshots.

## Follow-up: minimal resource-summary read change

The first recommendation was implemented locally after the investigation.
Only `internal/telemetry/allocation_resources.go` changes production behavior;
the regex validators, metrics collector, persisted report format and database
schema are unchanged. This follow-up has not been deployed to the stand.

Both completed-allocation history and Stage resource queries now project three
values from the selected report: its presence, its typed allocation identity,
and `runtime.resources`. They no longer transfer or decode Worker detail for
these summaries. Effective-report ordering, expiration, owner filtering and
pagination are unchanged.

The optional resource block still goes through the existing strict contract
decoder and semantic validator. An invalid block is omitted as before; a
persisted `invalid_report` sentinel preserves its diagnostic. A separate report
presence value preserves the distinction between a report with no measurements
and an allocation whose report has not arrived. Missing or non-string allocation
identities remain read errors. Complete reports continue to be validated and
normalized by the write path and are immutable in PostgreSQL.

Two binaries compared the old and new repository methods against the same live
data, with PostgreSQL `default_transaction_read_only=on`, one connection, three
warm-up calls and 20 measured calls per limit. `runtime.MemStats.TotalAlloc`
measures allocated bytes; the experiment also records object allocations and
wall-clock time. There is no HTTP or profiling-handler overhead in these values,
so they should be compared with each other, not treated as end-to-end UI latency.

| Records | Allocated/request before | After | Time/request before | After |
| --- | ---: | ---: | ---: | ---: |
| 10 | 2.20 MiB | 0.098 MiB | 7.62 ms | 1.55 ms |
| 50 | 14.88 MiB | 0.480 MiB | 47.21 ms | 6.14 ms |
| 100 | 36.00 MiB | 0.958 MiB | 112.02 ms | 13.60 ms |

For 50 records this is approximately **31 times less allocation** (96.8% less),
with object allocations falling from **531,506 to 12,374 per request**. The
serialized summary results have identical SHA-256 digests at all three limits.
This measures temporary allocation and read latency; a new steady-state Server
RSS reduction has not been measured or claimed.

Real PostgreSQL regression tests cover complete/partial/unavailable resources,
invalid sentinels, absent/null/malformed/future resource blocks, pinned policies
and legacy allocations, missing and expired reports, typed report identity,
effective-report selection, history pagination, terminal-only reads, ownership,
and equality between history and Stage summaries. Tests use a separately created
disposable database, not the stand's database.

Validation passed: `go test -race -count=1` for `internal/telemetry`,
`internal/contracts` and `internal/httpapi/public` with the test database enabled;
`go vet` for those packages; and builds of all four `cmd` executables. The
temporary test database was removed after verification.

Experiment binaries, runner, original implementation and result JSON are kept
under `.local/resource-history-optimization-20260919/`. The implementation's base
revision is `d2920a17`; the prior two commits after the deployed `be33cebb` only
add an A2A experiment and its task record.
