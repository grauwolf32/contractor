# V32 performance release review

Historical evidence for the V32 branch on 2026-09-06. The raw measurements below
retain their original toolchain and workload. Follow-up integration into main
is tracked in the [integration review](2026-09-07-main-uncommitted-review.md).

The V32 implementation now has an executable eight-group acceptance matrix,
independent Go/Python/PostgreSQL/browser gates and a reproducible measurement
harness. The machine-readable evidence, including all five raw samples per
scenario, is [stored beside this review](2026-09-06-performance-v32-release.yml).

## Measurement environment

- Reproducible benchmark revision: `cbf8871181e083eb36a067d17a58acd8caa99527`.
  Raw samples were captured immediately before a history-only rebase at
  `7ec550a3153d6572f9f80a664ea8dcebc51a1690`; `internal/performance` and
  `tools/performancebench` are byte-identical at the rebased revision.
- Go: `go1.25.6 linux/amd64`; kernel: `Linux 7.1.9-arch1-2`.
- CPU: Intel Core i7-7700K at 4.20 GHz, 8 logical CPUs.
- Memory at capture: 31 GiB total, 3.0 GiB available; the host had all 4 GiB of
  swap in use. RSS comparisons therefore have more environmental uncertainty
  than the isolated CPU micro-workloads.
- Harness command: `make benchmark-performance`. The structured harness used
  50,000 HTTP operations, 240 simulated 15-second collection frames, one-second
  pprof workloads and five fresh child processes per variant.

CPU values are process user+system time from `getrusage`. RSS is the child
process high-water value reported by Linux. Allocation counts are Go malloc
deltas per completed operation. p95 is computed from individual HTTP/collection
operations or fixed 128-hash batches for the profiling workload. Coefficient of
variation (CV) is population standard deviation divided by the mean.

## Observed results

| Scenario | Mean throughput | Mean p95 | Mean CPU / repetition | Mean RSS | Mean allocations / operation | CV throughput / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| authenticated JSON, metrics off | 6.38 M req/s | 138 ns | 7.88 ms | 10.77 MiB | 1.000 | 2.59% / 4.49% |
| authenticated JSON, metrics on | 1.37 M req/s | 911 ns | 38.68 ms | 14.89 MiB | 16.001 | 1.67% / 5.39% |
| collection disabled seam | 15.57 M advances/s | 32 ns | 0.049 ms | 9.17 MiB | 0.004 | 4.55% / 16.75% |
| collection enabled | 2,513 frames/s | 0.880 ms | 100.03 ms | 15.22 MiB | 2,626.528 | 1.82% / 1.76% |
| pprof off | 5.64 M hashes/s | 229 ns | 0.995 s | 10.52 MiB | 0.000000532 | 2.07% / 14.95% |
| pprof listener idle | 5.69 M hashes/s | 209 ns | 0.989 s | 12.62 MiB | 0.000000527 | 0.80% / 8.03% |
| pprof CPU capture active | 5.61 M hashes/s | 223 ns | 0.993 s | 12.46 MiB | 0.000002928 | 1.93% / 6.45% |

The lower-bound Go microbenchmarks from the same command measured the direct
no-op handler at roughly 3.8 ns/op and the fixed HTTP recorder around 0.43–0.44
µs/op, 400 B/op and 12 allocations/op. The representative harness additionally
does a credential-header check and writes a bounded JSON page; its enabled minus
disabled p95 was about 0.77 µs and 15 allocations per request on this host. This
is visible fixed work and a useful regression baseline, but its relative
slowdown is intentionally not projected onto network-, JSON- or DB-dominated
production requests.

The simulated collection hour retained exactly 240 frames and 625,120 encoded
bytes, below the 8 MiB live-history bound. Every repetition made exactly 60
logical database-stat reads, 12 size reads and 60 history flush attempts; the
disabled variant made none and allocated no retained frames or queue. The
collection benchmark compresses an hour into one process burst, so its
throughput is a stress measurement rather than expected wall-clock CPU load.

Idle-listener throughput was 0.88% above the off mean, while active capture was
0.63% below it. Those differences are smaller than sample variation and do not
support a universal overhead claim. Active captures returned readable non-empty
profiles of 2,374–2,542 bytes. Their cost remains operationally distinct from
enabling an idle listener.

## Acceptance evidence

The matrix at `tests/e2e/performance_matrix.yml` maps every normative acceptance
group to an exact named test and rejects missing/renamed owners. Its focused gate
covers:

1. all four startup switch combinations and truly absent disabled components;
2. 15/60/300-second fake-clock cadence, bounded rings/queues and reset/gap rules;
3. fixed HTTP dimensions, concurrent drains, streaming and WebSocket handshakes;
4. disposable PostgreSQL roles, locks, outages, deadlines, retention and plans;
5. allocation finalize/abort/lease/drain, partial/lost reports and old/new wire
   compatibility;
6. authenticated bounded APIs, owner isolation, browser stale/gap/disabled
   states and durable post-release history in the production stack;
7. loopback-only profiling, allowlisted routes, finite captures, readable output
   and shutdown cancellation;
8. this reproducible raw performance evidence.

## Verification outcome

All required gates passed on 2026-09-06:

- `make test-performance-metrics` passed the strict matrix/evidence validator,
  race-enabled focused Go checks, 86 focused Python Runtime checks, a real
  Go-to-Python mTLS finalize/abort lifecycle, disposable PostgreSQL fault and
  retention checks, and the 21-scenario production browser stack;
- `make benchmark-performance` completed five repetitions for each variant and
  the five-count Go microbenchmarks;
- `make verify` passed all Go packages, 1,286 Python tests with 24 expected
  skips, 244 UI tests across 43 files, the Node static-server suite, generated
  API checks, lint, typecheck and the production UI build;
- the independently built Node UI was directly regression-tested for both
  `/operations/performance` and `/operations/allocations/completed` deep links.

The browser build ran successfully with Node 26.7.0 while the repository declares
Node 24.20.x; pnpm reported that toolchain mismatch as a warning. The product
checks themselves passed, but a release pipeline should continue to use the
declared Node version. A noisy percentage threshold is deliberately not part of
the gate.
