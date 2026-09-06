# Private v2 performance contracts

The registration, allocation and runtime-report JSON schemas describe the strict
private DTOs. Shared performance fragments live in
`../v1alpha1/performance.schema.json`; the legacy final-report schema references
the same resource block. Full v2 schemas were generated from the corresponding
Python DTOs, then their performance definitions were linked to those shared
producer fragments. Go/Python shared fixtures additionally enforce cross-field
resource invariants that JSON Schema cannot express (peak versus boundaries,
sample counts and gap versus duration).

`supportedPerformanceMetricsVersions` is optional: absent/empty is unsupported,
`[1]` supports `performanceMetrics: {version: 1, intervalSeconds: 15}`. Current
Runtime registration still omits the capability until V32-004 installs collection.
Omitting the allocation instruction disables sampling. Neither capability nor
instruction changes placement eligibility or execution budgets.

Upgrade Server first; an old Server can reject the new registration field.
Rollback requires coordination, not automatic fallback to another protocol.

Unknown numbers are omitted. Resource integers are limited to `2^53 - 1` for
exact cross-language JSON/TypeScript representation. A complete resource report
requires CPU and RSS boundaries, at least two successful RSS observations and
no uncovered interval greater than 30 seconds. An observed peak is not a true
maximum or a process-lifetime high-water mark.

Producer schemas reject malformed resources. Receivers isolate such a block,
omit it and retain only the safe `invalid_report` diagnostic; valid Worker and
adapter reports remain intact. This exception does not relax duplicate-key,
JSON syntax, authentication or original final-report byte limits. Diagnostics
are not copied into model-visible context or serialized as raw input.

The startup switches are contracts only in V32-001. Collectors, allocation
sampling and the separate profiling listener are implemented by later tasks;
the three public performance routes remain `planned` until V32-005.
