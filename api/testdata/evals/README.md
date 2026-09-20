# Managed evaluation conformance examples

These language-neutral fixtures are consumed by Contractor's Go codecs and are
available to independent public API clients. They require neither Playground nor
live execution. All IDs, hashes, credentials and timestamps are illustrative.

- `cases.json`: document kind, exact fixture file and accepted/rejected outcome.
- `valid`: native/external Workflow and Audit, authoring/control, evidence,
  comparisons, charts and safe read DTOs. `review.json` is deliberately owner-only.
- `invalid`: malformed/ambiguous JSON, closed-schema failures and semantic faults.
- `http-mutations.json`: complete planned public mutation examples and CAS/key
  headers. These endpoints are not implemented by V38-002.
- `audit-accounting.json`: one parent Audit, check retry, discovery and assessment
  Runs, repeated identical observations and overlapping child lifetimes. Expected
  totals are 100 tokens, four model calls and 2000 ms parent wall time.

The trace-small fixture has eight expected members: A passes 3/4, B 2/4;
conditional assessment quality is 3/3 and 2/3. Terminal/quality/token paired
coverage is 3/2/2. B's known 130 tokens cover two members; the complete token pair
cohort is A 80/45, B 70/60, with deltas -10/+15. It cannot prove whole-experiment
savings. The failed-execution/passing-assessment fixture keeps those concepts
separate. Cases and rows with unknown measurements remain in denominators.

Private sentinels belong only to input/review/private plan fixtures. Their absence
from execution/public projections and owner-safe errors is asserted by Go tests.
Byte/depth bounds, Unicode, exact-byte preservation, immutable copies, mutation
replay and stale-revision behavior also have focused codec tests.
