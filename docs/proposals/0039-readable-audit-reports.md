# Readable Audit reports

Proposal 0039 · Status: **Draft; not approved for implementation.**

Recorded on 2026-10-23. Today an Audit report is a set of status counters. This
proposal makes it a document that answers what was checked, what was found
and what was left unchecked. The data mostly exists already. The report
generator does not use it, and a few projections that would make it readable
are not recorded. This is a design recommendation, not an enabled capability.

## User outcome

A person who opens a check's report, or receives its downloaded `report.md`,
can answer these questions without opening the check page:

1. What was checked: project, materials, check type, scope, dates.
2. What is wrong: confirmed issues by severity, with location, CWE and the
   analyst's rationale.
3. What still needs a person: possible issues awaiting review, the most
   important first.
4. How far the check got: each endpoint or requirement in plain words, with
   problem rows first.
5. What was not checked and why: blocked items, failed attempts, stop reason,
   inventory gaps and limits.

The owner who accepts a report (REV-R45) decides on this content, not on
counters.

## Current state

The example is the demo check `audit_387903b4030eb04b28969a71a23c5cc2`
(`openapi-operation-trace@1`, 12 endpoints of a deliberately vulnerable API).

### What the report says

`humanSummary` (`internal/auditimport/report.go:443`) writes `report.md`, and
the report page renders only that document
(`ui/src/routes/reports/report-view.tsx:332`). It contains:

- the title `Audit audit_387903…` (an identifier, not the project or target);
- the profile, the conclusion code `completed-with-gaps`, "Selected items: 12"
  and "Applicable coverage: 8.33%";
- three count tables with 21 rows, of which 17 are zero, labelled with
  internal codes (`traced-partial`, `collection-contract-invalid`);
- an empty "Limitations" heading, followed by the certification disclaimer.

### What the reader needs and does not get

- **Issues.** The report shows "confirmed 2, proposed 36" and no titles. The
  two confirmed issues (a hardcoded JWT fallback secret, high, CWE-798; a 24 h
  token without revocation, medium, CWE-613) are only in `report.json`. Among
  the 36 unreviewed possible issues is an unauthenticated `GET /admin/users`
  that returns password hashes and session tokens. The report does not name it.
- **Coverage.** 11 of 12 endpoints were traced (1 fully, 10 with gaps) and one,
  `POST /auth/password/reset/confirm`, is blocked by an execution failure. The
  report says 8.33%.
- **Limitations.** 5 failed attempts, 1 blocked endpoint and 27 trace gaps
  exist; the section is empty because it lists only the stop reason and
  inventory gaps.

### Spec and code disagree

- **REV-R79** already requires baseline, profile and standard versions, scope,
  the coverage matrix, confirmed and proposed findings separately, unresolved
  questions, exclusions, incomplete evidence, human decisions, limits and
  provenance. `report.json` carries most of it; `report.md` carries counts.
  REV-R71 asks only for "coverage, finding and attempt tables", which the
  count tables satisfy literally. REV-R79 is not met.
- **REV-R22** defines applicability coverage as
  `(satisfied + violated) / selected applicable`. `summarizeCoverage`
  (`internal/auditimport/report.go:404`) also counts `traced-complete` in the
  numerator, and has done so since `873e931c` (2026-09-06). Neither formula
  describes operation tracing: in this check it gives 0% or 8.33%, while the
  meaningful figures are 11 of 12 traced and 1 of 12 traced without gaps.

### Data that is missing or noisy

| Need | Today |
| --- | --- |
| Readable item name (`GET /admin/users`) | Report items carry only `subjectKey` `op-<sha256>`. The coverage read derives the name from the task artifact on every page (`describeCoverageTask`, `internal/auditservice/coverage.go:128`), and the UI derives it again from `details.taskDocument` (`itemOperation`, `ui/src/routes/projects/audits/check-title.ts:14`): one rule, two definitions. |
| Which endpoint a finding came from | All 38 findings have `subject: null`; the Runtime hardcodes it (`runtime/src/contractor_runtime/toolsets/security_findings/submission.py:80`). Collection resolves which item's result lists each proposal (`internal/auditimport/check_results.go:155`) but does not persist the item. |
| Order of possible issues | None of the 36 has a `severity_suggestion`; the Runtime hardcodes `""` (`submission.py:85`) and the model-facing tool has no such parameter. |
| Meaningful trace gaps | Gaps are free-form model strings (`jwt-secret-env-deployment-dependent`, `sibling-endpoints-out-of-scope`). Every caveat about the environment makes an item `traced-partial`, so "partial" does not separate an unfinished trace from a stated assumption. |
| Words | Labels for coverage states exist only in the UI (`COVERAGE_STATUS_LABELS`, `ui/src/app/vocabulary.ts`); the Go report uses codes. |

## Design

### 1. Report document

`report.md` keeps its role (REV-R71, REV-R73): one deterministic, frozen,
downloadable document built by the Server from recorded report entries. Its
sections and their content:

| Section | Content |
| --- | --- |
| Title and header | Project name, check type in words with `name@version`, materials (source and API spec names with short digests), scope, start and finish times. The Audit ID moves to the provenance appendix. |
| Summary | Three to five sentences built from counts: confirmed issues by severity; possible issues awaiting review; how many items were checked, checked with gaps, blocked or not checked; whether the check stopped early and why. Zero categories are omitted. |
| Issues | One table row per confirmed issue ordered by analyst severity (critical → informational), then by first location: severity, title, first location, CWE/standard references, item. Below the table, a detail block per issue: description, limitations, analyst verdict and rationale, assessment state. |
| Possible issues | One row per unreviewed finding ordered by suggested severity, then by item ordinal: suggested severity, title, first location, item. `needs-evidence` findings follow as their own group. Rejected and duplicate findings appear only as counts with a pointer to the check page. |
| Coverage | One row per item: item name, coverage state in words, gaps. Rows outside the completed bucket (REV-R193) come first; completed rows follow. |
| Not checked and limitations | Stop reason; blocked and excluded items with their reason; failed, cancelled, invalid and missing-output attempts by item with their terminal outcome; inventory gaps; the limits that were reached; exclusions and N/A rationales (REV-R21). |
| Provenance (appendix) | Audit ID, profile digest, baseline and inventory digests, Round manifests, limits. |

Counts stay in the document only inside the summary sentences and as
section totals ("Possible issues (36)"). Tables of zero counts are removed.

Example, from the demo check:

```markdown
# vulnyapi · API operation trace (openapi-operation-trace@1)

Materials: source code vulnyapi (sha256:23beb94e…), API spec openapi
(sha256:e7b97754…) · 12 endpoints · 22 Oct 2026 09:12–10:24 UTC+3

## Summary

2 confirmed issues: 1 high, 1 medium. 36 possible issues need review.
11 of 12 endpoints were traced, 10 of them with gaps; 1 endpoint was not
checked because its check failed.

## Issues (2)

| Severity | Issue | Location | Reference | Endpoint |
| --- | --- | --- | --- | --- |
| High | JWT signing secret falls back to a hardcoded dev secret | utils/security.py:16 | CWE-798 | POST /auth/register |
| Medium | Access-token TTL of 24 hours with no server-side revocation | utils/security.py:18 | CWE-613 | POST /auth/login |

## Possible issues (36)

| Suggested | Possible issue | Location | Endpoint |
| --- | --- | --- | --- |
| — | Unauthenticated admin user listing exposes all accounts | routers/admin.py:13 | GET /admin/users |
| — | Response leaks password hashes and live session tokens | routers/admin.py | GET /admin/users |
…

## Coverage

| Endpoint | State | Gaps |
| --- | --- | --- |
| POST /auth/password/reset/confirm | Blocked | Check failed |
| GET /admin/users | Partially traced | deployment controls unverifiable; … |
…
```

(The endpoint column of the issue tables assumes the origin projection in
section 2; the suggested severity column stays empty until section 5 lands.)

### 2. Projections the report needs

Finalization reads recorded entries and no proposal or task artifact
(REV-R195, REV-R196). Every new field is therefore recorded when its source is
already being read:

- **Item name.** The Server builds every task during inventory generation
  (`internal/auditdomain/inventory.go`), so it can compute a bounded
  `subjectLabel` there and store it with the item when the Round is
  materialized: `METHOD /path` for an operation,
  the scanner operation for a scan, the statement for a checklist item and the
  objective for a finding verification. The rule lives once, in Go. The
  coverage read returns the stored label, `describeCoverageTask` stops
  deriving it, the UI's `itemOperation` uses the field, and report items carry
  it. Bound: 512 Unicode characters (new LIM-AUD entry), longer statements are
  shortened with an ellipsis.
- **Finding origin.** Collection records the item whose result listed the
  proposal next to the finding contribution. With a batch size of 1 the
  receipt's Audit execution already identifies the item (checked on the demo
  check: receipt → `audit_execution_items` gives `GET /admin/users` for the
  admin-listing issues); a batch needs the membership that collection already
  resolves. The report projection of a finding (FND-R158) carries the names of
  its origin items, at most 16 with an omission count (new LIM-REV entry). A
  finding without an item origin (a proposal imported from an ordinary Run)
  says so.
- **Severity counts.** Finalization reads one aggregate per finding state and
  severity, which keeps the summary sentence independent of page size.

`report.json` gains these fields under a new schema id
`contractor.audit.report.v2`. Frozen v1 reports remain readable: the report
read returns their exact bytes and does not decode the schema
(`internal/httpapi/public/audit_types.go:198`).

### 3. Size bounds

`report.md` stays within 64 KiB (LIM-REV-8) while an Audit may hold 100,000
items (LIM-AUD-25) and 100,000 findings (LIM-REV-7). Each section therefore
has a fixed byte budget and ends with an explicit omission line when the
budget is reached ("… and 31 more possible issues; see report.json or the
check page"):

| Section | Budget |
| --- | --- |
| Title, header, summary | 4 KiB |
| Issues table and details (detail blocks are cut first, the table last) | 24 KiB |
| Possible issues | 12 KiB |
| Coverage (problem rows first) | 12 KiB |
| Not checked and limitations | 8 KiB |
| Provenance | 2 KiB |

Each budget reserves the bytes of its omission line, as the inventory-gap loop
does today. A single over-long value is shortened, never allowed to push out
the rest of its section. Ordering by severity moves into the recorded page
order (REV-R196): findings of each state are paged by severity rank, then
finding ID. Problem-first coverage is a second pass over the recorded item
pages, so assembly holds at most one page in memory.

### 4. Coverage figures per mode

REV-R22 is rewritten as one figure per profile mode, with a zero denominator
reported as N/A (REV-R23):

| Mode | Headline figure |
| --- | --- |
| `operation-tracing` | Traced: `(traced-complete + traced-partial) / selected`, with "without gaps" `traced-complete / selected` beside it |
| Every other mode | Assessed: `(satisfied + violated) / applicable`, as REV-R22 states today |

The machine report keeps every count. `summarizeCoverage` is changed to match
the rewritten requirement, and the UI workspace counter (UI-R144) uses the
same figure.

### 5. Data quality prerequisites

These belong to their own tasks; the report renders whatever they provide.

- **Suggested severity.** The model-facing finding tool gains an optional
  `severity_suggestion` from the closed set of REV-R56. The Runtime validates
  it; REV-R58 keeps it separate from analyst severity.
- **Finding subject.** For a one-item assignment the Runtime fills `subject`
  from the assigned item deterministically; the model does not supply it.
  Batch assignments keep `null` and rely on the origin projection.
- **Gap classes.** `submit_check_result` gaps gain a closed class next to the
  free-form detail: `not-traced` (part of the requested trace is missing),
  `environment-assumption` (the code was traced; behaviour depends on
  deployment), `out-of-scope` (the path leaves the item's subject). Only
  `not-traced` makes an item `traced-partial`; the other classes are reported
  as notes. This changes REV-R18 and the Worker result contract (AWC-R32), so
  it needs the cross-language fixture of AGENTS.md 3.1.

### 6. Shared words

The report uses the user-facing vocabulary of `docs/design/ui/README.md`
("Issue", "Possible issue", "Partially traced", "Not checked yet"). The labels
move into one fixture, `api/testdata/audit-vocabulary.json`, which the Go
report builder and `ui/src/app/vocabulary.ts` both validate against.

### 7. Report page

The page keeps rendering `report.md` (REV-R73) and adds two things above it:

- **Header from the check**, not from the report: project, check type, state,
  who accepted the report and when (acceptance happens after the candidate is
  frozen, REV-R45, so it cannot be inside `report.md`).
- **Changes since this report** (REV-R70): from the existing bounded findings
  totals (REV-R165), for example "Since this report: 12 possible issues
  reviewed, 3 confirmed". Each figure links to the filtered findings list. The
  report itself is never rewritten.

The page does not parse `machine` for these figures. The report read returns
the full `report.json` inline, which may reach 64 MiB (LIM-REV-8); whether the
page should stop requesting it is decision point 4.

## Failure, lifecycle and bounds

- Finalization keeps its page budget and resume rules (REV-R196, REV-R197);
  the new fields add bytes to existing entries, not new reads.
- A report candidate that cannot fit is impossible by construction for
  `report.md` (section budgets). `report.json` keeps its 64 MiB bound and
  FAIL-AUD-9.
- Accepted reports stay frozen. Triage after acceptance (REV-R68) shows only
  in the page's "changes since" block.
- Audit deletion removes reports as today; no new entity or state is added.

## Verification

- Golden `report.md` for each profile mode built from fixture entries,
  including an empty check, a check stopped early and a check at the
  LIM-REV-7 and LIM-AUD-25 bounds that proves each section budget and
  omission line.
- PostgreSQL test of finalization that resumes across page budgets and
  produces byte-identical artifacts to an uninterrupted build.
- Cross-language test of the vocabulary fixture (Go and UI) and of the gap
  classes (Go and Runtime).
- UI test of the page header and "changes since" block at 390 px in the
  three themes.

## Alternatives

### A. Build the readable report in the UI from `report.json`

Rich interaction (sorting, filters, links) at no Server cost. Rejected as the
primary path: the downloaded and accepted artifact would stay unreadable, the
owner would accept bytes that differ from what the page shows, and the page
would parse a document of up to 64 MiB.

### B. A model-written executive summary

Reads well. Rejected: it moves result assembly back to the model, against the
project direction, and the frozen candidate would depend on a model call
during finalization.

### C. Deterministic `report.md` plus a thin live page — recommended

Sections 1–7 above.

## Decision points

1. Section budgets in section 3: accept as written or change.
2. Gap classes (section 5): accept the three classes or keep free-form gaps
   and change only how the report groups them.
3. Coverage figure for `operation-tracing`: "traced" with "without gaps"
   beside it, or "without gaps" alone.
4. Stop returning `machine` inline from the report read and download
   `report.json` through the artifact read instead. This removes a field from
   a `/v1` response, which AGENTS.md 3.1 allows only additively, so it needs
   your decision.
5. Whether an owner can issue a new report edition after triage, or the
   "changes since" block is enough.

## Proposed tasks

| Order | Kind | Scope |
| --- | --- | --- |
| 1 | defect | `report.md` per REV-R79: sections, budgets, words, REV-R22 figure; spec REV-R71/R22/R79 updated |
| 2 | feature | `subjectLabel` on Audit items, coverage read and report items; UI uses it |
| 3 | feature | Finding origin item on contributions and report projections |
| 4 | feature | Report page header and "changes since this report" |
| 5 | feature | Runtime `severity_suggestion` and deterministic `subject` for one-item assignments |
| 6 | feature | Gap classes in the Worker result contract |

Tasks 1–3 deliver the readable document; 4 makes the page tell the truth
after triage; 5–6 improve the data the document shows.
