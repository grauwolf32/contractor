# Run a common Audit

An Audit pins selected exact materials and a program of ordinary Workflows.
It accepts bounded Tasks, dispatches ordinary Runs through RunService, and
retains their exact outputs. A completed Run does not imply a verified security
property. An exploratory result can contain observations and limitations
without a hypothesis, verdict or Project Finding.

The current shipped programs are:

| Profile | Selected inputs | Work |
| --- | --- | --- |
| `openapi-operation-observe@2` | Source ZIP and OpenAPI | Separate passive operation observations |
| `openapi-operation-trace@1` | Source ZIP and OpenAPI | Source tracing with optional evidenced Project problems |
| `openapi-sqlmap-scan@1` | OpenAPI, engagement and execution parameters | Separate selected request checks using SQLMap |
| `openapi-nuclei-scan@1` | OpenAPI, engagement and execution parameters | Separate selected URL/template checks using Nuclei |
| `source-openapi-sqlmap-scan@2` | Source ZIP, engagement and execution parameters | Ordinary source-to-OpenAPI Run, preparation and request checks |
| `web-scan@1` | Target, engagement and optional rulepack | One Task for the selected target |

Fixed assertions also run through this same path: `source-checklist`,
`owasp-top10-2025-source-risk`, ASVS 5.0 L1 pilot/full source review, and WSTG 4.2
full/fast source/active HTTP profiles. Standard package versions and the selected
original requirements stay pinned; creating additional checks does not change
that original total. HTTP profiles require an engagement and selected materials.

## Start with exact materials

In the UI select a Project, open **Start check**, choose a published check type
and select a material for each required input. Matching file formats do not
prove that a material is suitable. Advanced scope annotations are optional;
executable scan targets and restrictions come from the selected exact inputs
and Project policy. Keep the Project's configured model.

The public API exposes the exact program at
`GET /v1/audit-profiles/{name}/{version}`. Create a draft with
`POST /v1/projects/{projectId}/audits`, selecting `profile`, exact `inputs`,
optional `scope` and Runtime labels. Start it with
`POST /v1/audits/{auditId}/start`, the current `If-Match` and an
`Idempotency-Key`. Start returns `{audit}`. Tasks may appear asynchronously;
read `GET /v1/audits/{auditId}/tasks` and follow every returned cursor.

OpenAPI preparation is an ordinary Workflow with a Server-operation Stage. The
domain builder interprets OpenAPI and execution parameters; its output is a bounded list of
work. Common intake validates that list in slices and assigns Task identities.
A model or a Runtime Worker is not involved in deterministic preparation.
Each selected operation becomes a separate Task and can retry independently.
Compatible Tasks can share an ordinary checking Run.

## Check an assertion

For `source-checklist`, use **Write a checklist** beside its material selector.
Enter one observable claim per line with its conditions, grounds, version and
required evidence. This saves an ordinary exact Project JSON material. External
checklists may use equivalent YAML with the same strict fields, for example:

```yaml
schema: contractor.audit.assertions.v1
items:
- key: record-owner
  version: "1"
  statement: Only the record owner can read the selected record
  scope: {actor: authenticated reader, conditions: selected endpoints only}
  grounds: Explicit source review requirement
  applicability: selected endpoints
  allowed_methods: [static]
  required_evidence: [Exact inspected source and tested conditions]
```

At most 128 entries / 1 MiB are accepted. Curated-standard profiles derive their
selected assertions directly from the exact standard package without a model.

Open **Show assertions** inside an active or paused check to create/edit an
assertion or accept a check for it. **Save and check** commits the new revision
and Task together. The Server assigns identity, revision and origin, and pins
that exact revision in the Task. Previous checks keep their original wording.
Follow **View checks and results** for scoped support/refutation, limitations and
technical gaps. A successful Run alone proves no security property. Additional
assertions and split checks preserve the original requirement denominator.

The corresponding API is `POST /v1/audits/{auditId}/hypotheses`,
`PATCH /v1/audits/{auditId}/hypotheses/{hypothesisId}`, and paged `GET` on the
collection. Exact reads require `?revision=N`. Writes use `If-Match` and
`Idempotency-Key`. Task creation selects an owned hypothesis ID; the Server picks
and freezes its current revision. Public requests and model conclusions never
supply a persisted hypothesis ID/revision pair as result authority.

An evidenced problem can explicitly publish a Project Finding. An incidental
finding uses `incidental: true` with its trusted item key and has no hypothesis
association. Project provenance keeps the exact source assertion readable after
the originating Audit is purged.

## Time limits, pause and explicit continuation

Pause closes new Run submission and drains work already admitted. Collection
and explicit Task acceptance continue while paused. An empty paused Audit does
not automatically complete. Resume reopens dispatch under the existing policy,
budgets and deadline. A deadline or cancellation stops new submission; terminal
Runs and partial results are still collected truthfully.

For a two-Workflow handoff:

1. Start the Audit and pause after Run A is admitted.
2. Wait for A to finish and for its exact output to be retained in Audit scope.
3. Read the latest Audit revision and select that exact output as input to B.
4. Accept B using the common API below, then Resume with the current revision.

```json
{
  "work": [{
    "key": "follow-up-1",
    "workflow": "inspect",
    "objective": "Inspect this exact retained output",
    "inputs": {
      "material": {
        "scope_kind": "audit",
        "scope_id": "audit_example",
        "ref": {"namespace": "saved", "name": "output", "revision": "exact-r1"},
        "digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "media_type": "application/json",
        "size_bytes": 2
      }
    }
  }]
}
```

Send it to `POST /v1/audits/{auditId}/tasks` with `If-Match` and one stable
`Idempotency-Key`. `inspect` must be a Workflow alias declared by the pinned
program, and `material` must satisfy its input contract. The values above are
illustrative; use the actual returned exact reference, digest and size.
Acceptance returns original Task identities, creates no Run directly, and
replays before rereading mutable sources. A mutation accepts at most 128 Tasks,
1 MiB of metadata and 128 distinct exact inputs. The coordinator submits the
accepted work through RunService after Resume. Automatic proposals and a
next-work generator are outside this basic path.

## Read retained results and Project problems

The check page pages accepted Tasks and reads their saved ordinary results.
`GET /v1/audits/{auditId}/tasks/{taskId}` returns the exact `result` reference
when collected. Read it through the existing Artifact endpoint using
`key=task/{taskId}/result` and its exact revision. The UI previews at most
32 KiB and offers the complete result as a download. Producer Run deletion
does not erase this Audit material.

Project problems use one canonical Project Finding identity and independently
retained source contributions. Trusted execution membership derives Audit,
Task, Run and optional exact hypothesis origin. Model-supplied identifiers do
not establish that association. A saved ordinary result is not automatically a
Finding and needs no formal report acceptance.

## Scanner inputs

Use the [OpenAPI example](../../configs/scan/examples/audit-openapi-scan/openapi.json)
and the corresponding
[SQLMap parameters](../../configs/scan/examples/audit-openapi-scan/sqlmap-parameters.json)
or [Nuclei parameters](../../configs/scan/examples/audit-openapi-scan/nuclei-parameters.json)
as format examples. Supply a target you are authorized to check and the
Project policy/capabilities required by the selected operations. Nuclei URL
checks do not reproduce OpenAPI request methods or bodies. Authentication is
applied by engagement identities and private credentials. Upload an engagement
with the declared media type; supply optional per-operation request values as
a separate application/json material. Parameters are ordinary launch fields,
not uploaded artifacts.

The source-to-scan program uses the same parameters and optional request values; its ordinary generation
Workflow produces the OpenAPI input before deterministic preparation.
Web-scan uses the selected target/engagement artifacts and an optional exact
custom rulepack. See [scanner setup](../../runtime/README.md#cli-scanners).

## Verification

`make test-audit-basic-e2e` requires disposable PostgreSQL and real
Server/Runtime processes. It checks explicit exact handoff with and without a
Finding, pause/drain, restart and bounded prepared-work intake.
`make test-audit-common-programs-e2e` executes all six current programs and
requires actual Nuclei and SQLMap binaries. Scripted model responses test the
protocol; neither gate claims live-model quality. Both gates reject missing
mandatory cases, skips and stale evidence.
