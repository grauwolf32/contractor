# Public OpenAPI review — 2026-09-19

Reviewed `api/openapi/contractor-public-v1.yaml` against the current Server,
configurations and clients. Review baseline:
`b779dedf285f5d6645f064d423e8873b5c382ab6`. Line numbers below refer to that
state. These are review results, not a change to the normative contract.
Original observations and reproductions before the fixes are retained below.
All 12 findings were fixed in V59; clarifications, changes and re-verification
limits are described in the [2026-09-20 results](2026-09-20-public-openapi-corrections-results.md).

The file is structurally valid but cannot be considered fully aligned with the
implementation. Twelve findings were identified: one P1, eight P2 and three P3.
Some defects are in handlers or client generation. Editing YAML alone will not
resolve them. The listed fixes do not require rewriting the entire API.

| ID | Priority | Fix location | Problem |
| --- | --- | --- | --- |
| R01 | P1 | Public projection, then schema | `AuditReport.review` is lost before the HTTP response |
| R02 | P2 | YAML and generated clients | Summarizer omits the existing `instructions` field |
| R03 | P2 | YAML and generated clients | Telemetry export omits the existing `retry` field |
| R04 | P2 | RuntimeConfig schemas | Author request and stored document use one union |
| R05 | P2 | Audit pagination | Allowed `limit=200` becomes an invalid internal 201 |
| R06 | P2 | Finding decision schema | Verdict-dependent required fields are not described |
| R07 | P2 | Query schemas | Filtering by a label with an empty value is forbidden |
| R08 | P2 | Login schema and unit documentation | Character-based password length conflicts with the Server's byte policy |
| R09 | P2 | Go client generation | Explicit `null` becomes an omitted patch field |
| R10 | P3 | Git mutation parameters | Origin/CSRF requirements for cookie auth are undocumented |
| R11 | P3 | RunStatus schema | The 1024-attempt limit is not enforced for automatic retry |
| R12 | P3 | Git import description | An explicit create precondition is described as required but is actually optional |

1. **R01 — P1: losing review blocks approval of a proposed report in the UI.**

   `AuditReport.review` is declared at OpenAPI:4355. The service fetches review
   for `status=proposed` and returns it in `ReportProjection`
   ([service.go](../../internal/auditservice/service.go), lines 210–215, 264).
   However, `auditReportResponse` lacks the field, and `getAuditReport` does not
   copy it into the HTTP response
   ([audit_handlers.go](../../internal/httpapi/public/audit_handlers.go),
   lines 192–198, 781–785).

   The UI shows `ActionReviewControls` only when
   `report.data.review?.state === "pending"`; navigation with `?review=...` also
   checks the request ID
   ([detail.tsx](../../ui/src/routes/projects/audits/detail.tsx), lines 1811, 1901–1905).
   As a result, the proposed report has no approval actions on this page.

   Reproduction: the real HTTP handler with a fake service returning a non-nil
   pending review produced `200 {"status":"proposed"}` without `review`.
   Existing schema validation does not catch this because the field is optional.

   Fix: carry review through the public DTO and add a proposed-report contract
   scenario. Require review for proposed in the schema while preserving other
   statuses. This is primarily an implementation defect; removing the field
   from OpenAPI would be wrong.

2. **R02 — P2: the closed Summarizer schema rejects ordinary catalog responses.**

   `WorkerSummarizerConfigBody` at OpenAPI:6169–6176 has
   `additionalProperties: false` but no `instructions` field.
   Public projection adds `{ref, digest}` when Summarizer instructions are configured
   ([resources.go](../../internal/config/resources.go), lines 219–232).

   Reproduction: current `configs/` was loaded with real `config.Load`;
   serialized `ConfigurationResource` objects were checked against Draft 2020-12.
   **30 of 41 AgentTemplate versions failed validation** because of
   `body.summarizer.instructions`. The remaining 11 AgentTemplates, four
   ModelPolicies and one Gateway passed. Example:
   [artifact_builder_v2_memory.yaml](../../configs/agent-templates/artifact_builder_v2_memory.yaml).

   The mismatch affects list/detail configuration responses. A strict client
   rejects the resource; Go typed models lose the unknown field when decoding,
   and the TypeScript type does not describe it.

   Fix: add optional `instructions` using the existing instruction-ref schema
   and regenerate both clients. Validate real public projections of repository
   configurations, including optional nested fields. Removing
   `additionalProperties: false` from the entire object is unnecessary.

3. **R03 — P2: telemetry export retry is implemented but forbidden by the schema.**

   `WorkerTelemetryExportConfig` at OpenAPI:5158–5166 is closed and lists only
   `batchSizeBytes`, `maxAttempts`, `maxPendingSpans`, `maxPendingBytes`.
   The Server supports nested `retry` backoff settings
   ([normalize.go](../../internal/runtimeconfig/normalize.go), lines 481–507;
   [telemetry_export.go](../../internal/contracts/telemetry_export.go), line 9).
   The stored canonical document is included in the public response.

   Reproduction: `PreparePublication` and `Resolve` accepted
   `export.retry={"initialBackoffMilliseconds":17,"maxBackoffMilliseconds":43}`.
   The resulting document failed `RuntimeConfigDocument`: `retry was unexpected`.
   The mismatch affects publication requests, read responses and generated DTOs.

   Fix: add an optional retry schema with actual bounds/defaults; preserve
   acceptance of old documents without it. Check both the request and normalized
   response, since normalization adds defaults.

4. **R04 — P2: RuntimeConfig author/read schemas are mixed.**

   OpenAPI:5199–5203 permits `gateway` as a selector string, expanded ref or
   `null`; lines 5213–5216 permit `llmGateway: null`. The same
   `RuntimeConfigDocument` is used for POST at line 2446 and for reading the
   published document at line 5259.

   Actual rules differ: both null values are forbidden; at publication, gateway
   must be an exact selector string; after resolution, a ref object with digest
   is stored
   ([normalize.go](../../internal/runtimeconfig/normalize.go), lines 335–350).
   This matches the
   [RuntimeConfig specification](../spec/07-runtime-labels-and-infrastructure-config.md),
   line 133.

   Reproduction: JSON Schema accepts each worker block
   `{"llmGateway":null}`, `{"llmGateway":{"gateway":null}}` and
   `{"llmGateway":{"gateway":{"gatewayId":"local-litellm","version":"1","digest":"sha256:<64 hex>"}}}`.
   `PreparePublication` rejects all three. The test used a full valid digest,
   not the abbreviation shown in the example.

   Fix: separate author-document and resolved-document schemas, sharing only
   genuinely common components. Remove unsupported null branches for gateway.
   There is no need to broaden the Server parser to match the current overly
   permissive union.

5. **R05 — P2: Audit pagination fails at its declared upper bound.**

   The shared `Limit` parameter at OpenAPI:3265 permits 1–200. Findings, reviews
   and finding provenance use it. Their handlers request `limit + 1` to detect
   a subsequent page
   ([audit_review_handlers.go](../../internal/httpapi/public/audit_review_handlers.go),
   lines 55, 186, 339). However, service validators reject values above
   `MaxFindingPageSize = 200`
   ([finding_review.go](../../internal/auditservice/finding_review.go), lines 908–943).

   Reproduction: for `?limit=200`, real handlers with real service validators
   return HTTP 400 for all three lists. The provenance test uses a fake lookup
   of an existing owned finding; PostgreSQL is unnecessary to demonstrate
   the limit mismatch.

   Fix: align the public page size and internal extra row. Account for subsequent
   receipt batch hydration, which is also limited to 200 IDs
   ([audit_receipt_batch.go](../../internal/findingintake/audit_receipt_batch.go),
   line 21). Raising only one service bound is insufficient. Acceptance must
   check 199/200, presence of a 201st record and the next cursor.

6. **R06 — P2: conditional finding-decision fields are undocumented.**

   `DecideAuditFindingRequest` at OpenAPI:4704–4712 requires only verdict and
   rationale. The Server requires severity for `true_positive` and forbids it
   for other verdicts. `duplicateTargetId` is required only for `duplicate`
   ([finding_review.go](../../internal/auditservice/finding_review.go), lines 967–975).

   Reproduction: the schema accepts
   `{"verdict":"true_positive","rationale":"Observed evidence"}` and the analogous
   `duplicate` request; real `DecideFinding` rejects both before SQL.

   Fix: describe request variants or conditional rules for each verdict, including
   forbidden unrelated fields. Separately verify that the chosen form preserves
   useful generated types rather than merely passing a validator.

7. **R07 — P2: filter schema rejects supported labels with empty values.**

   Both `label` query schemas — OpenAPI:761 and 1633 — use `minLength: 3`
   and `pattern: '^[^=]+=.+'`. These reject `triaged=`. Yet `RunMetadataLabels`
   explicitly allows an empty value, and the handler splits the selector at
   the first `=` and accepts it
   ([run_handlers.go](../../internal/httpapi/public/run_handlers.go), lines 131–144).

   Reproduction: `GET /v1/runs?label=triaged%3D` passes the real authenticated
   handler with HTTP 200 but is rejected by the OpenAPI request validator.

   Fix: allow an empty right-hand side in both query schemas; account for the
   minimum length of `a=`. Cover global and Project lists with identical cases.

8. **R08 — P2: password length is described in different units.**

   `LoginRequest.password` at OpenAPI:6406 sets `minLength: 12`, counting Unicode
   code points. Auth uses a 12–1024 UTF-8 byte bound
   ([password.go](../../internal/auth/password.go), lines 20–31).

   Reproduction: the password `пароль` contains six characters and 12 bytes.
   Bootstrap accepts it; real login returns HTTP 200 and a cookie, while the
   request validator rejects it.

   Fix: remove the incompatible character minimum, explicitly document the byte
   policy and preserve exact Server validation. If needed, express byte bounds
   with a schema extension and client-side validation. Also test the upper
   boundary with multibyte characters. Do not change the existing password
   policy or require users to replace existing passwords merely to fit the schema.

9. **R09 — P2: the typed Go client loses explicit-null patch operations.**

   Nullable branches for telemetry/httpProxy/caido at OpenAPI:5217–5228 are correct.
   However, generated Go `RuntimeWorkerPatch` uses `*T` with `omitempty`
   ([public.gen.go](../../internal/publicclient/generated/public.gen.go), lines 3629–3633).
   Ordinary `json.Marshal` cannot distinguish an absent field from explicit null.
   Generator configuration:
   [oapi-codegen.yaml](../../internal/publicclient/generated/oapi-codegen.yaml).

   Reproduction: a raw document with worker `telemetry:null`, `httpProxy:null`,
   `caido:null` and planner `telemetry:null` passes `PreparePublication`.
   Unmarshal/marshal through generated `RuntimeConfigDocument` turns spec into
   `{"planner":{},"worker":{}}`; republication returns `spec.worker cannot be empty`.
   With adjacent fields present, a clear operation may disappear without an
   empty-object error.

   Fix: preserve all three states — absent/null/value — in generated Go request
   types and check actual typed POST bytes. This is a consumer defect: the HTTP
   API supports clear, TypeScript preserves `| null`, and the Go raw-body overload
   can send a correct document. Supported null values must not be removed from
   OpenAPI just to simplify Go types.

   Temporary regeneration with `output-options.nullable-type: true` confirmed
   that pinned oapi-codegen recognizes current `oneOf`/null and generates
   `nullable.Nullable[T]`. The option is global: it changes other nullable DTOs
   and adds `github.com/oapi-codegen/nullable`, so call sites need checking.
   The temporary modified client was not built or round-trip tested during this review.

10. **R10 — P3: Git mutations omit browser mutation headers.**

    `replaceGitKey`, `deleteGitKey`, `importGitArtifact`,
    `importProjectGitArtifact` at OpenAPI:106–214 do not reference
    `OptionalOrigin`/`OptionalCSRFToken`, unlike other ordinary mutations.
    Shared authentication permits a session cookie, but an unsafe request with
    that cookie requires both headers
    ([auth_handlers.go](../../internal/httpapi/public/auth_handlers.go), lines 148–153).

    The request description is incomplete: an external client using these
    operations is not told about required headers and may receive 403.
    The shared UI transport already adds CSRF; no Server protection bypass was found.

    Fix: add the same reusable parameters and document their conditional
    requirement for cookie auth. Bearer requests do not need CSRF.

11. **R11 — P3: the 1024-attempt RunStatus bound is not enforced.**

    OpenAPI:5612–5613 sets `maxItems: 1024` for attempts/transitions.
    Workflow validation accepts `retry.maxAttempts=1025`
    ([workflow.go](../../internal/config/workflow.go), lines 690, 957).
    Scheduler creates further attempts up to the configured maximum
    ([stage_finalization.go](../../internal/scheduler/stage_finalization.go),
    lines 204, 241). Store and public handler return the full history.
    A manual-resume limit does not constrain automatic retry.

    Reproduction: the real config validator accepted 1025; the HTTP handler with
    fake persisted history returned 1025 attempts, which the schema rejected.
    No run of 1025 real Scheduler attempts was performed.

    Fix within current-contract reconciliation: remove the unsupported
    `maxItems`. Agree on any global history quota or separate pagination
    independently; do not silently truncate data to satisfy a validator.

12. **R12 — P3: the Git create-precondition description is stricter than implementation.**

    OpenAPI:150,196 says `Requires explicit create or CAS precondition`.
    The shared parser treats absence of both headers as create
    ([request.go](../../internal/httpapi/public/request.go), lines 108–109).
    Git importer permits that request for an absent binding
    ([importer.go](../../internal/gitimport/importer.go), lines 166–180).

    Parser reproduction confirmed that no header is required. For an existing
    binding, a missing expected revision causes conflict; there is no CAS-update bypass.

    Fix: document the actual create default. If an explicit header is genuinely
    needed as a product guarantee, that is separate API tightening requiring
    compatibility checks for existing clients.

**Checks and confidence limits.** YAML was parsed as YAML 1.2 with no duplicate
keys; 338 component schemas passed the Draft 2020-12 metaschema. The file has
88 paths and 111 operations with implementation markers. All operations were
matched to route registration, including six archive routes registered by a helper.

Successfully executed:

```sh
make verify-public-api
go test -count=1 ./internal/httpapi/public ./internal/publicclient/...
```

Fresh generation with pinned `oapi-codegen@v2.8.0` and
`openapi-typescript@7.13.0` produced files byte-identical to the committed Go/TS
clients. Eight temporary Go overlay tests additionally ran with real
validators/handlers and the fake dependencies described above; real configuration
projections and canonical RuntimeConfig JSON were checked. Temporary tests were
not added to the production checkout. For relevant defects, this document
distinguishes behavior demonstrated by a test from consequences established
through code reading.

The review included no live PostgreSQL/LLM/browser end-to-end run. A green
`verify-public-api` verifies the checked fixtures; it does not prove alignment
of every optional field, decision branch or boundary value. No substantial
confirmed issues were found in SchedulerSettings, Credentials or Operations
enums/required arrays within this review.

**Proposed work order.** Fix R01 in a small separate change and add an HTTP
check for proposed reports. R05 can be fixed independently with full-page and
next-cursor tests. Then reconcile catalog and telemetry schemas (R02/R03) and
simple query/auth/decision contracts (R06/R07/R08/R10/R12), regenerating clients
after source changes.

R04/R09 should be addressed together: author/read schemas, nullable request
types and round-trip checks on generated bytes. This work overlaps with
RuntimeConfig contracts in future toolset work; compare branches touching the
same schemas before implementation. R11 can be closed with a local correction
to the documented bound; designing a general history-retention policy is outside
the OpenAPI fix.
