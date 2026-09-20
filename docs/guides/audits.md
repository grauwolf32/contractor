# Runnable Audit demo

An Audit coordinates Workflows into an application security assessment. Its
AuditProfile selects the check Workflows and assessment rules, while the Audit
tracks scope, coverage, evidence, findings and review across their Runs. See the
[Audit specification](../spec/19-audits.md) for the complete coordination model.

The operator catalog ships two bounded source-input demo programs:

- `source-checklist@1` turns a checklist plus a source ZIP into one check per
  checklist entry and executes up to two compatible checks in one ordinary Run.
- `openapi-operation-observe@1` turns an OpenAPI document plus a source ZIP into
  one trace item per supported path operation. Callbacks and webhooks are
  reported as inventory or coverage gaps; the Server never follows remote
  references.

The profiles launch `audit-source-check@1` and
`audit-openapi-operation-observe@1` as Audit-managed child Runs. Their trusted
completion binding selects `audit-results@2`. The model records each item's
result; Runtime publishes the complete package after normal completion and
leaves validation, evidence retention, coverage and settlement to
the Server. Each item remains independently visible even when it shares a Run.

## Time limits, pause and continuation

The play icon next to an Audit opens its start or continuation settings. The UI
defaults to seven days; choose 24 hours, a custom duration, or **No time limit**.
The clock includes queue waiting, but manual pauses preserve the remaining time.
At the limit, the Audit pauses new Run submissions while existing Runs finish
and their results are collected. Use the pause icon to do the same manually.

`POST /v1/audits/{auditId}/start` and `/resume` accept an optional JSON body:
`{"deadlineSeconds":604800}` for seven days, or `{"deadlineSeconds":0}` for no
Audit time limit. Omitting the body at start preserves the profile default;
omitting it at resume preserves the remaining paused time, or renews the profile
allowance if it was exhausted. These mutations still require `If-Match` and
`Idempotency-Key`.

**Continue Audit** is available for paused Audits, including those paused at
the time limit. Completed, failed and cancelled Audits remain final, including
older records closed with `deadline_exhausted`. Their results and reports remain
available for inspection.
Expired task approvals require a new human decision.

## Start a checklist Audit

The following commands assume a running local stack, `jq`, a source archive at
`./source.zip`, and the repository fixture checklist. Replace the bearer token
with the token configured for the Server.

```sh
export CONTRACTOR_URL=http://127.0.0.1:8080
export CONTRACTOR_TOKEN=contractor-local-token
export AUTHORIZATION="Authorization: Bearer ${CONTRACTOR_TOKEN}"

curl -fsS -H "$AUTHORIZATION" \
  "$CONTRACTOR_URL/v1/audit-profiles?limit=100" |
  jq '.items[] | {ref, serverCompatible, compatibilityReasons}'

project=$(
  curl -fsS -X POST -H "$AUTHORIZATION" \
    -H 'Content-Type: application/json' \
    -H 'Idempotency-Key: audit-demo-project' \
    --data '{"kind":"project","name":"Audit demo","description":"Bounded source audit"}' \
    "$CONTRACTOR_URL/v1/projects"
)
project_id=$(printf '%s' "$project" | jq -r .projectId)

source_ref=$(
  curl -fsS -X PUT -H "$AUTHORIZATION" -H 'If-None-Match: *' \
    -H 'Content-Type: application/zip' --data-binary @./source.zip \
    "$CONTRACTOR_URL/v1/projects/$project_id/artifacts/sources/source" |
    jq -c .artifact
)
checklist_ref=$(
  curl -fsS -X PUT -H "$AUTHORIZATION" -H 'If-None-Match: *' \
    -H 'Content-Type: application/yaml' \
    --data-binary @tests/fixtures/audits/checklist.yaml \
    "$CONTRACTOR_URL/v1/projects/$project_id/artifacts/checklists/checklist" |
    jq -c .artifact
)

audit=$(
  jq -nc --argjson source "$source_ref" --argjson checklist "$checklist_ref" \
    '{profile:{name:"source-checklist",version:"1"},
      inputs:{source:$source,checklist:$checklist},
      scope:{objective:"Trace the bounded checklist against this source revision."}}' |
  curl -fsS -X POST -H "$AUTHORIZATION" \
    -H 'Content-Type: application/json' \
    -H 'Idempotency-Key: audit-demo-checklist' --data-binary @- \
    "$CONTRACTOR_URL/v1/projects/$project_id/audits"
)
audit_id=$(printf '%s' "$audit" | jq -r .auditId)
revision=$(printf '%s' "$audit" | jq -r .revision)

curl -fsS -X POST -H "$AUTHORIZATION" \
  -H 'Idempotency-Key: audit-demo-checklist-start' \
  -H "If-Match: \"$revision\"" \
  "$CONTRACTOR_URL/v1/audits/$audit_id/start" | jq .
```

Poll the authoritative Audit projection and inspect its settled checks,
coverage and exact retained report:

```sh
curl -fsS -H "$AUTHORIZATION" "$CONTRACTOR_URL/v1/audits/$audit_id" | jq .
curl -fsS -H "$AUTHORIZATION" "$CONTRACTOR_URL/v1/audits/$audit_id/items?limit=100" | jq .
curl -fsS -H "$AUTHORIZATION" "$CONTRACTOR_URL/v1/audits/$audit_id/coverage?limit=100" | jq .
curl -fsS -H "$AUTHORIZATION" "$CONTRACTOR_URL/v1/audits/$audit_id/report" | jq .
```

A completed Audit can legitimately report `completed-with-gaps`,
`inconclusive`, `unmapped`, or partial trace coverage. Completion means the
bounded execution and collection lifecycle settled; it is not a security or
compliance certification. The same flow starts `openapi-operation-observe@1` by
uploading an `openapi` input and changing the exact profile selector.

## Curated Top 10, ASVS and WSTG programs

These operator profiles exercise the same ordinary Run path with an
exact licensed standard package:

- `owasp-top10-2025-source-risk@1` assesses ten bounded, independently authored
  source-risk scenarios mapped to the OWASP Top 10:2025 categories. It is a
  risk-awareness report, not exhaustive vulnerability discovery.
- `owasp-asvs-5-0-l1-source-pilot@1` verifies a selected five-requirement ASVS
  5.0.0 Level 1 pilot. Its selection, version-qualified requirement IDs,
  automated evidence, manual work, and not-applicable decisions remain separate
  in coverage and in the machine report.
- `owasp-asvs-5-0-l1-source-review@1` expands the source and documentation review
  to all 70 ASVS 5.0.0 Level 1 requirements using a new immutable package edition.
- `owasp-wstg-4-2-source-review@1` is a separate source-review Audit covering 94
  active WSTG 4.2 scenarios. It records source-backed violations and gaps; it
  does not execute live tests or report their success.
- `owasp-wstg-4-2-active-http@1` checks a running website/API using HTTP evidence
  and per-item active-check approval. It has the same 94 WSTG scenarios and
  explicitly records unsupported browser, network and identity checks as gaps.
- `owasp-wstg-4-2-fast-source-review@1` and
  `owasp-wstg-4-2-fast-active-http@1` provide a first pass over 16 selected WSTG
  scenarios using the same evidence rules. The other 78 scenarios are outside
  coverage. Each Fast Audit allows at most 16 Runs and one attempt per item.

Source-review profiles require the exact `source` ZIP input. Active HTTP WSTG
requires a text/Markdown `context` brief and the target and authorization scope.
See
[standard sources and scope](audit-standard-sources.md) for the exact upstream
revisions, evidence limits and reproducible package generation. The Audit baseline exposes
the pinned standard source revision, license, retained package digest, selected
denominator, and exact `trace` Skill source. Removing the current profile,
Workflow, AgentTemplate, or standard catalog entry does not rewrite a completed
Audit; its retained report and provenance remain readable until Audit deletion.

## Trace one finding after source deletion

The public API retains exact source and verifier provenance owned by the Audit,
even after a collected child Run is deleted or the mutable Workflow/checklist
catalog changes. The example script reads one revision-consistent snapshot. If
an analyst edits the finding between pages, the API returns `409` and the script
restarts the entire traversal instead of combining revisions.

```sh
export CONTRACTOR_API_URL="$CONTRACTOR_URL"
export CONTRACTOR_API_TOKEN="$CONTRACTOR_TOKEN"
export AUDIT_ID="$audit_id"
export FINDING_ID=finding_...
docs/examples/audit-finding-backtrace.sh
```

The output starts with the current analyst verdict/severity and then emits one
JSON line per source proposal, check attempt, or direct verification. Deleted
Runs remain identifiable through their retained exact Workflow closure and
`runDeleted` provenance; the script does not need database or catalog access.

## Audit release verification

The general strict conformance and fault map is
[`tests/e2e/audits_matrix.yml`](../../tests/e2e/audits_matrix.yml). The curated
program release contract is separately declared in
[`tests/e2e/audit_program_library_matrix.yml`](../../tests/e2e/audit_program_library_matrix.yml).
Every case names its executable owner and the durable boundary it proves. Run
the complete curated-program gate against a disposable PostgreSQL database:

```sh
export CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:contractor@127.0.0.1:5432/contractor_test?sslmode=disable'
make test-audit-program-library-e2e
```

For diagnosis, the curated gate is split into
`test-audit-program-library-matrix`, `test-audit-program-library-hardening`,
`test-audit-program-library-process`, and
`test-audit-program-library-browser`. Process tests launch real Server and
Runtime processes with mTLS; the browser gate serves the independently built
frontend against the public API. The aggregate `release-verify` target includes
this complete gate. A failed component is a release failure—coverage rows and a
terminal Audit alone are not evidence that recovery, retention, isolation, or
UI contracts passed.
