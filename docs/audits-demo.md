# Runnable Audit demo

The operator catalog ships two bounded, non-certifying Audit programs:

- `source-checklist@1` turns a checklist plus a source ZIP into one check per
  checklist entry and executes up to two compatible checks in one ordinary Run.
- `openapi-operation-trace@1` turns an OpenAPI document plus a source ZIP into
  one trace item per supported path operation. Callbacks and webhooks are
  reported as inventory or coverage gaps; the Server never follows remote
  references.

Both profiles use ordinary `audit-source-check@1` Runs. The Runtime's
`submit_check_result` tool derives immutable item identities and the execution
manifest digest from trusted Run inputs, writes one complete canonical result
package, and leaves validation, evidence retention, coverage and settlement to
the Server. Each item remains independently visible even when it shares a Run.

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
compliance certification. The same flow starts `openapi-operation-trace@1` by
uploading an `openapi` input and changing the exact profile selector.

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

The strict conformance and fault map is
[`tests/e2e/audits_matrix.yml`](../tests/e2e/audits_matrix.yml). Every case names
its executable owner and the durable boundary or fault it proves. Run the full
gate against a disposable PostgreSQL database:

```sh
export CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:contractor@127.0.0.1:5432/contractor_test?sslmode=disable'
make test-audits-e2e
```

For diagnosis, the gate is split into `test-audits-matrix`,
`test-audits-hardening`, `test-audits-process`, and `test-audits-browser`.
Process tests launch real Server and Runtime processes with mTLS; the browser
gate serves the independently built frontend against the public API. The
aggregate `release-verify` target includes the complete Audit gate. A failed
component is a release failure—coverage rows and a terminal Audit alone are not
evidence that recovery, retention, isolation, or UI contracts passed.
