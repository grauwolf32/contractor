# Runnable Audit demo

The operator catalog ships two bounded, non-certifying Audit programs:

- `source-checklist@1` turns a checklist plus a source ZIP into one check per
  checklist entry.
- `openapi-operation-trace@1` turns an OpenAPI document plus a source ZIP into
  one trace item per supported path operation. Callbacks and webhooks are
  reported as inventory or coverage gaps; the Server never follows remote
  references.

Both profiles use ordinary `audit-source-check@1` Runs. The Runtime's
`submit_check_result` tool derives the immutable item identity and execution
manifest digest from trusted Run inputs, writes a canonical result package,
and leaves validation, evidence retention, coverage and settlement to the
Server.

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
