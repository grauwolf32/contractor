#!/usr/bin/env bash
set -euo pipefail

# Read one analyst rating and every revision-consistent provenance page through
# the public API. The bearer token must belong to the Audit owner.
: "${CONTRACTOR_API_URL:?set CONTRACTOR_API_URL, for example http://127.0.0.1:8080}"
: "${CONTRACTOR_API_TOKEN:?set CONTRACTOR_API_TOKEN}"
: "${AUDIT_ID:?set AUDIT_ID}"
: "${FINDING_ID:?set FINDING_ID}"

api_url=${CONTRACTOR_API_URL%/}
authorization="Authorization: Bearer ${CONTRACTOR_API_TOKEN}"

audit=$(curl --fail --silent --show-error \
  -H "$authorization" \
  "$api_url/v1/audits/$AUDIT_ID")
finding=$(curl --fail --silent --show-error \
  -H "$authorization" \
  "$api_url/v1/audits/$AUDIT_ID/findings/$FINDING_ID")

audit_revision=$(jq -er '.revision' <<<"$audit")
finding_revision=$(jq -er '.revision' <<<"$finding")
jq '{findingId, state, analystVerdict, analystSeverity, analystDecision}' \
  <<<"$finding"

cursor=""
while true; do
  query="limit=200&auditRevision=$audit_revision&findingRevision=$finding_revision"
  if [[ -n "$cursor" ]]; then
    query="$query&cursor=$(jq -rn --arg value "$cursor" '$value|@uri')"
  fi
  page=$(curl --fail --silent --show-error \
    -H "$authorization" \
    "$api_url/v1/audits/$AUDIT_ID/findings/$FINDING_ID/provenance?$query")
  jq -c '.items[]' <<<"$page"
  if [[ $(jq -r '.page.hasMore' <<<"$page") != true ]]; then
    break
  fi
  cursor=$(jq -er '.page.nextCursor' <<<"$page")
done
