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

scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT

# A rating or provenance mutation invalidates the snapshot. Buffer the complete
# traversal and retry it from the current Audit/finding revisions so callers
# never mistake a mixed or partial history for a consistent backtrace.
for attempt in {1..5}; do
  audit=$(curl --fail --silent --show-error \
    -H "$authorization" \
    "$api_url/v1/audits/$AUDIT_ID")
  finding=$(curl --fail --silent --show-error \
    -H "$authorization" \
    "$api_url/v1/audits/$AUDIT_ID/findings/$FINDING_ID")
  audit_revision=$(jq -er '.revision' <<<"$audit")
  finding_revision=$(jq -er '.revision' <<<"$finding")
  jq '{findingId, state, analystVerdict, analystSeverity, analystDecision}' \
    <<<"$finding" >"$scratch/finding.json"
  : >"$scratch/provenance.jsonl"

  cursor=""
  restart=false
  while true; do
    query="limit=200&auditRevision=$audit_revision&findingRevision=$finding_revision"
    if [[ -n "$cursor" ]]; then
      query="$query&cursor=$(jq -rn --arg value "$cursor" '$value|@uri')"
    fi
    status=$(curl --silent --show-error \
      -o "$scratch/page.json" -w '%{http_code}' \
      -H "$authorization" \
      "$api_url/v1/audits/$AUDIT_ID/findings/$FINDING_ID/provenance?$query")
    if [[ "$status" == 409 ]]; then
      restart=true
      break
    fi
    if [[ "$status" != 200 ]]; then
      jq -c . "$scratch/page.json" >&2 2>/dev/null || true
      echo "provenance request failed with HTTP $status" >&2
      exit 1
    fi
    jq -c '.items[]' "$scratch/page.json" >>"$scratch/provenance.jsonl"
    if [[ $(jq -r '.page.hasMore' "$scratch/page.json") != true ]]; then
      break
    fi
    cursor=$(jq -er '.page.nextCursor' "$scratch/page.json")
  done

  if [[ "$restart" == false ]]; then
    cat "$scratch/finding.json" "$scratch/provenance.jsonl"
    exit 0
  fi
  echo "finding changed during traversal; restarting snapshot ($attempt/5)" >&2
done

echo "finding kept changing; no revision-consistent backtrace was produced" >&2
exit 1
