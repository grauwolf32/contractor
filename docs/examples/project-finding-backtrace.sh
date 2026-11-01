#!/usr/bin/env bash
set -euo pipefail
# One canonical assessment plus immutable producer receipts and result/history
# descriptors. Every request is bounded; complete output is buffered on disk
# until the Project revision and the Finding revision have been checked.
: "${CONTRACTOR_API_URL:?set CONTRACTOR_API_URL}"
: "${CONTRACTOR_API_TOKEN:?set CONTRACTOR_API_TOKEN}"
: "${PROJECT_ID:?set PROJECT_ID}"
: "${FINDING_ID:?set FINDING_ID}"
api_url=${CONTRACTOR_API_URL%/}
authorization="Authorization: Bearer ${CONTRACTOR_API_TOKEN}"
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
project_path="$api_url/v1/projects/$PROJECT_ID"

fetch() {
 local status
 status=$(curl --silent --show-error -o "$scratch/answer.json" -w '%{http_code}' -H "$authorization" "$1")
 if [[ "$status" == 409 ]]; then return 2; fi
 if [[ "$status" != 200 ]]; then
   echo "backtrace request failed with HTTP $status" >&2
   return 1
 fi
}
for attempt in {1..5}; do
 fetch "$project_path/findings/$FINDING_ID"
 cp "$scratch/answer.json" "$scratch/finding.json"
 finding_revision=$(jq -er '.revision' "$scratch/finding.json")
 : >"$scratch/history.jsonl"
 cursor=""; project_revision=""; restart=false
 while true; do
   query="limit=64&findingId=$FINDING_ID"
   if [[ -n "$cursor" ]]; then query="$query&cursor=$(jq -rn --arg value "$cursor" '$value|@uri')"; fi
   if fetch "$project_path/finding-sources?$query"; then :; else
     code=$?;if [[ "$code" != 2 ]];then exit "$code";fi;restart=true;break
   fi
   project_revision=$(jq -er '.projectRevision' "$scratch/answer.json")
   cp "$scratch/answer.json" "$scratch/sources.json"
   jq -r '.items[].receiptId' "$scratch/sources.json" >"$scratch/receipts.txt"
   while IFS= read -r receipt_id; do
     fetch "$project_path/finding-sources/$receipt_id"
     jq -c '{kind:"source",source:.}' "$scratch/answer.json" >>"$scratch/history.jsonl"
     history_cursor=""
     while true; do
       history_query="limit=64&projectRevision=$project_revision"
       if [[ -n "$history_cursor" ]];then history_query="$history_query&cursor=$(jq -rn --arg value "$history_cursor" '$value|@uri')";fi
       if fetch "$project_path/finding-sources/$receipt_id/history?$history_query";then :;else
         code=$?;if [[ "$code" != 2 ]];then exit "$code";fi;restart=true;break
       fi
       jq -c --arg receiptId "$receipt_id" '.items[]|{kind:"history",receiptId:$receiptId,artifact:.}' "$scratch/answer.json" >>"$scratch/history.jsonl"
       if [[ $(jq -r '.page.hasMore' "$scratch/answer.json") != true ]];then break;fi
       history_cursor=$(jq -er '.page.nextCursor' "$scratch/answer.json")
     done
     if [[ "$restart" == true ]];then break;fi
   done <"$scratch/receipts.txt"
   if [[ "$restart" == true || $(jq -r '.page.hasMore' "$scratch/sources.json") != true ]];then break;fi
   cursor=$(jq -er '.page.nextCursor' "$scratch/sources.json")
 done
 if [[ "$restart" == false ]];then
   if fetch "$project_path/finding-sources?limit=1&findingId=$FINDING_ID&projectRevision=$project_revision";then :;else
     code=$?;if [[ "$code" != 2 ]];then exit "$code";fi;restart=true
   fi
   fetch "$project_path/findings/$FINDING_ID"
   if [[ $(jq -er '.revision' "$scratch/answer.json") != "$finding_revision" ]];then restart=true;fi
 fi
 if [[ "$restart" == false ]];then
   jq -c . "$scratch/finding.json"
   cat "$scratch/history.jsonl"
   exit 0
 fi
 echo "finding changed during traversal; restarting snapshot ($attempt/5)" >&2
done
echo "finding kept changing; no revision-consistent backtrace was produced" >&2
exit 1
