package pgstore

// cleanupProposedItemsSQL removes private staged rows before ordinary closure.
// Both primary rows and exact-hold counter releases share the caller's bound;
// unpublished review requests count as related writes as well.
var cleanupProposedItemsSQL = `
WITH live_claim AS MATERIALIZED (
 ` + auditClaims.LiveClaimSQL("$1", "$2", "$3") + `
), target AS MATERIALIZED (
 SELECT audit.audit_id FROM audits AS audit JOIN live_claim USING(audit_id)
 WHERE audit.state IN ('finalizing','cancelling','deleting') FOR UPDATE OF audit
), candidates AS MATERIALIZED (
 SELECT item.item_id,1+item.retained_dependency_hold_count+
  CASE WHEN item.approval_kind='none' THEN 0 ELSE 1 END AS work
 FROM audit_items AS item JOIN audit_rounds AS round USING(audit_id,round_id) JOIN target USING(audit_id)
 WHERE round.state='proposed' ORDER BY item.item_id LIMIT $4
), weighted AS MATERIALIZED (
 SELECT *,sum(work) OVER(ORDER BY item_id) AS cumulative FROM candidates
), reviews AS (
 DELETE FROM audit_review_requests AS request USING weighted
 WHERE weighted.cumulative<=$4 AND request.audit_id=$1
 AND request.subject_kind='audit-item-action' AND request.subject_id=weighted.item_id RETURNING request.request_id
), removed AS (
 DELETE FROM audit_items AS item USING weighted
 WHERE weighted.cumulative<=$4 AND item.item_id=weighted.item_id
 AND (SELECT count(*) FROM reviews)>=0 RETURNING weighted.work
)
SELECT CASE WHEN EXISTS(SELECT 1 FROM weighted WHERE cumulative>$4) THEN $4
 ELSE COALESCE((SELECT sum(work) FROM removed),0)::integer END`
