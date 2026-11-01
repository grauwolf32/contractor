package pgstore

// Each selected item has at most 129 source associations plus its original
// task source. DISTINCT collapses receipts already admitted to one Finding.
const itemFindingIDsSQL = `WITH selected AS MATERIALIZED (
 SELECT item.item_id,item.proposal_receipt_id,audit.project_id,coalesce(member.finding_sources,'[]'::jsonb) AS finding_sources
 FROM unnest($3::text[]) AS requested(item_id)
 JOIN audit_items AS item USING(item_id)
 JOIN audits AS audit USING(audit_id)
 LEFT JOIN audit_execution_items AS member ON member.execution_item_id=item.last_execution_item_id AND member.audit_id=item.audit_id
 WHERE audit.owner_id=$1 AND item.audit_id=$2 AND `+acceptedItemSQL("item")+`
), related AS (
 SELECT item_id,project_id,proposal_receipt_id AS receipt_id FROM selected WHERE proposal_receipt_id IS NOT NULL
 UNION ALL
 SELECT selected.item_id,selected.project_id,association.receipt_id FROM selected
 CROSS JOIN LATERAL jsonb_to_recordset(selected.finding_sources) AS association(receipt_id text)
)
SELECT DISTINCT related.item_id,source.finding_id FROM related
 JOIN project_finding_sources AS source ON source.receipt_id=related.receipt_id AND source.project_id=related.project_id
 ORDER BY related.item_id,source.finding_id`
