package auditservice

// SQL statements for finding_collection.go.

// readCollectionReviewsSQL maps receipts $2 to the findings they contributed to
// or first created, within Audits of owner $1, returning each finding's
// revision, state and current decision, assessment and duplicate target
// (empty strings when unset). Ordered by receipt and capped at $3 rows.
// Used by Service.ReadCollectionReviews.
var readCollectionReviewsSQL = `
WITH membership AS (
    SELECT c.receipt_id, c.finding_id, c.audit_id FROM audit_finding_contributions AS c WHERE c.receipt_id=ANY($2::text[])
    UNION
    SELECT f.first_receipt_id, f.finding_id, f.audit_id FROM audit_findings AS f WHERE f.first_receipt_id=ANY($2::text[])
)
SELECT m.receipt_id, f.audit_id, f.finding_id, f.revision, f.state,
       COALESCE(f.current_decision_id,''), COALESCE(f.current_assessment_id,''), COALESCE(f.duplicate_target_id,'')
  FROM membership AS m JOIN audit_findings AS f ON f.audit_id=m.audit_id AND f.finding_id=m.finding_id
  JOIN audits AS a ON a.audit_id=f.audit_id
 WHERE a.owner_id=$1 ORDER BY m.receipt_id, f.audit_id, f.finding_id LIMIT $3`
