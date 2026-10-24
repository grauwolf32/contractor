package pgstore

// SQL statements for finding_collection.go.

// readCollectionReviewsSQL maps receipts $2 to the findings they contributed to
// or first created, within Audits of owner $1, returning each finding's
// revision, state and current decision, assessment and duplicate target
// (empty strings when unset). Ordered by receipt and capped at $3 rows.
// Used by QueryStore.CollectionReviews.
const readCollectionReviewsSQL = `
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

// readCollectionReceiptIDsSQL resolves findings $3 of owner $1's Audit $2 in
// one read. Rows of kind 0 give, for each requested finding's 1-based
// ordinal, its current revision, NULL when the owner's Audit has no such
// finding. Rows of kind 1 give the distinct receipts that contributed to or
// first proposed the requested findings, reading at most $4 contributions per
// finding and returning at most $4 receipts, in receipt order.
// Used by QueryStore.CollectionReceiptIDs.
const readCollectionReceiptIDsSQL = `
WITH request AS (
    SELECT requested.finding_id, requested.ordinality
      FROM unnest($3::text[]) WITH ORDINALITY AS requested(finding_id, ordinality)
), owned AS (
    SELECT a.audit_id FROM audits AS a WHERE a.audit_id=$2 AND a.owner_id=$1
)
SELECT 0 AS kind, request.ordinality, f.revision, NULL::text AS receipt_id
  FROM request
  LEFT JOIN audit_findings AS f
    ON f.audit_id IN (SELECT audit_id FROM owned) AND f.finding_id=request.finding_id
UNION ALL
(SELECT 1, NULL, NULL, receipts.receipt_id
   FROM (
       SELECT contribution.receipt_id
         FROM request
         JOIN owned ON true
        CROSS JOIN LATERAL (
            SELECT c.receipt_id FROM audit_finding_contributions AS c
             WHERE c.finding_id=request.finding_id AND c.audit_id=owned.audit_id
             ORDER BY c.receipt_id LIMIT $4
        ) AS contribution
       UNION
       SELECT f.first_receipt_id
         FROM request
         JOIN audit_findings AS f ON f.finding_id=request.finding_id
         JOIN owned ON owned.audit_id=f.audit_id
   ) AS receipts
  ORDER BY receipts.receipt_id
  LIMIT $4)`
