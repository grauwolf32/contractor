package findingintake

// SQL statements for postgres.go.

// retainProjectReceiptSQL upgrades ordinary source ownership exactly once.
// It changes no proposal, publication intent, finding identity or assessment.
const retainProjectReceiptSQL = `
UPDATE finding_proposal_receipts
SET project_proposal=$2::jsonb,project_evidence=$3::jsonb,
 retention_state='project-held',discarded_at=NULL,retention_updated_at=clock_timestamp()
WHERE receipt_id=$1 AND project_proposal IS NULL`

// readReceiptBySubmissionSQL reads the proposal receipt of submission $3 by
// invocation $2 in allocation $1. Used by readReceiptBySubmission.
var readReceiptBySubmissionSQL = `
SELECT ` + receiptProjection + `
  FROM finding_proposal_receipts AS receipt
 WHERE receipt.allocation_id = $1 AND receipt.invocation_id = $2
   AND receipt.submission_id = $3`

// listRunReceiptRowsSQL returns one page of owner $1's proposal receipts of
// Run $2, which the owner also owns, oldest first. ($3, $4) is the
// (created_at, receipt_id) cursor of the previous page; a NULL $3 starts at
// the first page. At most $5 rows. Used by Service.listRunReceiptRows.
var listRunReceiptRowsSQL = `
SELECT ` + receiptProjection + `
  FROM finding_proposal_receipts AS receipt
  JOIN workflow_runs AS run ON run.run_id = receipt.run_id
 WHERE receipt.owner_id = $1 AND receipt.run_id = $2 AND run.owner_id = $1
   AND ($3::timestamptz IS NULL OR (receipt.created_at, receipt.receipt_id) > ($3, $4))
 ORDER BY receipt.created_at, receipt.receipt_id
 LIMIT $5`

// Audit inbox pages are oldest first in receipt (created_at, receipt_id)
// order, a total order because receipt_id is unique. ($3, $4) is the cursor of
// the previous page, as for listRunReceiptRowsSQL; with a NULL $3 and an empty
// $4 the bound becomes minus infinity and the empty ID, which every receipt
// follows. Spelled as one row comparison over the (time, ID) columns of the
// index a branch walks, the bound stays an index condition in a generic plan,
// which a "$3 IS NULL OR ..." disjunction does not.
func auditInboxAfterCursorSQL(createdAt, receiptID string) string {
	return `(` + createdAt + `, ` + receiptID + `) >
           (COALESCE($3::timestamptz, '-infinity'), COALESCE($4::text, ''))`
}

// auditNativeInboxKeysSQL selects the keys of the first $5 receipts after the
// cursor that owner $1's Audit $2 originated while their source Run exists.
// It walks finding_proposal_receipts_audit_idx in page order, so it reads the
// Audit's own receipts after the cursor until $5 pass, never another Audit's.
var auditNativeInboxKeysSQL = `
    SELECT receipt.receipt_id, receipt.created_at
      FROM finding_proposal_receipts AS receipt
     WHERE receipt.audit_id = $2 AND receipt.owner_id = $1
       AND receipt.source_run_deleted_at IS NULL
       AND ` + auditInboxAfterCursorSQL("receipt.created_at", "receipt.receipt_id") + `
     ORDER BY receipt.created_at, receipt.receipt_id
     LIMIT $5`

// auditHeldInboxKeysSQL selects the keys of the first $5 receipts of owner $1
// after the cursor that the owner's Audit $2 holds in its own Project. Each
// hold keeps its receipt's creation time (receipt_created_at, copied when the
// hold is inserted), so the branch walks
// finding_proposal_audit_holds_audit_receipt_idx in page order after the
// cursor and probes the receipt of each hold it passes by primary key: it
// reads the page's holds, not every hold of the Audit (FND-R186). The Audit's
// Project is read once, so the walk is not joined with the Audit row and keeps
// its index order. OFFSET 0 keeps the receipt probe per hold, so a generic
// plan cannot turn it into a scan of every owner's receipts.
var auditHeldInboxKeysSQL = `
    SELECT receipt.receipt_id, receipt.created_at
      FROM finding_proposal_audit_holds AS hold
     CROSS JOIN LATERAL (
           SELECT candidate.receipt_id, candidate.created_at
             FROM finding_proposal_receipts AS candidate
            WHERE candidate.receipt_id = hold.receipt_id AND candidate.owner_id = $1
           OFFSET 0
       ) AS receipt
     WHERE hold.audit_id = $2
       AND hold.project_id = (
           SELECT audit.project_id FROM audits AS audit
            WHERE audit.audit_id = $2 AND audit.owner_id = $1
       )
       AND ` + auditInboxAfterCursorSQL("hold.receipt_created_at", "hold.receipt_id") + `
     ORDER BY hold.receipt_created_at, hold.receipt_id
     LIMIT $5`

// listAuditInboxSQL returns one page of the inbox of owner $1's Audit $2:
// the owner's proposal receipts that the Audit originated while their source
// Run exists, and those the Audit holds in its Project. The cursor ($3, $4)
// and the limit $5 are those of listRunReceiptRowsSQL.
//
// Each branch keeps its own first $5 receipts after the cursor in page order,
// and the first $5 of their union are the first $5 of the whole inbox: a
// receipt among the inbox's first $5 has fewer than $5 predecessors in its own
// branch. UNION keeps a receipt that is both native and held once, so it takes
// one slot, and the next page starts after the last receipt returned in both
// branches alike. Each branch reads its first $5 entries after the cursor, and
// only the page's rows are read whole.
// Used by Service.ListAuditInbox.
var listAuditInboxSQL = `
WITH page AS (
    (` + auditNativeInboxKeysSQL + `)
     UNION
    (` + auditHeldInboxKeysSQL + `)
     ORDER BY created_at, receipt_id
     LIMIT $5
)
SELECT ` + receiptProjection + `
  FROM page
  JOIN finding_proposal_receipts AS receipt ON receipt.receipt_id = page.receipt_id
 ORDER BY receipt.created_at, receipt.receipt_id`

// listAuditHeldInboxSQL returns one page of owner $1's proposal receipts that
// the owner's Audit $2 holds in its Project, oldest first: the held branch of
// listAuditInboxSQL alone. The cursor ($3, $4) and the limit $5 are those of
// listRunReceiptRowsSQL. Used by Service.ListAuditHeldInbox.
var listAuditHeldInboxSQL = `
WITH page AS (` + auditHeldInboxKeysSQL + `
)
SELECT ` + receiptProjection + `
  FROM page
  JOIN finding_proposal_receipts AS receipt ON receipt.receipt_id = page.receipt_id
 ORDER BY receipt.created_at, receipt.receipt_id`

// importIntoAuditWithTxSQL locks owner $1's retained proposal receipt for Run
// $3 and exact proposal ref $4, with its retention state. Destination Audit
// $2, which the caller holds locked, must be in the Run's Project and one of
// states $5. Returns the receipt's proposal and evidence data.
// Used by Service.importIntoAuditWithTx.
var importIntoAuditWithTxSQL = `
SELECT receipt.receipt_id, audit.project_id, receipt.proposal_ref, receipt.evidence,
       receipt.invocation_id, receipt.client_key, receipt.workflow_closure_digest,
       receipt.proposal_digest, receipt.proposal_media_type, receipt.proposal_size_bytes,
       receipt.project_proposal,receipt.project_evidence
  FROM finding_proposal_receipts AS receipt
  JOIN workflow_runs AS run ON run.run_id = receipt.run_id
  JOIN audits AS audit ON audit.audit_id = $2
 WHERE receipt.owner_id = $1 AND receipt.run_id = $3
   AND receipt.proposal_ref = $4::jsonb
   AND run.owner_id = $1 AND run.project_id = audit.project_id
   AND audit.owner_id = $1
   AND audit.state = ANY($5::text[])
 FOR UPDATE OF receipt`
