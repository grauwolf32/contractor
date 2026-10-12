package findingintake

import "github.com/grauwolf32/contractor/internal/config/auditprofile"

// SQL statements for postgres.go.

// readReceiptBySubmissionSQL reads the proposal receipt of submission $3 by
// invocation $2 in allocation $1. Used by readReceiptBySubmission.
var readReceiptBySubmissionSQL = `
SELECT ` + receiptProjection + `
  FROM finding_proposal_receipts AS receipt
 WHERE receipt.allocation_id = $1 AND receipt.invocation_id = $2
   AND receipt.submission_id = $3`

// readReceiptByIDSQL reads proposal receipt $1. Used by readReceiptByID.
var readReceiptByIDSQL = `
SELECT ` + receiptProjection + `
  FROM finding_proposal_receipts AS receipt
 WHERE receipt.receipt_id = $1`

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

// listAuditInboxSQL returns one page of the inbox of owner $1's Audit $2,
// oldest first: the owner's proposal receipts that the Audit originated while
// their source Run exists, and those the Audit holds in its Project. The
// cursor ($3, $4) and the limit $5 are those of listRunReceiptRowsSQL.
// Used by Service.ListAuditInbox.
var listAuditInboxSQL = `
SELECT ` + receiptProjection + `
  FROM finding_proposal_receipts AS receipt
 WHERE receipt.owner_id = $1
   AND ((receipt.audit_id = $2 AND receipt.source_run_deleted_at IS NULL) OR EXISTS (
       SELECT 1 FROM finding_proposal_audit_holds AS hold
        JOIN audits AS audit
          ON audit.audit_id = hold.audit_id AND audit.project_id = hold.project_id
       WHERE hold.receipt_id = receipt.receipt_id
         AND hold.audit_id = $2 AND audit.owner_id = $1
   ))
   AND ($3::timestamptz IS NULL OR (receipt.created_at, receipt.receipt_id) > ($3, $4))
 ORDER BY receipt.created_at, receipt.receipt_id
 LIMIT $5`

// listAuditHeldInboxSQL returns one page of owner $1's proposal receipts that
// the owner's Audit $2 holds in its Project, oldest first. The cursor
// ($3, $4) and the limit $5 are those of listRunReceiptRowsSQL.
// Used by Service.ListAuditHeldInbox.
var listAuditHeldInboxSQL = `
SELECT ` + receiptProjection + `
  FROM finding_proposal_receipts AS receipt
  JOIN finding_proposal_audit_holds AS hold
    ON hold.receipt_id = receipt.receipt_id AND hold.audit_id = $2
  JOIN audits AS audit
    ON audit.audit_id = hold.audit_id AND audit.project_id = hold.project_id
 WHERE receipt.owner_id = $1 AND audit.owner_id = $1
   AND ($3::timestamptz IS NULL OR (receipt.created_at, receipt.receipt_id) > ($3, $4))
 ORDER BY receipt.created_at, receipt.receipt_id
 LIMIT $5`

// importIntoAuditWithTxSQL locks owner $1's retained proposal receipt for Run
// $3 and exact proposal ref $4, with its retention state. Destination Audit
// $2, which the caller holds locked, must be in the Run's Project and one of
// states $5, require human finding confirmation and have no pending report
// review. Returns the receipt's proposal and evidence data.
// Used by Service.importIntoAuditWithTx.
var importIntoAuditWithTxSQL = `
SELECT receipt.receipt_id, audit.project_id, receipt.proposal_ref, receipt.evidence,
       receipt.invocation_id, receipt.client_key, receipt.workflow_closure_digest,
       receipt.proposal_digest, receipt.proposal_media_type, receipt.proposal_size_bytes
  FROM finding_proposal_receipts AS receipt
  JOIN workflow_runs AS run ON run.run_id = receipt.run_id
  JOIN audits AS audit ON audit.audit_id = $2
 WHERE receipt.owner_id = $1 AND receipt.run_id = $3
   AND receipt.proposal_ref = $4::jsonb
   AND run.owner_id = $1 AND run.project_id = audit.project_id
   AND audit.owner_id = $1
   AND audit.state = ANY($5::text[])
   AND NOT EXISTS (
       SELECT 1
         FROM audit_review_requests AS review
        WHERE review.audit_id = audit.audit_id AND review.subject_kind = 'audit-report'
          AND review.state = 'pending'
   )
   AND audit.profile_snapshot #>> '` + auditprofile.AuditProfileFindingConfirmationPath + `' = '` +
	string(auditprofile.AuditFindingHumanRequired) + `'
 FOR UPDATE OF receipt`
