package rundeletion

// SQL statements for deleter.go.

// markProposalSourceRunDeletedSQL tombstones the retention state of every
// finding proposal received from Run $1: it sets source_run_deleted_at and
// moves the state to 'audit-held' when an Audit hold exists, otherwise to
// 'discarded' with discarded_at. Existing timestamps are kept.
// Used by deleteReleasedTerminalRun.
var markProposalSourceRunDeletedSQL = `
UPDATE finding_proposal_receipts AS receipt
   SET source_run_deleted_at = COALESCE(receipt.source_run_deleted_at, clock_timestamp()),
       retention_state = CASE WHEN EXISTS (
           SELECT 1 FROM finding_proposal_audit_holds AS hold
            WHERE hold.receipt_id = receipt.receipt_id
       ) THEN 'audit-held' ELSE 'discarded' END,
       discarded_at = CASE WHEN EXISTS (
           SELECT 1 FROM finding_proposal_audit_holds AS hold
            WHERE hold.receipt_id = receipt.receipt_id
       ) THEN NULL ELSE COALESCE(receipt.discarded_at, clock_timestamp()) END,
       retention_updated_at = clock_timestamp()
 WHERE receipt.run_id = $1`
