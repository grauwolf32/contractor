package findingintake

// SQL statements for collection_list.go.

// listAuditCollectionSQL reports, per receipt in $1, its state in owner $3's
// Audit $2: whether an Audit hold exists, whether that hold was created at or
// after Run $4 finished, whether a direct-verification assessment exists, and
// whether a finding.proposal_rejected event names it. Returns no rows unless
// the Audit and Run both belong to $3. Used by Service.ListAuditCollection.
var listAuditCollectionSQL = `
SELECT input.receipt_id,
       hold.receipt_id IS NOT NULL,
       COALESCE(hold.created_at >= run.finished_at, false),
       EXISTS (SELECT 1 FROM audit_finding_assessments AS assessment
                WHERE assessment.receipt_id = input.receipt_id
                  AND assessment.audit_id = $2 AND assessment.direct_verification),
       EXISTS (SELECT 1 FROM audit_events AS event
                WHERE event.audit_id = $2 AND event.kind = 'finding.proposal_rejected'
                  AND event.entity_id = input.receipt_id)
  FROM unnest($1::text[]) AS input(receipt_id)
  JOIN audits AS audit ON audit.audit_id = $2 AND audit.owner_id = $3
  JOIN workflow_runs AS run ON run.run_id = $4 AND run.owner_id = $3
  LEFT JOIN finding_proposal_audit_holds AS hold
    ON hold.receipt_id = input.receipt_id AND hold.audit_id = $2`
