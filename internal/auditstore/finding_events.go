package auditstore

import "context"

type RejectedFindingProposalParams struct {
	AuditID   string
	ReceiptID string
	RunID     string
	Reason    string
}

// RecordRejectedFindingProposal advances one Audit revision for a proposal
// rejected by finding intake. The caller owns the receipt replay check and
// transaction; the Audit revision and event are written in one statement.
func (s *PostgresStore) RecordRejectedFindingProposal(
	ctx context.Context, params RejectedFindingProposalParams,
) error {
	var sequence int64
	return s.db.QueryRow(ctx, `
WITH advanced AS (
    UPDATE audits
       SET revision = revision + 1,
           next_event_sequence = next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
     WHERE audit_id = $1
    RETURNING audit_id, next_event_sequence
), recorded AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, summary)
    SELECT audit_id, next_event_sequence - 1, 'finding.proposal_rejected', $2,
           jsonb_build_object('runId', $3::text, 'reason', $4::text)
      FROM advanced
    RETURNING sequence_number
)
SELECT sequence_number FROM recorded`,
		params.AuditID, params.ReceiptID, params.RunID, params.Reason,
	).Scan(&sequence)
}

type DirectFindingAssessmentParams struct {
	AuditID         string
	FindingID       string
	FindingRevision int64
	AssessmentID    string
	RetainedBytes   int64
}

// RecordDirectFindingAssessment charges the Audit evidence quota and records
// the finding assessment in the caller's transaction. A failed quota check
// returns pgx.ErrNoRows without changing the Audit or reserving a sequence.
func (s *PostgresStore) RecordDirectFindingAssessment(
	ctx context.Context, params DirectFindingAssessmentParams,
) error {
	var sequence int64
	return s.db.QueryRow(ctx, `
WITH advanced AS (
    UPDATE audits
       SET retained_evidence_bytes = retained_evidence_bytes + $2,
           revision = revision + 1,
           next_event_sequence = next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
     WHERE audit_id = $1
       AND retained_evidence_bytes + $2 <= max_evidence_bytes
    RETURNING audit_id, next_event_sequence
), recorded AS (
    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    )
    SELECT audit_id, next_event_sequence - 1, 'finding.assessed', $3, $4,
           jsonb_build_object('assessmentId', $5::text, 'directVerification', true)
      FROM advanced
    RETURNING sequence_number
)
SELECT sequence_number FROM recorded`,
		params.AuditID, params.RetainedBytes, params.FindingID,
		params.FindingRevision, params.AssessmentID,
	).Scan(&sequence)
}
