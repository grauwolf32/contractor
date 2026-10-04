package auditstore

// SQL statements for finding_events.go.

// recordRejectedFindingProposalSQL bumps the revision and event sequence of
// Audit $1 and appends a finding.proposal_rejected event for receipt $2 naming
// source Run $3 and rejection reason $4. Returns the event's sequence number.
// Used by PostgresStore.RecordRejectedFindingProposal.
var recordRejectedFindingProposalSQL = `
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
SELECT sequence_number FROM recorded`

// recordDirectFindingAssessmentSQL charges $2 retained evidence bytes to Audit
// $1 only while the total stays within max_evidence_bytes, bumps its revision
// and appends a finding.assessed event for finding $3 at revision $4 with
// assessment $5. Over quota it returns no row and changes nothing.
// Used by PostgresStore.RecordDirectFindingAssessment.
var recordDirectFindingAssessmentSQL = `
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
SELECT sequence_number FROM recorded`
