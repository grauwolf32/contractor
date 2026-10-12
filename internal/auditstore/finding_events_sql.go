package auditstore

// SQL statements for finding_events.go.

// rejectedFindingProposalEvents record proposal receipt $2 of source Run $3
// rejected for reason $4.
var rejectedFindingProposalEvents = auditEvents(auditEvent(
	"'finding.proposal_rejected'", "$2::text", "NULL",
	"jsonb_build_object('runId', $3::text, 'reason', $4::text)",
))

// recordRejectedFindingProposalSQL advances Audit $1 and records a
// finding.proposal_rejected event for receipt $2 naming source Run $3 and
// rejection reason $4. Returns the event's sequence number.
// Used by PostgresStore.RecordRejectedFindingProposal.
var recordRejectedFindingProposalSQL = `
WITH advanced AS (
    UPDATE audits AS audit
       SET ` + rejectedFindingProposalEvents.advance() + `,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
     WHERE audit.audit_id = $1
    RETURNING ` + rejectedFindingProposalEvents.appended() + `
)
SELECT last_event_sequence FROM advanced`

// directFindingAssessmentEvents record assessment $5 of finding $3 at
// revision $4.
var directFindingAssessmentEvents = auditEvents(auditEvent(
	"'finding.assessed'", "$3::text", "$4::bigint",
	"jsonb_build_object('assessmentId', $5::text, 'directVerification', true)",
))

// recordDirectFindingAssessmentSQL charges $2 retained evidence bytes to Audit
// $1 only while the total stays within max_evidence_bytes, advances it and
// records a finding.assessed event for finding $3 at revision $4 with
// assessment $5. Over quota it returns no row and changes nothing.
// Used by PostgresStore.recordDirectFindingAssessment.
var recordDirectFindingAssessmentSQL = `
WITH advanced AS (
    UPDATE audits AS audit
       SET retained_evidence_bytes = audit.retained_evidence_bytes + $2,
           ` + directFindingAssessmentEvents.advance() + `,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
     WHERE audit.audit_id = $1
       AND audit.retained_evidence_bytes + $2 <= audit.max_evidence_bytes
    RETURNING ` + directFindingAssessmentEvents.appended() + `
)
SELECT last_event_sequence FROM advanced`
