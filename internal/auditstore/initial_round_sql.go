package auditstore

// initialRoundEvents record the first round $5 with the items $7 and the
// reviews they open.
var initialRoundEvents = auditEvents(auditEvent(
	"'round.accepted'", "$5::text", "1",
	`jsonb_build_object('round', 1, 'items', jsonb_array_length($7::jsonb),
                              'reviews', `+initialRoundReviewCount+`)`,
))

// The live claim is locked before the Audit revision. Cancelling, deleting,
// pausing or reaching the deadline fences acceptance after package publication.
var acceptInitialRoundSQL = `
WITH live_claim AS MATERIALIZED (
    SELECT audit_id FROM audit_controller_claims
     WHERE audit_id = $1 AND holder_id = $2 AND epoch = $3
       AND expires_at > clock_timestamp()
     FOR UPDATE
), round_input AS MATERIALIZED (
    SELECT $5::text AS round_id, 1 AS ordinal, $6::jsonb AS manifest_ref,
           $7::jsonb AS items, $8::text AS manifest_digest,
           $9::jsonb AS links, $11::text AS acceptance_digest
), audit_gate AS MATERIALIZED (
    SELECT audit.audit_id,
           contractor_require_active_audit_project(audit.project_id, audit.owner_id)
      FROM audits AS audit JOIN live_claim USING (audit_id)
     WHERE audit.audit_id = $1 AND audit.revision = $4
       AND audit.state = 'active' AND audit.dispatch_state = 'open'
       AND audit.phase = 'inventory' AND audit.current_round_id IS NULL
       AND audit.outstanding_run_count = 0
       AND (audit.deadline_at IS NULL OR audit.deadline_at > clock_timestamp())
       AND NOT EXISTS (SELECT 1 FROM audit_rounds WHERE audit_id = $1)
       AND jsonb_array_length($7::jsonb) BETWEEN 1 AND audit.max_items_per_round
       AND jsonb_array_length($7::jsonb) <= audit.max_items_total
       AND audit.retained_evidence_bytes + $10 <= audit.max_evidence_bytes
     FOR UPDATE OF audit
), started AS (
    UPDATE audits AS audit
       SET phase = 'rounds', current_round_id = $5,
           retained_evidence_bytes = audit.retained_evidence_bytes + $10,
           ` + initialRoundEvents.advance() + `,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM audit_gate WHERE audit.audit_id = audit_gate.audit_id
    RETURNING audit.*, ` + initialRoundEvents.appended() + `
` + initialRoundRowsSQL + `
)
SELECT ` + prefixedAuditColumns("started") + ` FROM started`
