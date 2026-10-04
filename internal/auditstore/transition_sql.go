package auditstore

// SQL statements for transition.go.

// transitionClaimedSQL moves the Audit from state $5 at revision $4 to $6
// under the live controller claim ($1-$3), requiring an active Project when the
// target is active. It sets stop reason $7/$8, closes dispatch for closing and
// terminal targets, stamps pause, finish and deletion times and appends
// audit.state_changed; the caller completes the trailing SELECT.
// Used by PostgresStore.TransitionClaimed.
var transitionClaimedSQL = `
	WITH live_claim AS MATERIALIZED (
	    SELECT claim.audit_id
	      FROM audit_controller_claims AS claim
	     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
	       AND claim.expires_at > clock_timestamp()
	     FOR UPDATE OF claim
	), claim_gate AS MATERIALIZED (
	    SELECT audit.audit_id,
	           CASE WHEN $6 = 'active'
	                THEN contractor_require_active_audit_project(audit.project_id, audit.owner_id)
	           END
	      FROM audits AS audit
	      JOIN live_claim USING (audit_id)
	     WHERE audit.audit_id = $1
	     FOR UPDATE OF audit
	), changed AS (
    UPDATE audits AS audit
       SET state = $6,
           paused_at = CASE WHEN $6 = 'paused' THEN clock_timestamp() ELSE NULL END,
           revision = audit.revision + 1,
           dispatch_state = CASE
               WHEN $6 IN ('finalizing', 'cancelling', 'completed', 'cancelled', 'failed', 'deleting')
                   THEN 'closed'
               ELSE audit.dispatch_state
           END,
           stop_reason_code = $7,
           stop_reason_message = $8,
           deletion_requested_at = CASE WHEN $6 = 'deleting'
               THEN COALESCE(audit.deletion_requested_at, clock_timestamp())
               ELSE audit.deletion_requested_at
           END,
           finished_at = CASE WHEN $6 IN ('completed', 'cancelled', 'failed')
                              THEN clock_timestamp() ELSE NULL END,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond'),
           next_event_sequence = audit.next_event_sequence + 1
      FROM claim_gate
     WHERE audit.audit_id = claim_gate.audit_id
       AND audit.revision = $4 AND audit.state = $5
    RETURNING audit.*
), event_row AS (
    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    )
    SELECT audit_id, next_event_sequence - 1, 'audit.state_changed', audit_id, revision,
           jsonb_build_object('from', $5::text, 'to', $6::text)
      FROM changed
)
SELECT `

// resumeSQL reactivates owner $1's paused Audit $2 at revision $3 behind the
// active-Project gate: dispatch reopens with a fresh hold and deadline $4, the
// stop reason is cleared, the audit.transition idempotency record ($6-$8) is
// stored and audit.resumed carries the previous stop reason and deadline JSON
// $5. The caller completes the trailing SELECT. Used by PostgresStore.Resume.
var resumeSQL = `
WITH previous AS MATERIALIZED (
    SELECT audit.audit_id, audit.stop_reason_code, audit.stop_reason_message,
           contractor_require_active_audit_project(audit.project_id, audit.owner_id) AS project_gate
      FROM audits AS audit
     WHERE audit.owner_id = $1 AND audit.audit_id = $2
     FOR UPDATE OF audit
), changed AS (
    UPDATE audits AS audit
       SET state = 'active', dispatch_state = 'open', hold_state = 'held',
           deadline_at = $4, paused_at = NULL, finished_at = NULL,
           stop_reason_code = NULL, stop_reason_message = NULL,
           revision = audit.revision + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond'),
           next_event_sequence = audit.next_event_sequence + 1
      FROM previous
     WHERE audit.audit_id = previous.audit_id
       AND audit.revision = $3 AND audit.state = 'paused'
    RETURNING audit.*, previous.stop_reason_code AS previous_code,
              previous.stop_reason_message AS previous_message
), idempotency_row AS (
    INSERT INTO audit_idempotency (
        owner_id, operation, idempotency_key, request_digest,
        audit_id, resource_id, response_snapshot
    )
    SELECT $1, 'audit.transition', $6, $7, audit_id, audit_id, $8::jsonb
      FROM changed
), event_row AS (
    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    )
    SELECT audit_id, next_event_sequence - 1, 'audit.resumed', audit_id, revision,
           jsonb_build_object(
               'from', 'paused', 'to', 'active',
               'previousStopReason', CASE WHEN previous_code IS NULL THEN 'null'::jsonb
                   ELSE jsonb_build_object('Code', previous_code,
                                           'Message', COALESCE(previous_message, ''))
               END,
               'deadlineAt', $5::jsonb
           )
      FROM changed
)
SELECT `

// transitionTrustedSQL is the claim-free form of transitionClaimedSQL for
// service-authorized changes: it locks Audit $1 and moves it from state $3 at
// revision $2 to $4 with stop reason $5/$6, applying the same active-Project
// gate, dispatch closing, timestamps and audit.state_changed event. The caller
// completes the trailing SELECT. Used by PostgresStore.TransitionTrusted.
var transitionTrustedSQL = `
WITH active_project_gate AS MATERIALIZED (
    SELECT audit.audit_id,
           CASE WHEN $4 = 'active'
                THEN contractor_require_active_audit_project(audit.project_id, audit.owner_id)
           END
      FROM audits AS audit
     WHERE audit.audit_id = $1
     FOR UPDATE OF audit
), changed AS (
    UPDATE audits AS audit
       SET state = $4,
           paused_at = CASE WHEN $4 = 'paused' THEN clock_timestamp() ELSE NULL END,
           revision = audit.revision + 1,
           dispatch_state = CASE
               WHEN $4 IN ('finalizing', 'cancelling', 'completed', 'cancelled', 'failed', 'deleting')
                   THEN 'closed'
               ELSE audit.dispatch_state
           END,
           stop_reason_code = $5,
           stop_reason_message = $6,
           deletion_requested_at = CASE WHEN $4 = 'deleting'
               THEN COALESCE(audit.deletion_requested_at, clock_timestamp())
               ELSE audit.deletion_requested_at
           END,
           finished_at = CASE WHEN $4 IN ('completed', 'cancelled', 'failed')
                              THEN clock_timestamp() ELSE NULL END,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond'),
           next_event_sequence = audit.next_event_sequence + 1
      FROM active_project_gate
     WHERE audit.audit_id = active_project_gate.audit_id
       AND audit.revision = $2 AND audit.state = $3
    RETURNING audit.*
), event_row AS (
    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    )
    SELECT audit_id, next_event_sequence - 1, 'audit.state_changed', audit_id, revision,
           jsonb_build_object('from', $3::text, 'to', $4::text)
      FROM changed
)
SELECT `
