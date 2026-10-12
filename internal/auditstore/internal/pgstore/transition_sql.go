package pgstore

// SQL statements for transition.go.

// transitionEvents record an owner transition and expire a pending report
// review after it.
var transitionEvents = auditEvents(
	auditEvent("'audit.state_changed'", "audit.audit_id", "audit.revision",
		"jsonb_build_object('from', $4::text, 'to', $5::text)"),
	reportReviewExpiryEvents,
)

// transitionSQL applies owner $1's pause or cancellation of Audit $2 from
// state $4 at revision $3 to $5 with stop reason $6/$7; cancelling closes
// dispatch. Leaving waiting_review expires the pending report review in the
// same revision step. It records audit.state_changed before the
// review.expired events. The caller completes the trailing SELECT and claims
// the request's receipt. Used by OwnerStore.Transition.
var transitionSQL = `
WITH target AS MATERIALIZED (
    SELECT audit.audit_id, audit.state
      FROM audits AS audit
     WHERE audit.audit_id = $2 AND audit.owner_id = $1
     FOR UPDATE OF audit
), ` + reportReviewSelectionCTE("target") + `, changed AS (
    UPDATE audits AS audit
       SET state = $5,
           paused_at = CASE WHEN $5 = 'paused' THEN clock_timestamp() ELSE NULL END,
           dispatch_state = CASE WHEN $5 = 'cancelling' THEN 'closed' ELSE dispatch_state END,
           stop_reason_code = $6,
           stop_reason_message = $7,
           ` + transitionEvents.advance() + `,
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
      FROM target
     WHERE audit.audit_id = target.audit_id
       AND audit.revision = $3 AND audit.state = $4
    RETURNING audit.*, ` + transitionEvents.appended() + `
), ` + reportReviewClosureCTE + `
SELECT `

// claimedTransitionEvents record a controller transition.
var claimedTransitionEvents = auditEvents(auditEvent(
	"'audit.state_changed'", "audit.audit_id", "audit.revision",
	"jsonb_build_object('from', $5::text, 'to', $6::text)",
))

// transitionClaimedSQL moves the Audit from state $5 at revision $4 to $6
// under the live controller claim ($1-$3), requiring an active Project when the
// target is active. It sets stop reason $7/$8, closes dispatch for closing and
// terminal targets, stamps pause, finish and deletion times and records
// audit.state_changed; the caller completes the trailing SELECT.
// Used by ClaimStore.TransitionClaimed.
var transitionClaimedSQL = `
	WITH live_claim AS MATERIALIZED (
	    ` + auditClaims.LiveClaimSQL("$1", "$2", "$3") + `
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
           ` + claimedTransitionEvents.advance() + `,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM claim_gate
     WHERE audit.audit_id = claim_gate.audit_id
       AND audit.revision = $4 AND audit.state = $5
    RETURNING audit.*, ` + claimedTransitionEvents.appended() + `
)
SELECT `

// resumeEvents record a resumed Audit with the stop reason it cleared and its
// deadline JSON $5.
var resumeEvents = auditEvents(auditEvent(
	"'audit.resumed'", "audit.audit_id", "audit.revision",
	`jsonb_build_object(
               'from', 'paused', 'to', 'active',
               'previousStopReason', CASE WHEN previous.stop_reason_code IS NULL THEN 'null'::jsonb
                   ELSE jsonb_build_object('Code', previous.stop_reason_code,
                                           'Message', COALESCE(previous.stop_reason_message, ''))
               END,
               'deadlineAt', $5::jsonb
           )`,
))

// resumeSQL reactivates owner $1's paused Audit $2 at revision $3 behind the
// active-Project gate: dispatch reopens with a fresh hold and deadline $4, the
// stop reason is cleared and audit.resumed carries the previous stop reason
// and deadline JSON $5. The caller completes the trailing SELECT and claims
// the request's receipt. Used by OwnerStore.Resume.
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
           ` + resumeEvents.advance() + `,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM previous
     WHERE audit.audit_id = previous.audit_id
       AND audit.revision = $3 AND audit.state = 'paused'
    RETURNING audit.*, ` + resumeEvents.appended() + `
)
SELECT `

// trustedTransitionEvents record a service-authorized transition.
var trustedTransitionEvents = auditEvents(auditEvent(
	"'audit.state_changed'", "audit.audit_id", "audit.revision",
	"jsonb_build_object('from', $3::text, 'to', $4::text)",
))

// transitionTrustedSQL is the claim-free form of transitionClaimedSQL for
// service-authorized changes: it locks Audit $1 and moves it from state $3 at
// revision $2 to $4 with stop reason $5/$6, applying the same active-Project
// gate, dispatch closing, timestamps and audit.state_changed event. The caller
// completes the trailing SELECT. Used by ReviewStore.transitionTrusted.
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
           ` + trustedTransitionEvents.advance() + `,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM active_project_gate
     WHERE audit.audit_id = active_project_gate.audit_id
       AND audit.revision = $2 AND audit.state = $3
    RETURNING audit.*, ` + trustedTransitionEvents.appended() + `
)
SELECT `
