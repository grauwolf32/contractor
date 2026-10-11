package auditstore

// SQL statements for store.go.

// auditCreatedEvents are the first events of a draft Audit.
var auditCreatedEvents = auditEvents(auditEvent(
	"'audit.created'", "audit.audit_id", "audit.revision", "jsonb_build_object('state', audit.state)",
))

var auditCreatedRevision, auditCreatedNextEventSequence = auditCreatedEvents.created()

// createDraftSQL inserts draft Audit $1 of owner $2 in Project $3 (raising if
// the Project is missing or deleting) with its empty controller claim row, the
// audit.create idempotency record ($16-$18) and the audit.created event.
// ON CONFLICT DO NOTHING leaves replay to the caller, which completes the
// trailing SELECT with the inserted Audit. Used by PostgresStore.CreateDraft.
var createDraftSQL = `
WITH project_gate AS MATERIALIZED (
    SELECT contractor_require_active_audit_project($3, $2)
), inserted AS (
    INSERT INTO audits AS audit (
        audit_id, owner_id, project_id,
        profile_name, profile_version, profile_digest, profile_snapshot, input_selection,
        max_rounds, batch_size, max_items_per_round, max_items_total,
        max_submitted_runs, max_item_run_attempts, max_evidence_bytes,
        revision, next_event_sequence
    )
    SELECT $1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb,
           $9, $10, $11, $12, $13, $14, $15,
           ` + auditCreatedRevision + `, ` + auditCreatedNextEventSequence + `
      FROM project_gate
    ON CONFLICT DO NOTHING
    RETURNING audit.*, ` + auditCreatedEvents.appended() + `
), claim_row AS (
    INSERT INTO audit_controller_claims (audit_id)
    SELECT audit_id FROM inserted
), idempotency_row AS (
    INSERT INTO audit_idempotency (
        owner_id, operation, idempotency_key, request_digest,
        audit_id, resource_id, response_snapshot
    )
    SELECT $2, 'audit.create', $16, $17, audit_id, audit_id, $18::jsonb
      FROM inserted
)
SELECT `
