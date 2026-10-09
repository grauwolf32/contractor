package auditstore

// startPreparationSQL pins the baseline, holds, links and start replay without a Round.
var startPreparationSQL = `
WITH project_gate AS MATERIALIZED (
    SELECT contractor_require_active_audit_project(project_id, owner_id)
      FROM audits WHERE audit_id = $2 AND owner_id = $1
), started AS (
    UPDATE audits AS audit
       SET baseline_snapshot = $4::jsonb, phase = 'preparing', state = 'active',
           hold_state = 'held', deadline_at = $5, retained_evidence_bytes = $7,
           started_at = clock_timestamp(), revision = revision + 1,
           next_event_sequence = next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
      FROM project_gate
     WHERE audit.owner_id = $1 AND audit.audit_id = $2 AND audit.revision = $3
       AND audit.state = 'draft' AND audit.phase = 'not-started' AND audit.current_round_id IS NULL
       AND $7 <= audit.max_evidence_bytes
       AND EXISTS (SELECT 1 FROM jsonb_each(audit.profile_snapshot->'workflows') AS role WHERE role.value->>'kind' = 'prepare')
    RETURNING audit.*
), link_input AS MATERIALIZED (
    SELECT * FROM jsonb_to_recordset($6::jsonb) AS link(
        logical_key text, artifact_ref jsonb, artifact_digest text,
        media_type text, size_bytes bigint, source_provenance jsonb, display_ref text
    )
), inserted_links AS (
    INSERT INTO audit_artifact_links (audit_id, logical_key, artifact_ref, artifact_digest, media_type, size_bytes, source_provenance, display_ref)
    SELECT started.audit_id, link.logical_key, link.artifact_ref, link.artifact_digest, link.media_type, link.size_bytes, link.source_provenance, link.display_ref
      FROM started CROSS JOIN link_input AS link
), idempotency_row AS (
    INSERT INTO audit_idempotency (owner_id, operation, idempotency_key, request_digest, audit_id, resource_id, response_snapshot)
    SELECT $1, 'audit.start', $8, $9, audit_id, audit_id, jsonb_build_object('auditId', audit_id) FROM started
), event_row AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, entity_revision, summary)
    SELECT audit_id, next_event_sequence - 1, 'audit.preparation_started', audit_id, revision, '{}'::jsonb FROM started
)
SELECT ` + prefixedAuditColumns("started") + ` FROM started`

// completePreparationSQL uses a live claim and revision to leave preparation
// only after every pinned prepare role has an immutable accepted receipt.
var completePreparationSQL = `
WITH live_claim AS MATERIALIZED (
    SELECT audit_id FROM audit_controller_claims
     WHERE audit_id = $1 AND holder_id = $2 AND epoch = $3 AND expires_at > clock_timestamp()
     FOR UPDATE
), project_gate AS MATERIALIZED (
    SELECT contractor_require_active_audit_project(project_id, owner_id)
      FROM audits JOIN live_claim USING (audit_id)
), changed AS (
    UPDATE audits AS audit
       SET phase = 'inventory', revision = revision + 1, next_event_sequence = next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
      FROM live_claim, project_gate
     WHERE audit.audit_id = live_claim.audit_id AND audit.revision = $4
       AND audit.phase = 'preparing' AND audit.state = 'active' AND audit.dispatch_state = 'open'
       AND (audit.deadline_at IS NULL OR audit.deadline_at > clock_timestamp())
       AND NOT EXISTS (
           SELECT 1 FROM jsonb_each(audit.profile_snapshot->'workflows') AS role
            WHERE role.value->>'kind' = 'prepare' AND NOT EXISTS (
                SELECT 1 FROM audit_executions AS execution
                 WHERE execution.audit_id = audit.audit_id AND execution.workflow_role = role.key
                   AND execution.role = 'prepare' AND execution.collection_disposition = 'accepted-result'
            )
       )
    RETURNING audit.*
), event_row AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, entity_revision, summary)
    SELECT audit_id, next_event_sequence - 1, 'audit.preparation_completed', audit_id, revision, '{}'::jsonb FROM changed
)
SELECT ` + prefixedAuditColumns("changed") + ` FROM changed`

// listPreparationRolesSQL derives bounded role status independently of Round history.
const listPreparationRolesSQL = `
SELECT role.key, COALESCE(attempts.count, 0), (role.value->>'maxRunAttempts')::integer,
       latest.execution_id,
       CASE WHEN latest.collection_disposition = 'accepted-result' THEN 'accepted'
            WHEN latest.execution_id IS NULL THEN 'pending'
            WHEN latest.state = 'collected' AND attempts.count >= (role.value->>'maxRunAttempts')::integer THEN 'failed'
            ELSE 'running' END
  FROM audits AS audit CROSS JOIN LATERAL jsonb_each(audit.profile_snapshot->'workflows') AS role
  LEFT JOIN LATERAL (
      SELECT count(*)::integer FROM audit_executions AS execution
       WHERE execution.audit_id = audit.audit_id AND execution.role = 'prepare' AND execution.workflow_role = role.key
  ) AS attempts ON true
  LEFT JOIN LATERAL (
      SELECT execution_id, state, collection_disposition FROM audit_executions AS execution
       WHERE execution.audit_id = audit.audit_id AND execution.role = 'prepare' AND execution.workflow_role = role.key
       ORDER BY role_attempt DESC LIMIT 1
  ) AS latest ON true
 WHERE audit.audit_id = $1 AND role.value->>'kind' = 'prepare' ORDER BY role.key`
