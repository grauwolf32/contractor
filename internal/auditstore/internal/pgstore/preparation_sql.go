package pgstore

// preparationStartedEvents record the start of preparation.
var preparationStartedEvents = auditEvents(auditEvent(
	"'audit.preparation_started'", "audit.audit_id", "audit.revision", "'{}'::jsonb",
))

// startPreparationSQL pins the baseline, holds and links of owner $1's draft
// Audit $2 at revision $3 and enters preparation without a Round. The caller
// claims the request's receipt. Used by OwnerStore.StartPreparation.
var startPreparationSQL = `
WITH project_gate AS MATERIALIZED (
    SELECT contractor_require_active_audit_project(project_id, owner_id)
      FROM audits WHERE audit_id = $2 AND owner_id = $1
), started AS (
    UPDATE audits AS audit
       SET baseline_snapshot = $4::jsonb, phase = 'preparing', state = 'active',
           hold_state = 'held', deadline_at = $5, retained_evidence_bytes = $7,
           started_at = clock_timestamp(),
           ` + preparationStartedEvents.advance() + `,
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
      FROM project_gate
     WHERE audit.owner_id = $1 AND audit.audit_id = $2 AND audit.revision = $3
       AND audit.state = 'draft' AND audit.phase = 'not-started' AND audit.current_round_id IS NULL
       AND $7 <= audit.max_evidence_bytes
       AND EXISTS (SELECT 1 FROM jsonb_each(` + profileWorkflowsSQL + `) AS role WHERE ` + prepareRoleSQL + `)
    RETURNING audit.*, ` + preparationStartedEvents.appended() + `
), link_input AS MATERIALIZED (
    SELECT * FROM jsonb_to_recordset($6::jsonb) AS link(
        logical_key text, artifact_ref jsonb, artifact_digest text,
        media_type text, size_bytes bigint, source_provenance jsonb, display_ref text
    )
), inserted_links AS (
    INSERT INTO audit_artifact_links (audit_id, logical_key, artifact_ref, artifact_digest, media_type, size_bytes, source_provenance, display_ref)
    SELECT started.audit_id, link.logical_key, link.artifact_ref, link.artifact_digest, link.media_type, link.size_bytes, link.source_provenance, link.display_ref
      FROM started CROSS JOIN link_input AS link
)
SELECT ` + prefixedAuditColumns("started") + ` FROM started`

// preparationCompletedEvents record the end of preparation.
var preparationCompletedEvents = auditEvents(auditEvent(
	"'audit.preparation_completed'", "audit.audit_id", "audit.revision", "'{}'::jsonb",
))

// completePreparationSQL uses a live claim and revision to leave preparation
// only after every pinned prepare role has an immutable accepted receipt.
var completePreparationSQL = `
WITH live_claim AS MATERIALIZED (
    ` + auditClaims.LiveClaimSQL("$1", "$2", "$3") + `
), project_gate AS MATERIALIZED (
    SELECT contractor_require_active_audit_project(project_id, owner_id)
      FROM audits JOIN live_claim USING (audit_id)
), changed AS (
    UPDATE audits AS audit
       SET phase = 'inventory', ` + preparationCompletedEvents.advance() + `,
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
      FROM live_claim, project_gate
     WHERE audit.audit_id = live_claim.audit_id AND audit.revision = $4
       AND audit.phase = 'preparing' AND audit.state = 'active' AND audit.dispatch_state = 'open'
       AND (audit.deadline_at IS NULL OR audit.deadline_at > clock_timestamp())
       AND NOT EXISTS (
           SELECT 1 FROM jsonb_each(` + profileWorkflowsSQL + `) AS role
            WHERE ` + prepareRoleSQL + ` AND NOT EXISTS (
                SELECT 1 FROM audit_executions AS execution
                 WHERE execution.audit_id = audit.audit_id AND execution.workflow_role = role.key
                   AND execution.role = 'prepare' AND execution.collection_disposition = 'accepted-result'
            )
       )
    RETURNING audit.*, ` + preparationCompletedEvents.appended() + `
)
SELECT ` + prefixedAuditColumns("changed") + ` FROM changed`

// preparationStatusSQL derives the public status of the jsonb_each row role
// (ASC-R42) from the attempts and latest rows that preparationAttemptsSQL
// joins. Both preparation reads use it, so the status has one definition.
const preparationStatusSQL = `CASE WHEN latest.collection_disposition = 'accepted-result' THEN 'accepted'
            WHEN latest.execution_id IS NULL THEN 'pending'
            WHEN latest.state = 'collected' AND (
                attempts.count >= ` + roleMaxRunAttemptsSQL + `
                OR latest.collection_disposition IN ('collection-contract-invalid', 'execution-cancelled')
                OR latest.collection_error_code = 'evidence-budget-exhausted'
            ) THEN 'failed'
            ELSE 'running' END`

// preparationAttemptsSQL joins to the jsonb_each row role of audit the number
// of its prepare attempts (attempts) and its latest attempt (latest) with the
// columns latestColumns, which include those preparationStatusSQL reads.
func preparationAttemptsSQL(latestColumns string) string {
	return `
  LEFT JOIN LATERAL (
      SELECT count(*)::integer FROM audit_executions AS execution
       WHERE execution.audit_id = audit.audit_id AND execution.role = 'prepare' AND execution.workflow_role = role.key
  ) AS attempts ON true
  LEFT JOIN LATERAL (
      SELECT ` + latestColumns + ` FROM audit_executions AS execution
       WHERE execution.audit_id = audit.audit_id AND execution.role = 'prepare' AND execution.workflow_role = role.key
       ORDER BY role_attempt DESC LIMIT 1
  ) AS latest ON true`
}

// listPreparationRolesSQL derives bounded role status independently of Round history.
var listPreparationRolesSQL = `
SELECT role.key, COALESCE(attempts.count, 0), ` + roleMaxRunAttemptsSQL + `,
       latest.execution_id,
       ` + preparationStatusSQL + `
  FROM audits AS audit CROSS JOIN LATERAL jsonb_each(` + profileWorkflowsSQL + `) AS role` +
	preparationAttemptsSQL("execution_id, state, collection_disposition, collection_error_code") + `
 WHERE audit.audit_id = $1 AND ` + prepareRoleSQL + ` ORDER BY role.key`

// listPreparationPageSQL reads, in one statement and so in one snapshot, the
// preparation of owner $1's Audits $2 that a list page read at revisions $3
// (ASC-R150). Every listed Audit yields at least one row. Its second column
// is false when the Audit is gone, belongs to another owner or has another
// revision; the role columns are NULL when it pins no prepare role. A role
// row carries the attempt count and only the latest attempt, whose
// preparation_outputs is non-empty only once that attempt is accepted.
// Used by QueryStore.ListPreparationPage.
var listPreparationPageSQL = `
WITH listed AS MATERIALIZED (
    SELECT listed.audit_id, listed.revision, listed.ordinal
      FROM unnest($2::text[], $3::bigint[]) WITH ORDINALITY AS listed(audit_id, revision, ordinal)
)
SELECT listed.audit_id, audit.audit_id IS NOT NULL, role.key, COALESCE(attempts.count, 0),
       ` + roleMaxRunAttemptsSQL + `, latest.execution_id, latest.role_attempt, latest.run_id,
       latest.state, latest.preparation_outputs,
       ` + preparationStatusSQL + `
  FROM listed
  LEFT JOIN audits AS audit
    ON audit.audit_id = listed.audit_id AND audit.audit_id = ANY ($2::text[])
   AND audit.owner_id = $1 AND audit.revision = listed.revision
  LEFT JOIN LATERAL jsonb_each(` + profileWorkflowsSQL + `) AS role ON ` + prepareRoleSQL +
	preparationAttemptsSQL("execution_id, role_attempt, run_id, state, collection_disposition, collection_error_code, preparation_outputs") + `
 ORDER BY listed.ordinal, role.key`
