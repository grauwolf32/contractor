package pgstore

import (
	"github.com/grauwolf32/contractor/internal/projectstore"
)

// SQL statements for store.go.

// heldAuditIDsByCredentialSQL lists, oldest first, at most $2 held Audits
// whose baseline names credential $1 in its credential list member, one of
// runtimeconfig.LLMCredentialIDsMember and RuntimeCredentialIDsMember: an
// Audit baseline repeats the RunSnapshot credential members. The expression
// and predicate match audits_held_llm_credentials_idx and
// audits_held_runtime_credentials_idx, which hold the member of held Audits
// only, so the check reads the held Audits naming the credential rather than
// every Audit and every held baseline (V407-025). Used by
// CredentialHolds.listHeldAuditIDsByCredential.
func heldAuditIDsByCredentialSQL(member string) string {
	return `
SELECT audit_id
  FROM audits
 WHERE hold_state = 'held'
   AND (baseline_snapshot->'` + member + `') ? $1
 ORDER BY created_at, audit_id
 LIMIT $2`
}

// auditCreatedEvents are the first events of a draft Audit.
var auditCreatedEvents = auditEvents(auditEvent(
	"'audit.created'", "audit.audit_id", "audit.revision", "jsonb_build_object('state', audit.state)",
))

var auditCreatedRevision, auditCreatedNextEventSequence = auditCreatedEvents.created()

// createDraftSQL inserts draft Audit $1 of owner $2 in Project $3 (raising if
// the Project is missing or deleting) with its empty controller claim row and
// the audit.created event. The Project's first Audit also creates the
// Project's finding correlation fence row (FND-R177). It inserts only a row
// its snapshot does not show, so it never waits on a finding writer that holds
// the existing row; only two first Audits of one Project wait on each other's
// insert. An Audit ID that exists already inserts nothing; the caller
// completes the trailing SELECT with the inserted Audit and claims the
// request's receipt. Used by OwnerStore.CreateDraft.
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
), correlation_fence AS (
    INSERT INTO finding_correlation_fences (project_id)
    SELECT project_id FROM inserted
     WHERE NOT EXISTS (SELECT 1 FROM finding_correlation_fences WHERE project_id = $3)
    ON CONFLICT DO NOTHING
)
SELECT `

// listAuditsSQL pages owner $1's Audits newest first before the
// (created_at, audit_id) cursor $6/$7, limited to $8 rows, optionally
// filtered by Project $2, states $3 and profile $4 at version $5. With $9 it
// lists only Audits of Projects listed to the owner, as every owner-wide list
// does. Used by QueryStore.List.
var listAuditsSQL = `
SELECT ` + prefixedAuditColumns("audits") + `
  FROM audits
 WHERE owner_id = $1
   AND ($2::text IS NULL OR project_id = $2)
   AND ($3::text[] IS NULL OR state = ANY($3))
   AND ($4::text IS NULL OR (profile_name = $4 AND profile_version = $5))
   AND ($6::timestamptz IS NULL OR (created_at, audit_id) < ($6, $7))
   AND (NOT $9::boolean OR ` + projectstore.OwnerListedProjectSQL("audits.project_id", "audits.owner_id") + `)
 ORDER BY created_at DESC, audit_id DESC
 LIMIT $8`
