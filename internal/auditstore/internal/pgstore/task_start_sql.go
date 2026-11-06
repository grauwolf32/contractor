package pgstore

// Activation uses the ordinary Audit event allocator and Artifact link owner.
var commonStartedEvents = auditEvents(auditEvent("'audit.started'", "audit.audit_id", "audit.revision", "'{}'::jsonb"))

var activateAuditSQL = `
WITH project_gate AS MATERIALIZED (
 SELECT contractor_require_active_audit_project(project_id,owner_id)
 FROM audits WHERE audit_id=$2 AND owner_id=$1
), started AS (
 UPDATE audits AS audit
 SET baseline_snapshot=$4::jsonb,state='active',hold_state='held',
     dispatch_state='open',deadline_at=$5,retained_evidence_bytes=$7,
     started_at=clock_timestamp(),updated_at=clock_timestamp(),` + commonStartedEvents.advance() + `
 FROM project_gate
 WHERE audit.owner_id=$1 AND audit.audit_id=$2 AND audit.revision=$3 AND audit.state='draft'
   AND $7<=audit.max_evidence_bytes
 RETURNING audit.*,` + commonStartedEvents.appended() + `
), link_input AS MATERIALIZED (
 SELECT * FROM jsonb_to_recordset($6::jsonb) AS link(
 logical_key text,artifact_ref jsonb,artifact_digest text,media_type text,size_bytes bigint,
 source_provenance jsonb,display_ref text)
), inserted_links AS (
 INSERT INTO audit_artifact_links(audit_id,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance,display_ref)
 SELECT started.audit_id,link.logical_key,link.artifact_ref,link.artifact_digest,link.media_type,
        link.size_bytes,link.source_provenance,link.display_ref
 FROM started CROSS JOIN link_input AS link
)
SELECT ` + prefixedAuditColumns("started") + ` FROM started`
