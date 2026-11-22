package pgstore

// Admission is bounded by one existing invocation's receipt prefix. Stop at
// the first over-bound row instead of hydrating other execution history.
const evidenceCollectionSizeSQL = `
SELECT count(*)::int,coalesce(sum(size_bytes+coalesce((source_provenance->>'sourceBytes')::bigint,0)),0)::bigint,
 coalesce(sum((source_provenance->>'sourceCount')::int),0)::int,
 (SELECT count(*)::int FROM audit_execution_items WHERE audit_id=$1 AND execution_id=$6)
FROM (SELECT size_bytes,source_provenance FROM audit_artifact_links
 WHERE audit_id=$1 AND ((logical_key >= $2 AND logical_key < $3)
 OR (logical_key >= $4 AND logical_key < $5))
 ORDER BY logical_key LIMIT 257) AS bounded`

// Same link/byte owner as ordinary Task and hypothesis collection; the
// evidence event also invalidates readers while the attempt is still live.
var insertEvidenceLinksSQL = recordedLinksSQL("'evidence.recorded'")
var insertJudgmentLinksSQL = recordedLinksSQL("'result.recorded'")

func recordedLinksSQL(kind string) string {
	events := auditEvents(auditEvent(kind, "audit.audit_id", "audit.revision", "jsonb_build_object('retainedBytes',$3::bigint)"))
	return `
WITH input AS (
 SELECT * FROM jsonb_to_recordset($2::jsonb) AS link(
 artifact_scope text,logical_key text,artifact_ref jsonb,artifact_digest text,
 media_type text,size_bytes bigint,source_provenance jsonb,display_ref text)
), inserted AS (
 INSERT INTO audit_artifact_links(audit_id,artifact_scope,logical_key,artifact_ref,
 artifact_digest,media_type,size_bytes,source_provenance,display_ref)
 SELECT $1,artifact_scope,logical_key,artifact_ref,artifact_digest,media_type,
 size_bytes,source_provenance,display_ref FROM input
 ON CONFLICT (audit_id,logical_key) DO NOTHING
), advanced AS (
 UPDATE audits AS audit SET retained_evidence_bytes=audit.retained_evidence_bytes+$3,
 ` + events.advance() + `,updated_at=GREATEST(clock_timestamp(),audit.updated_at+interval '1 microsecond')
 WHERE audit.audit_id=$1 RETURNING ` + events.appended() + `
) SELECT last_event_sequence FROM advanced`
}

// The execution is bounded to 256 accepted annotations, independent of the
// total Audit history. The index covers this ordered logical-key range.
const listExecutionEvidenceSQL = `
SELECT artifact_scope,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance
FROM audit_artifact_links WHERE audit_id=$1 AND logical_key >= $2 AND logical_key < $3
ORDER BY logical_key LIMIT 257`
