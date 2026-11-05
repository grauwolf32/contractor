package pgstore

// Links and cumulative evidence usage commit with the accepted hypothesis.
const insertHypothesisSourcesSQL = `
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
)
UPDATE audits SET retained_evidence_bytes=retained_evidence_bytes+$3 WHERE audit_id=$1`
