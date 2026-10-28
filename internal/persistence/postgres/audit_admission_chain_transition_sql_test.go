package postgres

// Compare stored business metadata and literal JSON text as well as immutable
// payloads. A successful preflight may append its ledger entry, nothing else.
const admissionChainTransitionDataSQL = `
SELECT jsonb_build_array(
 (SELECT jsonb_agg(to_jsonb(project) ORDER BY project_id) FROM projects AS project),
 (SELECT jsonb_agg(to_jsonb(audit) ORDER BY audit_id) FROM audits AS audit),
 (SELECT jsonb_agg(jsonb_build_array(to_jsonb(round),admission_checkpoint::text) ORDER BY round_id) FROM audit_rounds AS round),
 (SELECT jsonb_agg(jsonb_build_array(to_jsonb(item),prepared_execution::text,retained_dependencies::text) ORDER BY item_id) FROM audit_items AS item),
 (SELECT jsonb_agg(to_jsonb(link) ORDER BY audit_id,logical_key) FROM audit_artifact_links AS link),
 (SELECT jsonb_agg(to_jsonb(blob) ORDER BY sha256) FROM artifact_blobs AS blob),
 (SELECT jsonb_agg(to_jsonb(version) ORDER BY version_id) FROM artifact_versions AS version),
 (SELECT jsonb_agg(to_jsonb(binding) ORDER BY scope_kind,scope_id,namespace,name) FROM artifact_bindings AS binding),
 (SELECT jsonb_agg(to_jsonb(revision) ORDER BY scope_kind,scope_id,namespace,name,revision) FROM artifact_binding_revisions AS revision)
)::text`
