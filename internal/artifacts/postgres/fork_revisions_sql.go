package postgres

// forkTargetRevisionsSQL resolves at most MaxExactReadBatchSize selected
// bindings' target revisions, independently of the containing artifact count.
// Each indexed lateral lookup returns at most two rows so ambiguous matching
// lineage is rejected. Used by Repository.DescribeForkRevisions.
var forkTargetRevisionsSQL = `
SELECT selected.name, original.target_revision
FROM unnest($4::text[]) WITH ORDINALITY AS selected(name, ordinal)
CROSS JOIN LATERAL (
 SELECT target_revision FROM artifact_lineage
 WHERE target_scope_kind=$1 AND target_scope_id=$2
  AND target_namespace=$3 AND target_name=selected.name
  AND lineage_kind=$5
 ORDER BY target_revision
 LIMIT 2
) AS original
ORDER BY selected.ordinal, original.target_revision`

// forkSourceRevisionsSQL reads at most two matching fork edges per selected
// binding so missing or ambiguous provenance is detectable. Absent bindings
// are omitted; current_revision is never used to choose an original fork.
// Used by Repository.DescribeForkSources.
var forkSourceRevisionsSQL = `
SELECT binding.name, original.target_revision, original.source_scope_kind,
 original.source_scope_id, original.source_namespace, original.source_name,
 original.source_revision, original.created_at
FROM unnest($4::text[]) WITH ORDINALITY AS selected(name, ordinal)
JOIN artifact_bindings AS binding ON binding.scope_kind=$1 AND binding.scope_id=$2
 AND binding.namespace=$3 AND binding.name=selected.name
LEFT JOIN LATERAL (
 SELECT target_revision, source_scope_kind, source_scope_id, source_namespace,
  source_name, source_revision, created_at
 FROM artifact_lineage
 WHERE target_scope_kind=$1 AND target_scope_id=$2
  AND target_namespace=$3 AND target_name=selected.name AND lineage_kind=$5
 ORDER BY target_revision
 LIMIT 2
) AS original ON true
ORDER BY selected.ordinal, original.target_revision`
