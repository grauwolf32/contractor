package artifacts

// forkTargetRevisionsSQL resolves at most MaxExactReadBatchSize selected
// bindings' target revisions, independently of the containing artifact count.
// Each indexed lateral lookup returns at most two rows so ambiguous matching
// lineage is rejected. Used by PostgresRepository.DescribeForkRevisions.
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
