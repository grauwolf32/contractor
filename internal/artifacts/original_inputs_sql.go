package artifacts

// originalInputRevisionsSQL resolves at most MaxExactReadBatchSize selected
// original Run input forks, independently of the containing artifact count.
// Each indexed lateral lookup returns at most two rows so ambiguous original
// lineage is rejected. Used by PostgresRepository.DescribeOriginalInputs.
var originalInputRevisionsSQL = `
SELECT selected.name, original.target_revision
FROM unnest($2::text[]) WITH ORDINALITY AS selected(name, ordinal)
CROSS JOIN LATERAL (
 SELECT target_revision FROM artifact_lineage
 WHERE target_scope_kind='run' AND target_scope_id=$1
  AND target_namespace='inputs' AND target_name=selected.name
  AND lineage_kind='input_fork'
 ORDER BY target_revision
 LIMIT 2
) AS original
ORDER BY selected.ordinal, original.target_revision`
