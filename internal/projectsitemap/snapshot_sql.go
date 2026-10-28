package projectsitemap

// snapshotInputRevisionsSQL resolves the selected original Run input forks,
// not their Worker-editable current bindings. The snapshot admits at most eight
// names; each indexed lateral lookup returns at most two rows so inconsistent
// authority is detected without scanning the containing artifact collection.
var snapshotInputRevisionsSQL = `
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
