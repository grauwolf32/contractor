package projectsitemap

// allocationInputRevisionSQL resolves a named Run creation input, not its
// Worker-editable current binding. At most two rows prove the one original
// fork or reject inconsistent stored authority. Used by AllocationAccess.
var allocationInputRevisionSQL = `
SELECT target_revision FROM artifact_lineage
WHERE target_scope_kind='run' AND target_scope_id=$1
 AND target_namespace='inputs' AND target_name=$2
 AND lineage_kind='input_fork'
ORDER BY target_revision
LIMIT 2`
