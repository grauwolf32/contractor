package postgres

// Project material pages seek the derived current-binding key before joining
// metadata. Namespace-filtered reads use the companion namespace-leading index.
// Arguments: Project, optional namespace, cursor rank/namespace/name, fetch limit.
func projectMaterialPageSQL(namespacePredicate string) string {
	return `
WITH selected AS MATERIALIZED (
 SELECT * FROM artifact_bindings AS binding
 WHERE scope_kind='project' AND scope_id=$1
 AND namespace NOT LIKE 'audit-%'
 ` + namespacePredicate + `
 AND (material_rank,namespace,name)>($3::smallint,$4::text,$5::text)
 ORDER BY material_rank,namespace,name LIMIT $6
)
SELECT binding.namespace,binding.name,revision.revision,` + metadataPageColumnsSQL("true") + `,binding.material_rank
FROM selected AS binding` + currentMetadataJoinsSQL + `
ORDER BY binding.material_rank,binding.namespace,binding.name`
}

var listProjectMaterialMetadataSQL = projectMaterialPageSQL("AND $2::text IS NULL")
var listProjectNamespaceMaterialMetadataSQL = projectMaterialPageSQL("AND namespace=$2::text")
