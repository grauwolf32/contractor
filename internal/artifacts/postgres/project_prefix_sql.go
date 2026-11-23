package postgres

// listProjectPrefixMetadataSQL seeks the prefix and cursor in one C-collated
// index range. Only a bounded selected page joins current revision metadata.
var listProjectPrefixMetadataSQL = `
WITH selected AS MATERIALIZED (
 SELECT * FROM artifact_bindings
 WHERE scope_kind='project' AND scope_id=$1 AND namespace=$2
 AND name COLLATE "C">=$3::text COLLATE "C"
 AND name COLLATE "C"<($3::text||'~') COLLATE "C"
 AND name COLLATE "C">$4::text COLLATE "C"
 ORDER BY name COLLATE "C" LIMIT $5
)
SELECT binding.namespace,binding.name,revision.revision,` + metadataPageColumnsSQL("true") + `
FROM selected AS binding` + currentMetadataJoinsSQL + `
ORDER BY binding.name COLLATE "C"`
