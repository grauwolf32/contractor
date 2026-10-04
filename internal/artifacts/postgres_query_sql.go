package artifacts

// SQL statements for postgres_query.go.

// listLineageSQL returns one page of artifact_lineage edges in which exact
// revision $1..$5 is the source or the target, newest first, optionally
// excluding lineage kind $6. $7..$10 are the keyset cursor (created_at,
// target_revision, source_revision, lineage_kind); a NULL $7 starts the first
// page. Used by PostgresRepository.ListLineage.
var listLineageSQL = `
SELECT lineage.lineage_kind,
       lineage.source_scope_kind, lineage.source_scope_id,
       lineage.source_namespace, lineage.source_name, lineage.source_revision,
       lineage.target_scope_kind, lineage.target_scope_id,
       lineage.target_namespace, lineage.target_name, lineage.target_revision,
       lineage.created_at
FROM artifact_lineage AS lineage
WHERE ((
      lineage.source_scope_kind = $1 AND lineage.source_scope_id = $2
      AND lineage.source_namespace = $3 AND lineage.source_name = $4 AND lineage.source_revision = $5
    ) OR (
      lineage.target_scope_kind = $1 AND lineage.target_scope_id = $2
      AND lineage.target_namespace = $3 AND lineage.target_name = $4 AND lineage.target_revision = $5
    ))
  AND ($6::text = '' OR lineage.lineage_kind <> $6)
  AND ($7::timestamptz IS NULL OR (
    lineage.created_at, lineage.target_revision, lineage.source_revision, lineage.lineage_kind
  ) < ($7, $8, $9, $10))
ORDER BY lineage.created_at DESC, lineage.target_revision DESC,
         lineage.source_revision DESC, lineage.lineage_kind DESC
LIMIT $11`
