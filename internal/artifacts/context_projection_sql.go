package artifacts

// contextProjectionReplaySQL proves that the immutable target's exact source
// agrees with this replay. Payload bytes are checked through ordinary metadata.
// Used by PostgresTxRepository.CreateContextProjection.
var contextProjectionReplaySQL = `
SELECT EXISTS (
 SELECT 1 FROM artifact_lineage
 WHERE target_scope_kind=$1 AND target_scope_id=$2
  AND target_namespace=$3 AND target_name=$4 AND target_revision=$5
  AND source_scope_kind=$6 AND source_scope_id=$7
  AND source_namespace=$8 AND source_name=$9 AND source_revision=$10
  AND lineage_kind=$11
)`

// createContextProjectionLineageSQL uses the existing strict lineage holder.
// The ordinary holder trigger, FK and indexed revision protection serialize
// source selection against retention. Used by CreateContextProjection.
var createContextProjectionLineageSQL = `
INSERT INTO artifact_lineage (
 target_scope_kind,target_scope_id,target_namespace,target_name,target_revision,
 source_scope_kind,source_scope_id,source_namespace,source_name,source_revision,lineage_kind
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
ON CONFLICT DO NOTHING`
