package postgres

// SQL statements for postgres_replace_fork.go. Source selection uses the
// ordinary fork membership boundary, locking the exact revision until commit.
var replaceForkSourceSQL = `
    SELECT revision.version_id, revision.revision, version.media_type, blob.size_bytes
    FROM artifact_binding_revisions AS revision
    JOIN artifact_scopes AS source_scope
      ON source_scope.scope_kind = revision.scope_kind AND source_scope.scope_id = revision.scope_id
    JOIN artifact_bindings AS binding
      ON binding.scope_kind = revision.scope_kind
     AND binding.scope_id = revision.scope_id
     AND binding.namespace = revision.namespace
     AND binding.name = revision.name
    JOIN artifact_versions AS version ON version.version_id = revision.version_id
    JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
    WHERE revision.scope_kind = $1 AND revision.scope_id = $2
      AND revision.namespace = $3 AND revision.name = $4
      AND revision.revision = COALESCE($5::text, binding.current_revision)
      AND (NOT $8::boolean OR (binding.frozen AND binding.current_revision=revision.revision))

` + auditForkMembershipSQL + `
    FOR KEY SHARE OF source_scope, revision`

// replaceForkSQL advances one mutable target by CAS, charging ordinary history
// counters and retaining the source version without inserting a payload/version.
var replaceForkSQL = `
WITH changed AS (
    UPDATE artifact_bindings SET current_revision=$6, updated_at=clock_timestamp(),
        material_rank=(SELECT contractor_material_rank($3,media_type,git_resolved_commit IS NOT NULL)
                       FROM artifact_versions WHERE version_id=$7),
        retained_revisions=retained_revisions+1-$11,
        retained_bytes=retained_bytes+$8-$12
    WHERE scope_kind=$1 AND scope_id=$2 AND namespace=$3 AND name=$4
      AND current_revision=$5 AND NOT frozen
      AND retained_revisions+1-$11 <= $9 AND retained_bytes+$8-$12 <= $10
    RETURNING created_at
), revision AS (
    INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id)
    SELECT $1,$2,$3,$4,$6,$7 FROM changed RETURNING revision,created_at
), lineage AS (
    INSERT INTO artifact_lineage(target_scope_kind,target_scope_id,target_namespace,target_name,target_revision,
        source_scope_kind,source_scope_id,source_namespace,source_name,source_revision,lineage_kind)
    SELECT $1,$2,$3,$4,$6,$13,$14,$15,$16,$17,$18 FROM revision RETURNING 1
)
SELECT revision.revision,changed.created_at,revision.created_at FROM revision,changed,lineage`
