package artifacts

// SQL statements for git_source.go.

// recordGitSourceSQL sets the Git provenance of the version behind exact
// revision $5 of scope $1/$2, binding $3/$4, if that version has none yet. It
// returns whether the revision exists and whether the provenance was recorded;
// the version trigger allows this single change. Used by
// PostgresRepository.RecordGitSource.
var recordGitSourceSQL = `
WITH revision AS (
    SELECT version_id FROM artifact_binding_revisions
    WHERE scope_kind=$1 AND scope_id=$2 AND namespace=$3 AND name=$4 AND revision=$5
), recorded AS (
    UPDATE artifact_versions AS version
       SET git_repository_url=$6, git_requested_ref=$7, git_resolved_commit=$8, git_imported_at=$9
      FROM revision
     WHERE version.version_id = revision.version_id AND version.git_resolved_commit IS NULL
    RETURNING 1
)
SELECT EXISTS (SELECT 1 FROM revision), EXISTS (SELECT 1 FROM recorded)`
