-- Retained-history accounting of every logical Artifact binding (ART-R123,
-- LIM-ART-9). Each binding records how many revisions it retains and their
-- full encoded size: every retained revision is charged the size of its blob,
-- also when the version or blob is shared. ArtifactStore writes and forks keep
-- both counters in the statement that inserts the revision; this migration
-- backfills them from the stored revisions. Nothing here removes a revision.

ALTER TABLE artifact_bindings
    ADD COLUMN retained_revisions bigint,
    ADD COLUMN retained_bytes bigint;

-- The admission guard of Project bindings rejects every update of a binding
-- whose Project is deleting; the backfill changes no admitted content, so it
-- runs without that guard. No other row trigger of artifact_bindings fires for
-- these columns.
ALTER TABLE artifact_bindings DISABLE TRIGGER artifact_bindings_project_admission;

UPDATE artifact_bindings AS binding
SET retained_revisions = retained.revisions,
    retained_bytes = retained.bytes
FROM (
    SELECT revision.scope_kind, revision.scope_id, revision.namespace, revision.name,
           count(*) AS revisions, sum(blob.size_bytes) AS bytes
    FROM artifact_binding_revisions AS revision
    JOIN artifact_versions AS version ON version.version_id = revision.version_id
    JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
    GROUP BY revision.scope_kind, revision.scope_id, revision.namespace, revision.name
) AS retained
WHERE binding.scope_kind = retained.scope_kind AND binding.scope_id = retained.scope_id
  AND binding.namespace = retained.namespace AND binding.name = retained.name;

ALTER TABLE artifact_bindings ENABLE TRIGGER artifact_bindings_project_admission;

-- A binding always retains at least its current revision, so a binding the
-- backfill left without counters has no stored revision: corrupt data that
-- must stop the migration rather than be accounted as empty.
ALTER TABLE artifact_bindings
    ALTER COLUMN retained_revisions SET NOT NULL,
    ALTER COLUMN retained_bytes SET NOT NULL,
    ADD CONSTRAINT artifact_bindings_retained_revisions_check CHECK (retained_revisions >= 1),
    ADD CONSTRAINT artifact_bindings_retained_bytes_check CHECK (retained_bytes >= 0);
