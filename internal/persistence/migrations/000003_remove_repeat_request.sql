-- Repeat drafts use workflow_runs and exact input_fork lineage. Remove the
-- duplicate request records and their now-unused storage; ordinary artifacts
-- sharing a version or blob retain it. This namespace had no other writer.
SELECT set_config('contractor.lifecycle_purge', 'run', true);

CREATE TEMPORARY TABLE removed_repeat_versions ON COMMIT DROP AS
SELECT DISTINCT version.version_id, version.blob_sha256
FROM artifact_binding_revisions AS revision
JOIN artifact_versions AS version USING (version_id)
WHERE revision.namespace = 'contractor-system';

DELETE FROM artifact_lineage
WHERE target_namespace = 'contractor-system' OR source_namespace = 'contractor-system';
DELETE FROM artifact_bindings WHERE namespace = 'contractor-system';
DELETE FROM artifact_binding_revisions WHERE namespace = 'contractor-system';

DELETE FROM artifact_versions AS version
USING removed_repeat_versions AS removed
WHERE version.version_id = removed.version_id
  AND NOT EXISTS (
    SELECT 1 FROM artifact_binding_revisions AS revision
    WHERE revision.version_id = version.version_id
  );
DELETE FROM artifact_blobs AS blob
USING removed_repeat_versions AS removed
WHERE blob.sha256 = removed.blob_sha256
  AND NOT EXISTS (
    SELECT 1 FROM artifact_versions AS version WHERE version.blob_sha256 = blob.sha256
  );

SELECT set_config('contractor.lifecycle_purge', '', true);

DROP INDEX artifact_bindings_user_listing_idx;
CREATE INDEX artifact_bindings_user_listing_idx
ON artifact_bindings (scope_id, namespace, name)
WHERE scope_kind = 'user' AND namespace <> 'audit-standards';
