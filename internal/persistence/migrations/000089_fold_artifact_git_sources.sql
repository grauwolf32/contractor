-- Git provenance was a 0..1 extension of an artifact version. The version row
-- carries it; trusted Git import sets it once, in the transaction that created
-- the version, and it is immutable afterwards.
ALTER TABLE artifact_versions
    ADD COLUMN git_repository_url text CHECK (octet_length(git_repository_url) BETWEEN 1 AND 4096),
    ADD COLUMN git_requested_ref text CHECK (octet_length(git_requested_ref) BETWEEN 1 AND 1024),
    ADD COLUMN git_resolved_commit text CHECK (git_resolved_commit ~ '^[0-9a-f]{40}$'),
    ADD COLUMN git_imported_at timestamptz,
    ADD CONSTRAINT artifact_versions_git_source CHECK (
        (git_repository_url IS NULL) = (git_resolved_commit IS NULL)
        AND (git_repository_url IS NULL) = (git_imported_at IS NULL)
        AND (git_requested_ref IS NULL OR git_repository_url IS NOT NULL)
    );

ALTER TABLE artifact_versions DISABLE TRIGGER artifact_versions_immutable;
UPDATE artifact_versions AS version
   SET git_repository_url = source.repository_url,
       git_requested_ref = source.requested_ref,
       git_resolved_commit = source.resolved_commit,
       git_imported_at = source.imported_at
  FROM artifact_git_sources AS source
 WHERE source.version_id = version.version_id;
ALTER TABLE artifact_versions ENABLE TRIGGER artifact_versions_immutable;

DROP TABLE artifact_git_sources;

CREATE FUNCTION contractor_protect_artifact_version()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'DELETE' AND contractor_lifecycle_purge_enabled() THEN
        RETURN OLD;
    END IF;
    IF TG_OP = 'UPDATE' AND OLD.git_resolved_commit IS NULL AND NEW.git_resolved_commit IS NOT NULL
        AND ROW(NEW.version_id, NEW.blob_sha256, NEW.media_type, NEW.created_at)
            IS NOT DISTINCT FROM ROW(OLD.version_id, OLD.blob_sha256, OLD.media_type, OLD.created_at)
    THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'immutable ArtifactStore row cannot be changed' USING ERRCODE = '23514';
END;
$$;

DROP TRIGGER artifact_versions_immutable ON artifact_versions;
CREATE TRIGGER artifact_versions_immutable
BEFORE UPDATE OR DELETE ON artifact_versions
FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_version();
