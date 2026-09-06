CREATE TABLE artifact_git_sources (
    version_id text PRIMARY KEY REFERENCES artifact_versions(version_id) ON DELETE CASCADE,
    repository_url text NOT NULL CHECK (octet_length(repository_url) BETWEEN 1 AND 4096),
    requested_ref text CHECK (octet_length(requested_ref) BETWEEN 1 AND 1024),
    resolved_commit text NOT NULL CHECK (resolved_commit ~ '^[0-9a-f]{40}$'),
    imported_at timestamptz NOT NULL
);

CREATE TRIGGER artifact_git_source_immutable
BEFORE UPDATE OR DELETE ON artifact_git_sources
FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_immutable();
