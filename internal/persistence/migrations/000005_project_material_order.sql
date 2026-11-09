-- A derived ordering key on ordinary bindings keeps Project material paging
-- bounded before metadata hydration. It does not change artifact identity,
-- timestamps, payloads, revisions or retention. ArtifactStore owns every write.
CREATE FUNCTION contractor_material_rank(namespace text, media_type text, git_source boolean)
RETURNS smallint LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT CASE
        WHEN git_source THEN 0
        WHEN namespace = 'sources' THEN 0
        WHEN namespace = 'openapi' THEN 1
        WHEN namespace = 'engagement' THEN 2
        WHEN namespace = 'likec4' THEN 3
        WHEN namespace = 'docs' THEN 4
        WHEN namespace = 'diffs' THEN 5
        WHEN media_type IN ('application/vnd.oai.openapi', 'application/vnd.oai.openapi+json', 'application/vnd.oai.openapi+yaml') THEN 1
        WHEN media_type = 'application/vnd.contractor.engagement+json' THEN 2
        WHEN media_type = 'text/vnd.likec4' THEN 3
        WHEN media_type IN ('text/markdown', 'text/x-markdown', 'application/pdf') THEN 4
        WHEN media_type = 'text/x-diff' THEN 5
        ELSE 6
    END::smallint
$$;

ALTER TABLE artifact_bindings ADD COLUMN material_rank smallint NOT NULL DEFAULT 6
    CHECK (material_rank BETWEEN 0 AND 6);

UPDATE artifact_bindings AS binding
SET material_rank = contractor_material_rank(binding.namespace, version.media_type, version.git_resolved_commit IS NOT NULL)
FROM artifact_binding_revisions AS revision
JOIN artifact_versions AS version USING (version_id)
WHERE revision.scope_kind = binding.scope_kind AND revision.scope_id = binding.scope_id
  AND revision.namespace = binding.namespace AND revision.name = binding.name
  AND revision.revision = binding.current_revision;

CREATE INDEX artifact_bindings_project_material_order_idx
ON artifact_bindings (scope_id, material_rank, namespace, name)
WHERE scope_kind = 'project' AND namespace NOT LIKE 'audit-%';

CREATE INDEX artifact_bindings_project_material_namespace_order_idx
ON artifact_bindings (scope_id, namespace, material_rank, name)
WHERE scope_kind = 'project' AND namespace NOT LIKE 'audit-%';
