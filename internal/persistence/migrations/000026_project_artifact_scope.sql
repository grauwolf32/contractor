ALTER TABLE artifact_scopes
    DROP CONSTRAINT artifact_scopes_scope_kind_check,
    ADD CONSTRAINT artifact_scopes_scope_kind_check
        CHECK (scope_kind IN ('user', 'project', 'run')),
    ADD COLUMN project_id text GENERATED ALWAYS AS (
        CASE WHEN scope_kind = 'project' THEN scope_id ELSE NULL END
    ) STORED,
    ADD CONSTRAINT artifact_scopes_project_id_fkey
        FOREIGN KEY (project_id) REFERENCES projects(project_id);

CREATE INDEX artifact_scopes_project_idx
    ON artifact_scopes (project_id)
    WHERE project_id IS NOT NULL;
