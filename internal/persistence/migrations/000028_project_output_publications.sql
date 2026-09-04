ALTER TABLE workflow_runs
    ADD CONSTRAINT workflow_runs_run_project_key UNIQUE (run_id, project_id);

ALTER TABLE artifact_lineage
    DROP CONSTRAINT artifact_lineage_lineage_kind_check,
    ADD CONSTRAINT artifact_lineage_lineage_kind_check
        CHECK (lineage_kind IN ('input_fork', 'output_bind', 'project_output_publish'));

CREATE TABLE workflow_run_output_publications (
    run_id text NOT NULL,
    project_id text NOT NULL,
    output_name text NOT NULL CHECK (
        btrim(output_name) <> ''
        AND position('/' in output_name) = 0
        AND octet_length(output_name) <= 128
    ),
    status text NOT NULL CHECK (status IN ('published', 'already_present', 'failed')),
    source_scope_kind text GENERATED ALWAYS AS ('run'::text) STORED,
    source_namespace text NOT NULL CHECK (source_namespace = 'outputs'),
    source_name text NOT NULL,
    source_revision text NOT NULL CHECK (btrim(source_revision) <> ''),
    target_scope_kind text GENERATED ALWAYS AS ('project'::text) STORED,
    target_namespace text NOT NULL CHECK (target_namespace = 'outputs'),
    target_name text NOT NULL,
    target_revision text,
    error_code text,
    error_message text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (run_id, output_name),
    FOREIGN KEY (run_id, project_id)
        REFERENCES workflow_runs(run_id, project_id),
    FOREIGN KEY (
        source_scope_kind, run_id, source_namespace, source_name, source_revision
    ) REFERENCES artifact_binding_revisions(scope_kind, scope_id, namespace, name, revision),
    FOREIGN KEY (
        target_scope_kind, project_id, target_namespace, target_name, target_revision
    ) REFERENCES artifact_binding_revisions(scope_kind, scope_id, namespace, name, revision),
    CHECK (source_name = output_name AND target_name = output_name),
    CHECK (
        (status = 'published'
            AND target_revision IS NOT NULL
            AND error_code IS NULL
            AND error_message IS NULL)
        OR (status = 'already_present'
            AND target_revision IS NULL
            AND error_code IS NULL
            AND error_message IS NULL)
        OR (status = 'failed'
            AND target_revision IS NULL
            AND error_code IS NOT NULL
            AND btrim(error_code) <> ''
            AND octet_length(error_code) <= 128
            AND error_message IS NOT NULL
            AND btrim(error_message) <> ''
            AND octet_length(error_message) <= 2048)
    )
);

CREATE INDEX workflow_run_output_publications_project_idx
    ON workflow_run_output_publications (project_id, created_at DESC, run_id, output_name);

CREATE TRIGGER workflow_run_output_publications_immutable
BEFORE UPDATE OR DELETE ON workflow_run_output_publications
FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_immutable();
