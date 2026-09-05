ALTER TABLE runtime_credentials
    DROP CONSTRAINT runtime_credentials_credential_kind_check;

ALTER TABLE runtime_credentials
    ADD CONSTRAINT runtime_credentials_credential_kind_check
    CHECK (credential_kind IN (
        'otlp-headers@1',
        'http-proxy-basic@1',
        'http-proxy-bearer@1',
        'caido-bearer@1',
        'http-origin-basic@1',
        'http-origin-bearer@1'
    ));

ALTER TABLE runtime_credential_creations
    DROP CONSTRAINT runtime_credential_creations_credential_kind_check;

ALTER TABLE runtime_credential_creations
    ADD CONSTRAINT runtime_credential_creations_credential_kind_check
    CHECK (credential_kind IN (
        'otlp-headers@1',
        'http-proxy-basic@1',
        'http-proxy-bearer@1',
        'caido-bearer@1',
        'http-origin-basic@1',
        'http-origin-bearer@1'
    ));

ALTER TABLE projects
    ADD COLUMN http_target_url text,
    ADD COLUMN http_target_credential_id text,
    ADD COLUMN http_target_credential_kind text,
    ADD CONSTRAINT projects_http_target_shape CHECK (
        (http_target_url IS NULL
            AND http_target_credential_id IS NULL
            AND http_target_credential_kind IS NULL)
        OR
        (http_target_url IS NOT NULL
            AND btrim(http_target_url) = http_target_url
            AND octet_length(http_target_url) BETWEEN 1 AND 2048
            AND (
                (http_target_credential_id IS NULL AND http_target_credential_kind IS NULL)
                OR
                (http_target_credential_id IS NOT NULL
                    AND http_target_credential_kind IN ('http-origin-basic@1', 'http-origin-bearer@1'))
            ))
    ),
    ADD CONSTRAINT projects_http_target_credential_fkey
        FOREIGN KEY (http_target_credential_id, http_target_credential_kind)
        REFERENCES runtime_credentials(credential_id, credential_kind);

CREATE INDEX projects_http_target_credential_idx
    ON projects (http_target_credential_id, project_id)
    WHERE http_target_credential_id IS NOT NULL;

ALTER TABLE workflow_runs
    ADD COLUMN project_http_target_snapshot jsonb,
    ADD CONSTRAINT workflow_runs_project_http_target_shape CHECK (
        project_http_target_snapshot IS NULL
        OR (
            project_id IS NOT NULL
            AND jsonb_typeof(project_http_target_snapshot) = 'object'
            AND jsonb_typeof(project_http_target_snapshot->'url') = 'string'
            AND btrim(project_http_target_snapshot->>'url') = project_http_target_snapshot->>'url'
            AND octet_length(project_http_target_snapshot->>'url') BETWEEN 1 AND 2048
            AND (
                NOT (project_http_target_snapshot ? 'credential')
                OR (
                    jsonb_typeof(project_http_target_snapshot->'credential') = 'object'
                    AND jsonb_typeof(project_http_target_snapshot#>'{credential,credentialId}') = 'string'
                    AND project_http_target_snapshot#>>'{credential,kind}'
                        IN ('http-origin-basic@1', 'http-origin-bearer@1')
                )
            )
        )
    );

CREATE INDEX workflow_runs_project_http_target_credential_idx
    ON workflow_runs ((project_http_target_snapshot#>>'{credential,credentialId}'), created_at, run_id)
    WHERE state IN ('initializing', 'running', 'cancelling')
      AND project_http_target_snapshot ? 'credential';

CREATE OR REPLACE FUNCTION contractor_protect_workflow_run_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.run_id IS DISTINCT FROM OLD.run_id
        OR NEW.owner_id IS DISTINCT FROM OLD.owner_id
        OR NEW.project_id IS DISTINCT FROM OLD.project_id
        OR NEW.workflow_name IS DISTINCT FROM OLD.workflow_name
        OR NEW.workflow_version IS DISTINCT FROM OLD.workflow_version
        OR NEW.workflow_schema_version IS DISTINCT FROM OLD.workflow_schema_version
        OR NEW.workflow_snapshot IS DISTINCT FROM OLD.workflow_snapshot
        OR NEW.parameters IS DISTINCT FROM OLD.parameters
        OR NEW.request_idempotency_key IS DISTINCT FROM OLD.request_idempotency_key
        OR NEW.request_digest IS DISTINCT FROM OLD.request_digest
        OR NEW.run_event_generation IS DISTINCT FROM OLD.run_event_generation
        OR NEW.runtime_labels IS DISTINCT FROM OLD.runtime_labels
        OR NEW.runtime_config_snapshot IS DISTINCT FROM OLD.runtime_config_snapshot
        OR NEW.project_http_target_snapshot IS DISTINCT FROM OLD.project_http_target_snapshot
        OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'immutable WorkflowRun fields cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.run_cancellation IS NOT NULL
        AND (
            NEW.cancellation_schema_version IS DISTINCT FROM OLD.cancellation_schema_version
            OR NEW.run_cancellation IS DISTINCT FROM OLD.run_cancellation
        )
    THEN
        RAISE EXCEPTION 'WorkflowRun cancellation cannot be changed' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
