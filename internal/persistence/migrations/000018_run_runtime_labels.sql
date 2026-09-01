CREATE OR REPLACE FUNCTION contractor_valid_pinned_runtime_config(candidate jsonb, expected_label text, explicit boolean)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
STRICT
AS $$
    SELECT jsonb_typeof(candidate) = 'object'
       AND (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(candidate) AS key)
           = ARRAY['bindingRevision', 'config', 'explicit', 'label']::text[]
       AND candidate->>'label' = expected_label
       AND candidate->'explicit' = to_jsonb(explicit)
       AND jsonb_typeof(candidate->'bindingRevision') = 'number'
       AND (candidate->>'bindingRevision')::numeric >= 1
       AND (candidate->>'bindingRevision')::numeric <= 18446744073709551615
       AND scale((candidate->>'bindingRevision')::numeric) = 0
       AND jsonb_typeof(candidate->'config') = 'object'
       AND (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(candidate->'config') AS key)
           = ARRAY['digest', 'name', 'version']::text[]
       AND candidate->'config'->>'name' ~ '^[a-z][a-z0-9_-]*$'
       AND length(candidate->'config'->>'name') <= 63
       AND candidate->'config'->>'version' ~ '^[A-Za-z0-9][A-Za-z0-9._-]*$'
       AND length(candidate->'config'->>'version') <= 128
       AND candidate->'config'->>'digest' ~ '^sha256:[0-9a-f]{64}$'
$$;

CREATE OR REPLACE FUNCTION contractor_valid_run_runtime_snapshot(candidate jsonb, labels text[])
RETURNS boolean
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
    item jsonb;
    index_value integer := 1;
    previous_value text := '';
    credential_value text;
BEGIN
    IF jsonb_typeof(candidate) <> 'object'
       OR (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(candidate) AS key)
          <> ARRAY['default', 'labels', 'llmCredentialIds', 'runtimeCredentialIds']::text[]
       OR contractor_valid_pinned_runtime_config(candidate->'default', 'default', false) IS DISTINCT FROM true
       OR jsonb_typeof(candidate->'labels') <> 'array'
       OR jsonb_array_length(candidate->'labels') <> cardinality(labels)
       OR jsonb_typeof(candidate->'llmCredentialIds') <> 'array'
       OR jsonb_typeof(candidate->'runtimeCredentialIds') <> 'array'
    THEN
        RETURN false;
    END IF;

    FOR item IN SELECT value FROM jsonb_array_elements(candidate->'labels')
    LOOP
        IF index_value > cardinality(labels)
           OR contractor_valid_pinned_runtime_config(item, labels[index_value], true) IS DISTINCT FROM true
        THEN
            RETURN false;
        END IF;
        index_value := index_value + 1;
    END LOOP;

    FOR credential_value IN
        SELECT value #>> '{}' FROM jsonb_array_elements(candidate->'llmCredentialIds') AS value
    LOOP
        IF credential_value IS NULL
           OR credential_value !~ '^[a-z][a-z0-9_-]*$'
           OR length(credential_value) > 128
           OR credential_value <= previous_value
        THEN
            RETURN false;
        END IF;
        previous_value := credential_value;
    END LOOP;

    previous_value := '';
    FOR credential_value IN
        SELECT value #>> '{}' FROM jsonb_array_elements(candidate->'runtimeCredentialIds') AS value
    LOOP
        IF credential_value IS NULL
           OR credential_value !~ '^[a-z][a-z0-9_-]*$'
           OR length(credential_value) > 128
           OR credential_value <= previous_value
        THEN
            RETURN false;
        END IF;
        previous_value := credential_value;
    END LOOP;
    RETURN true;
EXCEPTION WHEN OTHERS THEN
    RETURN false;
END;
$$;

ALTER TABLE workflow_runs
    ADD COLUMN runtime_labels text[] NOT NULL DEFAULT '{}'::text[]
        CHECK (contractor_valid_runtime_agent_labels(runtime_labels)),
    ADD COLUMN runtime_config_snapshot jsonb;

WITH current_default AS (
    SELECT b.revision, b.config_name, b.config_version, b.config_digest,
           c.canonical_document::jsonb AS document
    FROM runtime_label_bindings AS b
    JOIN runtime_config_versions AS c
      ON c.name = b.config_name
     AND c.version = b.config_version
     AND c.digest = b.config_digest
    WHERE b.label = 'default'
), normalized AS (
    SELECT revision, config_name, config_version, config_digest, document,
           document #>> '{spec,worker,llmGateway,credential}' AS llm_credential,
           COALESCE((
               SELECT jsonb_agg(value ORDER BY value)
               FROM (
                   SELECT DISTINCT value
                   FROM (VALUES
                       (document #>> '{spec,worker,telemetry,credential}'),
                       (document #>> '{spec,worker,httpProxy,credential}'),
                       (document #>> '{spec,planner,telemetry,credential}')
                   ) AS candidates(value)
                   WHERE value IS NOT NULL
               ) AS exact_values
           ), '[]'::jsonb) AS runtime_credentials
    FROM current_default
)
UPDATE workflow_runs
SET runtime_config_snapshot = jsonb_build_object(
    'default', jsonb_build_object(
        'label', 'default',
        'explicit', false,
        'bindingRevision', normalized.revision,
        'config', jsonb_build_object(
            'name', normalized.config_name,
            'version', normalized.config_version,
            'digest', normalized.config_digest
        )
    ),
    'labels', '[]'::jsonb,
    'llmCredentialIds', CASE
        WHEN normalized.llm_credential IS NULL THEN '[]'::jsonb
        ELSE jsonb_build_array(normalized.llm_credential)
    END,
    'runtimeCredentialIds', normalized.runtime_credentials
)
FROM normalized;

ALTER TABLE workflow_runs
    ALTER COLUMN runtime_labels DROP DEFAULT,
    ALTER COLUMN runtime_config_snapshot SET NOT NULL,
    ADD CONSTRAINT workflow_runs_runtime_config_snapshot_shape CHECK (
        contractor_valid_run_runtime_snapshot(runtime_config_snapshot, runtime_labels)
    );

CREATE INDEX workflow_runs_runtime_labels_idx
    ON workflow_runs USING gin (runtime_labels);

CREATE INDEX workflow_runs_runtime_llm_credentials_idx
    ON workflow_runs USING gin ((runtime_config_snapshot->'llmCredentialIds'))
    WHERE state IN ('initializing', 'running', 'cancelling');

CREATE INDEX workflow_runs_runtime_credentials_idx
    ON workflow_runs USING gin ((runtime_config_snapshot->'runtimeCredentialIds'))
    WHERE state IN ('initializing', 'running', 'cancelling');

CREATE OR REPLACE FUNCTION contractor_protect_workflow_run_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.run_id IS DISTINCT FROM OLD.run_id
        OR NEW.owner_id IS DISTINCT FROM OLD.owner_id
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
