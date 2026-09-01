CREATE OR REPLACE FUNCTION contractor_valid_allocation_runtime_configuration(candidate jsonb)
RETURNS boolean
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
    model_policy jsonb := candidate->'modelPolicy';
    provenance jsonb := candidate->'provenance';
BEGIN
    IF jsonb_typeof(candidate) <> 'object'
       OR (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(candidate) AS key)
          <> ARRAY['modelPolicy', 'origins', 'provenance']::text[]
       OR jsonb_typeof(model_policy) <> 'object'
       OR (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(model_policy) AS key)
          <> ARRAY['digest', 'policyId', 'version']::text[]
       OR model_policy->>'policyId' !~ '^[a-z][a-z0-9_-]*$'
       OR model_policy->>'version' !~ '^[A-Za-z0-9][A-Za-z0-9._+-]*$'
       OR model_policy->>'digest' !~ '^sha256:[0-9a-f]{64}$'
       OR jsonb_typeof(candidate->'origins') <> 'object'
       OR jsonb_typeof(provenance) <> 'object'
       OR NOT (provenance ?& ARRAY[
              'agentLabels', 'default', 'llmGatewayConfig', 'runLabels',
              'runtimeAdapters', 'runtimeCredentialRefs'
          ]::text[])
       OR EXISTS (
           SELECT 1
           FROM jsonb_object_keys(provenance) AS key
           WHERE key <> ALL (ARRAY[
               'agentLabels', 'default', 'llmCredential', 'llmGatewayConfig',
               'runLabels', 'runtimeAdapters', 'runtimeCredentialRefs'
           ]::text[])
       )
       OR jsonb_typeof(provenance->'default') <> 'object'
       OR jsonb_typeof(provenance->'runLabels') <> 'array'
       OR jsonb_typeof(provenance->'agentLabels') <> 'array'
       OR jsonb_typeof(provenance->'runtimeAdapters') <> 'array'
       OR jsonb_typeof(provenance->'runtimeCredentialRefs') <> 'array'
       OR provenance ?| ARRAY[
           'llmGatewayUrl', 'llmGatewayToken', 'telemetry', 'httpProxy',
           'headers', 'basicAuth', 'bearerToken', 'password'
       ]
    THEN
        RETURN false;
    END IF;
    RETURN true;
EXCEPTION WHEN OTHERS THEN
    RETURN false;
END;
$$;

ALTER TABLE stage_allocations
    ADD COLUMN runtime_agent_id text,
    ADD COLUMN runtime_agent_label_revision numeric(20, 0),
    ADD COLUMN runtime_configuration_schema_version text,
    ADD COLUMN runtime_configuration jsonb,
    ADD CONSTRAINT stage_allocation_runtime_configuration_shape CHECK (
        (
            runtime_agent_id IS NULL
            AND runtime_agent_label_revision IS NULL
            AND runtime_configuration_schema_version IS NULL
            AND runtime_configuration IS NULL
        ) OR (
            runtime_agent_id ~ '^[0-9a-f]{64}$'
            AND runtime_agent_label_revision >= 1
            AND runtime_agent_label_revision <= 18446744073709551615
            AND runtime_configuration_schema_version = 'contractor.runtime-config-provenance/v2'
            AND contractor_valid_allocation_runtime_configuration(runtime_configuration)
        )
    );

CREATE INDEX stage_allocations_live_llm_credential_idx
    ON stage_allocations ((runtime_configuration #>> '{provenance,llmCredential,credentialId}'))
    WHERE release_completed_at IS NULL AND runtime_configuration IS NOT NULL;

CREATE INDEX stage_allocations_live_runtime_credentials_idx
    ON stage_allocations USING gin ((runtime_configuration->'provenance'->'runtimeCredentialRefs'))
    WHERE release_completed_at IS NULL AND runtime_configuration IS NOT NULL;

CREATE OR REPLACE FUNCTION contractor_protect_stage_allocation_identity()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.allocation_id IS DISTINCT FROM OLD.allocation_id
       OR NEW.stage_execution_id IS DISTINCT FROM OLD.stage_execution_id
       OR NEW.logical_agent_name IS DISTINCT FROM OLD.logical_agent_name
       OR NEW.namespace IS DISTINCT FROM OLD.namespace
       OR NEW.agent_template_ref IS DISTINCT FROM OLD.agent_template_ref
       OR NEW.worker_runtime_ref IS DISTINCT FROM OLD.worker_runtime_ref
       OR NEW.runtime_agent_instance_id IS DISTINCT FROM OLD.runtime_agent_instance_id
       OR NEW.runtime_agent_id IS DISTINCT FROM OLD.runtime_agent_id
       OR NEW.runtime_agent_label_revision IS DISTINCT FROM OLD.runtime_agent_label_revision
       OR NEW.runtime_configuration_schema_version IS DISTINCT FROM OLD.runtime_configuration_schema_version
       OR NEW.runtime_configuration IS DISTINCT FROM OLD.runtime_configuration
       OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'Stage allocation identity and Runtime configuration are immutable'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER stage_allocations_protect_identity
BEFORE UPDATE ON stage_allocations
FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_allocation_identity();
