-- tool@1 allocations omit modelPolicy and all model-route provenance. Keep the
-- existing model-bearing shape while allowing the model-free Go wire contract.
CREATE OR REPLACE FUNCTION contractor_valid_allocation_runtime_configuration(candidate jsonb)
RETURNS boolean
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
    model_policy jsonb := candidate->'modelPolicy';
    origins jsonb := candidate->'origins';
    provenance jsonb := candidate->'provenance';
    has_model boolean := candidate ? 'modelPolicy';
BEGIN
    IF jsonb_typeof(candidate) IS DISTINCT FROM 'object'
       OR NOT (candidate ?& ARRAY['origins', 'provenance']::text[])
       OR EXISTS (
           SELECT 1 FROM jsonb_object_keys(candidate) AS key
           WHERE key <> ALL (ARRAY['modelPolicy', 'origins', 'provenance']::text[])
       )
       OR jsonb_typeof(origins) IS DISTINCT FROM 'object'
       OR jsonb_typeof(provenance) IS DISTINCT FROM 'object'
       OR NOT (provenance ?& ARRAY[
           'agentLabels', 'default', 'runLabels', 'runtimeAdapters', 'runtimeCredentialRefs'
       ]::text[])
       OR EXISTS (
           SELECT 1 FROM jsonb_object_keys(provenance) AS key
           WHERE key <> ALL (ARRAY[
               'agentLabels', 'default', 'llmCredential', 'llmGatewayConfig',
               'runLabels', 'runtimeAdapters', 'runtimeCredentialRefs'
           ]::text[])
       )
       OR jsonb_typeof(provenance->'default') IS DISTINCT FROM 'object'
       OR jsonb_typeof(provenance->'runLabels') IS DISTINCT FROM 'array'
       OR jsonb_typeof(provenance->'agentLabels') IS DISTINCT FROM 'array'
       OR jsonb_typeof(provenance->'runtimeAdapters') IS DISTINCT FROM 'array'
       OR jsonb_typeof(provenance->'runtimeCredentialRefs') IS DISTINCT FROM 'array'
    THEN
        RETURN false;
    END IF;

    IF has_model THEN
        IF jsonb_typeof(model_policy) IS DISTINCT FROM 'object'
           OR (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(model_policy) AS key)
              IS DISTINCT FROM ARRAY['digest', 'policyId', 'version']::text[]
           OR jsonb_typeof(model_policy->'policyId') IS DISTINCT FROM 'string'
           OR jsonb_typeof(model_policy->'version') IS DISTINCT FROM 'string'
           OR jsonb_typeof(model_policy->'digest') IS DISTINCT FROM 'string'
           OR model_policy->>'policyId' !~ '^[a-z][a-z0-9_-]*$'
           OR model_policy->>'version' !~ '^[A-Za-z0-9][A-Za-z0-9._+-]*$'
           OR model_policy->>'digest' !~ '^sha256:[0-9a-f]{64}$'
           OR jsonb_typeof(provenance->'llmGatewayConfig') IS DISTINCT FROM 'object'
        THEN
            RETURN false;
        END IF;
    ELSIF provenance ?| ARRAY['llmGatewayConfig', 'llmCredential']::text[]
       OR origins ?| ARRAY['llmGateway', 'llmCredential']::text[]
    THEN
        RETURN false;
    END IF;
    RETURN true;
EXCEPTION WHEN OTHERS THEN
    RETURN false;
END;
$$;
