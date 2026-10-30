-- pg_restore evaluates generated columns, CHECK constraints, column defaults
-- and index expressions with an empty search_path (PER-R46). A function body
-- kept as text resolves its names only then, so a body that calls another
-- schema function failed the restore of every backup. Each function that a
-- table definition evaluates, directly or through another function, gets a
-- SQL-standard body here, whose names PostgreSQL binds when the function is
-- created and which it still inlines; the two PL/pgSQL validators that call
-- only built-in functions pin search_path = pg_catalog instead. Every body
-- keeps its statement and every function its volatility, strictness and
-- parallel safety.

-- Helpers of the artifact-hold projections, callees first.
CREATE OR REPLACE FUNCTION contractor_jsonb_elements(value jsonb) RETURNS SETOF jsonb
LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
SELECT element
FROM jsonb_array_elements(CASE WHEN jsonb_typeof(value) = 'array' THEN value ELSE '[]'::jsonb END)
    AS element;
END;

CREATE OR REPLACE FUNCTION contractor_jsonb_values(value jsonb) RETURNS SETOF jsonb
LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
SELECT member.value
FROM jsonb_each(CASE WHEN jsonb_typeof(value) = 'object' THEN value ELSE '{}'::jsonb END) AS member;
END;

CREATE OR REPLACE FUNCTION contractor_artifact_ref_key(scope_kind text, scope_id text, ref jsonb) RETURNS jsonb
LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
SELECT CASE
    WHEN scope_kind IS NOT NULL AND scope_id IS NOT NULL
     AND jsonb_typeof(ref -> 'namespace') = 'string' AND jsonb_typeof(ref -> 'name') = 'string'
     AND jsonb_typeof(ref -> 'revision') = 'string'
    THEN jsonb_build_array(scope_kind, scope_id, ref -> 'namespace', ref -> 'name', ref -> 'revision')
END;
END;

CREATE OR REPLACE FUNCTION contractor_artifact_key_set(keys jsonb[]) RETURNS jsonb
LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
SELECT COALESCE(jsonb_agg(DISTINCT key ORDER BY key), '[]'::jsonb)
FROM unnest(keys) AS key
WHERE key IS NOT NULL;
END;

-- Generated columns.
CREATE OR REPLACE FUNCTION contractor_audit_dependency_holds(project text, manifest jsonb) RETURNS jsonb
LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
SELECT contractor_artifact_key_set(ARRAY(
 SELECT contractor_artifact_ref_key('project',project,dependency #> '{retained,ref}')
 FROM contractor_jsonb_elements(manifest->'dependencies') AS dependency
));
END;

CREATE OR REPLACE FUNCTION contractor_planner_session_artifact_refs(state jsonb) RETURNS jsonb
LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
SELECT contractor_artifact_key_set(ARRAY(
    SELECT contractor_artifact_ref_key('run', '', state -> 'completion' -> 'result' -> 'summaryArtifact')
    UNION ALL
    SELECT contractor_artifact_ref_key('run', '', slot)
    FROM contractor_jsonb_values(state -> 'completion' -> 'result' -> 'artifacts') AS slot
    UNION ALL
    SELECT contractor_artifact_ref_key('run', '', state -> 'scan' -> 'state' -> 'plan')
    UNION ALL
    SELECT contractor_artifact_ref_key('run', '', job -> 'report')
    FROM contractor_jsonb_elements(state -> 'scan' -> 'state' -> 'jobs') AS job
    UNION ALL
    SELECT contractor_artifact_ref_key('run', '', input)
    FROM contractor_jsonb_elements(state -> 'scan' -> 'state' -> 'jobs') AS job,
         contractor_jsonb_values(job -> 'inputArtifacts') AS input
));
END;

CREATE OR REPLACE FUNCTION contractor_round_results_descriptor_bytes(
    key text, ref jsonb, digest text, media text, size bigint, provenance jsonb, display text
) RETURNS bigint
LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
SELECT octet_length(replace(replace(replace(replace(replace(
    jsonb_strip_nulls(jsonb_build_object(
      'logical_key',key,'classification','accepted-attachment',
      'artifact',jsonb_build_object('ref',ref,'digest',digest,'mediaType',media,'sizeBytes',size),
      'kind',provenance->'evidence'->>'kind','evidence_id',provenance->'evidence'->>'id',
      'content_member_id',provenance->'evidence'->>'content_member_id','display_ref',display,
      'original',CASE WHEN provenance->'evidence'->'artifact' IS NOT NULL
        AND provenance->'evidence'->'artifact'<>'null'::jsonb
        THEN jsonb_build_object('ref',provenance->'evidence'->'artifact',
          'digest',digest,'mediaType',media,'sizeBytes',size) END
    ))::text,'<','<'),'>','>'),'&','&'),U&'\2028',' '),U&'\2029',' '));
END;

-- CHECK constraints.
CREATE OR REPLACE FUNCTION contractor_finding_group_key(project_id text, algorithm text, value text) RETURNS text
LANGUAGE sql IMMUTABLE STRICT
BEGIN ATOMIC
SELECT encode(sha256(convert_to('[' || to_json('contractor/finding-group/v1'::text)::text || ',' ||
  to_json(project_id)::text || ',' || to_json(algorithm)::text || ',' || to_json(value)::text || ']', 'UTF8')), 'hex');
END;

CREATE OR REPLACE FUNCTION contractor_valid_audit_input_materialization(progress json) RETURNS boolean
LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
SELECT COALESCE(
 octet_length(progress::text)<=262144 AND jsonb_typeof(value)='object'
 AND value ?& ARRAY['identity','revision','complete','deadlineAt','checkpoint','refs']
 AND value-ARRAY['identity','revision','complete','deadlineAt','checkpoint','refs']='{}'::jsonb
 AND jsonb_typeof(value->'identity')='string'
 AND value->>'identity' ~ '^sha256:[0-9a-f]{64}$'
 AND jsonb_typeof(value->'revision')='number'
 AND (value->>'revision')::bigint BETWEEN 1 AND 4294967295
 AND jsonb_typeof(value->'complete')='boolean'
 AND jsonb_typeof(value->'deadlineAt')='string'
 AND jsonb_typeof(value->'checkpoint')='object'
 AND jsonb_typeof(value->'refs')='array'
 AND jsonb_array_length(value->'refs')<=1024, false)
 FROM (SELECT progress::jsonb AS value) AS parsed;
END;

CREATE OR REPLACE FUNCTION contractor_valid_model_selection(default_model text, roles jsonb) RETURNS boolean
LANGUAGE sql IMMUTABLE PARALLEL SAFE
BEGIN ATOMIC
    SELECT CASE WHEN jsonb_typeof(roles) IS DISTINCT FROM 'object' THEN false ELSE
        default_model ~ '^[a-z0-9][a-z0-9.-]{0,74}$'
        AND NOT EXISTS (
            SELECT 1 FROM jsonb_each(roles) AS r(key, value)
            WHERE key NOT IN ('planner', 'worker', 'summarizer', 'audit_completion_worker')
               OR jsonb_typeof(value) IS DISTINCT FROM 'string'
               OR (value #>> '{}') !~ '^[a-z0-9][a-z0-9.-]{0,74}$'
        ) END;
END;

CREATE OR REPLACE FUNCTION contractor_valid_run_cancellation(payload jsonb) RETURNS boolean
LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
    SELECT COALESCE(
        jsonb_typeof(payload) = 'object'
        AND payload->>'code' = 'user_cancelled'
        AND jsonb_typeof(payload->'requestedAt') = 'string'
        AND btrim(payload->>'requestedAt') <> ''
        AND (
            NOT (payload ? 'requestedBy')
            OR jsonb_typeof(payload->'requestedBy') = 'string'
                AND btrim(payload->>'requestedBy') <> ''
        )
        AND (
            NOT (payload ? 'reason')
            OR jsonb_typeof(payload->'reason') = 'string'
                AND btrim(payload->>'reason') <> ''
        ),
        false
    );
END;

CREATE OR REPLACE FUNCTION contractor_valid_run_metadata_labels(labels jsonb) RETURNS boolean
LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
    SELECT jsonb_typeof(labels) = 'object'
       AND (SELECT count(*) FROM jsonb_object_keys(labels)) <= 32
       AND NOT EXISTS (
           SELECT 1 FROM jsonb_each(labels) AS entry
           WHERE jsonb_typeof(entry.value) <> 'string'
              OR octet_length(entry.key) NOT BETWEEN 1 AND 63
              OR entry.key !~ '^[a-z][a-z0-9]*([._-][a-z0-9]+)*$'
              OR entry.key ~ '^contractor\.'
              OR octet_length(entry.value #>> '{}') > 256
       );
END;

CREATE OR REPLACE FUNCTION contractor_valid_runtime_agent_labels(candidate text[]) RETURNS boolean
LANGUAGE sql IMMUTABLE STRICT
BEGIN ATOMIC
    SELECT cardinality(candidate) <= 32
       AND candidate = ARRAY(
           SELECT value
           FROM unnest(candidate) AS value
           ORDER BY value COLLATE "C"
       )
       AND NOT EXISTS (
           SELECT 1
           FROM unnest(candidate) AS value
           WHERE value !~ '^[a-z][a-z0-9_-]*$'
              OR length(value) > 63
              OR value = 'default'
       )
       AND cardinality(candidate) = (
           SELECT count(DISTINCT value) FROM unnest(candidate) AS value
       );
END;

CREATE OR REPLACE FUNCTION contractor_valid_stage_result(payload jsonb) RETURNS boolean
LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
    SELECT COALESCE(
        jsonb_typeof(payload) = 'object'
        AND jsonb_typeof(payload->'apiVersion') = 'string'
        AND btrim(payload->>'apiVersion') <> ''
        AND payload->>'outcome' IN ('succeeded', 'failed')
        AND jsonb_typeof(payload->'summary') = 'string'
        AND btrim(payload->>'summary') <> ''
        AND jsonb_typeof(payload->'artifacts') = 'object'
        AND (NOT (payload ? 'summarized') OR jsonb_typeof(payload->'summarized') = 'boolean')
        AND (
            (COALESCE(payload->'summarized' = 'true'::jsonb, false)
                AND payload->>'outcome' = 'failed'
                AND payload->'error'->>'code' = 'worker_summarized'
                AND payload->'error'->'retryable' = 'false'::jsonb)
            OR
            (NOT COALESCE(payload->'summarized' = 'true'::jsonb, false)
                AND payload->'error'->>'code' IS DISTINCT FROM 'worker_summarized')
        )
        AND (
            NOT (payload ? 'summaryArtifact')
            OR (
                payload->'summarized' = 'true'::jsonb
                AND jsonb_typeof(payload->'summaryArtifact') = 'object'
                AND jsonb_typeof(payload->'summaryArtifact'->'namespace') = 'string'
                AND btrim(payload->'summaryArtifact'->>'namespace') <> ''
                AND jsonb_typeof(payload->'summaryArtifact'->'name') = 'string'
                AND btrim(payload->'summaryArtifact'->>'name') <> ''
                AND jsonb_typeof(payload->'summaryArtifact'->'revision') = 'string'
                AND btrim(payload->'summaryArtifact'->>'revision') <> ''
            )
        )
        AND (
            (payload->>'outcome' = 'succeeded'
                AND (NOT (payload ? 'error') OR payload->'error' = 'null'::jsonb))
            OR
            (payload->>'outcome' = 'failed'
                AND jsonb_typeof(payload->'error') = 'object'
                AND jsonb_typeof(payload->'error'->'code') = 'string'
                AND btrim(payload->'error'->>'code') <> ''
                AND jsonb_typeof(payload->'error'->'message') = 'string'
                AND btrim(payload->'error'->>'message') <> ''
                AND jsonb_typeof(payload->'error'->'retryable') = 'boolean')
        ),
        false
    );
END;

CREATE OR REPLACE FUNCTION contractor_valid_stage_termination(payload jsonb) RETURNS boolean
LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
    SELECT COALESCE(
        jsonb_typeof(payload) = 'object'
        AND jsonb_typeof(payload->'outcome') = 'string'
        AND payload->>'outcome' IN ('cancelled', 'interrupted')
        AND jsonb_typeof(payload->'code') = 'string'
        AND btrim(payload->>'code') <> ''
        AND jsonb_typeof(payload->'message') = 'string'
        AND btrim(payload->>'message') <> ''
        AND jsonb_typeof(payload->'retryable') = 'boolean'
        AND jsonb_typeof(payload->'phase') = 'string'
        AND payload->>'phase' IN ('preparing', 'running')
        AND jsonb_typeof(payload->'occurredAt') = 'string'
        AND btrim(payload->>'occurredAt') <> '',
        false
    );
END;

CREATE OR REPLACE FUNCTION contractor_valid_pinned_runtime_config(
    candidate jsonb, expected_label text, explicit boolean
) RETURNS boolean
LANGUAGE sql IMMUTABLE STRICT
BEGIN ATOMIC
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
       AND candidate->'config'->>'digest' ~ '^sha256:[0-9a-f]{64}$';
END;

-- The Run snapshot validator was PL/pgSQL and called the pinned-config
-- validator by name. As a SQL-standard body it keeps that call bound. CASE
-- arms run in order, so jsonb_object_keys and the array functions see only
-- the JSON types they accept, where the PL/pgSQL exception handler used to
-- turn their errors into false. Each credential list is strictly ascending:
-- every identifier is greater than the one before it, the first than ''.
CREATE OR REPLACE FUNCTION contractor_valid_run_runtime_snapshot(candidate jsonb, labels text[]) RETURNS boolean
LANGUAGE sql IMMUTABLE STRICT
BEGIN ATOMIC
SELECT CASE
    WHEN jsonb_typeof(candidate) <> 'object' THEN false
    WHEN (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(candidate) AS key)
         IS DISTINCT FROM ARRAY['default', 'labels', 'llmCredentialIds', 'runtimeCredentialIds']::text[]
        THEN false
    WHEN jsonb_typeof(candidate->'labels') <> 'array'
      OR jsonb_typeof(candidate->'llmCredentialIds') <> 'array'
      OR jsonb_typeof(candidate->'runtimeCredentialIds') <> 'array'
        THEN false
    ELSE contractor_valid_pinned_runtime_config(candidate->'default', 'default', false) IS TRUE
     AND jsonb_array_length(candidate->'labels') = cardinality(labels)
     AND NOT EXISTS (
         SELECT 1
         FROM jsonb_array_elements(candidate->'labels') WITH ORDINALITY AS item(value, position)
         WHERE contractor_valid_pinned_runtime_config(item.value, labels[item.position], true) IS NOT TRUE
     )
     AND NOT EXISTS (
         SELECT 1
         FROM (
             SELECT identifier, lag(identifier, 1, '') OVER (PARTITION BY list ORDER BY position) AS previous
             FROM (
                 SELECT 'llm' AS list, credential.value #>> '{}' AS identifier, credential.position
                 FROM jsonb_array_elements(candidate->'llmCredentialIds')
                     WITH ORDINALITY AS credential(value, position)
                 UNION ALL
                 SELECT 'runtime', credential.value #>> '{}', credential.position
                 FROM jsonb_array_elements(candidate->'runtimeCredentialIds')
                     WITH ORDINALITY AS credential(value, position)
             ) AS credentials
         ) AS ordered
         WHERE identifier IS NULL
            OR identifier !~ '^[a-z][a-z0-9_-]*$'
            OR length(identifier) > 128
            OR identifier <= previous
     )
END;
END;

-- PL/pgSQL validators that name only built-in objects.
ALTER FUNCTION contractor_valid_allocation_runtime_configuration(jsonb) SET search_path = pg_catalog;
ALTER FUNCTION contractor_valid_placement_labels(text[]) SET search_path = pg_catalog;
