-- Credential sets use the Go writer's ASCII byte order, independently of
-- database locale. Retain the installed function identity and every snapshot.
CREATE OR REPLACE FUNCTION contractor_valid_run_runtime_snapshot(candidate jsonb, labels text[]) RETURNS boolean
    LANGUAGE sql IMMUTABLE STRICT
    BEGIN ATOMIC
 SELECT
         CASE
             WHEN (jsonb_typeof(candidate) <> 'object'::text) THEN false
             WHEN (( SELECT array_agg(key.key ORDER BY key.key COLLATE "C") AS array_agg
                FROM jsonb_object_keys(contractor_valid_run_runtime_snapshot.candidate) key(key)) IS DISTINCT FROM ARRAY['default'::text, 'labels'::text, 'llmCredentialIds'::text, 'runtimeCredentialIds'::text]) THEN false
             WHEN ((jsonb_typeof((candidate -> 'labels'::text)) <> 'array'::text) OR (jsonb_typeof((candidate -> 'llmCredentialIds'::text)) <> 'array'::text) OR (jsonb_typeof((candidate -> 'runtimeCredentialIds'::text)) <> 'array'::text)) THEN false
             ELSE ((contractor_valid_pinned_runtime_config((candidate -> 'default'::text), 'default'::text, false) IS TRUE) AND (jsonb_array_length((candidate -> 'labels'::text)) = cardinality(labels)) AND (NOT (EXISTS ( SELECT 1
                FROM jsonb_array_elements((contractor_valid_run_runtime_snapshot.candidate -> 'labels'::text)) WITH ORDINALITY item(value, "position")
               WHERE (contractor_valid_pinned_runtime_config(item.value, (contractor_valid_run_runtime_snapshot.labels)[item."position"], true) IS NOT TRUE)))) AND (NOT (EXISTS ( SELECT 1
                FROM ( SELECT credentials.identifier,
                         lag(credentials.identifier, 1, ''::text) OVER (PARTITION BY credentials.list ORDER BY credentials."position") AS previous
                        FROM ( SELECT 'llm'::text AS list,
                                 (credential.value #>> '{}'::text[]) AS identifier,
                                 credential."position"
                                FROM jsonb_array_elements((contractor_valid_run_runtime_snapshot.candidate -> 'llmCredentialIds'::text)) WITH ORDINALITY credential(value, "position")
                             UNION ALL
                              SELECT 'runtime'::text AS text,
                                 (credential.value #>> '{}'::text[]),
                                 credential."position"
                                FROM jsonb_array_elements((contractor_valid_run_runtime_snapshot.candidate -> 'runtimeCredentialIds'::text)) WITH ORDINALITY credential(value, "position")) credentials) ordered
               WHERE ((ordered.identifier IS NULL) OR (ordered.identifier COLLATE "C" !~ '^[a-z][a-z0-9_-]*$'::text) OR (length(ordered.identifier) > 128) OR (ordered.identifier COLLATE "C" <= ordered.previous COLLATE "C"))))))
         END AS "case";
END;
