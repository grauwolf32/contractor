-- Run metadata labels are immutable and created with the Run, so they live on
-- the Run row as one jsonb object. A GIN index serves the all-labels-match
-- filter through containment; the key, value and count rules are unchanged.
CREATE FUNCTION contractor_valid_run_metadata_labels(labels jsonb)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT jsonb_typeof(labels) = 'object'
       AND (SELECT count(*) FROM jsonb_object_keys(labels)) <= 32
       AND NOT EXISTS (
           SELECT 1 FROM jsonb_each(labels) AS entry
           WHERE jsonb_typeof(entry.value) <> 'string'
              OR octet_length(entry.key) NOT BETWEEN 1 AND 63
              OR entry.key !~ '^[a-z][a-z0-9]*([._-][a-z0-9]+)*$'
              OR entry.key ~ '^contractor\.'
              OR octet_length(entry.value #>> '{}') > 256
       )
$$;

ALTER TABLE workflow_runs
    ADD COLUMN metadata_labels jsonb NOT NULL DEFAULT '{}'::jsonb
        CHECK (contractor_valid_run_metadata_labels(metadata_labels));

-- Backfill without the Run lifecycle and projection triggers: no Run changes
-- state here, so no event or Eval invalidation may be emitted.
ALTER TABLE workflow_runs DISABLE TRIGGER USER;
UPDATE workflow_runs AS run
   SET metadata_labels = labels.value
  FROM (
      SELECT run_id, jsonb_object_agg(label_key, label_value) AS value
      FROM workflow_run_metadata_labels
      GROUP BY run_id
  ) AS labels
 WHERE labels.run_id = run.run_id;
ALTER TABLE workflow_runs ENABLE TRIGGER USER;

DROP TABLE workflow_run_metadata_labels;
DROP FUNCTION contractor_protect_workflow_run_metadata_label_immutable();

CREATE INDEX workflow_runs_metadata_labels_idx
    ON workflow_runs USING gin (metadata_labels jsonb_path_ops);

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
        OR NEW.metadata_labels IS DISTINCT FROM OLD.metadata_labels
        OR NEW.request_idempotency_key IS DISTINCT FROM OLD.request_idempotency_key
        OR NEW.request_digest IS DISTINCT FROM OLD.request_digest
        OR NEW.run_event_generation IS DISTINCT FROM OLD.run_event_generation
        OR NEW.runtime_labels IS DISTINCT FROM OLD.runtime_labels
        OR NEW.runtime_config_snapshot IS DISTINCT FROM OLD.runtime_config_snapshot
        OR NEW.project_http_target_snapshot IS DISTINCT FROM OLD.project_http_target_snapshot
        OR NEW.publication_mode IS DISTINCT FROM OLD.publication_mode
        OR NEW.audit_execution_id IS DISTINCT FROM OLD.audit_execution_id
        OR NEW.audit_submission_key IS DISTINCT FROM OLD.audit_submission_key
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
