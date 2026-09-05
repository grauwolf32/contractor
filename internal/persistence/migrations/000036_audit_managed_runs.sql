ALTER TABLE audit_executions
    ADD CONSTRAINT audit_executions_submission_identity
        UNIQUE (execution_id, submission_key);

ALTER TABLE workflow_runs
    ADD COLUMN publication_mode text NOT NULL DEFAULT 'ordinary'
        CHECK (publication_mode IN ('ordinary', 'audit-managed')),
    ADD COLUMN audit_execution_id text,
    ADD COLUMN audit_submission_key text,
    ADD CONSTRAINT workflow_runs_audit_authority_shape CHECK (
        (publication_mode = 'ordinary'
            AND audit_execution_id IS NULL
            AND audit_submission_key IS NULL)
        OR
        (publication_mode = 'audit-managed'
            AND project_id IS NOT NULL
            AND audit_execution_id IS NOT NULL
            AND audit_submission_key IS NOT NULL)
    );

-- V25-003 could already have produced bindings through its low-level store
-- tests or an experimental Controller. Promote those rows before the new
-- authority constraints become active; the previous immutability trigger does
-- not yet know about the three new columns.
UPDATE workflow_runs AS run
SET publication_mode = 'audit-managed',
    audit_execution_id = execution.execution_id,
    audit_submission_key = execution.submission_key
FROM audit_executions AS execution
WHERE execution.run_id = run.run_id;

ALTER TABLE workflow_runs
    ADD CONSTRAINT workflow_runs_audit_execution_unique UNIQUE (audit_execution_id),
    ADD CONSTRAINT workflow_runs_audit_submission_unique UNIQUE (audit_submission_key),
    ADD CONSTRAINT workflow_runs_audit_submission_fkey
        FOREIGN KEY (audit_execution_id, audit_submission_key)
        REFERENCES audit_executions(execution_id, submission_key)
        ON DELETE RESTRICT;

CREATE INDEX workflow_runs_publication_mode_idx
    ON workflow_runs (publication_mode, run_id)
    WHERE publication_mode = 'audit-managed';

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

CREATE OR REPLACE FUNCTION contractor_require_complete_audit_run_binding()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.publication_mode = 'audit-managed' AND NOT EXISTS (
        SELECT 1
        FROM audit_executions AS execution
        WHERE execution.execution_id = NEW.audit_execution_id
          AND execution.submission_key = NEW.audit_submission_key
          AND execution.run_id = NEW.run_id
    ) THEN
        RAISE EXCEPTION 'Audit-managed WorkflowRun association is incomplete'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE CONSTRAINT TRIGGER workflow_runs_require_complete_audit_binding
AFTER INSERT OR UPDATE ON workflow_runs
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION contractor_require_complete_audit_run_binding();

CREATE OR REPLACE FUNCTION contractor_require_matching_audit_run()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.run_id IS NOT NULL AND NOT EXISTS (
        SELECT 1
        FROM workflow_runs AS run
        WHERE run.run_id = NEW.run_id
          AND run.audit_execution_id = NEW.execution_id
          AND run.audit_submission_key = NEW.submission_key
          AND run.publication_mode = 'audit-managed'
    ) THEN
        RAISE EXCEPTION 'AuditExecution Run association is not authoritative'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE CONSTRAINT TRIGGER audit_executions_require_matching_run
AFTER INSERT OR UPDATE ON audit_executions
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION contractor_require_matching_audit_run();
