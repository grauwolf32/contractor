-- Fire the WorkflowRun row triggers only when what they guard or check can
-- change (V370-028).
--
-- A Run event append updates only workflow_runs.next_run_event_sequence, yet
-- every append called the three BEFORE UPDATE guards, queued the lifecycle
-- capture and queued both deferred Audit checks for commit, and each of them
-- returned without effect. Now:
--
-- * A guard fires only for an UPDATE that assigns a column it compares
--   (UPDATE OF). No BEFORE trigger of workflow_runs assigns one of those
--   columns, so the SET list is the only way such a value changes;
--   TestWorkflowRunGuardsFireOnTheColumnsTheyCompare keeps each list equal
--   to the columns its function compares.
-- * The lifecycle capture and the deferred Audit completion check fire on
--   every INSERT, and on an UPDATE only when the state, or the completion,
--   changed. A WHEN condition of an INSERT trigger cannot read OLD, so each
--   is split into an INSERT and an UPDATE trigger; the INSERT trigger keeps
--   the old name.
-- * The deferred Audit binding check fires only for audit-managed Runs; its
--   function does nothing for any other publication mode.
--
-- Each condition is false only where the function returns without effect,
-- with one exception: the completion check no longer repeats on an update
-- that keeps an already validated completion, because the completion cannot
-- change once set and the profile binding it is checked against is
-- immutable (PER-R21, AWC-R7).

DROP TRIGGER workflow_runs_protect_audit_completion ON workflow_runs;
CREATE TRIGGER workflow_runs_protect_audit_completion
    BEFORE UPDATE OF audit_completion ON workflow_runs
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_completion();

DROP TRIGGER workflow_runs_protect_immutable ON workflow_runs;
CREATE TRIGGER workflow_runs_protect_immutable
    BEFORE UPDATE OF run_id, owner_id, project_id, workflow_name, workflow_version,
        workflow_schema_version, workflow_snapshot, parameters, metadata_labels,
        request_idempotency_key, request_digest, run_event_generation, runtime_labels,
        runtime_config_snapshot, project_http_target_snapshot, publication_mode,
        audit_execution_id, audit_submission_key, created_at, cancellation_schema_version,
        run_cancellation
    ON workflow_runs
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_workflow_run_immutable();

DROP TRIGGER workflow_runs_protect_skill_snapshot ON workflow_runs;
CREATE TRIGGER workflow_runs_protect_skill_snapshot
    BEFORE UPDATE OF skill_snapshot ON workflow_runs
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_workflow_run_skill_snapshot();

DROP TRIGGER workflow_runs_capture_lifecycle_event ON workflow_runs;
CREATE TRIGGER workflow_runs_capture_lifecycle_event
    AFTER INSERT ON workflow_runs
    FOR EACH ROW EXECUTE FUNCTION contractor_capture_workflow_run_lifecycle_event();
CREATE TRIGGER workflow_runs_capture_lifecycle_event_on_update
    AFTER UPDATE ON workflow_runs
    FOR EACH ROW WHEN (OLD.state IS DISTINCT FROM NEW.state)
    EXECUTE FUNCTION contractor_capture_workflow_run_lifecycle_event();

DROP TRIGGER workflow_runs_require_audit_completion ON workflow_runs;
CREATE CONSTRAINT TRIGGER workflow_runs_require_audit_completion
    AFTER INSERT ON workflow_runs
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION contractor_require_audit_completion();
CREATE CONSTRAINT TRIGGER workflow_runs_require_audit_completion_on_update
    AFTER UPDATE ON workflow_runs
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW WHEN (OLD.audit_completion IS DISTINCT FROM NEW.audit_completion)
    EXECUTE FUNCTION contractor_require_audit_completion();

DROP TRIGGER workflow_runs_require_complete_audit_binding ON workflow_runs;
CREATE CONSTRAINT TRIGGER workflow_runs_require_complete_audit_binding
    AFTER INSERT OR UPDATE ON workflow_runs
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW WHEN (NEW.publication_mode = 'audit-managed')
    EXECUTE FUNCTION contractor_require_complete_audit_run_binding();
