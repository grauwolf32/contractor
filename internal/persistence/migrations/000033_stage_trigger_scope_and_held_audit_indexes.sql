-- Scope the StageExecution row triggers to what they guard, and index the
-- held-Audit credential checks (V407-025).
--
-- StageExecution triggers (PER-R22). The immutability guard, the manual
-- continuation guard and the lifecycle capture fired on every UPDATE of
-- stage_executions, so an admission (admitted_at) or a counted output
-- publication failure (output_publication_failures) ran both guards and
-- queued the capture, and each returned without effect. Now:
--
-- * The immutability guard fires only for an UPDATE that assigns a column its
--   function compares (UPDATE OF). No BEFORE trigger of stage_executions
--   assigns a column, so the SET list is the only way such a value changes;
--   TestStageExecutionGuardsFireOnTheColumnsTheyCompare keeps the list equal
--   to the columns the function compares.
-- * The manual continuation guard keeps validating every INSERT and, on
--   UPDATE, fires only when resume_source_execution_id is assigned, the one
--   column its UPDATE branch compares.
-- * The lifecycle capture fires on every INSERT and, on UPDATE, only when the
--   state changed, the first test of its UPDATE branch. A WHEN condition of
--   an INSERT trigger cannot read OLD, so it is split into an INSERT and an
--   UPDATE trigger; the INSERT trigger keeps the old name. The function and
--   its StartPlanner exception are unchanged.
--
-- Each condition is false only where the function returns without effect.

DROP TRIGGER stage_executions_protect_immutable ON stage_executions;
CREATE TRIGGER stage_executions_protect_immutable
    BEFORE UPDATE OF stage_execution_id, run_id, stage_name, attempt, previous_execution_id,
        execution_config_variant, escalation_ordinal, stage_spec_schema_version,
        stage_spec_snapshot, stage_context_schema_version, stage_context_snapshot, created_at,
        planner_session_id, planner_invocation_id, candidate_stage_result,
        candidate_result_schema_version, accepted_stage_result, accepted_result_schema_version,
        stage_termination, termination_schema_version, finalization_id, finalization_deadline,
        abort_id, abort_deadline
    ON stage_executions
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_execution_immutable();

DROP TRIGGER stage_executions_manual_continuation ON stage_executions;
CREATE TRIGGER stage_executions_manual_continuation
    BEFORE INSERT OR UPDATE OF resume_source_execution_id ON stage_executions
    FOR EACH ROW EXECUTE FUNCTION contractor_validate_manual_stage_execution();

DROP TRIGGER stage_executions_capture_lifecycle_event ON stage_executions;
CREATE TRIGGER stage_executions_capture_lifecycle_event
    AFTER INSERT ON stage_executions
    FOR EACH ROW EXECUTE FUNCTION contractor_capture_stage_execution_lifecycle_event();
CREATE TRIGGER stage_executions_capture_lifecycle_event_on_update
    AFTER UPDATE ON stage_executions
    FOR EACH ROW WHEN (OLD.state IS DISTINCT FROM NEW.state)
    EXECUTE FUNCTION contractor_capture_stage_execution_lifecycle_event();

-- Held-Audit credential checks (AUD-R16). Managed credential deletion lists
-- the held Audits whose baseline names the credential in its LLM or Runtime
-- credential member (CredentialHolds.listHeldAuditIDsByCredential). Without
-- an index the check read every Audit row and detoasted the baseline of
-- every held Audit, although only the held Audits naming the credential can
-- match. Each index holds the member of held Audits only: an Audit enters it
-- when its start holds the baseline credentials and leaves it when the hold
-- is released, so its size follows the open Audits, not the Audit history,
-- and a lookup reads only the baselines that name the credential. The member
-- names are runtimeconfig.LLMCredentialIDsMember and
-- runtimeconfig.RuntimeCredentialIDsMember;
-- TestPostgresHeldAuditCredentialChecksReadOnlyMatchingHeldAudits fails when
-- the query stops using them.
CREATE INDEX audits_held_llm_credentials_idx
    ON audits USING gin ((baseline_snapshot -> 'llmCredentialIds'))
    WHERE hold_state = 'held';
CREATE INDEX audits_held_runtime_credentials_idx
    ON audits USING gin ((baseline_snapshot -> 'runtimeCredentialIds'))
    WHERE hold_state = 'held';
