-- Fire the row triggers of stage_allocations, audits, artifact_bindings,
-- audit_executions and the Eval retention trigger of workflow_runs only when
-- what they guard, check or project can change (V409-006, PER-R22).
--
-- Each condition below is false only where the trigger has nothing to
-- decide: its function would return without effect, or, for the Eval output
-- projection, would mark a member dirty although nothing Eval reads of the
-- binding changed. No accepted write, rejection or lock changes. No
-- foreign key of these tables cascades an UPDATE, no column a guard lists is
-- generated, and no BEFORE UPDATE trigger of these tables assigns a column,
-- so the SET list is the only way a listed value changes; a trigger that
-- writes one of these tables (the report-received mark on stage_allocations,
-- the finding and event counters on audits) runs an UPDATE whose SET list
-- the same lists filter. TestRowTriggerInventory pins every trigger of the
-- five tables, RequireColumnScopedGuards holds each column list to its
-- function, and the store tests count the triggers each operation calls.
--
-- stage_allocations. A release attempt, a completed release and the
-- report-received mark called all three guards, and the completion check
-- joined stage_executions and workflow_runs on every one of them.
-- * The identity and performance policy guards fire only for an UPDATE that
--   assigns a column they compare (UPDATE OF).
-- * The completion check keeps its parent lookup on INSERT and, on UPDATE,
--   fires only for completion_contract and only rejects its change. The
--   lookup repeated on UPDATE compared an unchanged contract with authority
--   that cannot change under an allocation: the allocation's Stage and agent
--   are immutable (stage_allocations_protect_identity), the Stage's name and
--   Run too (stage_executions_protect_immutable), and the Run's completion is
--   fixed by the transaction that creates the Run: its commit checks the
--   completion against the Audit's profile binding
--   (workflow_runs_require_audit_completion), a set completion never changes
--   and an unset one may only be set while the Run is initializing
--   (contractor_protect_audit_completion), before any Stage exists (RUN-R26).
--
-- audits. Every event append, finding admission and revision bump updates
-- the Audit row, and the preparation guard probed audit_executions for a
-- prepare execution on each of them unless the Audit was preparing or taking
-- its inventory.
-- * The guard fires only for an UPDATE that assigns the phase or a baseline
--   column; audit_id only selects the Audit's executions.
-- * It compares the baseline and the phase before probing, so an assignment
--   that changes neither probes nothing. The probe still decides every
--   baseline change and every not-started phase, as before.
--
-- artifact_bindings.
-- * Project admission (DEL-R40) keeps firing on every INSERT and UPDATE of a
--   Project binding, without a column list: every such UPDATE is an artifact
--   write (a revision CAS or a freeze) that a deleting Project must reject at
--   the write itself, and the decision depends on projects and audits, not on
--   the binding's columns. Its function returned at once for any other scope,
--   which the WHEN condition now skips.
-- * The identity guard fires only for an UPDATE that assigns a column it
--   compares.
-- * The Eval output projection fires only for Run outputs: on INSERT and
--   DELETE of such a binding, and on UPDATE only when its revision or its
--   frozen flag changed. A WHEN condition of a DELETE trigger cannot read
--   NEW and one of an INSERT trigger cannot read OLD, so the trigger is
--   split per event; the INSERT trigger keeps the old name.
--
-- audit_executions.
-- * The collection guard fires only for an UPDATE that assigns a column of
--   the receipt it protects; the preparation check validates every INSERT
--   and, on UPDATE, fires only for the identity, the outputs and the
--   collection receipt it compares. The other columns it reads only qualify
--   the assignment of the receipt.
-- * The deferred Run association check fires on every INSERT and, on UPDATE,
--   only when a column it reads changed. The Run columns it reads are
--   immutable (workflow_runs_protect_immutable), a Run that references the
--   execution cannot be created without the execution pointing back to it
--   (workflow_runs_require_complete_audit_binding), and the only Run delete
--   marks run_deleted_at in its own transaction (rundeletion), so the check no
--   longer repeats at commit for a terminal observation or a collection.
-- * The Eval Audit child projection keeps firing on every change: each Audit
--   execution write changes the state, Run, terminal outcome, collection or
--   Run deletion that Eval inventory reads, and the function's work is one
--   indexed probe of eval_submissions.
--
-- workflow_runs. The Eval retention trigger matches a deleted Run to a
-- run-create suboperation by its Project, request key and request digest;
-- it now fires only when the Run has all three, so deleting a standalone
-- Run, or an Audit child Run without a request key, calls nothing.

DROP TRIGGER stage_allocations_protect_identity ON stage_allocations;
CREATE TRIGGER stage_allocations_protect_identity
    BEFORE UPDATE OF allocation_id, stage_execution_id, logical_agent_name, namespace,
        agent_template_ref, worker_runtime_ref, runtime_agent_instance_id, runtime_agent_id,
        runtime_agent_label_revision, runtime_configuration_schema_version, runtime_configuration,
        created_at
    ON stage_allocations
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_allocation_identity();

DROP TRIGGER stage_allocations_protect_performance_policy ON stage_allocations;
CREATE TRIGGER stage_allocations_protect_performance_policy
    BEFORE UPDATE OF performance_collection_policy ON stage_allocations
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_allocation_performance_policy();

CREATE OR REPLACE FUNCTION contractor_require_allocation_completion() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE pinned jsonb; stage_name text;
BEGIN
 IF TG_OP = 'UPDATE' THEN
  IF NEW.completion_contract IS DISTINCT FROM OLD.completion_contract THEN
   RAISE EXCEPTION 'Allocation completion is immutable' USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
 END IF;
 SELECT r.audit_completion, s.stage_name INTO pinned, stage_name
 FROM stage_executions s JOIN workflow_runs r USING (run_id)
 WHERE s.stage_execution_id = NEW.stage_execution_id;
 IF pinned ->> 'stage' = stage_name AND pinned ->> 'agent' = NEW.logical_agent_name THEN
  IF NEW.completion_contract IS DISTINCT FROM pinned -> 'contract' THEN
   RAISE EXCEPTION 'Allocation completion differs from Run authority' USING ERRCODE = '23514';
  END IF;
 ELSIF NEW.completion_contract IS NOT NULL THEN
  RAISE EXCEPTION 'Allocation has no completion authority' USING ERRCODE = '23514';
 END IF;
 RETURN NEW;
END;
$$;

DROP TRIGGER stage_allocations_require_completion ON stage_allocations;
CREATE TRIGGER stage_allocations_require_completion
    BEFORE INSERT OR UPDATE OF completion_contract ON stage_allocations
    FOR EACH ROW EXECUTE FUNCTION contractor_require_allocation_completion();

CREATE OR REPLACE FUNCTION contractor_protect_audit_preparation_baseline() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    baseline_changed boolean := ROW(NEW.baseline_snapshot, NEW.profile_snapshot, NEW.input_selection,
            NEW.profile_name, NEW.profile_version, NEW.profile_digest, NEW.owner_id, NEW.project_id)
        IS DISTINCT FROM ROW(OLD.baseline_snapshot, OLD.profile_snapshot, OLD.input_selection,
            OLD.profile_name, OLD.profile_version, OLD.profile_digest, OLD.owner_id, OLD.project_id);
BEGIN
    -- Only a changed baseline or a not-started phase is rejected once
    -- preparation began, so only they need to know whether it did.
    IF baseline_changed OR NEW.phase = 'not-started' THEN
        IF OLD.phase IN ('preparing', 'inventory') OR EXISTS (
            SELECT 1 FROM audit_executions WHERE audit_id = OLD.audit_id AND role = 'prepare'
        ) THEN
            IF baseline_changed THEN
                RAISE EXCEPTION 'Audit preparation baseline is immutable' USING ERRCODE = '23514';
            END IF;
            RAISE EXCEPTION 'Audit preparation cannot return to not-started' USING ERRCODE = '23514';
        END IF;
    END IF;
    IF NEW.phase IN ('preparing', 'inventory') THEN
        IF NEW.baseline_snapshot->>'schema' IS DISTINCT FROM 'contractor.audit.baseline.v1'
           OR jsonb_typeof(NEW.baseline_snapshot->'inputs') IS DISTINCT FROM 'object'
           OR NEW.baseline_snapshot ? 'inventory'
           OR NOT EXISTS (SELECT 1 FROM jsonb_each(NEW.profile_snapshot->'workflows') AS role WHERE role.value->>'kind' = 'prepare')
        THEN RAISE EXCEPTION 'Audit preparation baseline or profile is invalid' USING ERRCODE = '23514'; END IF;
    END IF;
    IF NEW.phase = 'inventory' AND OLD.phase IS DISTINCT FROM 'inventory' AND EXISTS (
        SELECT 1 FROM jsonb_each(NEW.profile_snapshot->'workflows') AS role
         WHERE role.value->>'kind' = 'prepare' AND NOT EXISTS (
             SELECT 1 FROM audit_executions AS execution
              WHERE execution.audit_id = NEW.audit_id AND execution.role = 'prepare'
                AND execution.workflow_role = role.key AND execution.collection_disposition = 'accepted-result'
         )
    ) THEN RAISE EXCEPTION 'Audit preparation is incomplete' USING ERRCODE = '23514'; END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER audits_protect_preparation_baseline ON audits;
CREATE TRIGGER audits_protect_preparation_baseline
    BEFORE UPDATE OF phase, baseline_snapshot, profile_snapshot, input_selection, profile_name,
        profile_version, profile_digest, owner_id, project_id
    ON audits
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_preparation_baseline();

DROP TRIGGER artifact_bindings_project_admission ON artifact_bindings;
CREATE TRIGGER artifact_bindings_project_admission
    BEFORE INSERT OR UPDATE ON artifact_bindings
    FOR EACH ROW WHEN (NEW.scope_kind = 'project')
    EXECUTE FUNCTION contractor_guard_project_artifact_binding_admission();

DROP TRIGGER artifact_bindings_protect_identity ON artifact_bindings;
CREATE TRIGGER artifact_bindings_protect_identity
    BEFORE UPDATE OF scope_kind, scope_id, namespace, name, created_at, current_revision, frozen
    ON artifact_bindings
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_binding();

DROP TRIGGER eval_outputs_projection ON artifact_bindings;
CREATE TRIGGER eval_outputs_projection
    AFTER INSERT ON artifact_bindings
    FOR EACH ROW WHEN (NEW.scope_kind = 'run' AND NEW.namespace = 'outputs')
    EXECUTE FUNCTION contractor_eval_output_changed();
CREATE TRIGGER eval_outputs_projection_on_update
    AFTER UPDATE ON artifact_bindings
    FOR EACH ROW WHEN (NEW.scope_kind = 'run' AND NEW.namespace = 'outputs'
        AND (OLD.current_revision IS DISTINCT FROM NEW.current_revision OR OLD.frozen IS DISTINCT FROM NEW.frozen))
    EXECUTE FUNCTION contractor_eval_output_changed();
CREATE TRIGGER eval_outputs_projection_on_delete
    AFTER DELETE ON artifact_bindings
    FOR EACH ROW WHEN (OLD.scope_kind = 'run' AND OLD.namespace = 'outputs')
    EXECUTE FUNCTION contractor_eval_output_changed();

DROP TRIGGER audit_executions_protect_collection ON audit_executions;
CREATE TRIGGER audit_executions_protect_collection
    BEFORE UPDATE OF execution_id, audit_id, run_id, terminal_outcome, terminal_run_generation,
        terminal_run_sequence, collection_receipt_id, collection_disposition,
        collection_source_output_ref, collection_source_output_digest, collection_retained_refs,
        collection_error_code, collection_request_digest, collected_at
    ON audit_executions
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_execution_collection();

DROP TRIGGER audit_executions_validate_preparation ON audit_executions;
CREATE TRIGGER audit_executions_validate_preparation
    BEFORE INSERT OR UPDATE OF execution_id, audit_id, round_id, role, workflow_role, role_attempt,
        manifest_ref, manifest_digest, submission_key, request_digest, preparation_snapshot,
        preparation_outputs, collection_receipt_id
    ON audit_executions
    FOR EACH ROW EXECUTE FUNCTION contractor_validate_audit_preparation_execution();

DROP TRIGGER audit_executions_require_matching_run ON audit_executions;
CREATE CONSTRAINT TRIGGER audit_executions_require_matching_run
    AFTER INSERT ON audit_executions
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION contractor_require_matching_audit_run();
CREATE CONSTRAINT TRIGGER audit_executions_require_matching_run_on_update
    AFTER UPDATE ON audit_executions
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW WHEN (OLD.run_id IS DISTINCT FROM NEW.run_id
        OR OLD.run_deleted_at IS DISTINCT FROM NEW.run_deleted_at
        OR OLD.execution_id IS DISTINCT FROM NEW.execution_id
        OR OLD.submission_key IS DISTINCT FROM NEW.submission_key)
    EXECUTE FUNCTION contractor_require_matching_audit_run();

DROP TRIGGER workflow_runs_eval_tombstone ON workflow_runs;
CREATE TRIGGER workflow_runs_eval_tombstone
    BEFORE DELETE ON workflow_runs
    FOR EACH ROW WHEN (OLD.project_id IS NOT NULL AND OLD.request_idempotency_key IS NOT NULL
        AND OLD.request_digest IS NOT NULL)
    EXECUTE FUNCTION contractor_eval_retain_deleted_execution();
