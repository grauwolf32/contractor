-- The reviewed pre-release cutover archived old Audit facts as opaque Project
-- Artifacts. Refuse surviving operational history rather than discard or infer
-- it. Common Tasks and exact Artifact bytes are untouched in every lifecycle state.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM audit_items)
       OR EXISTS (SELECT 1 FROM audit_rounds)
       OR EXISTS (SELECT 1 FROM audit_round_result_pages)
       OR EXISTS (SELECT 1 FROM audit_report_pages)
       OR EXISTS (SELECT 1 FROM audits WHERE current_round_id IS NOT NULL OR phase IN ('preparing','inventory'))
       OR EXISTS (SELECT 1 FROM audit_executions WHERE role <> 'task'
           OR round_id IS NOT NULL OR role_attempt IS NOT NULL
           OR manifest_ref IS NOT NULL OR manifest_digest IS NOT NULL
           OR preparation_snapshot IS NOT NULL OR preparation_outputs <> '[]'::jsonb)
       OR EXISTS (SELECT 1 FROM audit_execution_items WHERE task_id IS NULL
           OR round_id IS NOT NULL OR item_id IS NOT NULL OR task_ref IS NOT NULL
           OR task_digest IS NOT NULL OR input_refs <> '[]'::jsonb OR finding_sources <> '[]'::jsonb)
    THEN
        RAISE EXCEPTION 'unarchived Audit pipeline records block retirement' USING ERRCODE='23514';
    END IF;
END;
$$;

DROP TRIGGER audits_protect_preparation_baseline ON audits;
DROP TRIGGER audits_discard_report_pages ON audits;
DROP TRIGGER audit_executions_validate_preparation ON audit_executions;
DROP TRIGGER audit_task_execution_protect_identity ON audit_executions;
DROP TRIGGER audit_task_attempt_protect_identity ON audit_execution_items;
ALTER TABLE audit_items DROP CONSTRAINT audit_items_last_execution_item_fkey;
ALTER TABLE audits DROP COLUMN current_round_id, DROP COLUMN phase;
ALTER TABLE audit_execution_items
    DROP COLUMN round_id, DROP COLUMN item_id, DROP COLUMN task_ref,
    DROP COLUMN task_digest, DROP COLUMN input_refs, DROP COLUMN finding_sources,
    ALTER COLUMN task_id SET NOT NULL,
    ADD CONSTRAINT audit_execution_items_common_execution_fk
        FOREIGN KEY(execution_id,audit_id) REFERENCES audit_executions(execution_id,audit_id) ON DELETE CASCADE,
    ADD CONSTRAINT audit_execution_items_common_member UNIQUE(execution_id,task_id);

ALTER TABLE audit_executions
    DROP COLUMN round_id, DROP COLUMN role, DROP COLUMN role_attempt,
    DROP COLUMN manifest_ref, DROP COLUMN manifest_digest,
    DROP COLUMN preparation_snapshot, DROP COLUMN preparation_outputs;

DROP TABLE audit_round_result_pages;
DROP TABLE audit_report_pages;
DROP TABLE audit_items;
DROP TABLE audit_rounds;
DROP INDEX audit_admission_decisions_idx;
DROP INDEX audit_admission_pages_idx;

DROP FUNCTION contractor_validate_audit_dependencies();
DROP FUNCTION contractor_hold_audit_dependencies();
DROP FUNCTION contractor_audit_dependency_closure_valid(text,jsonb,jsonb);
DROP FUNCTION contractor_audit_dependency_origin_valid(text,jsonb);
DROP FUNCTION contractor_audit_dependency_holds(text,jsonb);
DROP FUNCTION contractor_audit_dependency_namespace(text);
DROP FUNCTION contractor_protect_audit_item_source();
DROP FUNCTION contractor_protect_audit_preparation_baseline();
DROP FUNCTION contractor_validate_audit_preparation_execution();
DROP FUNCTION contractor_protect_prepared_audit_item();
DROP FUNCTION contractor_protect_prepared_audit_round();
DROP FUNCTION contractor_protect_round_results_page();
DROP FUNCTION contractor_protect_round_results_root();
DROP FUNCTION contractor_discard_audit_report_pages();

CREATE TRIGGER audit_task_execution_protect_identity
    BEFORE UPDATE OF execution_id,audit_id,workflow_role,submission_key,request_digest,lane_class
    ON audit_executions FOR EACH ROW
    WHEN (NEW.execution_id IS DISTINCT FROM OLD.execution_id OR NEW.audit_id IS DISTINCT FROM OLD.audit_id
       OR NEW.workflow_role IS DISTINCT FROM OLD.workflow_role OR NEW.submission_key IS DISTINCT FROM OLD.submission_key
       OR NEW.request_digest IS DISTINCT FROM OLD.request_digest OR NEW.lane_class IS DISTINCT FROM OLD.lane_class)
    EXECUTE FUNCTION contractor_protect_immutable_history();
CREATE TRIGGER audit_task_attempt_protect_identity
    BEFORE UPDATE OF execution_item_id,execution_id,audit_id,task_id,batch_ordinal,item_attempt
    ON audit_execution_items FOR EACH ROW
    WHEN (NEW.execution_item_id IS DISTINCT FROM OLD.execution_item_id OR NEW.execution_id IS DISTINCT FROM OLD.execution_id
       OR NEW.audit_id IS DISTINCT FROM OLD.audit_id OR NEW.task_id IS DISTINCT FROM OLD.task_id
       OR NEW.batch_ordinal IS DISTINCT FROM OLD.batch_ordinal OR NEW.item_attempt IS DISTINCT FROM OLD.item_attempt)
    EXECUTE FUNCTION contractor_protect_immutable_history();

-- Inventory pages must seek within one Audit after the retired Round index
-- disappears; execution_id first would scan interleaved executions of others.
CREATE INDEX audit_executions_inventory_idx ON audit_executions(audit_id,execution_id);
