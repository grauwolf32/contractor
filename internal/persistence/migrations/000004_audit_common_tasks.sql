-- Common accepted work replaces the item/Round execution pipeline. Task
-- documents are canonical JSON owned by the Audit service; outputs remain
-- ordinary Artifacts. The final cutover removes the retired relations below.
ALTER TABLE audits
    ADD COLUMN task_count integer NOT NULL DEFAULT 0 CHECK(task_count >= 0 AND task_count <= max_items_total),
    ADD COLUMN task_iteration integer NOT NULL DEFAULT 1 CHECK(task_iteration >= 1 AND task_iteration <= max_rounds),
    ADD COLUMN iteration_task_count integer NOT NULL DEFAULT 0 CHECK(iteration_task_count >= 0 AND iteration_task_count <= max_items_per_round),
    ADD COLUMN initial_work_offset integer NOT NULL DEFAULT 0 CHECK(initial_work_offset >= 0 AND initial_work_offset <= 10000);

CREATE TABLE audit_tasks (
    task_id text PRIMARY KEY CHECK(task_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'),
    audit_id text NOT NULL REFERENCES audits(audit_id) ON DELETE CASCADE,
    source_key text NOT NULL CHECK(source_key ~ '^sha256:[0-9a-f]{64}$'),
    source json NOT NULL CHECK(json_typeof(source)='object' AND octet_length(source::text) <= 4096),
    document json NOT NULL CHECK(json_typeof(document)='object' AND octet_length(document::text) <= 1048576),
    workflow_alias text GENERATED ALWAYS AS (document->>'workflow') STORED NOT NULL,
    batch_digest text NOT NULL CHECK(batch_digest ~ '^sha256:[0-9a-f]{64}$'),
    ordinal integer NOT NULL CHECK(ordinal >= 1 AND ordinal <= 100000),
    iteration integer NOT NULL CHECK(iteration >= 1 AND iteration <= 32),
    state text NOT NULL DEFAULT 'ready' CHECK(state IN ('ready','selected','settled')),
    attempts integer NOT NULL DEFAULT 0 CHECK(attempts >= 0 AND attempts <= 10),
    last_execution_id text,
    result json CHECK(result IS NULL OR (json_typeof(result)='object' AND octet_length(result::text) <= 4096)),
    gap_code text CHECK(gap_code IS NULL OR (btrim(gap_code) <> '' AND octet_length(gap_code) <= 128)),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    metadata_bytes integer GENERATED ALWAYS AS
        (8192 + octet_length(document::text) + octet_length(source::text)) STORED,
    UNIQUE(audit_id, source_key),
    UNIQUE(audit_id, ordinal),
    UNIQUE(task_id, audit_id),
    CHECK(document->>'schema'='contractor.audit.task.v1'),
    CHECK(document->>'audit_id'=audit_id AND document->>'task_id'=task_id),
    CHECK(state='settled' OR (result IS NULL AND gap_code IS NULL))
);
CREATE INDEX audit_tasks_state_idx ON audit_tasks(audit_id, state, ordinal);
CREATE INDEX audit_tasks_ready_workflow_idx ON audit_tasks(audit_id,workflow_alias,ordinal)
    INCLUDE(task_id,metadata_bytes) WHERE state='ready';
CREATE TRIGGER audit_tasks_protect_identity
    BEFORE UPDATE OF task_id,audit_id,source_key,source,document,batch_digest,ordinal,iteration
    ON audit_tasks FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_history();

ALTER TABLE audit_executions
    ADD COLUMN prepared_work_source json
        CHECK(prepared_work_source IS NULL OR (json_typeof(prepared_work_source)='object' AND octet_length(prepared_work_source::text) <= 4096)),
    ADD COLUMN prepared_work_offset integer NOT NULL DEFAULT 0 CHECK(prepared_work_offset >= 0 AND prepared_work_offset <= 256),
    ADD COLUMN prepared_work_count integer CHECK(prepared_work_count >= 0 AND prepared_work_count <= 256),
    ADD CONSTRAINT audit_executions_prepared_work_shape CHECK
        ((prepared_work_source IS NULL AND prepared_work_count IS NULL AND prepared_work_offset=0)
         OR (prepared_work_source IS NOT NULL AND prepared_work_count IS NOT NULL AND prepared_work_offset <= prepared_work_count));

CREATE FUNCTION contractor_protect_audit_prepared_source() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.prepared_work_source IS NOT NULL AND (
        NEW.prepared_work_source IS NULL OR
        OLD.prepared_work_source::jsonb IS DISTINCT FROM NEW.prepared_work_source::jsonb OR
        OLD.prepared_work_count IS DISTINCT FROM NEW.prepared_work_count OR
        NEW.prepared_work_offset < OLD.prepared_work_offset OR
        (OLD.state='collected' AND NEW.prepared_work_offset <> OLD.prepared_work_offset)
    ) THEN
        RAISE EXCEPTION 'prepared source and committed progress are immutable' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER audit_execution_protect_prepared_source
    BEFORE UPDATE OF prepared_work_source,prepared_work_count,prepared_work_offset
    ON audit_executions FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_prepared_source();

-- During the caller cutover the shared execution/attempt relations also hold
-- common tasks. These rows have no retired Round, manifest or task ZIP.
ALTER TABLE audit_executions
    ALTER COLUMN manifest_ref DROP NOT NULL,
    ALTER COLUMN manifest_digest DROP NOT NULL,
    DROP CONSTRAINT audit_executions_role_check,
    ADD CONSTRAINT audit_executions_role_check CHECK(role IN ('task','prepare','discovery','check','assessment')),
    DROP CONSTRAINT audit_executions_role_shape,
    ADD CONSTRAINT audit_executions_role_shape CHECK
        ((role='task' AND round_id IS NULL AND role_attempt IS NULL AND manifest_ref IS NULL AND manifest_digest IS NULL)
         OR (role='check' AND round_id IS NOT NULL AND role_attempt IS NULL AND manifest_ref IS NOT NULL AND manifest_digest IS NOT NULL)
         OR (role IN ('prepare','discovery','assessment') AND role_attempt IS NOT NULL AND manifest_ref IS NOT NULL AND manifest_digest IS NOT NULL));
ALTER TABLE audit_execution_items
    ADD COLUMN task_id text,
    ADD CONSTRAINT audit_execution_items_task_fk FOREIGN KEY(task_id,audit_id) REFERENCES audit_tasks(task_id,audit_id) ON DELETE CASCADE,
    ALTER COLUMN round_id DROP NOT NULL,
    ALTER COLUMN item_id DROP NOT NULL,
    ALTER COLUMN task_ref DROP NOT NULL,
    ALTER COLUMN task_digest DROP NOT NULL,
    ADD CONSTRAINT audit_execution_items_task_shape CHECK
        ((task_id IS NOT NULL AND round_id IS NULL AND item_id IS NULL AND task_ref IS NULL AND task_digest IS NULL AND input_refs='[]'::jsonb)
         OR (task_id IS NULL AND round_id IS NOT NULL AND item_id IS NOT NULL AND task_ref IS NOT NULL AND task_digest IS NOT NULL));
CREATE UNIQUE INDEX audit_execution_items_task_attempt_idx ON audit_execution_items(task_id,item_attempt) WHERE task_id IS NOT NULL;
ALTER TABLE audit_tasks ADD CONSTRAINT audit_tasks_last_execution_fk FOREIGN KEY(last_execution_id,audit_id) REFERENCES audit_executions(execution_id,audit_id) DEFERRABLE INITIALLY DEFERRED;
CREATE INDEX audit_tasks_last_execution_idx ON audit_tasks(last_execution_id) WHERE last_execution_id IS NOT NULL;
CREATE TRIGGER audit_task_execution_protect_identity
    BEFORE UPDATE OF execution_id,audit_id,role,workflow_role,submission_key,request_digest,lane_class
    ON audit_executions FOR EACH ROW WHEN (OLD.role='task' OR NEW.role='task')
    EXECUTE FUNCTION contractor_protect_immutable_history();
CREATE TRIGGER audit_task_attempt_protect_identity
    BEFORE UPDATE OF execution_item_id,execution_id,audit_id,task_id,batch_ordinal,item_attempt,input_refs
    ON audit_execution_items FOR EACH ROW WHEN (OLD.task_id IS NOT NULL OR NEW.task_id IS NOT NULL)
    EXECUTE FUNCTION contractor_protect_immutable_history();
CREATE TRIGGER audit_task_protect_settled_result
    BEFORE UPDATE OF state,attempts,last_execution_id,result,gap_code
    ON audit_tasks FOR EACH ROW WHEN (OLD.state='settled')
    EXECUTE FUNCTION contractor_protect_immutable_history();
