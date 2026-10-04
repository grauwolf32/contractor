-- An experiment's workspace Project dependency was an exact copy of
-- eval_submissions.execution_project_id, written in the same transaction and
-- kept after the workspace is deleted. Readers and triggers use the submission.
UPDATE eval_submissions AS submission
   SET execution_project_id = dependency.project_id
  FROM eval_project_dependencies AS dependency
 WHERE dependency.experiment_id = submission.experiment_id
   AND dependency.member_id = submission.member_id
   AND submission.execution_project_id IS NULL;

DROP TABLE eval_project_dependencies;

CREATE OR REPLACE FUNCTION contractor_eval_fence_project() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.lifecycle_state = 'deleting' AND OLD.lifecycle_state = 'active' THEN
        UPDATE eval_experiments
           SET deletion_requested_at = CASE
                   WHEN project_id = NEW.project_id THEN COALESCE(deletion_requested_at, clock_timestamp())
                   ELSE deletion_requested_at
               END,
               state = 'cancelling',
               revision = revision + 1,
               updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
         WHERE project_id = NEW.project_id
            OR (state NOT IN ('finished', 'cancelled') AND experiment_id IN (
                SELECT experiment_id
                  FROM eval_submissions
                 WHERE execution_project_id = NEW.project_id
            ));
    END IF;
    RETURN NEW;
END; $$;

CREATE OR REPLACE FUNCTION contractor_eval_retain_deleted_execution() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE bound_experiment text; bound_member text; kind text; execution text; outcome text; never_started boolean;
BEGIN
    IF TG_TABLE_NAME='workflow_runs' THEN
        SELECT op.experiment_id,op.member_id INTO bound_experiment,bound_member
        FROM eval_suboperations op JOIN eval_experiments e USING(experiment_id)
        WHERE op.kind='run-create' AND op.operation_key=OLD.request_idempotency_key
          AND e.owner_id=OLD.owner_id AND e.project_id=OLD.project_id AND op.request_sha256=OLD.request_digest;
        kind:='run';execution:=OLD.run_id;outcome:=OLD.state;never_started:=false;
    ELSE
        SELECT op.experiment_id,op.member_id INTO bound_experiment,bound_member
        FROM audit_idempotency i JOIN eval_suboperations op ON op.operation_key=i.idempotency_key
        JOIN eval_experiments e ON e.experiment_id=op.experiment_id
        JOIN eval_submissions s ON s.experiment_id=op.experiment_id AND s.member_id=op.member_id
        WHERE i.audit_id=OLD.audit_id AND i.operation='audit.create' AND op.kind='audit-create'
          AND e.owner_id=OLD.owner_id AND i.owner_id=OLD.owner_id AND s.execution_project_id=OLD.project_id AND op.request_sha256=i.request_digest;
        kind:='audit';execution:=OLD.audit_id;never_started:=(OLD.started_at IS NULL);
        -- Audit deletion can replace its old terminal state with deleting.
        -- Drain is confirmed by ordinary purge, but quality remains unknown.
        outcome:=CASE WHEN OLD.state IN ('completed','cancelled','failed') THEN OLD.state ELSE 'unknown' END;
    END IF;
    IF bound_experiment IS NOT NULL THEN
        INSERT INTO eval_execution_tombstones(experiment_id,member_id,execution_kind,execution_id,terminal_state,never_started)
        VALUES(bound_experiment,bound_member,kind,execution,outcome,never_started);
        UPDATE eval_experiments SET view_generation=view_generation+1,
            updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond')
        WHERE experiment_id=bound_experiment;
    END IF;
    RETURN OLD;
END; $$;
