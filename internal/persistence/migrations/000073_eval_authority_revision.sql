-- The public experiment revision is the owner's If-Match authority token.
-- Coordinator observations still advance updated_at, but must not consume it.
CREATE OR REPLACE FUNCTION contractor_eval_protect_experiment() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF ROW(NEW.experiment_id,NEW.owner_id,NEW.project_id,NEW.portable_id,NEW.control_mode,NEW.created_at)
       IS DISTINCT FROM ROW(OLD.experiment_id,OLD.owner_id,OLD.project_id,OLD.portable_id,OLD.control_mode,OLD.created_at)
       OR NEW.revision NOT IN (OLD.revision, OLD.revision + 1)
       OR (NEW.revision = OLD.revision
           AND ROW(NEW.name,NEW.draft,NEW.dataset_id,NEW.dataset_revision,NEW.max_in_flight,
                   NEW.wall_ms,NEW.token_limit,NEW.deletion_requested_at)
               IS DISTINCT FROM
               ROW(OLD.name,OLD.draft,OLD.dataset_id,OLD.dataset_revision,OLD.max_in_flight,
                   OLD.wall_ms,OLD.token_limit,OLD.deletion_requested_at))
       OR NEW.updated_at <= OLD.updated_at
       OR NEW.observed_tokens < OLD.observed_tokens
       OR (OLD.started_at IS NOT NULL AND ROW(NEW.started_at,NEW.deadline_at) IS DISTINCT FROM ROW(OLD.started_at,OLD.deadline_at))
       OR (OLD.deletion_requested_at IS NOT NULL AND NEW.deletion_requested_at IS DISTINCT FROM OLD.deletion_requested_at)
       OR (EXISTS (SELECT 1 FROM eval_frozen_plans WHERE experiment_id=OLD.experiment_id)
           AND ROW(NEW.draft,NEW.dataset_id,NEW.dataset_revision,NEW.expected_count,NEW.max_in_flight,NEW.wall_ms,NEW.token_limit)
               IS DISTINCT FROM ROW(OLD.draft,OLD.dataset_id,OLD.dataset_revision,OLD.expected_count,OLD.max_in_flight,OLD.wall_ms,OLD.token_limit))
    THEN RAISE EXCEPTION 'Evaluation revision or immutable fields changed' USING ERRCODE = '23514'; END IF;
    RETURN NEW;
END; $$;

-- A deleted execution invalidates views, not the owner's lifecycle authority.
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
        JOIN eval_project_dependencies d ON d.experiment_id=op.experiment_id AND d.member_id=op.member_id
        WHERE i.audit_id=OLD.audit_id AND i.operation='audit.create' AND op.kind='audit-create'
          AND e.owner_id=OLD.owner_id AND i.owner_id=OLD.owner_id AND d.project_id=OLD.project_id AND op.request_sha256=i.request_digest;
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
