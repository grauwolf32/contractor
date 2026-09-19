-- Exact portable closure and trusted preparation snapshots share private Eval
-- retention. None of these documents becomes a model-visible Artifact binding.
CREATE TABLE eval_plan_resources (
    experiment_id text NOT NULL REFERENCES eval_frozen_plans ON DELETE CASCADE,
    resource_path text NOT NULL CHECK (length(resource_path) BETWEEN 1 AND 1024),
    document_kind text NOT NULL,
    document bytea NOT NULL CHECK (octet_length(document) BETWEEN 2 AND 1048576),
    document_sha256 text GENERATED ALWAYS AS ('sha256:' || encode(sha256(document),'hex')) STORED,
    PRIMARY KEY (experiment_id,resource_path)
);
CREATE TRIGGER eval_resources_immutable BEFORE UPDATE OR DELETE ON eval_plan_resources
FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();
ALTER TABLE eval_submissions ADD COLUMN observed_tokens bigint NOT NULL DEFAULT 0 CHECK (observed_tokens>=0);
CREATE INDEX eval_submissions_reconcile_idx ON eval_submissions(experiment_id,updated_at,member_id) WHERE state IN ('intent','accepted');

-- Ordinary owner deletion can win before an Eval observes a terminal result.
-- Retain exact effect identity so recovery cannot resubmit the same member.
CREATE TABLE eval_execution_tombstones (
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    execution_kind text NOT NULL,
    execution_id text NOT NULL,
    terminal_state text NOT NULL,
    never_started boolean NOT NULL,
    deleted_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY(experiment_id,member_id),
    FOREIGN KEY(experiment_id,member_id) REFERENCES eval_members ON DELETE CASCADE
);
CREATE FUNCTION contractor_eval_retain_deleted_execution() RETURNS trigger LANGUAGE plpgsql AS $$
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
        UPDATE eval_experiments SET view_generation=view_generation+1,revision=revision+1,
            updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond')
        WHERE experiment_id=bound_experiment;
    END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workflow_runs_eval_tombstone BEFORE DELETE ON workflow_runs
FOR EACH ROW EXECUTE FUNCTION contractor_eval_retain_deleted_execution();
CREATE TRIGGER audits_eval_tombstone BEFORE DELETE ON audits
FOR EACH ROW EXECUTE FUNCTION contractor_eval_retain_deleted_execution();

CREATE TRIGGER eval_tombstones_immutable BEFORE UPDATE OR DELETE ON eval_execution_tombstones
FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();
