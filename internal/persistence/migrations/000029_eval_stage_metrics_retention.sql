-- Telemetry retention keeps the StageMetrics an Eval member reads (V404-001,
-- OBS-R43, EVM-R239).
--
-- contractor_eval_run_members is the one definition of the Eval members that
-- executed a Run: directly, or as a child Run of the Audit the member
-- executed. The projection trigger marks these members dirty when the Run's
-- metrics change, and retention cleanup renews an expired StageMetrics row
-- instead of deleting it while any of them exist.
CREATE FUNCTION contractor_eval_run_members(selected_run text)
    RETURNS TABLE (experiment_id text, member_id text)
    LANGUAGE sql STABLE AS $$
    SELECT s.experiment_id, s.member_id
      FROM eval_submissions s
     WHERE s.execution_kind = 'run' AND s.execution_id = selected_run
    UNION
    SELECT s.experiment_id, s.member_id
      FROM audit_executions x
      JOIN eval_submissions s ON s.execution_kind = 'audit' AND s.execution_id = x.audit_id
     WHERE x.run_id = selected_run
$$;

CREATE OR REPLACE FUNCTION contractor_eval_metrics_changed() RETURNS trigger
    LANGUAGE plpgsql AS $$
DECLARE
    r record;
    selected_run text;
BEGIN
    IF TG_TABLE_NAME = 'stage_executions' THEN
        selected_run := COALESCE(NEW.run_id, OLD.run_id);
    ELSE
        SELECT run_id INTO selected_run FROM stage_executions
        WHERE stage_execution_id = COALESCE(NEW.stage_execution_id, OLD.stage_execution_id);
    END IF;

    FOR r IN SELECT * FROM contractor_eval_run_members(selected_run) LOOP
        PERFORM contractor_eval_dirty(r.experiment_id, r.member_id);
    END LOOP;
    RETURN NULL;
END;
$$;

-- A renewal moves only expires_at, which no Eval member reads, so it must not
-- mark them for recollection.
DROP TRIGGER eval_metrics_projection ON stage_metrics;
CREATE TRIGGER eval_metrics_projection
    AFTER INSERT OR DELETE OR UPDATE OF metrics_schema_version, metrics, summary ON stage_metrics
    FOR EACH ROW EXECUTE FUNCTION contractor_eval_metrics_changed();
