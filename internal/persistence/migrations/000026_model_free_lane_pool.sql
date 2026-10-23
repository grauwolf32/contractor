-- Model-free Stages run in their own Scheduler lane pool (V400-001,
-- docs/proposals/0040-model-free-stage-lanes.md).
--
-- scheduler_settings.max_concurrent_tool_runs sizes the `tool` pool, as
-- max_concurrent_runs sizes the `model` pool. Both share the singleton's
-- revision, so an update must change at least one of them.
ALTER TABLE scheduler_settings
    ADD COLUMN max_concurrent_tool_runs integer NOT NULL DEFAULT 2,
    ADD CONSTRAINT scheduler_settings_max_concurrent_tool_runs_check
        CHECK (max_concurrent_tool_runs >= 1 AND max_concurrent_tool_runs <= 32);
ALTER TABLE scheduler_settings ALTER COLUMN max_concurrent_tool_runs DROP DEFAULT;

CREATE OR REPLACE FUNCTION contractor_protect_scheduler_settings() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Scheduler settings singleton cannot be deleted'
            USING ERRCODE = '23514';
    END IF;
    IF NEW.singleton IS DISTINCT FROM OLD.singleton
        OR (NEW.max_concurrent_runs IS NOT DISTINCT FROM OLD.max_concurrent_runs
            AND NEW.max_concurrent_tool_runs IS NOT DISTINCT FROM OLD.max_concurrent_tool_runs)
        OR NEW.revision IS DISTINCT FROM OLD.revision + 1
        OR NEW.updated_at <= OLD.updated_at
    THEN
        RAISE EXCEPTION 'Scheduler settings update must be one semantic revision'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

-- workflow_runs.scheduler_lane_class is the pool that progresses the Run's
-- next StageExecution: its active one, otherwise its entry Stage. The Server
-- writes it when it creates the Run and in the statement that creates each
-- StageExecution. Existing Runs take the class of that Stage's spec here;
-- this expression is the Server's rule (resolved.StageLaneClass) as of this
-- migration: `tool` when the Planner is passthrough@1 or scan-plan@1 and
-- every Worker runs on tool@1.
ALTER TABLE workflow_runs ADD COLUMN scheduler_lane_class text;

WITH next_stage AS (
    SELECT run.run_id,
           COALESCE(
               (SELECT stage.stage_spec_snapshot
                  FROM stage_executions AS stage
                 WHERE stage.run_id = run.run_id
                   AND stage.state IN ('preparing', 'running', 'finalizing', 'aborting')
                 ORDER BY stage.created_at, stage.stage_execution_id
                 LIMIT 1),
               run.workflow_snapshot -> 'stages' -> (run.workflow_snapshot ->> 'entryStage')
           ) AS spec
      FROM workflow_runs AS run
)
UPDATE workflow_runs AS run
   SET scheduler_lane_class = CASE
           WHEN (next_stage.spec -> 'planner' ->> 'plannerId') || '@' || (next_stage.spec -> 'planner' ->> 'version')
                    IN ('passthrough@1', 'scan-plan@1')
            AND NOT EXISTS (
                    SELECT 1
                      FROM jsonb_each(COALESCE(next_stage.spec -> 'agents', '{}'::jsonb)) AS agent(name, binding)
                     WHERE agent.binding -> 'template' -> 'runtime'
                           IS DISTINCT FROM '{"runtimeId": "tool", "version": "1"}'::jsonb)
           THEN 'tool'
           ELSE 'model'
       END
  FROM next_stage
 WHERE next_stage.run_id = run.run_id;

ALTER TABLE workflow_runs
    ALTER COLUMN scheduler_lane_class SET NOT NULL,
    ADD CONSTRAINT workflow_runs_scheduler_lane_class_check
        CHECK (scheduler_lane_class IN ('model', 'tool'));

-- Each pool claims only its class, so the class leads both claim indexes and
-- a claim still reads candidates in claim order without sorting (SCH-R164).
DROP INDEX workflow_runs_claim_idx;
DROP INDEX workflow_runs_deferred_claim_idx;

CREATE INDEX workflow_runs_claim_idx
    ON workflow_runs USING btree (scheduler_lane_class, scheduler_claim_tier, updated_at, created_at, run_id)
    WHERE scheduler_claim_tier IS NOT NULL;

CREATE INDEX workflow_runs_deferred_claim_idx
    ON workflow_runs USING btree (scheduler_lane_class, updated_at, created_at, run_id)
    WHERE scheduler_deferred AND scheduler_claim_tier > 0;
