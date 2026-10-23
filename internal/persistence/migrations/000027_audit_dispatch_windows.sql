-- An Audit paces its child Runs per Scheduler lane class (V400-004,
-- docs/proposals/0040-model-free-stage-lanes.md, AUX-R3).
--
-- audit_executions.lane_class is the class the execution counts against:
-- `tool` when every Stage of its role's pinned Workflow is `tool`, otherwise
-- `model`. The Server writes it with the intent and the terminal observation
-- reads it, so an execution leaves the count it entered. Existing executions
-- take the class of their role's Workflow in the Audit's profile snapshot;
-- the Stage expression is the Server's rule (resolved.StageLaneClass) as of
-- this migration, as in 000026: `tool` when the Planner is passthrough@1 or
-- scan-plan@1 and every Worker runs on tool@1.
ALTER TABLE audit_executions ADD COLUMN lane_class text;

-- The deferred Run check would leave the backfill's trigger events pending,
-- and PostgreSQL refuses the ALTER below while any are; check them as the
-- rows change instead.
SET CONSTRAINTS audit_executions_require_matching_run IMMEDIATE;

UPDATE audit_executions AS execution
   SET lane_class = CASE
           -- A role missing from the snapshot leaves the class NULL, which
           -- fails the migration below instead of defaulting it.
           WHEN audit.profile_snapshot -> 'workflows' -> execution.workflow_role -> 'workflow' -> 'stages' IS NULL
           THEN NULL
           WHEN NOT EXISTS (
                   SELECT 1
                     FROM jsonb_each(audit.profile_snapshot -> 'workflows' -> execution.workflow_role -> 'workflow' -> 'stages')
                          AS stage(name, spec)
                    WHERE (stage.spec -> 'planner' ->> 'plannerId') || '@' || (stage.spec -> 'planner' ->> 'version')
                              NOT IN ('passthrough@1', 'scan-plan@1')
                       OR EXISTS (
                              SELECT 1
                                FROM jsonb_each(COALESCE(stage.spec -> 'agents', '{}'::jsonb)) AS agent(name, binding)
                               WHERE agent.binding -> 'template' -> 'runtime'
                                     IS DISTINCT FROM '{"runtimeId": "tool", "version": "1"}'::jsonb))
           THEN 'tool'
           ELSE 'model'
       END
  FROM audits AS audit
 WHERE audit.audit_id = execution.audit_id;

ALTER TABLE audit_executions
    ALTER COLUMN lane_class SET NOT NULL,
    ADD CONSTRAINT audit_executions_lane_class_check CHECK (lane_class IN ('model', 'tool'));

-- audits.outstanding_tool_run_count is the `tool` part of
-- outstanding_run_count: executions that hold a dispatch reservation, from
-- the intent until the terminal observation (states intent and submitted).
-- outstanding_run_count stays the total, so the `model` part is the
-- difference.
ALTER TABLE audits ADD COLUMN outstanding_tool_run_count integer DEFAULT 0 NOT NULL;

UPDATE audits AS audit
   SET outstanding_tool_run_count = outstanding.count
  FROM (SELECT execution.audit_id, count(*)::integer AS count
          FROM audit_executions AS execution
         WHERE execution.lane_class = 'tool' AND execution.state IN ('intent', 'submitted')
         GROUP BY execution.audit_id) AS outstanding
 WHERE outstanding.audit_id = audit.audit_id;

ALTER TABLE audits
    ADD CONSTRAINT audits_outstanding_tool_run_count_check
        CHECK (outstanding_tool_run_count >= 0 AND outstanding_tool_run_count <= outstanding_run_count);

-- The reconcile snapshot reads the first ready items of each Workflow role, so
-- a full window of one lane class cannot hide the other class's items. This
-- index serves both the walk over the distinct roles and each role's read in
-- item order.
CREATE INDEX audit_items_ready_role_idx
    ON audit_items (audit_id, round_id, workflow_role, ordinal, item_id)
 WHERE state = 'ready';
