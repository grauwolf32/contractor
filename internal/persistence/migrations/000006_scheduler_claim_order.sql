-- The Scheduler claim and tick read through indexes (V370-020).
--
-- The claim ranked candidates by a CASE over state and the deferral window,
-- an expression no index delivers, so every claim sorted every unclaimed
-- active Run. scheduler_claim_tier stores the state part of that rank, and
-- workflow_runs_claim_idx delivers Runs in claim order (SCH-R121): cancelling
-- first, then running and waiting, then pending Runs and Runs whose Skill
-- initialization is pending, each by updated_at, created_at and run_id. NULL
-- means not claimable. A Run deferred within its window ranks behind all of
-- them (SCH-R123); workflow_runs_deferred_claim_idx delivers those in
-- updated_at order for the claim's second probe.
--
-- Neither index holds scheduler_claim_expires_at, so a claim renewal that
-- writes only the lease expiry can be a heap-only update. The two indexes
-- replace workflow_runs_admission_idx and
-- workflow_runs_skill_initialization_scheduler_idx, which served only the
-- sorting claim.
ALTER TABLE workflow_runs
    ADD COLUMN scheduler_claim_tier smallint GENERATED ALWAYS AS (
        CASE
            WHEN state = 'cancelling' THEN 0
            WHEN state IN ('running', 'waiting') THEN 1
            WHEN state = 'pending'
              OR (state = 'initializing' AND state_reason_code = 'skill_initialization_pending') THEN 2
        END
    ) STORED;

DROP INDEX workflow_runs_admission_idx;
DROP INDEX workflow_runs_skill_initialization_scheduler_idx;

CREATE INDEX workflow_runs_claim_idx
    ON workflow_runs USING btree (scheduler_claim_tier, updated_at, created_at, run_id)
    WHERE scheduler_claim_tier IS NOT NULL;

CREATE INDEX workflow_runs_deferred_claim_idx
    ON workflow_runs USING btree (updated_at, created_at, run_id)
    WHERE scheduler_deferred AND scheduler_claim_tier > 0;

-- A Scheduler tick reads the one active StageExecution of its Run (SCH-R114),
-- not the Run's whole history of up to 1,024 attempts (LIM-RUN-2); the claim
-- probes the same rows for a paused owner's pending Run.
CREATE INDEX stage_executions_run_active_idx
    ON stage_executions USING btree (run_id, created_at, stage_execution_id)
    WHERE state IN ('preparing', 'running', 'finalizing', 'aborting');
