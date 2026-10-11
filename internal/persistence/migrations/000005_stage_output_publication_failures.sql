-- Bound replays of a terminal commit that keeps failing on publication
-- (V368-035).
--
-- A transient database failure while publishing a Project output aborts the
-- terminal Run transaction, and finalization recovery replays it (PRJ-R58).
-- The aborted transaction can record nothing, so the Scheduler counts each
-- such failure here in its own statement, against the finalizing
-- StageExecution whose commit is replayed. Once LIM-PRJ-4 failures are counted
-- the next replay records a failed publication instead of trying again. A new
-- StageExecution, such as a manual continuation, starts from zero.
ALTER TABLE stage_executions
    ADD COLUMN output_publication_failures integer DEFAULT 0 NOT NULL,
    ADD CONSTRAINT stage_executions_output_publication_failures_check
        CHECK (output_publication_failures >= 0);
