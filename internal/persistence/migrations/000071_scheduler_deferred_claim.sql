-- A deferred running Run must yield a claim to pending work. The marker lasts
-- until a non-deferred claim release; a wall-clock backoff could expire before
-- another owner's pending Run is polled and recreate starvation.
ALTER TABLE workflow_runs ADD COLUMN scheduler_deferred boolean NOT NULL DEFAULT false;
