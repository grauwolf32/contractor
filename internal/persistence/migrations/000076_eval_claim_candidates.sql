-- The coordinator selects actionable experiments without scanning terminal
-- experiments retained for history. Paused experiments use a deadline range;
-- accepted commands and unpublished projections have independent indexes.
CREATE INDEX eval_experiments_claim_active_idx
    ON eval_experiments (experiment_id)
    WHERE state IN ('preparing', 'running', 'settling', 'pausing', 'cancelling');

CREATE INDEX eval_experiments_claim_paused_deadline_idx
    ON eval_experiments (deadline_at, experiment_id)
    WHERE state = 'paused';

CREATE INDEX eval_projection_queue_dirty_idx
    ON eval_projection_queue (experiment_id)
    WHERE revision <> published_revision;
