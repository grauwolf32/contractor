-- Audit deadlines are optional. Pausing freezes the remaining admission time;
-- a continuation retains previous reports under their historical artifact refs.
ALTER TABLE audits ADD COLUMN paused_at timestamptz;
ALTER TABLE audits ADD COLUMN continuation_count integer NOT NULL DEFAULT 0
    CHECK (continuation_count >= 0);

UPDATE audits SET paused_at = updated_at WHERE state = 'paused';

ALTER TABLE audits DROP CONSTRAINT audits_time_shape;
ALTER TABLE audits ADD CONSTRAINT audits_time_shape CHECK (
    (state = 'draft' AND started_at IS NULL AND finished_at IS NULL)
    OR (state IN ('active', 'waiting_review', 'paused', 'finalizing', 'cancelling')
        AND started_at IS NOT NULL AND finished_at IS NULL)
    OR (state IN ('completed', 'cancelled', 'failed')
        AND started_at IS NOT NULL AND finished_at IS NOT NULL)
    OR state = 'deleting'
);
