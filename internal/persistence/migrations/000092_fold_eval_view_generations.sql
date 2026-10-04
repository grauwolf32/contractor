-- Since publication prunes superseded generations, an experiment keeps one
-- view generation, so its header lives on the experiment's projection queue
-- row. Pair and chart rows reference the queue and stay immutable while they
-- belong to the current generation.
ALTER TABLE eval_projection_queue
    ADD COLUMN generation bigint CHECK (generation > 0),
    ADD COLUMN summary bytea,
    ADD COLUMN suites bytea,
    ADD COLUMN pins_verified boolean,
    ADD COLUMN content_sha256 text CHECK (content_sha256 ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN published_at timestamptz;

UPDATE eval_projection_queue AS queue
   SET generation = view.generation,
       summary = view.summary,
       suites = view.suites,
       pins_verified = view.pins_verified,
       content_sha256 = view.content_sha256,
       published_at = view.created_at
  FROM eval_view_generations AS view
 WHERE view.experiment_id = queue.experiment_id
   AND view.snapshot_id = queue.snapshot_id;
-- A snapshot without its generation row never resolved to a view.
UPDATE eval_projection_queue SET snapshot_id = NULL
 WHERE snapshot_id IS NOT NULL AND generation IS NULL;
ALTER TABLE eval_projection_queue
    ADD CONSTRAINT eval_projection_queue_view_shape CHECK (
        (snapshot_id IS NULL) = (generation IS NULL)
        AND (generation IS NULL) = (summary IS NULL)
        AND (generation IS NULL) = (suites IS NULL)
        AND (generation IS NULL) = (pins_verified IS NULL)
        AND (generation IS NULL) = (published_at IS NULL)
    );

-- Rows of a generation the queue no longer names cannot be read; the current
-- trigger already lets them go. Remove them before the foreign keys move.
DELETE FROM eval_view_pairs AS pair
 WHERE NOT EXISTS (
     SELECT 1 FROM eval_projection_queue AS queue
      WHERE queue.experiment_id = pair.experiment_id AND queue.generation = pair.generation
 );
DELETE FROM eval_view_charts AS chart
 WHERE NOT EXISTS (
     SELECT 1 FROM eval_projection_queue AS queue
      WHERE queue.experiment_id = chart.experiment_id AND queue.generation = chart.generation
 );

ALTER TABLE eval_view_pairs
    DROP CONSTRAINT eval_view_pairs_experiment_id_generation_fkey,
    ADD CONSTRAINT eval_view_pairs_experiment_id_fkey
        FOREIGN KEY (experiment_id) REFERENCES eval_projection_queue ON DELETE CASCADE;
ALTER TABLE eval_view_charts
    DROP CONSTRAINT eval_view_charts_experiment_id_generation_fkey,
    ADD CONSTRAINT eval_view_charts_experiment_id_fkey
        FOREIGN KEY (experiment_id) REFERENCES eval_projection_queue ON DELETE CASCADE;

DROP TABLE eval_view_generations;

CREATE OR REPLACE FUNCTION contractor_eval_view_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF current_setting('contractor.eval_purge', true) = 'on' THEN RETURN OLD; END IF;
        IF NOT EXISTS (
            SELECT 1 FROM eval_projection_queue q
            WHERE q.experiment_id = OLD.experiment_id AND q.generation = OLD.generation
        ) THEN RETURN OLD; END IF;
    END IF;
    RAISE EXCEPTION 'Evaluation record is immutable' USING ERRCODE = '23514';
END;
$$;
