-- A view generation is a pure function of its selected member documents,
-- comparison policy, verified pins and selection revision. Their digest lets a
-- publication that changed none of them keep the current generation. Older
-- generations have no digest; their next publication replaces them.
ALTER TABLE eval_view_generations
    ADD COLUMN content_sha256 text CHECK (content_sha256 ~ '^sha256:[0-9a-f]{64}$');

-- Every reader resolves the queue's current snapshot and reads its rows in one
-- snapshot transaction, so no reader can reach a superseded generation. Such a
-- generation may be deleted; the current one and every update stay immutable
-- outside Purge.
CREATE FUNCTION contractor_eval_view_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF current_setting('contractor.eval_purge', true) = 'on' THEN RETURN OLD; END IF;
        IF NOT EXISTS (
            SELECT 1
            FROM eval_projection_queue q
            JOIN eval_view_generations v ON v.experiment_id = q.experiment_id AND v.snapshot_id = q.snapshot_id
            WHERE q.experiment_id = OLD.experiment_id AND v.generation = OLD.generation
        ) THEN RETURN OLD; END IF;
    END IF;
    RAISE EXCEPTION 'Evaluation record is immutable' USING ERRCODE = '23514';
END; $$;

DROP TRIGGER eval_views_immutable ON eval_view_generations;
CREATE TRIGGER eval_views_immutable BEFORE UPDATE OR DELETE ON eval_view_generations
FOR EACH ROW EXECUTE FUNCTION contractor_eval_view_immutable();
DROP TRIGGER eval_view_members_immutable ON eval_view_members;
CREATE TRIGGER eval_view_members_immutable BEFORE UPDATE OR DELETE ON eval_view_members
FOR EACH ROW EXECUTE FUNCTION contractor_eval_view_immutable();
DROP TRIGGER eval_view_pairs_immutable ON eval_view_pairs;
CREATE TRIGGER eval_view_pairs_immutable BEFORE UPDATE OR DELETE ON eval_view_pairs
FOR EACH ROW EXECUTE FUNCTION contractor_eval_view_immutable();
DROP TRIGGER eval_view_charts_immutable ON eval_view_charts;
CREATE TRIGGER eval_view_charts_immutable BEFORE UPDATE OR DELETE ON eval_view_charts
FOR EACH ROW EXECUTE FUNCTION contractor_eval_view_immutable();

-- Publication now prunes as it replaces; drop what earlier publications kept.
DELETE FROM eval_view_generations v
WHERE NOT EXISTS (
    SELECT 1
    FROM eval_projection_queue q
    WHERE q.experiment_id = v.experiment_id AND q.snapshot_id = v.snapshot_id
);
