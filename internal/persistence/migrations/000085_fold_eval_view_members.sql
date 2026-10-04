-- Every member is one side of exactly one pair, and the pair document embeds
-- both sides' member views, so member pages and report provenance read pairs.
-- Only each side's collection completeness was unique to eval_view_members.
ALTER TABLE eval_view_pairs
    ADD COLUMN complete_a boolean,
    ADD COLUMN complete_b boolean;

-- Retained pairs are immutable outside purge; backfill them once here. Publication
-- always wrote both member rows, so the fallback to incomplete never applies to
-- a published generation.
ALTER TABLE eval_view_pairs DISABLE TRIGGER eval_view_pairs_immutable;
UPDATE eval_view_pairs AS pair
   SET complete_a = COALESCE((
           SELECT side.collection_complete FROM eval_view_members AS side
            WHERE side.experiment_id = pair.experiment_id AND side.generation = pair.generation
              AND side.member_id = convert_from(pair.document, 'UTF8')::jsonb #>> '{a,member,memberId}'
       ), false),
       complete_b = COALESCE((
           SELECT side.collection_complete FROM eval_view_members AS side
            WHERE side.experiment_id = pair.experiment_id AND side.generation = pair.generation
              AND side.member_id = convert_from(pair.document, 'UTF8')::jsonb #>> '{b,member,memberId}'
       ), false);
ALTER TABLE eval_view_pairs ENABLE TRIGGER eval_view_pairs_immutable;

ALTER TABLE eval_view_pairs
    ALTER COLUMN complete_a SET NOT NULL,
    ALTER COLUMN complete_b SET NOT NULL;

DROP TABLE eval_view_members;
