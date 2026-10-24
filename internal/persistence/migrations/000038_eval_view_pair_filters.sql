-- A view pair row stores, for each side, the member fields that member page
-- filters and sorts by (V409-014, EVM-R245).
--
-- A member page used to unpack both sides of every pair of the generation
-- from the pair document, as jsonb, before its filter, sort and limit; it
-- now reads these columns, and only the pairs of the page's rows send their
-- document. For side a and side b of the pair: the member's frozen ordinal
-- (eval_members.ordinal), its variant and eligibility, the state of its
-- execution (NULL when the member view has no execution), its assessment and
-- whether its evidence conflicts. The view publication derives them from the
-- same member views it encodes into the pair document and inserts them with
-- it (projectComparison), and eval_view_pairs_immutable keeps them as
-- immutable as the rest of the row.
ALTER TABLE eval_view_pairs
    ADD COLUMN member_ordinal_a integer,
    ADD COLUMN member_ordinal_b integer,
    ADD COLUMN variant_a text,
    ADD COLUMN variant_b text,
    ADD COLUMN eligibility_a text,
    ADD COLUMN eligibility_b text,
    ADD COLUMN execution_state_a text,
    ADD COLUMN execution_state_b text,
    ADD COLUMN assessment_a text,
    ADD COLUMN assessment_b text,
    ADD COLUMN conflicting_a boolean,
    ADD COLUMN conflicting_b boolean;

-- PostgreSQL cannot process JSON that holds a NUL escape (\u0000) anywhere,
-- with json as with jsonb, yet a member view may carry one in a free string
-- such as a reason. The backfill therefore turns every \u0000 of the stored
-- text into \u0001 before the jsonb cast. The result is still valid JSON:
-- the six characters are a valid escape after the change wherever they were
-- one before, and literal characters wherever their backslash was escaped.
-- Only free strings can change, because member IDs, variant IDs and the
-- eligibility, execution state and assessment enums contain no backslash
-- (portable common schema), so every value the backfill reads is unchanged.
-- The server encodes the pair document, which escapes no other code point
-- that jsonb rejects.
CREATE FUNCTION eval_view_pair_sides(stored bytea) RETURNS jsonb
    LANGUAGE sql IMMUTABLE STRICT AS $$
SELECT replace(convert_from(stored, 'UTF8'), E'\\u0000', E'\\u0001')::jsonb
$$;

-- Backfill every retained pair row from its own document and the frozen
-- ordinals of its two members. OFFSET 0 keeps the subquery from being pulled
-- up, so each document is parsed once. Only DDL passes
-- eval_view_pairs_immutable, and only for this statement of the migration
-- transaction.
ALTER TABLE eval_view_pairs DISABLE TRIGGER eval_view_pairs_immutable;

UPDATE eval_view_pairs AS p
   SET member_ordinal_a = member_a.ordinal,
       member_ordinal_b = member_b.ordinal,
       variant_a = sides.a #>> '{member,variantId}',
       variant_b = sides.b #>> '{member,variantId}',
       eligibility_a = sides.a #>> '{member,eligibility}',
       eligibility_b = sides.b #>> '{member,eligibility}',
       execution_state_a = sides.a #>> '{execution,state}',
       execution_state_b = sides.b #>> '{execution,state}',
       assessment_a = sides.a ->> 'assessment',
       assessment_b = sides.b ->> 'assessment',
       conflicting_a = (sides.a ->> 'conflicting')::boolean,
       conflicting_b = (sides.b ->> 'conflicting')::boolean
  FROM (SELECT parsed.experiment_id, parsed.generation, parsed.ordinal,
               parsed.document -> 'a' AS a, parsed.document -> 'b' AS b
          FROM (SELECT experiment_id, generation, ordinal, eval_view_pair_sides(document) AS document
                  FROM eval_view_pairs
                OFFSET 0) AS parsed) AS sides
  JOIN eval_members AS member_a ON member_a.experiment_id = sides.experiment_id
       AND member_a.member_id = sides.a #>> '{member,memberId}'
  JOIN eval_members AS member_b ON member_b.experiment_id = sides.experiment_id
       AND member_b.member_id = sides.b #>> '{member,memberId}'
 WHERE p.experiment_id = sides.experiment_id
   AND p.generation = sides.generation
   AND p.ordinal = sides.ordinal;

ALTER TABLE eval_view_pairs ENABLE TRIGGER eval_view_pairs_immutable;

DROP FUNCTION eval_view_pair_sides(bytea);

-- Every side names a member of its experiment, so every row is backfilled;
-- a row the backfill missed stops the migration here.
ALTER TABLE eval_view_pairs
    ALTER COLUMN member_ordinal_a SET NOT NULL,
    ALTER COLUMN member_ordinal_b SET NOT NULL,
    ALTER COLUMN variant_a SET NOT NULL,
    ALTER COLUMN variant_b SET NOT NULL,
    ALTER COLUMN eligibility_a SET NOT NULL,
    ALTER COLUMN eligibility_b SET NOT NULL,
    ALTER COLUMN assessment_a SET NOT NULL,
    ALTER COLUMN assessment_b SET NOT NULL,
    ALTER COLUMN conflicting_a SET NOT NULL,
    ALTER COLUMN conflicting_b SET NOT NULL;
