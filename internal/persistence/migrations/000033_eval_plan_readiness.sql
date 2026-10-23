-- A native frozen plan stores the readiness its experiment view shows
-- (V407-029, EVM-R242, LIM-EVM-19).
--
-- eval_frozen_plans.readiness is the Readiness JSON of a playground.plan/v1
-- plan: for each pinned dimension its equality status and the origin of each
-- arm's pin, and for each arm its member eligibility counts. The freeze
-- transaction derives it from the validated plan (evaldomain.NativeReadiness)
-- and inserts it with the plan, so an experiment read neither transfers the
-- plan document of up to 1 MiB nor recomputes readiness from it. An external
-- registration has none. eval_plans_immutable keeps it as immutable as the
-- rest of the row.
ALTER TABLE eval_frozen_plans ADD COLUMN readiness bytea;

-- Backfill each existing native plan with the readiness its reads computed so
-- far: every dimension any variant pins, in byte order; an origin of observed
-- or producer-supplied, otherwise unavailable; status equal or different only
-- when both arms observed a non-null value; then the baseline's and the
-- candidate's member counts. Only DDL passes eval_plans_immutable, and only
-- for this statement of the migration transaction.
ALTER TABLE eval_frozen_plans DISABLE TRIGGER eval_plans_immutable;

UPDATE eval_frozen_plans AS p
   SET readiness = convert_to(jsonb_build_object(
        'pins', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                    'dimension', d.dimension,
                    'requiredEqual', COALESCE(plan.comparison->'requiredEqual' ? d.dimension, false),
                    'baselineOrigin', CASE WHEN a.pin->>'origin' IN ('observed', 'producer-supplied')
                                           THEN a.pin->>'origin' ELSE 'unavailable' END,
                    'candidateOrigin', CASE WHEN b.pin->>'origin' IN ('observed', 'producer-supplied')
                                            THEN b.pin->>'origin' ELSE 'unavailable' END,
                    'status', CASE WHEN a.pin->>'origin' = 'observed' AND b.pin->>'origin' = 'observed'
                                        AND a.pin->>'value' IS NOT NULL AND b.pin->>'value' IS NOT NULL
                                   THEN CASE WHEN a.pin->>'value' = b.pin->>'value' THEN 'equal' ELSE 'different' END
                                   ELSE 'unavailable' END)
                    ORDER BY d.dimension COLLATE "C")
              FROM (SELECT DISTINCT dimension.name AS dimension
                      FROM jsonb_each(plan.document->'pins') AS arm(variant, pins),
                           jsonb_object_keys(arm.pins) AS dimension(name)) AS d
             CROSS JOIN LATERAL (SELECT plan.document->'pins'->(plan.comparison->>'baseline')->d.dimension) AS a(pin)
             CROSS JOIN LATERAL (SELECT plan.document->'pins'->(plan.comparison->>'candidate')->d.dimension) AS b(pin)
        ), '[]'::jsonb),
        'arms', (
            SELECT jsonb_agg(arm.counts ORDER BY arm.position)
              FROM (SELECT side.position, jsonb_build_object(
                            'variantId', side.variant,
                            'expected', count(m.member),
                            'eligible', count(m.member) FILTER (WHERE m.member->>'eligibility' = 'eligible'),
                            'unsupported', count(m.member) FILTER (WHERE m.member->>'eligibility' = 'unsupported'),
                            'blocked', count(m.member) FILTER (WHERE m.member->>'eligibility' = 'blocked')) AS counts
                      FROM unnest(ARRAY[plan.comparison->>'baseline', plan.comparison->>'candidate'])
                               WITH ORDINALITY AS side(variant, position)
                      LEFT JOIN jsonb_array_elements(plan.document->'members') AS m(member)
                             ON m.member->>'variant_id' = side.variant
                     GROUP BY side.position, side.variant) AS arm)
       )::text, 'UTF8')
  FROM (SELECT experiment_id,
               convert_from(document, 'UTF8')::jsonb AS document,
               convert_from(setup, 'UTF8')::jsonb->'comparison' AS comparison
          FROM eval_frozen_plans
         WHERE document_kind = 'playground.plan/v1') AS plan
 WHERE p.experiment_id = plan.experiment_id;

ALTER TABLE eval_frozen_plans ENABLE TRIGGER eval_plans_immutable;

-- A Readiness has at most 2,048 pins and 2 arms, about 531 KiB when every
-- string has its maximum length (LIM-EVM-19); the bound is that of the setup.
ALTER TABLE eval_frozen_plans
    ADD CONSTRAINT eval_frozen_plans_readiness_kind_check
        CHECK ((document_kind = 'playground.plan/v1') = (readiness IS NOT NULL)),
    ADD CONSTRAINT eval_frozen_plans_readiness_check
        CHECK (octet_length(readiness) >= 2 AND octet_length(readiness) <= 1048576);
