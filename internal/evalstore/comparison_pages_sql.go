package evalstore

// SQL statements for comparison_pages.go.

// pairPageSQL is a template (%s verbs are the tokens or duration pair columns)
// for the "filtered" CTE that readSelectedPage completes with a count or a
// keyset page. It selects pairs of the owner's view generation $3 by suite $4,
// filter $5 (regressions/unresolved), an optional measure bin $6-$9 matching
// either side and, when metric $10 is set, both measures present; difference
// is |b - a|. Used by Store.PairPage.
var pairPageSQL = `
WITH filtered AS MATERIALIZED (
    SELECT p.ordinal, p.document, abs(%s - %s) AS difference
    FROM eval_view_pairs p
    JOIN eval_experiments e USING (experiment_id)
    WHERE e.owner_id = $1 AND p.experiment_id = $2 AND p.generation = $3
        AND ($4 = '' OR p.suite_id = $4)
        AND ($5 IN ('', 'all')
            OR ($5 = 'regressions' AND p.regression)
            OR ($5 = 'unresolved' AND p.unresolved))
        AND (NOT $6 OR (
            (%s >= $7 AND (%s < $8 OR ($9 AND %s = $8)))
            OR (%s >= $7 AND (%s < $8 OR ($9 AND %s = $8)))
        ))
        AND ($10 = '' OR (%s IS NOT NULL AND %s IS NOT NULL))
)
`

// selectedMemberPageSQL is a template (%s verbs are the tokens or duration pair
// columns) for the "filtered" CTE that readSelectedPage completes. Every member
// is one side of exactly one pair, so it unpacks both sides of the pairs in the
// owner's view generation $3 and orders them by the member's frozen ordinal.
// It filters by suite $4, variant $5 and filter $6 (unresolved, failed,
// unscored, eligibility or conflicting), and bins $7-$10 on the member's own
// side, which then needs both measures; difference is NULL.
// Used by Store.SelectedMemberPage.
var selectedMemberPageSQL = `
WITH members AS (
    SELECT member.ordinal, convert_to(side.value::text, 'UTF8') AS document,
        side.collection_complete, side.value,
        CASE side.arm WHEN 'a' THEN %s ELSE %s END AS measure
    FROM eval_view_pairs p
    JOIN eval_experiments e USING (experiment_id)
    CROSS JOIN LATERAL (VALUES
        ('a', convert_from(p.document, 'UTF8')::jsonb -> 'a', p.complete_a),
        ('b', convert_from(p.document, 'UTF8')::jsonb -> 'b', p.complete_b)
    ) AS side(arm, value, collection_complete)
    JOIN eval_members member ON member.experiment_id = p.experiment_id
        AND member.member_id = side.value #>> '{member,memberId}'
    WHERE e.owner_id = $1 AND p.experiment_id = $2 AND p.generation = $3
        AND ($4 = '' OR p.suite_id = $4)
        AND (%s IS NOT NULL AND %s IS NOT NULL OR NOT $7)
), filtered AS MATERIALIZED (
    SELECT ordinal, document, NULL::double precision AS difference
    FROM members
    WHERE ($5 = '' OR value #>> '{member,variantId}' = $5)
        AND ($6 IN ('', 'all')
            OR ($6 = 'unresolved' AND (
                value #>> '{execution,state}' NOT IN ('succeeded', 'failed', 'cancelled')
                OR value ->> 'assessment' NOT IN ('pass', 'fail')
                OR NOT collection_complete OR (value ->> 'conflicting')::boolean
            ))
            OR ($6 = 'failed' AND value #>> '{execution,state}' = 'failed')
            OR ($6 = 'unscored' AND value ->> 'assessment' = 'unscored')
            OR ($6 IN ('unsupported', 'blocked') AND value #>> '{member,eligibility}' = $6)
            OR ($6 = 'conflicting' AND (value ->> 'conflicting')::boolean)
        )
        AND (NOT $7 OR (measure >= $8 AND (measure < $9 OR ($10 AND measure = $9))))
)
`
