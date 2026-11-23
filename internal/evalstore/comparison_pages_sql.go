package evalstore

// SQL statements for comparison_pages.go.

// pairCohortSQL is a template (%s verbs are the tokens or duration pair
// columns) for the "filtered" CTE of a pair page, which selectedPageSQL
// completes. It selects the ordinals of the pairs of the owner's view
// generation $3 by suite $4, filter $5 (regressions/unresolved), an optional
// measure bin $6-$9 matching either side and, when metric $10 is set, both
// measures present; difference is |b - a|. No document enters the cohort.
// The owner check does not depend on the pair, so it runs once.
// Used by Store.PairPage.
var pairCohortSQL = `
WITH filtered AS MATERIALIZED (
    SELECT p.ordinal, p.ordinal AS pair_ordinal, NULL::text AS arm, abs(%s - %s) AS difference
    FROM eval_view_pairs p
    WHERE EXISTS (SELECT 1 FROM eval_experiments e WHERE e.experiment_id = $2 AND e.owner_id = $1)
        AND p.experiment_id = $2 AND p.generation = $3
        AND ($4 = '' OR p.suite_id = $4)
        AND ($5 IN ('', 'all')
            OR ($5 = 'regressions' AND p.regression)
            OR ($5 = 'unresolved' AND p.unresolved))
        AND (NOT $6 OR (
            (%s >= $7 AND (%s < $8 OR ($9 AND %s = $8)))
            OR (%s >= $7 AND (%s < $8 OR ($9 AND %s = $8)))
        ))
        AND ($10 = '' OR (%s IS NOT NULL AND %s IS NOT NULL))
)`

// memberCohortSQL is a template (%s verbs are the tokens or duration pair
// columns) for the "filtered" CTE of a member page, which selectedPageSQL
// completes. Every member is one side of exactly one pair, so it unpacks the
// member columns of both sides of the pairs in the owner's view generation $3
// (EVM-R245) and yields each member's frozen ordinal with its pair's ordinal
// and side. It filters by suite $4, variant $5 and filter $6 (unresolved,
// failed, unscored, eligibility or conflicting), and bins $7-$10 on the
// member's own side, which then needs both measures; difference is NULL. No
// document enters the cohort. Used by Store.SelectedMemberPage.
var memberCohortSQL = `
WITH filtered AS MATERIALIZED (
    SELECT side.ordinal, p.ordinal AS pair_ordinal, side.arm, NULL::double precision AS difference
    FROM eval_view_pairs p
    CROSS JOIN LATERAL (VALUES
        ('a', p.member_ordinal_a, p.variant_a, p.eligibility_a, p.execution_state_a, p.assessment_a,
            p.conflicting_a, p.complete_a, %s),
        ('b', p.member_ordinal_b, p.variant_b, p.eligibility_b, p.execution_state_b, p.assessment_b,
            p.conflicting_b, p.complete_b, %s)
    ) AS side(arm, ordinal, variant_id, eligibility, execution_state, assessment, conflicting,
        collection_complete, measure)
    WHERE EXISTS (SELECT 1 FROM eval_experiments e WHERE e.experiment_id = $2 AND e.owner_id = $1)
        AND p.experiment_id = $2 AND p.generation = $3
        AND ($4 = '' OR p.suite_id = $4)
        AND (%s IS NOT NULL AND %s IS NOT NULL OR NOT $7)
        AND ($5 = '' OR side.variant_id = $5)
        AND ($6 IN ('', 'all')
            OR ($6 = 'unresolved' AND (
                side.execution_state NOT IN ('succeeded', 'failed', 'cancelled')
                OR side.assessment NOT IN ('pass', 'fail')
                OR NOT side.collection_complete OR side.conflicting
            ))
            OR ($6 = 'failed' AND side.execution_state = 'failed')
            OR ($6 = 'unscored' AND side.assessment = 'unscored')
            OR ($6 IN ('unsupported', 'blocked') AND side.eligibility = $6)
            OR ($6 = 'conflicting' AND side.conflicting)
        )
        AND (NOT $7 OR (side.measure >= $8 AND (side.measure < $9 OR ($10 AND side.measure = $9))))
)`

// selectedPageSQL is a template that completes pairCohortSQL or
// memberCohortSQL into the one statement of a page (EVM-R245): %[1]s is the
// sort, %[2]s the keyset condition after the cursor and %[3]s the placeholder
// of the page size. Every row carries the exact count of the filtered cohort.
// The rows are the page in sort order, followed by one lookahead row when
// another page follows; an empty page is one row without an ordinal. A row of
// the page reads its pair's document by primary key, unless an earlier row of
// the page has the same pair; the lookahead row reads none. The scalar
// subquery keeps that lookup per row, so no plan reads the documents of the
// cohort. Used by selectedPageStatement.
var selectedPageSQL = `, candidates AS (
    SELECT ordinal, pair_ordinal, arm, difference
    FROM filtered
    WHERE %[2]s
    ORDER BY %[1]s
    LIMIT %[3]s + 1
), page AS (
    SELECT ordinal, pair_ordinal, arm, difference,
        row_number() OVER (ORDER BY %[1]s) AS position,
        row_number() OVER (PARTITION BY pair_ordinal ORDER BY %[1]s) AS pair_position
    FROM candidates
)
SELECT total.filtered_count, page.ordinal, page.pair_ordinal, page.arm, page.difference,
    CASE WHEN page.position <= %[3]s AND page.pair_position = 1 THEN (
        SELECT p.document
        FROM eval_view_pairs p
        WHERE p.experiment_id = $2 AND p.generation = $3 AND p.ordinal = page.pair_ordinal
    ) END AS document
FROM (SELECT count(*) AS filtered_count FROM filtered) AS total
LEFT JOIN page ON true
ORDER BY page.position
`
