package evalstore

// SQL statements for progress.go.

// progressSQL splits the owner's eval_progress_observations between $3 and $4
// into $5-millisecond buckets from $3 and keeps the latest observation (by
// observed_at, then sequence) per bucket, limited to $9 buckets. It returns
// per-arm counts: the overall a/b counts when suite $6 is empty, otherwise
// suite $6's terminal counts for variants $7 and $8. Used by Store.Progress.
var progressSQL = `
WITH bucketed AS (
    SELECT observed_at, sequence,
        floor(extract(epoch FROM (observed_at - $3::timestamptz)) * 1000 / $5)::bigint AS bucket,
        CASE WHEN $6 = '' THEN (counts ->> 'a')::integer
            ELSE (counts #>> ARRAY['suites', $6, 'counts', $7, 'terminal'])::integer
        END AS a,
        CASE WHEN $6 = '' THEN (counts ->> 'b')::integer
            ELSE (counts #>> ARRAY['suites', $6, 'counts', $8, 'terminal'])::integer
        END AS b
    FROM eval_progress_observations p
    JOIN eval_experiments e USING (experiment_id)
    WHERE e.owner_id = $1 AND p.experiment_id = $2
        AND observed_at >= $3 AND observed_at <= $4
), last_in_bucket AS (
    SELECT DISTINCT ON (bucket) observed_at, a, b, bucket
    FROM bucketed
    ORDER BY bucket, observed_at DESC, sequence DESC
)
SELECT observed_at, a, b
FROM last_in_bucket
ORDER BY bucket
LIMIT $9
`
