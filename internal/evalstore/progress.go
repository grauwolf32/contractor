package evalstore

import (
	"context"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// Progress returns at most one real observation per display bucket. The range
// ends at the selected publication, so later observations cannot leak into it.
func (s *Store) Progress(ctx context.Context, e Experiment, v View, suite, baseline, candidate string) ([]evaldomain.ProgressPoint, int64, error) {
	if e.StartedAt == nil {
		return []evaldomain.ProgressPoint{}, 1, nil
	}
	width := max(int64(1), v.CreatedAt.Sub(*e.StartedAt).Milliseconds()/evaldomain.MaxProgressBuckets+1)
	rows, err := s.db.Query(ctx, `
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
`, e.OwnerID, e.ID, *e.StartedAt, v.CreatedAt, width, suite, baseline, candidate, evaldomain.MaxProgressBuckets)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	points := []evaldomain.ProgressPoint{}
	for rows.Next() {
		var observed time.Time
		var a, b *int
		if err = rows.Scan(&observed, &a, &b); err != nil {
			return nil, 0, err
		}
		elapsed := observed.Sub(*e.StartedAt).Milliseconds()
		gap := len(points) == 0 && elapsed > 0
		if len(points) > 0 {
			gap = elapsed-points[len(points)-1].ElapsedMS > max(width, evaldomain.ProgressGap.Milliseconds())
		}
		points = append(points, evaldomain.ProgressPoint{ElapsedMS: elapsed, ObservedAt: observed.UTC().Format(time.RFC3339Nano), A: a, B: b, GapBefore: gap || a == nil || b == nil})
	}
	return points, width, rows.Err()
}
