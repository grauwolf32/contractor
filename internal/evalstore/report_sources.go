package evalstore

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// ViewSources attributes the exact selected records. Hidden rubrics and free
// text assessment reasons never enter this report projection.
func (s *Store) ViewSources(ctx context.Context, owner, id string, generation int64) ([]evaldomain.Source, error) {
	rows, err := s.db.Query(ctx, `
WITH selected AS (
    SELECT m.member_id, convert_from(m.document, 'UTF8')::jsonb AS document
    FROM eval_view_members m
    JOIN eval_experiments e USING (experiment_id)
    WHERE e.owner_id = $1 AND m.experiment_id = $2 AND m.generation = $3
), sources AS (
    SELECT r.kind, convert_from(r.document, 'UTF8')::jsonb -> 'source' AS source
    FROM selected m
    CROSS JOIN LATERAL (VALUES
        ('result', m.document ->> 'resultSha256'),
        ('assessment', m.document ->> 'assessmentSha256')
    ) chosen(kind, digest)
    JOIN eval_records r ON r.experiment_id = $2
        AND r.member_id = m.member_id
        AND r.kind = chosen.kind
        AND r.record_sha256 = chosen.digest
)
SELECT DISTINCT CASE WHEN kind = 'result' THEN source
    ELSE jsonb_build_object(
        'system', source ->> 'kind',
        'id', CASE source ->> 'kind'
            WHEN 'external' THEN source ->> 'producerId'
            WHEN 'human' THEN 'owner-review'
            ELSE 'native-checks'
        END,
        'revision', NULL, 'sourceSha256', NULL
    )
END AS provenance
FROM sources
ORDER BY provenance
LIMIT $4
`, owner, id, generation, evaldomain.MaxReportSources+1)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []evaldomain.Source{}
	for rows.Next() {
		var raw []byte
		if err = rows.Scan(&raw); err != nil {
			return nil, err
		}
		var source evaldomain.Source
		if err = json.Unmarshal(raw, &source); err != nil {
			return nil, err
		}
		out = append(out, source)
	}
	if len(out) > evaldomain.MaxReportSources {
		return nil, evaldomain.Failure("eval_limit_exceeded")
	}
	return out, rows.Err()
}
