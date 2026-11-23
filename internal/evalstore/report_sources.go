package evalstore

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// ViewSources attributes the exact selected records. Hidden rubrics and free
// text assessment reasons never enter this report projection.
func (s *Store) ViewSources(ctx context.Context, owner, id string, generation int64) ([]evaldomain.Source, error) {
	rows, err := s.db.Query(ctx, viewSourcesSQL, owner, id, generation, evaldomain.MaxReportSources+1)
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
