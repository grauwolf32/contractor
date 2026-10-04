package evalstore

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

// MetricSnapshot projects only counters/completeness. It excludes prompts,
// tool arguments, errors and physical artifact locations.
type MetricSnapshot struct {
	RunID            string
	StageExecutionID string
	Document         json.RawMessage
	ExpectedWorkers  int
	Planner          config.PlannerRef
}

// The caller supplies the Run IDs from its authoritative member inventory.
// Ownership is checked again here. One extra row exposes truncation explicitly.
func (s *Store) MetricSnapshots(ctx context.Context, owner string, runs []string) ([]MetricSnapshot, error) {
	if len(runs) > evaldomain.MaxInventoryExecutions {
		return nil, evaldomain.Failure("eval_limit_exceeded")
	}
	rows, err := s.db.Query(ctx, metricSnapshotsQuery, runs, owner, evaldomain.MaxMetricSnapshots+1, evaldomain.MaxCollectionBytes, evaldomain.MaxDocumentBytes)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	snapshots := []MetricSnapshot{}
	for rows.Next() {
		var item MetricSnapshot
		if err = rows.Scan(&item.RunID, &item.StageExecutionID, &item.Document, &item.ExpectedWorkers, &item.Planner.PlannerID, &item.Planner.Version); err != nil {
			return nil, err
		}
		snapshots = append(snapshots, item)
	}
	return snapshots, rows.Err()
}
