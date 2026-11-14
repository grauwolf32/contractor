package telemetry

import (
	"context"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
)

// WorkerProgressReader reads a selected, already owner-authorized allocation.
// Physical routing and live allocation authority remain with Control Plane.
type WorkerProgressReader interface {
	ReadWorkerProgress(context.Context, string, string, string, string) (*reporting.WorkerCompactionProgress, *reporting.WorkerModelPhases, error)
}

// WithWorkerProgressReader enables bounded live diagnostics using the ordinary
// State reader. It adds no retained State cache or report/lifecycle writes.
func (r *Repository) WithWorkerProgressReader(reader WorkerProgressReader) *Repository {
	r.workerProgress = reader
	return r
}

func (r *Repository) readLiveWorkerDiagnostics(ctx context.Context, p RuntimeDiagnosticParams) (RuntimeDiagnosticPage, error) {
	result := RuntimeDiagnosticPage{Status: "pending", Reason: "report_missing", Items: []RuntimeDiagnostic{}}
	rows, err := r.db.Query(ctx, liveWorkerDiagnosticPageSQL, p.OwnerID, p.RunID, p.StageExecutionID, p.After, p.Limit+1)
	if err != nil {
		return result, fmt.Errorf("select live Worker diagnostics: %w", err)
	}
	// Release database resources before external I/O, including MaxConns=1.
	type selected struct{ id, name, generation string }
	selectedRows := []selected{}
	for rows.Next() {
		var row selected
		if err := rows.Scan(&row.id, &row.name, &row.generation); err != nil {
			rows.Close()
			return result, err
		}
		selectedRows = append(selectedRows, row)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return result, err
	}
	// All calls share a request-local deadline, distinct from Worker execution.
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	for _, row := range selectedRows {
		result.Generation = row.generation
		if p.Generation != "" && p.Generation != result.Generation {
			return result, fmt.Errorf("%w: live allocation generation changed", ErrConflict)
		}
		if len(result.Items) == p.Limit {
			result.More = true
			break
		}
		compaction, phases, err := r.workerProgress.ReadWorkerProgress(ctx, p.RunID, p.StageExecutionID, row.id, row.name)
		if err != nil {
			return result, fmt.Errorf("read live Worker diagnostics: %w", err)
		}
		// Even a valid snapshot with no observation must not imply idle/applied.
		item := RuntimeDiagnostic{LogicalAgent: row.name, Status: "unavailable", Reason: "report_missing", Adapters: []RuntimeAdapterDiagnostic{}, ModelPhases: phases}
		if compaction != nil {
			if err := compaction.Validate(); err != nil {
				return result, err
			}
			value := compaction.Clone()
			item.Compaction = &value
		}
		result.Items = append(result.Items, item)
		result.After = row.name
	}
	if len(result.Items) > 0 {
		result.Status = "available"
		result.Reason = ""
	}
	return result, nil
}
