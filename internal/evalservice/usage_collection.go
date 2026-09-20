package evalservice

import (
	"context"
	"encoding/json"
	"math"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// A minimal projection of authoritative stage metrics. Optional pointers retain
// unavailable counters instead of using telemetry's display-summary zeros.
type metricReport struct {
	Complete  bool                       `json:"complete"`
	Truncated bool                       `json:"truncated"`
	Metrics   contracts.ExecutionMetrics `json:"metrics"`
}
type metricSnapshot struct {
	Planner *metricReport           `json:"planner"`
	Workers map[string]metricReport `json:"workers"`
	Runtime map[string]struct {
		Complete bool `json:"complete"`
	} `json:"runtime"`
}

func attemptMetrics(run, stage string, raw []byte, expectedWorkers int, planner config.PlannerRef) (evaldomain.AttemptObservation, error) {
	out := evaldomain.AttemptObservation{RunID: run, StageExecutionID: stage, SourceSHA256: evaldomain.Digest(raw), Metrics: map[string]int64{}, ReportsComplete: true}
	var snap metricSnapshot
	if err := json.Unmarshal(raw, &snap); err != nil {
		return out, err
	}
	participants := []metricReport{}
	if snap.Planner != nil {
		// Older passthrough@1 reports omitted counters for model work it never
		// performs. Only the immutable Stage planner identity permits these zeros;
		// missing Worker or model-backed Planner counters remain incomplete.
		if planner.PlannerID == "passthrough" && planner.Version == "1" {
			for _, counter := range []**int64{
				&snap.Planner.Metrics.ModelCalls, &snap.Planner.Metrics.InputTokens,
				&snap.Planner.Metrics.OutputTokens, &snap.Planner.Metrics.TotalTokens,
			} {
				if *counter == nil {
					zero := int64(0)
					*counter = &zero
				}
			}
		}
		participants = append(participants, *snap.Planner)
	}
	for _, r := range snap.Workers {
		participants = append(participants, r)
	}
	if len(participants) == 0 || len(snap.Workers) != expectedWorkers || len(snap.Runtime) != expectedWorkers {
		out.ReportsComplete = false
	}
	for _, r := range snap.Runtime {
		out.ReportsComplete = out.ReportsComplete && r.Complete
	}
	missing := map[string]bool{}
	add := func(name string, n *int64) {
		if n == nil || *n < 0 {
			missing[name] = true
			return
		}
		if *n > math.MaxInt64-out.Metrics[name] {
			missing[name] = true
			return
		}
		out.Metrics[name] += *n
	}
	for _, r := range participants {
		out.ReportsComplete = out.ReportsComplete && r.Complete
		out.Truncated = out.Truncated || r.Truncated
		add("modelCalls", r.Metrics.ModelCalls)
		add("inputTokens", r.Metrics.InputTokens)
		add("outputTokens", r.Metrics.OutputTokens)
		add("totalTokens", r.Metrics.TotalTokens)
		if r.Metrics.Tools == nil {
			missing["toolCalls"], missing["toolFailures"] = true, true
		} else {
			zero := int64(0)
			add("toolCalls", &zero)
			add("toolFailures", &zero)
			for _, tool := range r.Metrics.Tools {
				add("toolCalls", tool.Calls)
				add("toolFailures", tool.Failed)
			}
		}
		if b := r.Metrics.WorkerBudget; b != nil && b.TokenUsageUnavailable > 0 {
			missing["inputTokens"], missing["outputTokens"], missing["totalTokens"] = true, true, true
		}
	}
	for name := range missing {
		out.IncompleteMetrics = append(out.IncompleteMetrics, name)
	}
	sort.Strings(out.IncompleteMetrics)
	return out, nil
}
func observedUsage(ctx context.Context, db pg.DBTX, owner, member string, execution ExecutionView, inventory evalstore.Inventory) (evaldomain.Usage, error) {
	if execution.Ref == nil {
		return evaldomain.Usage{}, evaldomain.Failure("eval_not_ready")
	}
	observation := evaldomain.UsageObservation{
		MemberID:          member,
		Parent:            *execution.Ref,
		Executions:        []evaldomain.ExecutionRef{},
		InventoryComplete: inventory.Complete,
		Terminal:          isTerminal(execution.State),
		Missing:           append([]string{}, inventory.Gaps...),
		Attempts:          []evaldomain.AttemptObservation{},
	}
	if execution.StartedAt != nil {
		s := execution.StartedAt.UTC().Format(time.RFC3339Nano)
		observation.StartedAt = &s
	}
	if execution.FinishedAt != nil {
		s := execution.FinishedAt.UTC().Format(time.RFC3339Nano)
		observation.FinishedAt = &s
	}
	parentBytes, err := json.Marshal(execution)
	if err != nil {
		return evaldomain.Usage{}, err
	}
	observation.ParentSourceSHA256 = evaldomain.Digest(parentBytes)
	runs := []string{}
	seen := map[string]bool{}
	for _, entry := range inventory.Entries {
		if entry.Execution == nil {
			continue
		}
		ref := *entry.Execution
		observation.Executions = append(observation.Executions, ref)
		if ref.Kind == "run" && entry.Available && !seen[ref.ID] {
			runs = append(runs, ref.ID)
			seen[ref.ID] = true
		}
	}

	snapshots, err := evalstore.NewPostgresStore(db).MetricSnapshots(ctx, owner, runs)
	if err != nil {
		return evaldomain.Usage{}, err
	}
	for i, snapshot := range snapshots {
		if i >= evaldomain.MaxMetricSnapshots {
			observation.Missing = append(observation.Missing, "Stage metrics exceed the collection bound.")
			break
		}
		if snapshot.Document == nil {
			observation.Missing = append(observation.Missing, "Stage metrics are unavailable or exceed the payload bound.")
			continue
		}
		attempt, err := attemptMetrics(snapshot.RunID, snapshot.StageExecutionID, snapshot.Document, snapshot.ExpectedWorkers, snapshot.Planner)
		if err != nil {
			return evaldomain.Usage{}, err
		}
		observation.Attempts = append(observation.Attempts, attempt)
	}
	return evaldomain.NormalizeUsage(observation)
}
