package evalservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"sort"
	"time"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// metricReport is one report of the stage metrics projection that
// MetricSnapshots reads (metricSnapshotsQuery): the report's completeness
// flags and its metrics, whose optional counters stay absent instead of
// becoming telemetry's display-summary zeros.
type metricReport struct {
	Complete  bool                       `json:"complete" strictjson:"required"`
	Truncated bool                       `json:"truncated" strictjson:"required"`
	Metrics   reporting.ExecutionMetrics `json:"metrics" strictjson:"required"`
}

// metricSnapshot is the whole projection of one StageMetrics document. Every
// member it names is always written; the Planner is null for a Stage without
// a Planner report.
type metricSnapshot struct {
	Planner *metricReport           `json:"planner" strictjson:"required,nullable"`
	Workers map[string]metricReport `json:"workers" strictjson:"required"`
	Runtime map[string]struct {
		Complete bool `json:"complete" strictjson:"required"`
	} `json:"runtime" strictjson:"required"`
}

// errStoredMetrics marks a stored StageMetrics projection that fails the
// rules its reports were validated against on write (OBS-R110, OBS-R111): a
// data-integrity failure of the member's collection, never a missing counter.
var errStoredMetrics = errors.New("stored StageMetrics violate their contract")

// decodeMetricSnapshot decodes a projection as strictly as its reports were
// decoded on write and applies their metric rules, so a duplicate or unknown
// member, a missing one, a negative counter or one above 2^53-1 is rejected.
func decodeMetricSnapshot(raw []byte) (metricSnapshot, error) {
	var snap metricSnapshot
	if err := strictjson.DecodeNonNull(raw, &snap); err != nil {
		return metricSnapshot{}, err
	}
	if snap.Planner != nil {
		if err := snap.Planner.Metrics.Validate(); err != nil {
			return metricSnapshot{}, fmt.Errorf("planner report: %w", err)
		}
	}
	for name, report := range snap.Workers {
		if err := contracts.ValidateOpaqueID("worker logical Agent name", name); err != nil {
			return metricSnapshot{}, err
		}
		if err := report.Metrics.Validate(); err != nil {
			return metricSnapshot{}, fmt.Errorf("worker report %q: %w", name, err)
		}
	}
	for name := range snap.Runtime {
		if err := contracts.ValidateOpaqueID("runtime logical Agent name", name); err != nil {
			return metricSnapshot{}, err
		}
	}
	return snap, nil
}

// attemptMetrics reads every Planner and Worker counter as reported, whatever
// the Planner: a model-free Planner's report, or the Scheduler's placeholder
// for it, states its zero model counters (EVM-R81), and a counter absent by
// contract leaves its counter incomplete.
func attemptMetrics(run, stage string, raw []byte, expectedWorkers int) (evaldomain.AttemptObservation, error) {
	out := evaldomain.AttemptObservation{RunID: run, StageExecutionID: stage, SourceSHA256: evaldomain.Digest(raw), Metrics: map[string]int64{}, ReportsComplete: true}
	snap, err := decodeMetricSnapshot(raw)
	if err != nil {
		return evaldomain.AttemptObservation{}, fmt.Errorf("%w: Stage execution %s: %w", errStoredMetrics, stage, err)
	}
	participants := []metricReport{}
	if snap.Planner != nil {
		participants = append(participants, *snap.Planner)
	}
	for _, name := range slices.Sorted(maps.Keys(snap.Workers)) {
		participants = append(participants, snap.Workers[name])
	}
	if len(participants) == 0 || len(snap.Workers) != expectedWorkers || len(snap.Runtime) != expectedWorkers {
		out.ReportsComplete = false
	}
	for _, r := range snap.Runtime {
		out.ReportsComplete = out.ReportsComplete && r.Complete
	}
	// A contribution absent by contract leaves its counter incomplete; one
	// whose known contributions exceed the int64 range is overflowed and has no
	// value, whichever participant or tool is read first (EVM-R240). Validation
	// has bounded every present contribution to 0..2^53-1.
	missing := map[string]bool{}
	sums := map[string]*evaldomain.CounterSum{}
	add := func(name string, n *int64) {
		if n == nil {
			missing[name] = true
			return
		}
		if sums[name] == nil {
			sums[name] = &evaldomain.CounterSum{}
		}
		sums[name].Add(*n)
	}
	for _, r := range participants {
		out.ReportsComplete = out.ReportsComplete && r.Complete
		out.Truncated = out.Truncated || r.Truncated
		add("modelCalls", r.Metrics.ModelCalls)
		add("inputTokens", r.Metrics.InputTokens)
		add("outputTokens", r.Metrics.OutputTokens)
		add("totalTokens", r.Metrics.TotalTokens)
		if summary := r.Metrics.Summarizer; summary != nil {
			// These calls are outside the normal-loop counters. Phase/cost
			// breakdowns are views and must never be added again.
			for name, value := range map[string]uint64{"modelCalls": summary.ModelCalls, "inputTokens": summary.InputTokens, "outputTokens": summary.OutputTokens, "totalTokens": summary.TotalTokens} {
				contribution := int64(value) // Validate bounded it to 2^53-1.
				add(name, &contribution)
			}
			if summary.TokenUsageUnavailable > 0 {
				missing["inputTokens"], missing["outputTokens"], missing["totalTokens"] = true, true, true
			}
		}
		// Validation requires the tools map, so a report without tools adds
		// known zeros.
		zero := int64(0)
		add("toolCalls", &zero)
		add("toolFailures", &zero)
		for _, name := range slices.Sorted(maps.Keys(r.Metrics.Tools)) {
			tool := r.Metrics.Tools[name]
			add("toolCalls", tool.Calls)
			add("toolFailures", tool.Failed)
		}
		if b := r.Metrics.WorkerBudget; b != nil && b.TokenUsageUnavailable > 0 {
			missing["inputTokens"], missing["outputTokens"], missing["totalTokens"] = true, true, true
		}
	}
	for name, sum := range sums {
		if total, fits := sum.Value(); fits {
			out.Metrics[name] = total
		} else {
			out.OverflowMetrics = append(out.OverflowMetrics, name)
		}
	}
	sort.Strings(out.OverflowMetrics)
	for name := range missing {
		out.IncompleteMetrics = append(out.IncompleteMetrics, name)
	}
	sort.Strings(out.IncompleteMetrics)
	return out, nil
}
func observedUsage(ctx context.Context, db pg.DBTX, owner, member string, execution ExecutionView, inventory evalstore.Inventory, storage *artifactpostgres.Storage) (evaldomain.Usage, error) {
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

	snapshots, err := evalstore.NewPostgresStore(db, storage).MetricSnapshots(ctx, owner, runs)
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
		attempt, err := attemptMetrics(snapshot.RunID, snapshot.StageExecutionID, snapshot.Document, snapshot.ExpectedWorkers)
		if err != nil {
			return evaldomain.Usage{}, err
		}
		observation.Attempts = append(observation.Attempts, attempt)
	}
	return evaldomain.NormalizeUsage(observation)
}
