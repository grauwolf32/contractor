package telemetry_test

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// TestMetricsPostgresStageSummaryKeepsTokenCountersInRange records a Planner
// report whose counters are at the report integer bound and a Worker report
// with positive counters: the rebuilt and stored summary is their exact,
// non-negative sum. A report above the bound never reaches the summary; it is
// invalid (OBS-R111, OBS-R112).
func TestMetricsPostgresStageSummaryKeepsTokenCountersInRange(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool)
	reports := telemetry.NewRepository(pool)
	createTelemetryRun(t, ctx, runs)
	receivedAt := time.Now().UTC()
	stage := createTelemetryStage(t, ctx, runs, "stage-counter-bound", "allocation-counter-bound", "session-counter-bound")

	planner := plannerEnvelope(stage, "session-counter-bound", receivedAt)
	beyond := int64(math.MaxInt64)
	planner.Report.Metrics.TotalTokens = &beyond
	if err := reports.RecordPlannerReport(ctx, planner); !errors.Is(err, telemetry.ErrInvalid) {
		t.Fatalf("Planner report above the counter bound = %v, want ErrInvalid", err)
	}
	bound := int64(contracts.MaxExactInteger)
	planner.Report.Metrics.InputTokens = &bound
	planner.Report.Metrics.TotalTokens = &bound
	if err := reports.RecordPlannerReport(ctx, planner); err != nil {
		t.Fatal(err)
	}
	allocation := allocationEnvelope(stage, "allocation-counter-bound", receivedAt, true)
	workerTokens := int64(7)
	allocation.Report.Worker.Metrics.InputTokens = &workerTokens
	allocation.Report.Worker.Metrics.TotalTokens = &workerTokens
	if err := reports.RecordAllocationReport(ctx, allocation); err != nil {
		t.Fatal(err)
	}
	metrics, err := reports.RebuildStageMetrics(ctx, stage, contracts.APIVersion)
	if err != nil {
		t.Fatal(err)
	}
	if metrics.Summary.TotalTokens != bound+workerTokens || metrics.Summary.InputTokens != bound+workerTokens {
		t.Fatalf("rebuilt summary = %+v, want exact non-negative token sums", metrics.Summary)
	}
	var stored int64
	if err := pool.QueryRow(ctx, `SELECT (summary->>'totalTokens')::bigint FROM stage_metrics WHERE stage_execution_id = $1`, stage).Scan(&stored); err != nil {
		t.Fatal(err)
	}
	if stored != bound+workerTokens {
		t.Fatalf("stored summary totalTokens = %d, want %d", stored, bound+workerTokens)
	}
}
