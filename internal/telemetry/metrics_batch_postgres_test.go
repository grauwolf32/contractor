package telemetry_test

import (
	"context"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// A StageMetrics row that no longer decodes is left out of the batch on its
// own: its healthy sibling is still returned and the read is not an error,
// so Run detail keeps every diagnostic it can show (RUN-R90).
func TestPostgresStageMetricsBatchOmitsOnlyUndecodableRows(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool)
	createTelemetryRun(t, ctx, runs)
	for _, stage := range []struct{ id, metrics, summary string }{
		{"stage-healthy", `{"workers":{},"runtime":{}}`, `{"reportsComplete":true,"modelCalls":2}`},
		{"stage-bad-summary", `{"workers":{},"runtime":{}}`, `{"modelCalls":"two"}`},
		{"stage-bad-metrics", `{"workers":{"builder":{"toolCalls":"none"}},"runtime":{}}`, `{}`},
	} {
		createTelemetryStage(t, ctx, runs, stage.id, "allocation-"+stage.id, "session-"+stage.id)
		finishTelemetryStage(t, ctx, runs, stage.id)
		if _, err := pool.Exec(ctx, `INSERT INTO stage_metrics(stage_execution_id, metrics_schema_version, metrics, summary)
VALUES ($1, 'contractor/v1alpha1', $2::jsonb, $3::jsonb)`, stage.id, stage.metrics, stage.summary); err != nil {
			t.Fatal(err)
		}
	}
	metrics, err := telemetry.NewRepository(pool).GetStageMetricsBatch(ctx,
		[]string{"stage-bad-summary", "stage-healthy", "stage-bad-metrics", "stage-without-metrics"})
	if err != nil {
		t.Fatalf("batch with undecodable rows = %v, want the healthy rows", err)
	}
	healthy, found := metrics["stage-healthy"]
	if len(metrics) != 1 || !found || healthy.Summary.ModelCalls != 2 || !healthy.Summary.ReportsComplete {
		t.Fatalf("StageMetrics batch = %+v, want only the healthy row", metrics)
	}
}
