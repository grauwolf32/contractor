package telemetry_test

import (
	"context"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// A StageMetrics row is decoded as strictly as it was validated on write. A
// row that fails that decoding is left out of the batch on its own: its
// healthy sibling is still returned and the read is not an error, so Run
// detail keeps every diagnostic it can show (RUN-R90, OBS-R110). jsonb keeps
// one member of a duplicated name, so no stored row can hold a duplicate.
func TestPostgresStageMetricsBatchOmitsOnlyRowsThatBreakTheirContract(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	reports := telemetry.NewRepository(pool)
	createTelemetryRun(t, ctx, runs)
	const healthy = "stage-healthy"
	createTelemetryStage(t, ctx, runs, healthy, "allocation-"+healthy, "session-"+healthy)
	now := time.Now().UTC()
	if err := reports.RecordAllocationReport(ctx, allocationEnvelope(healthy, "allocation-"+healthy, now, true)); err != nil {
		t.Fatal(err)
	}
	if err := reports.RecordPlannerReport(ctx, plannerEnvelope(healthy, "session-"+healthy, now)); err != nil {
		t.Fatal(err)
	}
	rebuilt, err := reports.RebuildStageMetrics(ctx, healthy, contracts.APIVersion)
	if err != nil {
		t.Fatal(err)
	}
	finishTelemetryStage(t, ctx, runs, healthy)
	// Each broken row changes one member of the healthy row's stored metrics
	// or summary.
	broken := map[string]string{
		"stage-summary-type":            `summary = jsonb_set(summary, '{modelCalls}', '"two"')`,
		"stage-summary-missing-member":  `summary = summary - 'errorCount'`,
		"stage-summary-unknown-member":  `summary = summary || '{"cachedTokens":0}'`,
		"stage-summary-negative":        `summary = jsonb_set(summary, '{toolCalls}', '-1')`,
		"stage-metrics-type":            `metrics = jsonb_set(metrics, '{workers,builder,toolCalls}', '"none"')`,
		"stage-metrics-unknown-member":  `metrics = jsonb_set(metrics, '{planner,prompt}', '"x"')`,
		"stage-metrics-unknown-counter": `metrics = jsonb_set(metrics, '{workers,builder,metrics,cachedTokens}', '1')`,
		"stage-metrics-negative":        `metrics = jsonb_set(metrics, '{planner,metrics,modelCalls}', '-1')`,
		"stage-metrics-above-bound":     `metrics = jsonb_set(metrics, '{workers,builder,metrics,modelCalls}', '9007199254740992')`,
		"stage-metrics-missing-member":  `metrics = metrics #- '{planner,reportId}'`,
		"stage-metrics-null-member":     `metrics = jsonb_set(metrics, '{workers,builder,errors}', 'null')`,
	}
	ids := []string{healthy, "stage-without-metrics"}
	for id, change := range broken {
		createTelemetryStage(t, ctx, runs, id, "allocation-"+id, "session-"+id)
		finishTelemetryStage(t, ctx, runs, id)
		if _, err := pool.Exec(ctx, `INSERT INTO stage_metrics(stage_execution_id, metrics_schema_version, metrics, summary)
SELECT $1, metrics_schema_version, metrics, summary FROM stage_metrics WHERE stage_execution_id = $2`, id, healthy); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `UPDATE stage_metrics SET `+change+` WHERE stage_execution_id = $1`, id); err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		ids = append(ids, id)
	}
	metrics, err := reports.GetStageMetricsBatch(ctx, ids)
	if err != nil {
		t.Fatalf("batch with broken rows = %v, want the healthy row", err)
	}
	read, found := metrics[healthy]
	if len(metrics) != 1 || !found {
		keys := make([]string, 0, len(metrics))
		for id := range metrics {
			keys = append(keys, id)
		}
		t.Fatalf("StageMetrics batch holds %s, want only the healthy row", strings.Join(keys, ", "))
	}
	if !reflect.DeepEqual(read.Metrics, rebuilt.Metrics) || read.Summary != rebuilt.Summary ||
		read.Summary.ModelCalls != 5 || read.Summary.ToolCalls != 1 {
		t.Fatalf("healthy row = %+v, want the rebuilt %+v", read, rebuilt)
	}
}

// The StageMetrics rebuild decodes a stored Planner report as strictly as the
// allocation report beside it: a stored report with an unknown member fails
// the rebuild instead of being read without it (OBS-R110).
func TestPostgresStageMetricsRebuildRejectsStoredReportsWithUnknownMembers(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	reports := telemetry.NewRepository(pool)
	createTelemetryRun(t, ctx, runs)
	for _, tc := range []struct{ stage, insert string }{
		{"stage-planner", `INSERT INTO planner_execution_reports (
    report_id, stage_execution_id, session_id, invocation_id, started_at, finished_at,
    report_schema_version, report
) VALUES ('planner-' || $1, $1, 'session-' || $1, 'invocation-' || $1, clock_timestamp(), clock_timestamp(), $2,
    jsonb_build_object('reportId', 'planner-' || $1, 'complete', true, 'metrics', '{"tools":{}}'::jsonb,
        'toolCalls', '[]'::jsonb, 'errors', '[]'::jsonb, 'truncated', false, 'prompt', 'x'))`},
		{"stage-allocation", `INSERT INTO allocation_execution_reports (
    report_id, stage_execution_id, allocation_id, logical_agent_name, report_schema_version, report
) VALUES ('allocation-final-' || $1, $1, 'allocation-' || $1, 'builder', $2,
    jsonb_build_object('reportId', 'allocation-final-' || $1, 'allocationId', 'allocation-' || $1,
        'startedAt', '2026-10-25T00:00:00Z', 'finishedAt', '2026-10-25T00:00:01Z',
        'worker', jsonb_build_object('reportId', 'worker-' || $1, 'complete', true,
            'metrics', '{"tools":{}}'::jsonb, 'toolCalls', '[]'::jsonb, 'errors', '[]'::jsonb,
            'truncated', false, 'prompt', 'x'),
        'runtime', '{"complete":true,"adapters":{}}'::jsonb))`},
	} {
		createTelemetryStage(t, ctx, runs, tc.stage, "allocation-"+tc.stage, "session-"+tc.stage)
		if _, err := pool.Exec(ctx, tc.insert, tc.stage, contracts.APIVersion); err != nil {
			t.Fatalf("%s: %v", tc.stage, err)
		}
		if record, err := reports.RebuildStageMetrics(ctx, tc.stage, contracts.APIVersion); err == nil ||
			!strings.Contains(err.Error(), "unknown member") {
			t.Fatalf("%s rebuild = (%+v, %v), want the unknown member rejected", tc.stage, record, err)
		}
	}
}
