package telemetry_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// A database CHECK rejection of a report is deterministic, so it must stay a
// permanent invalid report and never become a retryable conflict.
func TestReportWriteKeepsDatabaseRejectionsPermanent(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool)
	reports := telemetry.NewRepository(pool)
	createTelemetryRun(t, ctx, runs)
	stage := createTelemetryStage(t, ctx, runs, "stage-rejected", "allocation-rejected", "session-rejected")
	if _, err := pool.Exec(ctx, `ALTER TABLE planner_execution_reports
ADD CONSTRAINT planner_report_rejected_by_test CHECK (report_schema_version <> 'rejected-by-test')`); err != nil {
		t.Fatal(err)
	}

	rejected := plannerEnvelope(stage, "session-rejected", time.Now().UTC())
	rejected.ReportSchemaVersion = "rejected-by-test"
	err := reports.RecordPlannerReport(ctx, rejected)
	if !errors.Is(err, telemetry.ErrInvalid) || errors.Is(err, telemetry.ErrConflict) {
		t.Fatalf("CHECK-rejected report error = %v, want permanent ErrInvalid", err)
	}
	err = runs.RecordPlannerExecutionReport(ctx, runstore.RecordPlannerExecutionReportParams{
		StageExecutionID: rejected.StageExecutionID, SessionID: rejected.SessionID,
		InvocationID: rejected.InvocationID, StartedAt: rejected.StartedAt, FinishedAt: rejected.FinishedAt,
		ReportSchemaVersion: rejected.ReportSchemaVersion, Report: rejected.Report, Secrets: rejected.Secrets,
	})
	if !errors.Is(err, runstore.ErrInvalid) || errors.Is(err, runstore.ErrConflict) {
		t.Fatalf("Run store CHECK-rejected report error = %v, want permanent ErrInvalid", err)
	}

	orphan := plannerEnvelope("stage-missing", "session-missing", time.Now().UTC())
	if err := reports.RecordPlannerReport(ctx, orphan); !errors.Is(err, telemetry.ErrNotFound) {
		t.Fatalf("report for a missing StageExecution error = %v, want ErrNotFound", err)
	}
}
