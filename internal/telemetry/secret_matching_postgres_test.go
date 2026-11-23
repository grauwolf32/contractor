package telemetry_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// A short username, a password that starts with it and a 16-byte token reach
// PostgreSQL matched by the shared rule (OBS-R31, OBS-R32, LIM-OBS-3): the
// username only as a complete string, the long secrets anywhere without a
// suffix, and error codes whole.
func TestMetricsPostgresPersistsReportsMatchedByTheSharedSecretRule(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	reports := telemetry.NewRepository(pool)
	createTelemetryRun(t, ctx, runs)
	const (
		username = "admin"
		password = "admin-0123456789-password-xyz"
		token    = "tok-0123456789ab"
	)
	receivedAt := time.Now().UTC()
	stage := createTelemetryStage(t, ctx, runs, "stage-secret-rule", "allocation-secret-rule", "session-secret-rule")
	envelope := allocationEnvelope(stage, "allocation-secret-rule", receivedAt, true)
	envelope.Secrets = []string{username, password, token}
	stop := "administrator stopped after " + token
	envelope.Report.Runtime.StopReason = &stop
	envelope.Report.Worker.ToolCalls[0].Arguments = map[string]any{
		"path": "data/main.go", "user": username, "note": "login " + password,
	}
	envelope.Report.Worker.Errors = []reporting.ExecutionError{
		{Code: "admin_rejected", Message: "administrator login failed"},
		{Code: "bad_" + token, Message: "cannot read main.go"},
	}
	if err := reports.RecordAllocationReport(ctx, envelope); err != nil {
		t.Fatal(err)
	}
	var rendered string
	if err := pool.QueryRow(ctx,
		`SELECT report::text FROM allocation_execution_reports WHERE stage_execution_id = $1`, stage,
	).Scan(&rendered); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(rendered, "0123456789") {
		t.Fatal("persisted report keeps part of a long secret")
	}
	stored, err := reports.ListAllocationReports(ctx, stage)
	if err != nil || len(stored) != 1 {
		t.Fatalf("stored allocation reports = %d (%v)", len(stored), err)
	}
	worker := stored[0].Report.Worker
	arguments := worker.ToolCalls[0].Arguments
	if arguments["path"] != "data/main.go" || arguments["user"] != "[REDACTED]" ||
		arguments["note"] != "login [REDACTED]" {
		t.Fatal("persisted arguments were not matched by the shared rule")
	}
	if worker.Errors[0].Code != "admin_rejected" || worker.Errors[0].Message != "administrator login failed" ||
		worker.Errors[1].Code != "tool_call_failed" || worker.Errors[1].Message != "cannot read main.go" {
		t.Fatal("persisted error codes or messages were not matched by the shared rule")
	}
	if *stored[0].Report.Runtime.StopReason != "administrator stopped after [REDACTED]" {
		t.Fatal("persisted stop reason was not matched by the shared rule")
	}
}
