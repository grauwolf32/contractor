package telemetry_test

import (
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

func TestPostgresCostFieldsKeepExactProjectSpendThroughReplayRetentionAndDeletion(t *testing.T) {
	ctx := t.Context()
	pool := isolatedTelemetryPool(t, ctx)
	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: "cost-project", OwnerID: "user", Kind: projectstore.KindProject, Name: "Cost", IdempotencyKey: "create-cost", RequestDigest: "sha256:" + strings.Repeat("a", 64)})
	if err != nil {
		t.Fatal(err)
	}
	store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	_, err = runstoretest.CreateRun(ctx, store, runstore.CreateRunParams{RunID: "run-telemetry", OwnerID: "user", ProjectID: &project.ProjectID, LaneClass: resolved.LaneClassModel, WorkflowName: "workflow", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: []byte(`{"name":"workflow"}`), Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = store.TransitionRun(ctx, "run-telemetry", runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
	if _, err = store.TransitionRun(ctx, "run-telemetry", runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
		t.Fatal(err)
	}
	stage := createTelemetryStage(t, ctx, store, "stage-cost", "allocation-cost", "session-cost")
	reports := runservice.NewCostReports(pool)
	when := time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)
	e := allocationEnvelope(stage, "allocation-cost", when, true)
	e.Report.Worker.Metrics.LLMCost = &reporting.LLMCost{Currency: "USD", Calls: 3, Groups: []reporting.LLMCostGroup{{Model: "worker-model", Phase: "main", Source: "provider", Calls: 3, Amount: "0.1"}}}
	// A parent write failure must roll back the diagnostic receipt, participant
	// snapshot and Run subtotal together, so retry remains safe.
	if _, err := pool.Exec(ctx, `UPDATE projects SET llm_cost='{"currency":"EUR"}'::jsonb WHERE project_id=$1`, project.ProjectID); err != nil {
		t.Fatal(err)
	}
	if err := reports.RecordAllocationReport(ctx, e); err == nil {
		t.Fatal("invalid parent cost committed a report")
	}
	assertTelemetryCount(t, ctx, pool, "allocation_execution_reports", stage, 0)
	zero, _ := reporting.NewLLMCostTotals(false).Encode()
	if _, err := pool.Exec(ctx, `UPDATE projects SET llm_cost=$2::jsonb WHERE project_id=$1`, project.ProjectID, zero); err != nil {
		t.Fatal(err)
	}
	partial := allocationEnvelope(stage, "allocation-cost", when.Add(-time.Minute), false)
	partial.Report.ReportID = "allocation-partial-cost"
	partial.Report.Worker.ReportID = "worker-partial-cost"
	partial.Report.Worker.Metrics.LLMCost = &reporting.LLMCost{Currency: "USD", Calls: 1, Groups: []reporting.LLMCostGroup{{Model: "worker-model", Phase: "main", Source: "provider", Calls: 1, Amount: "0.05"}}}
	if err := reports.RecordAllocationReport(ctx, partial); err != nil {
		t.Fatal(err)
	}
	if v, err := reports.GetLLMCost(ctx, "user", "project", project.ProjectID); err != nil || v.Coverage != "partial" || v.Amount == nil || *v.Amount != "0.05" {
		t.Fatalf("partial: %+v %v", v, err)
	}
	var wg sync.WaitGroup
	failures := make(chan error, 8)
	for range 8 {
		wg.Go(func() { failures <- reports.RecordAllocationReport(ctx, e) })
	}
	wg.Wait()
	close(failures)
	for err := range failures {
		if err != nil {
			t.Fatal(err)
		}
	}
	p := plannerEnvelope(stage, "session-cost", when)
	p.Report.Metrics.LLMCost = &reporting.LLMCost{Currency: "USD", Calls: 2, Groups: []reporting.LLMCostGroup{{Model: "planner-model", Phase: "main", Source: "gateway", Calls: 2, Amount: "0.2"}}}
	if err := reports.RecordPlannerReport(ctx, p); err != nil {
		t.Fatal(err)
	}
	require := func(scope, id string) {
		t.Helper()
		v, err := reports.GetLLMCost(ctx, "user", scope, id)
		if err != nil || v.Amount == nil || *v.Amount != "0.3" || v.Calls != "5" || v.Reports != "2" || v.Basis != "mixed" || v.HistoricalGap {
			t.Fatalf("%s cost=%+v: %v", scope, v, err)
		}
	}
	require("run", "run-telemetry")
	require("project", project.ProjectID)
	if _, err := reports.GetLLMCost(ctx, "foreign", "project", project.ProjectID); !errors.Is(err, telemetry.ErrNotFound) {
		t.Fatalf("foreign cost: %v", err)
	}
	finishTelemetryStage(t, ctx, store, stage)
	if _, err := reports.CleanupExpired(ctx, when.Add(60*24*time.Hour), 100); err != nil {
		t.Fatal(err)
	}
	assertTelemetryCount(t, ctx, pool, "allocation_execution_reports", stage, 0)
	require("project", project.ProjectID)
	// A replay after full diagnostics expired still finds its last counted
	// contribution in the existing allocation/Stage record.
	if err := reports.RecordAllocationReport(ctx, e); err != nil {
		t.Fatal(err)
	}
	require("run", "run-telemetry")
	if _, err := store.TransitionRun(ctx, "run-telemetry", runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "finished"}); err != nil {
		t.Fatal(err)
	}
	if err := store.MarkStageAllocationReleased(ctx, "allocation-cost"); err != nil {
		t.Fatal(err)
	}
	if err := rundeletion.New(pool, artifactpgtest.NewStorage()).DeleteReleasedTerminalRun(ctx, "user", "run-telemetry"); err != nil {
		t.Fatal(err)
	}
	require("project", project.ProjectID)
	if _, err := pool.Exec(ctx, `DELETE FROM projects WHERE project_id=$1`, project.ProjectID); err != nil {
		t.Fatal(err)
	}
	if _, err := reports.GetLLMCost(ctx, "user", "project", project.ProjectID); !errors.Is(err, telemetry.ErrNotFound) {
		t.Fatalf("deleted Project retained cost resource: %v", err)
	}
}
