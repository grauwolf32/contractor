package runservice

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

func TestSharedPostgresAuditLLMCostUsesExistingFieldsAndDoesNotAdvanceRevision(t *testing.T) {
	f := newCommonTaskRunFixture(t, false)
	ctx := t.Context()
	created, err := f.service.CreateAudit(ctx, f.params)
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(f.pool, artifactpgtest.NewStorage())
	run, err := runs.GetRun(ctx, created.Run.RunID)
	if err != nil {
		t.Fatal(err)
	}
	if run.State == runstore.RunInitializing {
		run, err = runs.TransitionRun(ctx, run.RunID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"})
		if err != nil {
			t.Fatal(err)
		}
	}
	if run.State == runstore.RunPending {
		run, err = runs.TransitionRun(ctx, run.RunID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"})
		if err != nil {
			t.Fatal(err)
		}
	}
	_, err = runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{LaneClass: resolved.LaneClassModel, StageExecutionID: "cost-stage", RunID: run.RunID, StageName: "cost-stage", Attempt: 1, StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{"objective":"cost"}`), StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{}})
	if err != nil {
		t.Fatal(err)
	}
	started, _ := json.Marshal(map[string]string{"stageExecutionId": "cost-stage", "sessionId": "cost-session", "invocationId": "cost-invocation"})
	if err := runs.StartPlanner(ctx, runstore.StartPlannerParams{StageExecutionID: "cost-stage", SessionID: "cost-session", InvocationID: "cost-invocation", StateSchemaVersion: contracts.APIVersion, InitialState: json.RawMessage(`{"step":0}`), RunEvent: runstore.RunEventAppend{EventID: "cost-planner-started", EventSchemaVersion: contracts.APIVersion, Kind: runstore.RunEventPlannerStarted, Data: started}, Reason: runstore.Reason{Code: "planner_started"}}); err != nil {
		t.Fatal(err)
	}
	before, err := f.audits.Get(ctx, "owner", f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	when := time.Date(2026, 11, 8, 0, 0, 0, 0, time.UTC)
	calls := int64(1)
	e := telemetry.PlannerReportEnvelope{StageExecutionID: "cost-stage", SessionID: "cost-session", InvocationID: "cost-invocation", StartedAt: when.Add(-time.Second), FinishedAt: when, ReportSchemaVersion: contracts.APIVersion, ReceivedAt: when, Report: reporting.ExecutionReport{ReportID: "cost-report", Complete: true, Metrics: reporting.ExecutionMetrics{ModelCalls: &calls, Tools: map[string]reporting.ToolMetrics{}, LLMCost: &reporting.LLMCost{Currency: "USD", Calls: 1, Groups: []reporting.LLMCostGroup{{Model: "openrouter/test", Phase: "main", Source: "provider", Calls: 1, Amount: "0.00000625"}}}}, ToolCalls: []reporting.ToolCallRecord{}, Errors: []reporting.ExecutionError{}}}
	reports := NewCostReports(f.pool)
	for range 2 {
		if err := reports.RecordPlannerReport(ctx, e); err != nil {
			t.Fatal(err)
		}
	}
	for scope, id := range map[string]string{"run": run.RunID, "audit": f.audit.AuditID, "project": f.audit.ProjectID} {
		got, err := reports.GetLLMCost(ctx, "owner", scope, id)
		if err != nil || got.Amount == nil || *got.Amount != "0.00000625" || got.Reports != "1" || got.Calls != "1" || got.Basis != "reported" {
			t.Fatalf("%s: %+v %v", scope, got, err)
		}
		if scope == "project" && (got.ChecksAmount == nil || *got.ChecksAmount != "0.00000625" || got.StandaloneAmount == nil || *got.StandaloneAmount != "0") {
			t.Fatalf("child Run counted twice: %+v", got)
		}
	}
	after, err := f.audits.Get(ctx, "owner", f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if after.Revision != before.Revision || !after.UpdatedAt.Equal(before.UpdatedAt) {
		t.Fatal("monetary observation changed authoritative Audit revision")
	}
	var raw []byte
	if err := f.pool.QueryRow(ctx, `SELECT llm_cost FROM audit_executions WHERE execution_id=$1`, f.params.ExecutionID).Scan(&raw); err != nil {
		t.Fatal(err)
	}
	snapshot, err := reporting.DecodeLLMCostTotals(raw)
	if err != nil || snapshot.Amount != "0.00000625" {
		t.Fatalf("retained child cost: %+v %v", snapshot, err)
	}
	items, err := reports.GetAuditExecutionCosts(ctx, "owner", f.audit.AuditID, []string{f.params.ExecutionID})
	if err != nil || len(items) != 1 || items[0].Amount == nil || *items[0].Amount != "0.00000625" {
		t.Fatalf("bounded Work cost: %+v %v", items, err)
	}
	if _, err := reports.GetAuditExecutionCosts(ctx, "foreign", f.audit.AuditID, []string{f.params.ExecutionID}); err == nil {
		t.Fatal("foreign execution cost readable")
	}
}
