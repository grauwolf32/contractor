package telemetry_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

func TestToolDiagnosticsPostgresRetainedPagesPrivacyAndBoundaries(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool)
	repo := telemetry.NewRepository(pool)
	createTelemetryRun(t, ctx, runs)
	stage := createTelemetryStage(t, ctx, runs, "tool-stage", "tool-allocation", "tool-session")
	p := telemetry.ToolDiagnosticParams{OwnerID: "user", RunID: "run-telemetry", StageExecutionID: stage, Participant: "worker", LogicalAgent: "builder", Limit: 100}
	if page, err := repo.ReadToolCallDiagnostics(ctx, p); err != nil || page.Status != "pending" {
		t.Fatalf("missing: %+v %v", page, err)
	}
	envelope := allocationEnvelope(stage, "tool-allocation", time.Now().UTC(), false)
	envelope.Report.Worker.ToolCalls = nil
	duration, size := int64(math.MaxInt64), int64(9007199254740993)
	for i := range 1000 {
		envelope.Report.Worker.ToolCalls = append(envelope.Report.Worker.ToolCalls, reporting.ToolCallRecord{
			CallID: fmt.Sprintf("private-call-%04d", i), Tool: "probe", Outcome: reporting.ToolCallFailed, DurationMS: &duration, ResultSizeBytes: &size,
			Arguments: reporting.ToolArguments{"large": json.Number("9007199254740993"), "quoted": strings.Repeat(`"`, 200)},
			Error:     &reporting.ExecutionError{Code: "probe_failed", Message: strings.Repeat("наблюдение", 10)},
		})
	}
	envelope.Report.Worker.ToolCalls[0].Arguments["authorization"] = postgresSecret
	envelope.Report.Worker.ToolCalls[0].Arguments["url"] = "https://url-sentinel.invalid/no-secret"
	if err := repo.RecordAllocationReport(ctx, envelope); err != nil {
		t.Fatal(err)
	}
	planner := plannerEnvelope(stage, "tool-session", time.Now().UTC())
	planner.Report.ToolCalls = []reporting.ToolCallRecord{{CallID: "private-planner", Tool: "a2a.invoke", Outcome: reporting.ToolCallSucceeded}}
	if err := repo.RecordPlannerReport(ctx, planner); err != nil {
		t.Fatal(err)
	}
	record, err := repo.RebuildStageMetrics(ctx, stage, contracts.APIVersion)
	if err != nil {
		t.Fatal(err)
	}
	if len(record.Metrics.Workers["builder"].ToolCalls) != 1000 {
		t.Fatalf("fixture did not admit 1000 calls: %d", len(record.Metrics.Workers["builder"].ToolCalls))
	}
	// The containing Worker map exceeds the selected-report budget. A large
	// corrupt sibling must neither cross the read boundary nor hide this report.
	if _, err := pool.Exec(ctx, `UPDATE stage_metrics SET metrics=jsonb_set(metrics,'{workers,unselected}',to_jsonb(repeat('unselected-sentinel-',600000))) WHERE stage_execution_id=$1`, stage); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `DELETE FROM allocation_execution_reports WHERE stage_execution_id=$1`, stage); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `DELETE FROM planner_execution_reports WHERE stage_execution_id=$1`, stage); err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	var generation string
	pages := 0
	for {
		page, err := repo.ReadToolCallDiagnostics(ctx, p)
		if err != nil {
			t.Fatal(err)
		}
		pages++
		if page.Status != "available" || page.Complete == nil || *page.Complete || page.DetailTruncated || len(page.Items) == 0 || len(page.Items) > 100 {
			t.Fatalf("page facts: %+v", page)
		}
		encoded, _ := json.Marshal(page)
		if len(encoded) > telemetry.MaxDiagnosticPageBytes-2048 {
			t.Fatalf("response bound: %d", len(encoded))
		}
		for _, sentinel := range []string{"private-call-", postgresSecret, "url-sentinel.invalid", "private-planner", "unselected-sentinel-"} {
			if strings.Contains(string(encoded), sentinel) {
				t.Fatalf("leak: %s", sentinel)
			}
		}
		for _, item := range page.Items {
			if seen[item.CallReference] {
				t.Fatalf("duplicate %s", item.CallReference)
			}
			seen[item.CallReference] = true
			if item.DurationMS == nil || *item.DurationMS != "9223372036854775807" || item.ResultSizeBytes == nil || *item.ResultSizeBytes != "9007199254740993" || item.ArgumentsJSON == nil || !strings.Contains(*item.ArgumentsJSON, "9007199254740993") {
				t.Fatalf("rounded facts: %+v", item)
			}
		}
		generation = page.Generation
		if !page.More {
			break
		}
		if page.NextOffset <= p.Offset {
			t.Fatal("nonadvancing page")
		}
		p.Generation, p.Offset = page.Generation, page.NextOffset
	}
	if len(seen) != 1000 || pages <= 10 {
		t.Fatalf("count/byte paging not exercised: %d records %d pages", len(seen), pages)
	}
	for _, change := range []func(*telemetry.ToolDiagnosticParams){func(q *telemetry.ToolDiagnosticParams) { q.OwnerID = "foreign" }, func(q *telemetry.ToolDiagnosticParams) { q.RunID = "foreign" }, func(q *telemetry.ToolDiagnosticParams) { q.StageExecutionID = "foreign" }} {
		q := p
		change(&q)
		if _, err := repo.ReadToolCallDiagnostics(ctx, q); !errors.Is(err, telemetry.ErrNotFound) {
			t.Fatalf("foreign: %v", err)
		}
	}
	q := p
	q.Participant, q.LogicalAgent, q.Generation, q.Offset = "planner", "", "", 0
	if page, err := repo.ReadToolCallDiagnostics(ctx, q); err != nil || len(page.Items) != 1 || page.Items[0].Tool != "a2a.invoke" || page.Items[0].DurationMS != nil {
		t.Fatalf("planner selection: %+v %v", page, err)
	}
	if _, err := pool.Exec(ctx, `UPDATE stage_metrics SET summary=summary WHERE stage_execution_id=$1`, stage); err != nil {
		t.Fatal(err)
	}
	p.Generation, p.Offset = generation, 1
	if _, err := repo.ReadToolCallDiagnostics(ctx, p); !errors.Is(err, telemetry.ErrConflict) {
		t.Fatalf("stale: %v", err)
	}
	p.Generation, p.Offset = "", 0
	cancelled, cancelRead := context.WithCancel(ctx)
	cancelRead()
	if _, err := repo.ReadToolCallDiagnostics(cancelled, p); err == nil {
		t.Fatal("cancelled read succeeded")
	}
	// One corrupt participant never hides the independently selected healthy one.
	if _, err := pool.Exec(ctx, `UPDATE stage_metrics SET metrics=jsonb_set(metrics,'{workers,builder}','{"unknown":true}'::jsonb) WHERE stage_execution_id=$1`, stage); err != nil {
		t.Fatal(err)
	}
	if page, err := repo.ReadToolCallDiagnostics(ctx, p); err != nil || page.Status != "unavailable" || page.Reason != "invalid_report" {
		t.Fatalf("corrupt: %+v %v", page, err)
	}
	if page, err := repo.ReadToolCallDiagnostics(ctx, q); err != nil || len(page.Items) != 1 {
		t.Fatalf("healthy sibling: %+v %v", page, err)
	}
	if _, err := pool.Exec(ctx, `UPDATE stage_metrics SET metrics=jsonb_set(metrics,'{workers,builder}',to_jsonb(repeat('x',9*1024*1024))) WHERE stage_execution_id=$1`, stage); err != nil {
		t.Fatal(err)
	}
	if page, err := repo.ReadToolCallDiagnostics(ctx, p); err != nil || page.Status != "unavailable" || page.Reason != "report_oversized" {
		t.Fatalf("oversize: %+v %v", page, err)
	}
}

func TestToolDiagnosticsPostgresRejectsArrayWorkerContainerWithNumericName(t *testing.T) {
	ctx := t.Context()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool)
	repo := telemetry.NewRepository(pool)
	createTelemetryRun(t, ctx, runs)
	stage := createTelemetryStage(t, ctx, runs, "array-tool-stage", "array-allocation", "array-session")
	if err := repo.RecordAllocationReport(ctx, allocationEnvelope(stage, "array-allocation", time.Now().UTC(), true)); err != nil {
		t.Fatal(err)
	}
	if _, err := repo.RebuildStageMetrics(ctx, stage, contracts.APIVersion); err != nil {
		t.Fatal(err)
	}
	p := telemetry.ToolDiagnosticParams{OwnerID: "user", RunID: "run-telemetry", StageExecutionID: stage, Participant: "worker", LogicalAgent: "0", Limit: 1}
	// Numeric names are valid, but must remain object keys rather than become
	// array indices when a direct JSONB path selects the retained participant.
	if _, err := pool.Exec(ctx, `UPDATE stage_metrics SET metrics=jsonb_set(metrics,'{workers}',jsonb_build_object('0',metrics->'workers'->'builder')) WHERE stage_execution_id=$1`, stage); err != nil {
		t.Fatal(err)
	}
	if page, err := repo.ReadToolCallDiagnostics(ctx, p); err != nil || page.Status != "available" {
		t.Fatalf("numeric object key: %+v %v", page, err)
	}
	// Model malformed retained storage only in this isolated test schema; the
	// normal write constraint rejects this shape before it can be persisted.
	if _, err := pool.Exec(ctx, `ALTER TABLE stage_metrics DROP CONSTRAINT stage_metrics_metrics_check`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE stage_metrics SET metrics=jsonb_set(metrics,'{workers}',jsonb_build_array(metrics->'workers'->'0')) WHERE stage_execution_id=$1`, stage); err != nil {
		t.Fatal(err)
	}
	if page, err := repo.ReadToolCallDiagnostics(ctx, p); err != nil || page.Status != "pending" || page.Reason != "report_missing" || len(page.Items) != 0 {
		t.Fatalf("array element became a selected report: %+v %v", page, err)
	}
}

func TestToolDiagnosticsPostgresAdmittedNearMiBSourceStringsRemainBounded(t *testing.T) {
	for _, field := range []string{"callId", "tool"} {
		t.Run(field, func(t *testing.T) {
			ctx := t.Context()
			pool := isolatedTelemetryPool(t, ctx)
			runs := runstore.NewPostgresStore(pool)
			repo := telemetry.NewRepository(pool)
			createTelemetryRun(t, ctx, runs)
			stage := createTelemetryStage(t, ctx, runs, "large-tool-stage", "large-allocation", "large-session")
			envelope := plannerEnvelope(stage, "large-session", time.Now().UTC())
			call := reporting.ToolCallRecord{CallID: "private-call", Tool: "probe", Outcome: reporting.ToolCallSucceeded}
			large := strings.Repeat("x", 1040000)
			if field == "callId" {
				call.CallID = large
			} else {
				call.Tool = large
			}
			envelope.Report.ToolCalls = []reporting.ToolCallRecord{call}
			if err := repo.RecordPlannerReport(ctx, envelope); err != nil {
				t.Fatal(err)
			}
			if _, err := repo.RebuildStageMetrics(ctx, stage, contracts.APIVersion); err != nil {
				t.Fatal(err)
			}
			page, err := repo.ReadToolCallDiagnostics(ctx, telemetry.ToolDiagnosticParams{OwnerID: "user", RunID: "run-telemetry", StageExecutionID: stage, Participant: "planner", Limit: 1})
			if err != nil || len(page.Items) != 1 || page.More || page.Items[0].ToolTruncated != (field == "tool") {
				t.Fatalf("large admitted record concealed: %+v %v", page, err)
			}
			encoded, _ := json.Marshal(page)
			if len(encoded) > 4096 || strings.Contains(string(encoded), large[:1024]) {
				t.Fatalf("source identity/name overread: %d", len(encoded))
			}
		})
	}
}

func TestToolDiagnosticsPostgresEmptyTruncatedReportAndActualExpiryAreDistinct(t *testing.T) {
	ctx := t.Context()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool)
	repo := telemetry.NewRepository(pool)
	createTelemetryRun(t, ctx, runs)
	stage := createTelemetryStage(t, ctx, runs, "lifecycle-tool-stage", "lifecycle-allocation", "lifecycle-session")
	envelope := allocationEnvelope(stage, "lifecycle-allocation", time.Now().UTC(), false)
	envelope.Report.Worker.ToolCalls = []reporting.ToolCallRecord{}
	envelope.Report.Worker.Truncated = true
	if err := repo.RecordAllocationReport(ctx, envelope); err != nil {
		t.Fatal(err)
	}
	if _, err := repo.RebuildStageMetrics(ctx, stage, contracts.APIVersion); err != nil {
		t.Fatal(err)
	}
	p := telemetry.ToolDiagnosticParams{OwnerID: "user", RunID: "run-telemetry", StageExecutionID: stage, Participant: "worker", LogicalAgent: "builder", Limit: 25}
	page, err := repo.ReadToolCallDiagnostics(ctx, p)
	if err != nil || page.Status != "available" || len(page.Items) != 0 || page.Complete == nil || *page.Complete || !page.DetailTruncated {
		t.Fatalf("empty truncated report: %+v %v", page, err)
	}
	finishTelemetryStage(t, ctx, runs, stage)
	if _, err := repo.CleanupExpired(ctx, time.Now().UTC().Add(31*24*time.Hour), 100); err != nil {
		t.Fatal(err)
	}
	page, err = repo.ReadToolCallDiagnostics(ctx, p)
	if err != nil || page.Status != "unavailable" || page.Reason != "report_missing" || len(page.Items) != 0 || page.Complete != nil {
		t.Fatalf("expired report became empty success: %+v %v", page, err)
	}
}
