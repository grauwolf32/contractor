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

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

func TestRuntimeDiagnosticsPostgresBoundsOwnershipGenerationAndRetainedAuthority(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	reports := telemetry.NewRepository(pool)
	createTelemetryRun(t, ctx, runs)
	stage := createTelemetryStage(t, ctx, runs, "diagnostic-stage", "diagnostic-allocation", "diagnostic-session")
	params := telemetry.RuntimeDiagnosticParams{OwnerID: "user", RunID: "run-telemetry", StageExecutionID: stage, Limit: 10}
	missing, err := reports.ReadRuntimeDiagnostics(ctx, params)
	if err != nil || missing.Status != "pending" || len(missing.Items) != 0 {
		t.Fatalf("missing diagnostics: %+v %v", missing, err)
	}
	envelope := allocationEnvelope(stage, "diagnostic-allocation", time.Now().UTC(), true)
	secret := "diagnostic-secret-sentinel-421"
	envelope.Secrets = []string{secret}
	envelope.Report.Runtime.StopReason = &secret
	if err := reports.RecordAllocationReport(ctx, envelope); err != nil {
		t.Fatal(err)
	}
	record, err := reports.RebuildStageMetrics(ctx, stage, contracts.APIVersion)
	if err != nil {
		t.Fatal(err)
	}
	stop := strings.Repeat("наблюдение", 2000)
	for i := range 1000 {
		name := fmt.Sprintf("worker-%04d", i)
		record.Metrics.Runtime[name] = reporting.RuntimeReport{Complete: true, StopReason: &stop, Adapters: map[runtimesettings.RuntimeAdapterRef]reporting.RuntimeAdapterMetrics{
			runtimesettings.RuntimeAdapterOTLPHTTP: {Operations: math.MaxUint64, FailedOperations: 1, DroppedSpans: &reporting.DroppedSpanCounts{Shutdown: math.MaxUint64}},
		}}
		record.Metrics.Workers[name] = reporting.ExecutionReport{ReportID: name, Complete: true, Metrics: reporting.ExecutionMetrics{Tools: map[string]reporting.ToolMetrics{}}, ToolCalls: []reporting.ToolCallRecord{}, Errors: []reporting.ExecutionError{}}
	}
	if err := record.Metrics.Validate(); err != nil {
		t.Fatal(err)
	}
	// Real logical Worker names include upper case, periods and initial digits.
	for _, name := range []string{"Scanner.v2", "1scanner"} {
		record.Metrics.Runtime[name] = reporting.RuntimeReport{Complete: true, Adapters: map[runtimesettings.RuntimeAdapterRef]reporting.RuntimeAdapterMetrics{}}
	}
	// An invalid sibling in retained storage must not hide healthy participants
	// or consume a cursor window, including an oversized key that sorts first.
	record.Metrics.Runtime["!"+strings.Repeat("x", 10000)] = reporting.RuntimeReport{Complete: true}
	encoded, err := json.Marshal(record.Metrics)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE stage_metrics SET metrics=$2::jsonb WHERE stage_execution_id=$1`, stage, encoded); err != nil {
		t.Fatal(err)
	}
	// The retained read remains useful after original report cleanup (as with
	// Eval-held StageMetrics); it never reselects vanished source reports.
	if _, err := pool.Exec(ctx, `DELETE FROM allocation_execution_reports WHERE stage_execution_id=$1`, stage); err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	var generation string
	for {
		page, err := reports.ReadRuntimeDiagnostics(ctx, params)
		if err != nil {
			t.Fatal(err)
		}
		if page.Status != "available" || page.Reason != "invalid_report" || len(page.Items) > 10 {
			t.Fatalf("page: %+v", page)
		}
		data, _ := json.Marshal(page)
		if strings.Contains(string(data), secret) {
			t.Fatal("persisted secret reached diagnostics")
		}
		if len(data) > telemetry.MaxDiagnosticPageBytes-2048 {
			t.Fatalf("oversized page: %d", len(data))
		}
		for _, item := range page.Items {
			if seen[item.LogicalAgent] {
				t.Fatalf("duplicate %s", item.LogicalAgent)
			}
			seen[item.LogicalAgent] = true
			if strings.HasPrefix(item.LogicalAgent, "worker-") {
				if !item.StopReasonTruncated || len(*item.StopReason) > 4096 || item.Adapters[0].Operations != "18446744073709551615" {
					t.Fatalf("bounded exact facts lost: %+v", item)
				}
			}
		}
		generation = page.Generation
		if !page.More {
			break
		}
		params.Generation, params.After = page.Generation, page.After
	}
	if len(seen) != 1003 || !seen["Scanner.v2"] || !seen["1scanner"] {
		t.Fatalf("read %d participants", len(seen))
	}
	for _, change := range []func(*telemetry.RuntimeDiagnosticParams){func(p *telemetry.RuntimeDiagnosticParams) { p.OwnerID = "foreign" }, func(p *telemetry.RuntimeDiagnosticParams) { p.RunID = "another-run" }, func(p *telemetry.RuntimeDiagnosticParams) { p.StageExecutionID = "another-stage" }} {
		other := params
		change(&other)
		if _, err := reports.ReadRuntimeDiagnostics(ctx, other); !errors.Is(err, telemetry.ErrNotFound) {
			t.Fatalf("foreign read: %v", err)
		}
	}
	// An update in a new transaction invalidates cursors even if a fixture
	// preserves the same wall-clock timestamp: tuple generation also changes.
	if _, err := pool.Exec(ctx, `UPDATE stage_metrics SET summary=summary WHERE stage_execution_id=$1`, stage); err != nil {
		t.Fatal(err)
	}
	params.After = "builder"
	params.Generation = generation
	if _, err := reports.ReadRuntimeDiagnostics(ctx, params); !errors.Is(err, telemetry.ErrConflict) {
		t.Fatalf("stale cursor: %v", err)
	}
	cancelled, cancelRead := context.WithCancel(ctx)
	cancelRead()
	params.Generation, params.After = "", ""
	if _, err := reports.ReadRuntimeDiagnostics(cancelled, params); err == nil {
		t.Fatal("cancelled read succeeded")
	}
	pool.Close()
	if _, err := reports.ReadRuntimeDiagnostics(ctx, params); err == nil {
		t.Fatal("unavailable real store became a successful diagnostic read")
	}
}
