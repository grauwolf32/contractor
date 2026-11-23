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

type progressReader struct {
	calls int
	fail  bool
}

func (r *progressReader) ReadWorkerProgress(_ context.Context, run, stage, allocation, name string) (*reporting.WorkerCompactionProgress, *reporting.WorkerModelPhases, error) {
	r.calls++
	if r.fail {
		return nil, nil, errors.New("State unavailable")
	}
	if run != "run-telemetry" || stage != "live-stage" || allocation != "live-allocation" || name != "builder" {
		return nil, nil, errors.New("incorrect authority")
	}
	return &reporting.WorkerCompactionProgress{InvocationID: "worker-1", Phase: "idle"}, nil, nil
}

func TestLiveWorkerDiagnosticsUseExactOwnedPageAndReleasePoolBeforeIO(t *testing.T) {
	ctx := t.Context()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	createTelemetryRun(t, ctx, runs)
	stage := createTelemetryStage(t, ctx, runs, "live-stage", "live-allocation", "live-session")
	reader := &progressReader{}
	reports := telemetry.NewRepository(pool).WithWorkerProgressReader(reader)
	params := telemetry.RuntimeDiagnosticParams{OwnerID: "foreign", RunID: "run-telemetry", StageExecutionID: stage, Limit: 1}
	if _, err := reports.ReadRuntimeDiagnostics(ctx, params); !errors.Is(err, telemetry.ErrNotFound) || reader.calls != 0 {
		t.Fatalf("foreign read reached Runtime: %v", err)
	}
	params.OwnerID = "user"
	page, err := reports.ReadRuntimeDiagnostics(ctx, params)
	if err != nil || len(page.Items) != 1 || page.Items[0].Compaction == nil || page.Items[0].Complete != nil || page.Items[0].Status != "unavailable" || page.More {
		t.Fatalf("live facts: %+v %v", page, err)
	}
	reader.fail = true
	if _, err := reports.ReadRuntimeDiagnostics(ctx, params); err == nil {
		t.Fatal("State transport failure became success")
	}
	reader.fail = false
	// A final report takes precedence: live State is no longer consulted.
	final := allocationEnvelope(stage, "live-allocation", time.Now().UTC(), true)
	final.Report.Worker.Compaction = &reporting.WorkerCompactionProgress{InvocationID: "worker-1", Phase: "applied", Generation: 1, Attempts: 1, InputBytes: 100, SummaryBytes: 20, BeforePrompt: &reporting.CompactionPromptMeasurement{Generation: 0}, AfterPrompt: &reporting.CompactionPromptMeasurement{Generation: 1}}
	final.Report.Worker.Metrics.ModelPhases = &reporting.WorkerModelPhases{Compaction: reporting.WorkerModelUsage{ModelCalls: 1, TokenUsageUnavailable: 1}}
	if err := reports.RecordAllocationReport(ctx, final); err != nil {
		t.Fatal(err)
	}
	if _, err := reports.RebuildStageMetrics(ctx, stage, contracts.APIVersion); err != nil {
		t.Fatal(err)
	}
	calls := reader.calls
	page, err = reports.ReadRuntimeDiagnostics(ctx, params)
	if err != nil || reader.calls != calls || page.Items[0].Complete == nil || page.Items[0].Compaction == nil || page.Items[0].Compaction.Generation != 1 || page.Items[0].ModelPhases == nil || page.Items[0].ModelPhases.Compaction.ModelCalls != 1 {
		t.Fatalf("retained report precedence: %+v %v", page, err)
	}
}
