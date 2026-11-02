package public

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5/pgconn"
)

func TestToolDiagnosticOperationHasAnExplicitUnavailableResponse(t *testing.T) {
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(s *testServices) {
		s.RunDiagnostics = emptyAttemptDiagnostics{OwnerID: "user-1", RunID: "run-one", StageExecutionID: "stage-one"}
	})
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	response := contractCall(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, "/v1/runs/run-one/attempts/stage-one/tool-calls?participant=planner", nil), http.StatusOK)
	if !strings.Contains(response.Body.String(), `"status":"unavailable"`) {
		t.Fatalf("unavailable: %s", response.Body.String())
	}
}

func TestPublicToolDiagnosticsReadsExactOwnedRetainedFacts(t *testing.T) {
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	pool := isolatedPublicPool(t, ctx)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{LaneClass: resolved.LaneClassModel, RunID: "tool-run", OwnerID: "user-1", WorkflowName: "workflow", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{"name":"workflow"}`), Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()}); err != nil {
		t.Fatal(err)
	}
	for _, transition := range [][2]runstore.WorkflowRunState{{runstore.RunInitializing, runstore.RunPending}, {runstore.RunPending, runstore.RunRunning}} {
		if _, err := runs.TransitionRun(ctx, "tool-run", transition[0], transition[1], runstore.Reason{Code: "test"}); err != nil {
			t.Fatal(err)
		}
	}
	if err := runstoretest.CreateRunningStage(ctx, runs, "tool-run", "tool-stage", contracts.APIVersion); err != nil {
		t.Fatal(err)
	}
	repository := telemetry.NewRepository(pool)
	record, err := repository.RebuildStageMetrics(ctx, "tool-stage", contracts.APIVersion)
	if err != nil {
		t.Fatal(err)
	}
	metric := int64(9007199254740993)
	report := reporting.ExecutionReport{ReportID: "private-report-sentinel", Complete: true, Truncated: true, Metrics: reporting.ExecutionMetrics{Tools: map[string]reporting.ToolMetrics{}}, Errors: []reporting.ExecutionError{}, ToolCalls: []reporting.ToolCallRecord{
		{CallID: "private-call-sentinel", Tool: "probe", Outcome: reporting.ToolCallSucceeded, DurationMS: &metric, Arguments: reporting.ToolArguments{"number": json.Number("9007199254740993"), "url": "https://url-sentinel.invalid"}},
		{CallID: "private-second-call", Tool: "probe", Outcome: reporting.ToolCallFailed, Error: &reporting.ExecutionError{Code: "tool_call_failed", Message: "failed https://error-sentinel.invalid"}},
	}}
	record.Metrics.Planner = &report
	record.Metrics.Workers["Scanner.v2"] = report
	data, err := json.Marshal(record.Metrics)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE stage_metrics SET metrics=$2::jsonb WHERE stage_execution_id=$1`, "tool-stage", data); err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(s *testServices) { s.RunDiagnostics = repository })
	path := "/v1/runs/tool-run/attempts/tool-stage/tool-calls"
	read := func(path string, want int) string {
		response := contractCall(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, path, nil), want)
		if response.Header().Get("Cache-Control") != "no-store" {
			t.Fatal("diagnostics may be cached")
		}
		return response.Body.String()
	}
	first := read(path+"?participant=worker&logicalAgent=Scanner.v2&limit=1", http.StatusOK)
	var page toolDiagnosticPageResponse
	if err := json.Unmarshal([]byte(first), &page); err != nil {
		t.Fatal(err)
	}
	if len(page.Items) != 1 || page.Items[0].DurationMS == nil || *page.Items[0].DurationMS != "9007199254740993" || !page.DetailTruncated || page.Page.NextCursor == nil {
		t.Fatalf("first facts: %s", first)
	}
	for _, sentinel := range []string{"private-report-sentinel", "private-call-sentinel", "url-sentinel.invalid"} {
		if strings.Contains(first, sentinel) {
			t.Fatalf("leak: %s", sentinel)
		}
	}
	cursor := url.QueryEscape(*page.Page.NextCursor)
	next := read(path+"?participant=worker&logicalAgent=Scanner.v2&cursor="+cursor, http.StatusOK)
	if !strings.Contains(next, `"outcome":"failed"`) || strings.Contains(next, "error-sentinel.invalid") {
		t.Fatalf("next: %s", next)
	}
	read(path+"?participant=planner&cursor="+cursor, http.StatusBadRequest)
	read(path+"?participant=planner", http.StatusOK)
	for target, want := range map[string]int{
		path: http.StatusBadRequest, path + "?participant=runtime": http.StatusBadRequest, path + "?participant=worker": http.StatusBadRequest,
		path + "?participant=planner&logicalAgent=": http.StatusBadRequest, path + "?participant=planner&limit=101": http.StatusBadRequest,
		path + "?participant=planner&cursor=invalid": http.StatusBadRequest, strings.Replace(path, "tool-run", "foreign-run", 1) + "?participant=planner": http.StatusNotFound,
		strings.Replace(path, "tool-stage", "foreign-stage", 1) + "?participant=planner": http.StatusNotFound,
	} {
		read(target, want)
	}
	if _, err := pool.Exec(ctx, `UPDATE stage_metrics SET summary=summary WHERE stage_execution_id=$1`, "tool-stage"); err != nil {
		t.Fatal(err)
	}
	read(path+"?participant=worker&logicalAgent=Scanner.v2&cursor="+cursor, http.StatusConflict)
}

type failingToolDiagnostics struct {
	emptyAttemptDiagnostics
	err error
}

func (f failingToolDiagnostics) ReadToolCallDiagnostics(ctx context.Context, p telemetry.ToolDiagnosticParams) (telemetry.ToolDiagnosticPage, error) {
	if _, err := f.emptyAttemptDiagnostics.ReadToolCallDiagnostics(ctx, p); err != nil {
		return telemetry.ToolDiagnosticPage{}, err
	}
	return telemetry.ToolDiagnosticPage{}, f.err
}

func TestToolDiagnosticStorageConflictPreservesRetryPolicy(t *testing.T) {
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(s *testServices) {
		s.RunDiagnostics = failingToolDiagnostics{emptyAttemptDiagnostics: emptyAttemptDiagnostics{OwnerID: "user-1", RunID: "run-one", StageExecutionID: "stage-one"}, err: &pgconn.PgError{Code: persistencepostgres.SQLStateSerializationFailure}}
	})
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	response := contractCall(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, "/v1/runs/run-one/attempts/stage-one/tool-calls?participant=planner", nil), http.StatusServiceUnavailable)
	if !strings.Contains(response.Body.String(), "storage_transaction_conflict") || !strings.Contains(response.Body.String(), `"retryable":true`) {
		t.Fatalf("retry policy lost: %s", response.Body.String())
	}
}

func TestDiagnosticCursorScopeFramingCannotCollide(t *testing.T) {
	cursors := pageCursors{}
	for _, family := range []string{"runtime-diagnostics", "tool-diagnostics"} {
		left := diagnosticCursorKind(family, "owner", "a:b", "c", "worker", "Scanner.v2")
		right := diagnosticCursorKind(family, "owner", "a", "b:c", "worker", "Scanner.v2")
		if left == right {
			t.Fatal("scope tuple collision")
		}
		cursor, err := cursors.encodePageCursor(left, "generation", "1")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := cursors.decodePageCursor(cursor, right, 2); err == nil {
			t.Fatal("cursor crossed different Run/Stage tuples")
		}
	}
}
