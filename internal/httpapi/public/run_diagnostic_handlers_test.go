package public

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/routers/gorillamux"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

func TestPublicRuntimeDiagnosticsReadsExactOwnedRetainedFacts(t *testing.T) {
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	pool := isolatedPublicPool(t, ctx)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{
		LaneClass: resolved.LaneClassModel, RunID: "diagnostic-run", OwnerID: "user-1", WorkflowName: "workflow", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{"name":"workflow"}`), Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	for _, transition := range [][2]runstore.WorkflowRunState{{runstore.RunInitializing, runstore.RunPending}, {runstore.RunPending, runstore.RunRunning}} {
		if _, err := runs.TransitionRun(ctx, "diagnostic-run", transition[0], transition[1], runstore.Reason{Code: "test"}); err != nil {
			t.Fatal(err)
		}
	}
	if err := runstoretest.CreateRunningStage(ctx, runs, "diagnostic-run", "diagnostic-stage", contracts.APIVersion); err != nil {
		t.Fatal(err)
	}
	repository := telemetry.NewRepository(pool)
	record, err := repository.RebuildStageMetrics(ctx, "diagnostic-stage", contracts.APIVersion)
	if err != nil {
		t.Fatal(err)
	}
	flush := false
	for _, name := range []string{"first", "second"} {
		record.Metrics.Runtime[name] = reporting.RuntimeReport{Complete: false, Adapters: map[runtimesettings.RuntimeAdapterRef]reporting.RuntimeAdapterMetrics{
			runtimesettings.RuntimeAdapterOTLPHTTP: {Operations: math.MaxUint64, FailedOperations: 1, FlushAttempted: &flush, FlushSucceeded: &flush, DroppedSpans: &reporting.DroppedSpanCounts{Shutdown: 9007199254740993}},
		}}
	}
	data, err := json.Marshal(record.Metrics)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE stage_metrics SET metrics=$2::jsonb WHERE stage_execution_id=$1`, "diagnostic-stage", data); err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(services *testServices) { services.RunDiagnostics = repository })
	path := "/v1/runs/diagnostic-run/attempts/diagnostic-stage/runtime-diagnostics"
	read := func(path string) *httptest.ResponseRecorder {
		return serveAndValidatePublicContract(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, path, nil), false)
	}
	first := read(path + "?limit=1")
	var page runtimeDiagnosticPageResponse
	if first.Code != http.StatusOK || json.Unmarshal(first.Body.Bytes(), &page) != nil {
		t.Fatalf("diagnostic read: %d %s", first.Code, first.Body.String())
	}
	if first.Header().Get("Cache-Control") != "no-store" || len(page.Items) != 1 || !page.Page.HasMore || page.Page.NextCursor == nil {
		t.Fatalf("first page: %+v", page)
	}
	if page.Items[0].Status != "partial" || page.Items[0].DurationMS != nil || page.Items[0].Adapters[0].Operations != "18446744073709551615" || page.Items[0].Adapters[0].DroppedSpans.Shutdown != "9007199254740993" {
		t.Fatalf("facts lost: %+v", page.Items)
	}
	next := read(path + "?cursor=" + url.QueryEscape(*page.Page.NextCursor))
	if next.Code != http.StatusOK || !strings.Contains(next.Body.String(), `"logicalAgent":"second"`) {
		t.Fatalf("next page: %d %s", next.Code, next.Body.String())
	}
	if _, err := pool.Exec(ctx, `UPDATE stage_metrics SET summary=summary WHERE stage_execution_id=$1`, "diagnostic-stage"); err != nil {
		t.Fatal(err)
	}
	if stale := read(path + "?cursor=" + url.QueryEscape(*page.Page.NextCursor)); stale.Code != http.StatusConflict {
		t.Fatalf("stale page: %d %s", stale.Code, stale.Body.String())
	}
	for target, want := range map[string]int{
		path + "?limit=11":       http.StatusBadRequest,
		path + "?cursor=invalid": http.StatusBadRequest,
		strings.Replace(path, "diagnostic-run", "foreign-run", 1):     http.StatusNotFound,
		strings.Replace(path, "diagnostic-stage", "foreign-stage", 1): http.StatusNotFound,
	} {
		if response := read(target); response.Code != want {
			t.Fatalf("%s: %d %s", target, response.Code, response.Body.String())
		}
	}
}

func TestRuntimeDiagnosticOperationHasAnExplicitUnavailableResponse(t *testing.T) {
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(s *testServices) {
		s.RunDiagnostics = emptyAttemptDiagnostics{OwnerID: "user-1", RunID: "run-one", StageExecutionID: "stage-one"}
	})
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	response := contractCall(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, "/v1/runs/run-one/attempts/stage-one/runtime-diagnostics", nil), http.StatusOK)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"status":"unavailable"`) {
		t.Fatalf("unavailable: %d %s", response.Code, response.Body.String())
	}
}
