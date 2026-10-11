package public

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// runReadInterleaving commits a change through another pool right after the
// Run detail has read the Run row and before its next read. It interleaves
// deterministically, without sleeps or production hooks.
type runReadInterleaving struct {
	mu      sync.Mutex
	runRead bool
	done    bool
	commit  func()
}

func (trace *runReadInterleaving) TraceQueryStart(
	ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData,
) context.Context {
	trace.mu.Lock()
	defer trace.mu.Unlock()
	switch {
	case trace.done:
	case strings.Contains(data.SQL, "metadata_labels FROM workflow_runs WHERE run_id = $1"):
		trace.runRead = true
	case trace.runRead:
		trace.done = true
		trace.commit()
	}
	return ctx
}

func (*runReadInterleaving) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

// API-R48: the event cursor of a Run detail belongs to the state it is shown
// with. A terminal transition committed while the detail is being read must
// either be in the response or after its cursor, never only in the cursor.
func TestPostgresRunDetailResponseKeepsStateWithItsCursor(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	runs := runstore.NewPostgresStore(pool)
	const runID = "run-detail-snapshot"
	if _, err := runs.CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: "user-1", WorkflowName: "artifact-copy", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`),
		RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, runID, runstore.RunInitializing, runstore.RunRunning, runstore.Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
	before, err := runs.GetRunEventCursor(ctx, runID)
	if err != nil {
		t.Fatal(err)
	}
	var transitionErr error
	trace := &runReadInterleaving{commit: func() {
		_, transitionErr = runs.TransitionRun(
			ctx, runID, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "stage_failed"},
		)
	}}
	configuration := pool.Config()
	configuration.MaxConns, configuration.MinConns = 1, 0
	configuration.ConnConfig.Tracer = trace
	readerPool, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(readerPool.Close)
	details, err := runservice.NewRunDetailReader(runservice.PostgresRunDetailTransaction(readerPool))
	if err != nil {
		t.Fatal(err)
	}
	h := &handler{dependencies: Dependencies{RunDetails: details}}
	request := httptest.NewRequest(http.MethodGet, "/v1/runs/"+runID, nil)
	request.SetPathValue("runID", runID)
	request = request.WithContext(auth.WithPrincipal(ctx, auth.Principal{UserID: "user-1"}))
	response := httptest.NewRecorder()
	h.getRun(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("GET Run = %d %s", response.Code, response.Body.String())
	}
	if !trace.done || transitionErr != nil {
		t.Fatalf("terminal transition was not committed between the reads: %v", transitionErr)
	}
	var detail runStatusResponse
	if err := json.Unmarshal(response.Body.Bytes(), &detail); err != nil {
		t.Fatal(err)
	}
	if detail.State != runstore.RunRunning || detail.Deletable || detail.FinishedAt != nil ||
		detail.EventCursor == nil || detail.EventCursor.Generation != before.Generation ||
		detail.EventCursor.Sequence != strconv.FormatInt(before.Sequence, 10) {
		t.Fatalf("response state %s deletable %t cursor %+v, want running with cursor %+v",
			detail.State, detail.Deletable, detail.EventCursor, before)
	}
	// Subscribing after the response cursor delivers the terminal transition.
	events, err := runs.ListRunEvents(ctx, runID, before.Sequence, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].Kind != runstore.RunEventLifecycleChanged ||
		!strings.Contains(string(events[0].Data), string(runstore.RunFailed)) {
		t.Fatalf("events after the response cursor = %+v", events)
	}
}
