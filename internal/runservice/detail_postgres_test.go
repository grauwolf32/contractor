package runservice

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// runDetailInterleaving commits a concurrent change through another pool
// right after a Run detail has read the Run row and before its next read. It
// interleaves deterministically, without sleeps or production hooks.
type runDetailInterleaving struct {
	mu      sync.Mutex
	runRead bool
	done    bool
	commit  func()
}

func (trace *runDetailInterleaving) TraceQueryStart(
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

func (*runDetailInterleaving) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

func (trace *runDetailInterleaving) interleaved() bool {
	trace.mu.Lock()
	defer trace.mu.Unlock()
	return trace.done
}

// newRunDetailReaderPool opens a one-connection pool on the same database, so a
// reader that took a second connection for a lookup would block instead of
// passing.
func newRunDetailReaderPool(t *testing.T, ctx context.Context, pool *pgxpool.Pool, tracer pgx.QueryTracer) *pgxpool.Pool {
	t.Helper()
	configuration := pool.Config()
	configuration.MaxConns, configuration.MinConns = 1, 0
	configuration.ConnConfig.Tracer = tracer
	readerPool, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(readerPool.Close)
	return readerPool
}

func createRunningDetailRun(t *testing.T, ctx context.Context, runs *runstore.PostgresStore, owner, runID string) {
	t.Helper()
	if _, err := runs.CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: owner, WorkflowName: "detail", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`),
		RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
		t.Fatal(err)
	}
}

// A terminal transition that commits between the Run read and the cursor read
// must not move the cursor past the state the detail shows: the UI subscribes
// from that cursor and would never see the transition (API-R48).
func TestPostgresRunDetailReadsStateAndCursorFromOneSnapshot(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunServicePool(t, ctx, databaseURL)
	runs := runstore.NewPostgresStore(pool)
	const owner, runID = "owner-detail", "run-detail"
	createRunningDetailRun(t, ctx, runs, owner, runID)
	before, err := runs.GetRunEventCursor(ctx, runID)
	if err != nil {
		t.Fatal(err)
	}
	var transitionErr error
	trace := &runDetailInterleaving{commit: func() {
		_, transitionErr = runs.TransitionRun(
			ctx, runID, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "stage_failed"},
		)
	}}
	reader, err := NewRunDetailReader(PostgresRunDetailTransaction(newRunDetailReaderPool(t, ctx, pool, trace)))
	if err != nil {
		t.Fatal(err)
	}

	detail, err := reader.Read(ctx, owner, runID)
	if err != nil {
		t.Fatal(err)
	}
	if !trace.interleaved() || transitionErr != nil {
		t.Fatalf("terminal transition was not committed between the reads: %v", transitionErr)
	}
	after, err := runs.GetRunEventCursor(ctx, runID)
	if err != nil {
		t.Fatal(err)
	}
	if after.Generation != before.Generation || after.Sequence != before.Sequence+1 {
		t.Fatalf("terminal transition cursor = %+v, before %+v", after, before)
	}
	// Every part of the detail shows the state the cursor belongs to.
	if detail.Run.State != runstore.RunRunning || detail.Run.FinishedAt != nil ||
		detail.EventCursor != before || detail.ResumeStageExecutionID != nil ||
		detail.DeletionBlocker == nil || *detail.DeletionBlocker != rundeletion.RunNotTerminal {
		t.Fatalf("Run detail mixes snapshots: state=%s finished=%v cursor=%+v (before %+v, after %+v) blocker=%v",
			detail.Run.State, detail.Run.FinishedAt, detail.EventCursor, before, after, detail.DeletionBlocker)
	}
	// A subscription after the detail cursor delivers the terminal transition.
	events, err := runs.ListRunEvents(ctx, runID, detail.EventCursor.Sequence, 10)
	if err != nil {
		t.Fatal(err)
	}
	var lifecycle struct {
		Resource string `json:"resource"`
		State    string `json:"state"`
	}
	if len(events) != 1 || events[0].Kind != runstore.RunEventLifecycleChanged ||
		json.Unmarshal(events[0].Data, &lifecycle) != nil ||
		lifecycle.Resource != "run" || lifecycle.State != string(runstore.RunFailed) {
		t.Fatalf("events after the detail cursor = %+v", events)
	}

	current, err := reader.Read(ctx, owner, runID)
	if err != nil {
		t.Fatal(err)
	}
	if current.Run.State != runstore.RunFailed || current.Run.FinishedAt == nil ||
		current.EventCursor != after || current.DeletionBlocker != nil {
		t.Fatalf("current Run detail = state %s cursor %+v blocker %v", current.Run.State, current.EventCursor, current.DeletionBlocker)
	}
	if _, err := reader.Read(ctx, "another-owner", runID); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("another owner's Run detail error = %v", err)
	}
}

// StageMetrics are optional diagnostics. A failing metrics statement aborts a
// PostgreSQL transaction, so the detail reads it in a savepoint and still
// returns the durable Run state from the same snapshot.
func TestPostgresRunDetailSurvivesFailedOptionalMetricsRead(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunServicePool(t, ctx, databaseURL)
	runs := runstore.NewPostgresStore(pool)
	const owner, runID = "owner-metrics", "run-metrics"
	createRunningDetailRun(t, ctx, runs, owner, runID)
	if _, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: "stage-metrics", RunID: runID, StageName: "copy", Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{}`),
		StageContextSchemaVersion: contracts.APIVersion,
	}); err != nil {
		t.Fatal(err)
	}
	// The isolated database is this test's own; hiding the table makes only
	// the metrics statement fail on the server.
	if _, err := pool.Exec(ctx, `ALTER TABLE stage_metrics RENAME TO stage_metrics_hidden`); err != nil {
		t.Fatal(err)
	}
	reader, err := NewRunDetailReader(PostgresRunDetailTransaction(newRunDetailReaderPool(t, ctx, pool, nil)))
	if err != nil {
		t.Fatal(err)
	}
	detail, err := reader.Read(ctx, owner, runID)
	if err != nil {
		t.Fatalf("failed optional metrics read lost the Run detail: %v", err)
	}
	if detail.Run.State != runstore.RunRunning || len(detail.Executions) != 1 ||
		detail.Executions[0].StageExecutionID != "stage-metrics" || len(detail.Metrics) != 0 {
		t.Fatalf("Run detail = state %s executions %+v metrics %+v", detail.Run.State, detail.Executions, detail.Metrics)
	}
}
