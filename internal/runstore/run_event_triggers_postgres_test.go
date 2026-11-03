package runstore

import (
	"context"
	"encoding/json"
	"maps"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
)

// TestPostgresRunEventAppendsFireNoWorkflowRunTrigger requires that the
// Planner's Run event appends, which only advance next_run_event_sequence,
// fire no trigger of workflow_runs, deferred checks included, while a state
// change still fires the lifecycle capture and an assignment of a guarded
// column still fires its guard (PER-R22). The lifecycle events that the
// database appends from its own triggers advance the same single column, but
// they run nested, where EXPLAIN does not see them.
func TestPostgresRunEventAppendsFireNoWorkflowRunTrigger(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	run := createTestRun(t, ctx, NewPostgresStore(pool, artifactpgtest.NewStorage()), "run-event-triggers")
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	if _, err := tx.Exec(ctx, `SET CONSTRAINTS ALL IMMEDIATE`); err != nil {
		t.Fatal(err)
	}
	// The probe runs each statement first under EXPLAIN ANALYZE in a
	// savepoint, so a store operation that joins its caller's transaction is
	// probed too. EXPLAIN does not see the triggers of statements a trigger
	// runs.
	probe := explaintest.NewProbe(t, tx)
	probed := NewPostgresStore(probe, artifactpgtest.NewStorage())
	expectFired := func(step string, want map[string]int64, action func() error) {
		t.Helper()
		probe.Reset()
		if err := action(); err != nil {
			t.Fatalf("%s: %v", step, err)
		}
		if fired := probe.Fired("workflow_runs"); !maps.Equal(fired, want) {
			t.Errorf("%s fired the workflow_runs triggers %v, want %v", step, fired, want)
		}
	}
	expectFired("initialization", map[string]int64{
		"eval_runs_projection": 1, "workflow_runs_capture_lifecycle_event_on_update": 1,
	}, func() error {
		_, err := probed.TransitionRun(ctx, run.RunID, RunInitializing, RunPending, Reason{Code: "initialized"})
		return err
	})
	expectFired("Skill snapshot assignment", map[string]int64{"workflow_runs_protect_skill_snapshot": 1}, func() error {
		_, err := probe.Exec(ctx, `UPDATE workflow_runs SET skill_snapshot = skill_snapshot WHERE run_id = $1`, run.RunID)
		return err
	})

	store := NewPostgresStore(tx, artifactpgtest.NewStorage())
	if _, err := store.TransitionRun(ctx, run.RunID, RunPending, RunRunning, Reason{Code: "admitted"}); err != nil {
		t.Fatal(err)
	}
	execution, err := store.CreateStageExecution(ctx, CreateStageExecutionParams{
		LaneClass:        resolved.LaneClassModel,
		StageExecutionID: "stage-event-triggers", RunID: run.RunID, StageName: "copy", Attempt: 1,
		StageSpecSchemaVersion:    contracts.APIVersion,
		StageSpecSnapshot:         json.RawMessage(`{"objective":"copy"}`),
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext:              StageContextSnapshot{Parameters: map[string]string{}, Artifacts: map[string]PinnedContextArtifact{}},
	})
	if err != nil {
		t.Fatal(err)
	}
	identity := plannerRunEventData{
		StageExecutionID: execution.StageExecutionID, SessionID: "session-triggers", InvocationID: "invocation-triggers",
	}
	expectFired("Planner start", map[string]int64{}, func() error {
		return probed.StartPlanner(ctx, StartPlannerParams{
			StageExecutionID: execution.StageExecutionID,
			SessionID:        identity.SessionID, InvocationID: identity.InvocationID,
			StateSchemaVersion: contracts.APIVersion, InitialState: json.RawMessage(`{"step":0}`),
			RunEvent: RunEventAppend{
				EventID: "event-triggers-started", EventSchemaVersion: contracts.APIVersion,
				Kind: RunEventPlannerStarted, Data: mustRunEventJSON(t, identity),
			},
			Reason: Reason{Code: "planner_started"},
		})
	})
	expectFired("Planner event", map[string]int64{}, func() error {
		return probed.AppendPlannerEvent(ctx, AppendPlannerEventParams{
			SessionID: identity.SessionID, StageExecutionID: execution.StageExecutionID,
			InvocationID: identity.InvocationID, SequenceNumber: 2,
			NewStateSchemaVersion: contracts.APIVersion, NewState: json.RawMessage(`{"step":1}`),
			RunEvent: RunEventAppend{
				EventID: "event-triggers-request", EventSchemaVersion: contracts.APIVersion,
				Kind: RunEventPlannerRequestRecorded, Data: mustRunEventJSON(t, identity),
			},
		})
	})
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	page, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).ListRunEvents(ctx, run.RunID, 0, 20)
	events := page.Events
	if err != nil || len(events) < 2 || page.More || events[len(events)-2].EventID != "event-triggers-started" ||
		events[len(events)-1].EventID != "event-triggers-request" {
		t.Fatalf("Run events = (%+v, %v)", events, err)
	}
}
