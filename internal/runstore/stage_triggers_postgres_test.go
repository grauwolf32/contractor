package runstore

import (
	"context"
	"encoding/json"
	"maps"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

// TestPostgresStageExecutionUpdatesFireOnlyTheTriggersTheyConcern drives two
// StageExecutions through every store transition and counts the triggers of
// stage_executions each one calls (PER-R22). An admission and a counted
// output publication failure, which change no guarded column and no state,
// call none of them; a transition calls the lifecycle capture and the Eval
// projection, and the immutability guard only when it assigns a guarded
// column; an INSERT still runs the manual continuation check and acquires
// the exact revisions the Stage holds, and only recording a candidate or an
// accepted result updates those holds (DEL-R62). Assigning a
// guarded column calls its guard, changing one is still rejected, and the
// lifecycle events are exactly one per state the Stages entered.
func TestPostgresStageExecutionUpdatesFireOnlyTheTriggersTheyConcern(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	run := createTestRun(t, ctx, NewPostgresStore(pool), "run-stage-triggers")
	if _, err := admitTestRun(ctx, NewPostgresStore(pool), run.RunID); err != nil {
		t.Fatal(err)
	}
	copied := storeTestRunArtifact(t, ctx, pool, run.RunID, "builder", "result")
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()
	if _, err := tx.Exec(ctx, `SET CONSTRAINTS ALL IMMEDIATE`); err != nil {
		t.Fatal(err)
	}
	probe := explaintest.NewProbe(t, tx)
	store := NewPostgresStore(probe)
	const (
		capture      = "stage_executions_capture_lifecycle_event"
		captureState = "stage_executions_capture_lifecycle_event_on_update"
		continuation = "stage_executions_manual_continuation"
		immutable    = "stage_executions_protect_immutable"
		evalStages   = "eval_stages_projection"
		holds        = "stage_executions_hold_artifacts"
		holdsUpdate  = "stage_executions_hold_artifacts_on_update"
	)
	transition := map[string]int64{captureState: 1, evalStages: 1}
	guardedTransition := map[string]int64{immutable: 1, captureState: 1, evalStages: 1}
	// Recording a candidate or an accepted result also updates the exact
	// revisions it holds (DEL-R62).
	resultTransition := map[string]int64{immutable: 1, captureState: 1, evalStages: 1, holdsUpdate: 1}
	creation := map[string]int64{capture: 1, continuation: 1, evalStages: 1, holds: 1}
	expectFired := func(step string, want map[string]int64, action func() error) {
		t.Helper()
		probe.Reset()
		if err := action(); err != nil {
			t.Fatalf("%s: %v", step, err)
		}
		if fired := probe.Fired("stage_executions"); !maps.Equal(fired, want) {
			t.Errorf("%s fired the stage_executions triggers %v, want %v", step, fired, want)
		}
	}
	createStage := func(id string) func() error {
		return func() error {
			_, err := store.CreateStageExecution(ctx, CreateStageExecutionParams{
				LaneClass: resolved.LaneClassModel, StageExecutionID: id, RunID: run.RunID, StageName: id, Attempt: 1,
				StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{"objective":"copy"}`),
				StageContextSchemaVersion: contracts.APIVersion,
				StageContext:              StageContextSnapshot{Parameters: map[string]string{}, Artifacts: map[string]PinnedContextArtifact{}},
			})
			return err
		}
	}

	const finished, aborted = "stage-triggers-finished", "stage-triggers-aborted"
	expectFired("creation", creation, createStage(finished))
	expectFired("admission", map[string]int64{}, func() error {
		_, err := store.MarkStageAdmitted(ctx, finished)
		return err
	})
	identity := plannerRunEventData{StageExecutionID: finished, SessionID: "session-stage-triggers", InvocationID: "invocation-stage-triggers"}
	expectFired("Planner start", guardedTransition, func() error {
		return store.StartPlanner(ctx, StartPlannerParams{
			StageExecutionID: finished, SessionID: identity.SessionID, InvocationID: identity.InvocationID,
			StateSchemaVersion: contracts.APIVersion, InitialState: json.RawMessage(`{}`),
			Reason: Reason{Code: "planner_started"},
			RunEvent: RunEventAppend{
				EventID: "event-stage-triggers", EventSchemaVersion: contracts.APIVersion,
				Kind: RunEventPlannerStarted, Data: mustRunEventJSON(t, identity),
			},
		})
	})
	candidate := stageresult.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded, Summary: "copied",
		Artifacts: map[string]contracts.ArtifactRef{"copied": copied},
	}
	expectFired("finalization", resultTransition, func() error {
		return store.EnterFinalizing(ctx, EnterFinalizingParams{
			StageExecutionID: finished, ResultSchemaVersion: contracts.APIVersion, Candidate: candidate,
			FinalizationID: "finalization-stage-triggers", Deadline: time.Now().Add(time.Minute),
			Reason: Reason{Code: "planner_completed"},
		})
	})
	expectFired("output publication failure", map[string]int64{}, func() error {
		return store.RecordStageOutputPublicationFailure(ctx, run.RunID, finished)
	})
	expectFired("accepted result", resultTransition, func() error {
		return store.CompleteStageResult(ctx, finished, contracts.APIVersion, candidate)
	})

	expectFired("second creation", creation, createStage(aborted))
	expectFired("abort", guardedTransition, func() error {
		return store.EnterAborting(ctx, EnterAbortingParams{
			StageExecutionID: aborted, ExpectedState: StagePreparing, TerminationSchemaVersion: contracts.APIVersion,
			Termination: StageTermination{
				Outcome: TerminationInterrupted, Code: "allocation_lost", Message: "allocation lost",
				Phase: TerminationPreparing, OccurredAt: time.Now(),
			},
			AbortID: "abort-stage-triggers", Deadline: time.Now().Add(time.Minute), Reason: Reason{Code: "allocation_lost"},
		})
	})
	expectFired("committed termination", transition, func() error {
		return store.CompleteStageTermination(ctx, aborted)
	})

	// Assigning a guarded column calls its guard even when the value stays.
	expectFired("unchanged immutable assignment", map[string]int64{immutable: 1}, func() error {
		_, err := probe.Exec(ctx, `UPDATE stage_executions SET stage_name = stage_name WHERE stage_execution_id = $1`, finished)
		return err
	})
	expectFired("unchanged continuation assignment", map[string]int64{continuation: 1}, func() error {
		_, err := probe.Exec(ctx, `UPDATE stage_executions SET resume_source_execution_id = resume_source_execution_id
WHERE stage_execution_id = $1`, aborted)
		return err
	})
	// A guarded change is still rejected. Explain rolls each attempt back.
	for _, rejected := range []struct{ sql, id, message string }{
		{`UPDATE stage_executions SET stage_name = 'renamed' WHERE stage_execution_id = $1`, finished,
			"immutable StageExecution fields cannot be changed"},
		{`UPDATE stage_executions SET planner_invocation_id = 'other' WHERE stage_execution_id = $1`, finished,
			"Planner identity cannot be changed"},
		{`UPDATE stage_executions SET candidate_result_schema_version = 'other' WHERE stage_execution_id = $1`, finished,
			"candidate StageResult cannot be changed"},
		{`UPDATE stage_executions SET abort_id = 'other' WHERE stage_execution_id = $1`, aborted,
			"abort identity cannot be changed"},
		{`UPDATE stage_executions SET resume_source_execution_id = 'stage-triggers-finished' WHERE stage_execution_id = $1`, aborted,
			"manual continuation identity is immutable"},
	} {
		if _, err := explaintest.Explain(ctx, tx, rejected.sql, rejected.id); err == nil || !strings.Contains(err.Error(), rejected.message) {
			t.Errorf("%s = %v, want %q", rejected.sql, err, rejected.message)
		}
	}
	// An INSERT still validates a manual continuation against its source.
	if _, err := explaintest.Explain(ctx, tx, `
INSERT INTO stage_executions (
    stage_execution_id, run_id, stage_name, attempt, previous_execution_id, resume_source_execution_id,
    stage_spec_schema_version, stage_spec_snapshot, stage_context_schema_version, stage_context_snapshot,
    state, state_reason_code
)
SELECT 'stage-triggers-continuation', run_id, stage_name, 2, stage_execution_id, stage_execution_id,
       stage_spec_schema_version, '{"objective":"other"}', stage_context_schema_version, stage_context_snapshot,
       'preparing', 'created'
FROM stage_executions WHERE stage_execution_id = $1`, aborted); err == nil ||
		!strings.Contains(err.Error(), "manual continuation must copy its terminal source") {
		t.Errorf("manual continuation with another Stage spec = %v, want rejected", err)
	}

	events, err := NewPostgresStore(tx).ListRunEvents(ctx, run.RunID, 0, 100)
	if err != nil {
		t.Fatal(err)
	}
	if events.More {
		t.Fatal("Run event page reports more events, want the whole history")
	}
	var stageStates []string
	for _, event := range events.Events {
		var data struct {
			Resource         string `json:"resource"`
			StageExecutionID string `json:"stageExecutionId"`
			State            string `json:"state"`
		}
		if event.Kind != RunEventLifecycleChanged {
			continue
		}
		if err := json.Unmarshal(event.Data, &data); err != nil {
			t.Fatal(err)
		}
		if data.Resource == "stageExecution" {
			stageStates = append(stageStates, data.StageExecutionID+" "+data.State)
		}
	}
	want := []string{
		finished + " preparing", finished + " running", finished + " finalizing", finished + " succeeded",
		aborted + " preparing", aborted + " aborting", aborted + " interrupted",
	}
	if !slices.Equal(stageStates, want) {
		t.Errorf("StageExecution lifecycle events = %v, want %v", stageStates, want)
	}
}
