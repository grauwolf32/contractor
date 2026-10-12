package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner/stageresult"
)

// The transient publication failure count belongs to one finalizing
// StageExecution (PRJ-R58): it grows only while that Stage is finalizing and
// only under its own Run, and it is not a lifecycle change.
func TestPostgresStageOutputPublicationFailuresCountOnlyWhileFinalizing(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	run := createTestRun(t, ctx, store, "run-publication-failures")
	if _, err := admitTestRun(ctx, store, run.RunID); err != nil {
		t.Fatal(err)
	}
	other := createTestRun(t, ctx, store, "run-publication-failures-other")
	execution, err := store.CreateStageExecution(ctx, CreateStageExecutionParams{
		StageExecutionID: "stage-publication-failures", RunID: run.RunID, StageName: "copy", Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{"objective":"copy"}`),
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: StageContextSnapshot{
			Parameters: map[string]string{}, Artifacts: map[string]PinnedContextArtifact{},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	stageID := execution.StageExecutionID
	if err := store.RecordStageOutputPublicationFailure(ctx, run.RunID, stageID); !errors.Is(err, ErrConflict) {
		t.Fatalf("count for a preparing StageExecution = %v, want conflict", err)
	}
	if err := store.StartPlanner(ctx, StartPlannerParams{
		StageExecutionID: stageID, SessionID: "session-publication-failures",
		InvocationID: "invocation-publication-failures", StateSchemaVersion: contracts.APIVersion,
		InitialState: json.RawMessage(`{}`), Reason: Reason{Code: "planner_started"},
		RunEvent: RunEventAppend{
			EventID: "event-publication-failures", EventSchemaVersion: contracts.APIVersion,
			Kind: RunEventPlannerStarted,
			Data: mustRunEventJSON(t, plannerRunEventData{
				StageExecutionID: stageID, SessionID: "session-publication-failures",
				InvocationID: "invocation-publication-failures",
			}),
		},
	}); err != nil {
		t.Fatal(err)
	}
	revision := "revision-publication-failures"
	candidate := stageresult.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded, Summary: "copied",
		Artifacts: map[string]contracts.ArtifactRef{
			"copied": {Namespace: "builder", Name: "result", Revision: &revision},
		},
	}
	if err := store.EnterFinalizing(ctx, EnterFinalizingParams{
		StageExecutionID: stageID, ResultSchemaVersion: contracts.APIVersion, Candidate: candidate,
		FinalizationID: "finalization-publication-failures", Deadline: time.Now().Add(time.Minute),
		Reason: Reason{Code: "planner_completed"},
	}); err != nil {
		t.Fatal(err)
	}
	finalizing, err := store.GetStageExecution(ctx, stageID)
	if err != nil {
		t.Fatal(err)
	}
	cursor, err := store.GetRunEventCursor(ctx, run.RunID)
	if err != nil {
		t.Fatal(err)
	}

	for range 2 {
		if err := store.RecordStageOutputPublicationFailure(ctx, run.RunID, stageID); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.RecordStageOutputPublicationFailure(ctx, other.RunID, stageID); !errors.Is(err, ErrConflict) {
		t.Fatalf("count under another Run = %v, want conflict", err)
	}
	if err := store.RecordStageOutputPublicationFailure(ctx, run.RunID, " "); !errors.Is(err, ErrInvalid) {
		t.Fatalf("count without a StageExecution = %v, want invalid", err)
	}
	if failures, err := store.StageOutputPublicationFailures(ctx, run.RunID, stageID); err != nil || failures != 2 {
		t.Fatalf("counted failures = (%d, %v), want 2", failures, err)
	}
	if _, err := store.StageOutputPublicationFailures(ctx, other.RunID, stageID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("count read under another Run = %v, want not found", err)
	}
	counted, err := store.GetStageExecution(ctx, stageID)
	if err != nil || !counted.UpdatedAt.Equal(finalizing.UpdatedAt) || counted.State != StageFinalizing {
		t.Fatalf("counting changed the StageExecution lifecycle: %+v, %v", counted, err)
	}
	if after, err := store.GetRunEventCursor(ctx, run.RunID); err != nil || after != cursor {
		t.Fatalf("counting appended Run events: %+v -> %+v, %v", cursor, after, err)
	}

	if err := store.CompleteStageResult(ctx, stageID, contracts.APIVersion, candidate); err != nil {
		t.Fatal(err)
	}
	if err := store.RecordStageOutputPublicationFailure(ctx, run.RunID, stageID); !errors.Is(err, ErrConflict) {
		t.Fatalf("count for a succeeded StageExecution = %v, want conflict", err)
	}
	if failures, err := store.StageOutputPublicationFailures(ctx, run.RunID, stageID); err != nil || failures != 2 {
		t.Fatalf("counted failures after completion = (%d, %v), want 2", failures, err)
	}
}
