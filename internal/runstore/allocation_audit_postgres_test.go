package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/config/resolved"
)

func TestPostgresAllocationAuditBindingRejectsForeignStage(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool, artifactpgtest.NewStorage())
	run := createTestRun(t, ctx, store, "allocation-owner")
	other := createTestRun(t, ctx, store, "allocation-other")
	if _, err := store.TransitionRun(ctx, run.RunID, RunInitializing, RunPending, Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(ctx, run.RunID, RunPending, RunRunning, Reason{Code: "admitted"}); err != nil {
		t.Fatal(err)
	}
	stage, err := store.CreateStageExecution(ctx, CreateStageExecutionParams{
		LaneClass: resolved.LaneClassModel, StageExecutionID: "allocation-stage", RunID: run.RunID, StageName: "copy", Attempt: 1,
		StageSpecSchemaVersion: "contractor/v1alpha1", StageSpecSnapshot: json.RawMessage(`{"objective":"copy"}`),
		StageContextSchemaVersion: "contractor/v1alpha1", StageContext: StageContextSnapshot{Parameters: map[string]string{}, Artifacts: map[string]PinnedContextArtifact{}},
	})
	if err != nil {
		t.Fatal(err)
	}
	binding, err := store.ReadAllocationAuditBinding(ctx, run.RunID, stage.StageExecutionID)
	if err != nil || binding.OwnerID != run.OwnerID || binding.ExecutionID != nil {
		t.Fatalf("ordinary binding %+v: %v", binding, err)
	}
	if _, err := store.ReadAllocationAuditBinding(ctx, other.RunID, stage.StageExecutionID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("foreign Stage accepted: %v", err)
	}
	if _, err := store.ReadAllocationAuditBinding(ctx, run.RunID, "missing"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing Stage accepted: %v", err)
	}
}
