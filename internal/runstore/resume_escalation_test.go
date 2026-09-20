package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func TestPostgresResumeEscalatedStage(t *testing.T) {
	for _, variant := range []StageExecutionConfigVariant{StageExecutionConfigBase, StageExecutionConfigFailedEscalation, StageExecutionConfigInterruptedEscalation} {
		t.Run(string(variant), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			pool := isolatedRunStorePool(t, ctx)
			store := NewPostgresStore(pool)
			runID := "review-resume"
			run := createTestRun(t, ctx, store, runID)
			if _, err := store.TransitionRun(ctx, runID, RunInitializing, RunRunning, Reason{Code: "started"}); err != nil {
				t.Fatal(err)
			}
			var ordinal *int
			if variant != StageExecutionConfigBase {
				value := 1
				ordinal = &value
			}
			stageID := "review-stage"
			_, err := store.CreateStageExecution(ctx, CreateStageExecutionParams{
				StageExecutionID: stageID, RunID: runID, StageName: "build", Attempt: 1,
				ExecutionConfigVariant: variant, EscalationOrdinal: ordinal,
				StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{"objective":"build"}`),
				StageContextSchemaVersion: contracts.APIVersion,
			})
			if err != nil {
				t.Fatal(err)
			}
			failResumeStage(t, ctx, store, runID, stageID)
			previous, err := store.GetStageExecution(ctx, stageID)
			if err != nil {
				t.Fatal(err)
			}
			source, err := store.ResumableStage(ctx, run.OwnerID, runID)
			if err != nil || source == nil {
				t.Fatalf("expected advertised resumable stage, source=%v err=%v", source, err)
			}
			// A caller cannot bypass automatic ordinal uniqueness by merely marking
			// an attempt manual: it must commit its matching receipt atomically.
			tx, err := pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			_, err = NewPostgresStore(tx).CreateStageExecution(ctx, CreateStageExecutionParams{
				StageExecutionID: "forged-manual", RunID: runID, StageName: previous.StageName,
				Attempt: previous.Attempt + 1, PreviousExecutionID: &stageID, ResumeSourceExecutionID: &stageID,
				ExecutionConfigVariant: variant, EscalationOrdinal: ordinal,
				StageSpecSchemaVersion: previous.StageSpecSchemaVersion, StageSpecSnapshot: previous.StageSpecSnapshot,
				StageContextSchemaVersion: previous.StageContextSchemaVersion, StageContext: previous.StageContext,
			})
			if err != nil {
				_ = tx.Rollback(ctx)
				t.Fatal(err)
			}
			if err := tx.Commit(ctx); persistencepostgres.SQLState(err) != "23503" {
				t.Fatalf("missing receipt accepted: %v", err)
			}
			if _, err := store.GetStageExecution(ctx, "forged-manual"); !errors.Is(err, ErrNotFound) {
				t.Fatalf("failed commit left attempt: %v", err)
			}
			result, err := store.ResumeFailedRun(ctx, run.OwnerID, runID, *source, "review-resumed")
			t.Logf("variant=%s advertised=%s result=%+v err=%v", variant, *source, result, err)
			if err != nil {
				t.Fatalf("advertised eligible resume failed: %v", err)
			}
			next, err := store.GetStageExecution(ctx, result.StageExecutionID)
			if err != nil {
				t.Fatal(err)
			}
			if next.ResumeSourceExecutionID == nil || *next.ResumeSourceExecutionID != stageID ||
				next.Attempt != previous.Attempt+1 || next.ExecutionConfigVariant != variant ||
				!reflect.DeepEqual(next.EscalationOrdinal, ordinal) ||
				!reflect.DeepEqual(next.StageSpecSnapshot, previous.StageSpecSnapshot) ||
				!reflect.DeepEqual(next.StageContext, previous.StageContext) {
				t.Fatalf("manual continuation changed the pinned execution: %+v", next)
			}
			unchanged, err := store.GetStageExecution(ctx, stageID)
			if err != nil || !reflect.DeepEqual(previous, unchanged) {
				t.Fatalf("source changed: %v", err)
			}
			if _, err := pool.Exec(ctx, `UPDATE stage_executions SET resume_source_execution_id=NULL WHERE stage_execution_id=$1`, next.StageExecutionID); persistencepostgres.SQLState(err) != "23514" {
				t.Fatalf("manual identity rewrite: %v", err)
			}
			failResumeStage(t, ctx, store, runID, next.StageExecutionID)
			replay, err := store.ResumeFailedRun(ctx, run.OwnerID, runID, stageID, "must-not-exist")
			if err != nil || replay != result {
				t.Fatalf("response replay: %+v %v", replay, err)
			}
			stillFailed, _ := store.GetRun(ctx, runID)
			if stillFailed.State != RunFailed {
				t.Fatal("stale replay reopened run")
			}
			second, err := store.ResumeFailedRun(ctx, run.OwnerID, runID, next.StageExecutionID, "resumed-again")
			if err != nil {
				t.Fatal(err)
			}
			latest, err := store.GetStageExecution(ctx, second.StageExecutionID)
			if err != nil || latest.Attempt != 3 || !reflect.DeepEqual(latest.EscalationOrdinal, ordinal) {
				t.Fatalf("second continuation: %+v %v", latest, err)
			}
			failResumeStage(t, ctx, store, runID, latest.StageExecutionID)
			if err := store.DeleteReleasedTerminalRun(ctx, run.OwnerID, runID); err != nil {
				t.Fatalf("delete resumed terminal Run: %v", err)
			}
			var remaining int
			if err := pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM stage_executions WHERE run_id=$1) + (SELECT count(*) FROM run_stage_resumptions WHERE run_id=$1)`, runID).Scan(&remaining); err != nil || remaining != 0 {
				t.Fatalf("resume history did not cascade: remaining=%d err=%v", remaining, err)
			}

		})
	}
}
