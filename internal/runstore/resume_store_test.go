package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5/pgxpool"
)

func failedResumeFixture(t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID string) (*PostgresStore, StageExecution) {
	t.Helper()
	store := NewPostgresStore(pool)
	run := createTestRun(t, ctx, store, runID)
	if _, err := store.TransitionRun(ctx, runID, RunInitializing, RunRunning, Reason{Code: "started"}); err != nil {
		t.Fatal(err)
	}
	stageID := runID + "-stage"
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	scoped, _ := service.Run(runID)
	written, err := scoped.Write(ctx, contracts.ArtifactRef{Namespace: "analysis", Name: "upstream"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("retained successful output")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.BindOutputExact(ctx, runID, "upstream", written.Ref, nil); err != nil {
		t.Fatal(err)
	}
	priorID := runID + "-success"
	_, err = store.CreateStageExecution(ctx, CreateStageExecutionParams{
		StageExecutionID: priorID, RunID: runID, StageName: "analyze", Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{"objective":"analyze"}`),
		StageContextSchemaVersion: contracts.APIVersion,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = pool.Exec(ctx, `INSERT INTO planner_sessions(session_id,stage_execution_id,invocation_id,state_schema_version,state)
   VALUES($1||'-session',$1,$1||'-invocation','contractor/v1alpha1','{}')`, priorID)
	if err != nil {
		t.Fatal(err)
	}
	_, err = pool.Exec(ctx, `UPDATE stage_executions SET state='succeeded',
  planner_session_id=$1||'-session',planner_invocation_id=$1||'-invocation',
  candidate_result_schema_version='contractor/v1alpha1',accepted_result_schema_version='contractor/v1alpha1',
  candidate_stage_result=$2::jsonb,accepted_stage_result=$2::jsonb,
  finalization_id=$1||'-finalization',finalization_deadline=clock_timestamp(),
  planner_started_at=clock_timestamp(),terminal_at=clock_timestamp() WHERE stage_execution_id=$1`, priorID,
		`{"apiVersion":"contractor/v1alpha1","outcome":"succeeded","summary":"retained success","artifacts":{}}`)
	if err != nil {
		t.Fatal(err)
	}
	_, err = store.CreateStageExecution(ctx, CreateStageExecutionParams{
		StageExecutionID: stageID, RunID: runID, StageName: "build", Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{"objective":"build"}`),
		StageContextSchemaVersion: contracts.APIVersion, StageContext: StageContextSnapshot{Parameters: run.Parameters, Artifacts: map[string]PinnedContextArtifact{"upstream": {Required: true, Artifact: &written.Ref}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	failResumeStage(t, ctx, store, runID, stageID)
	stage, err := store.GetStageExecution(ctx, stageID)
	if err != nil {
		t.Fatal(err)
	}
	return store, stage
}

func failResumeStage(t *testing.T, ctx context.Context, store *PostgresStore, runID, stageID string) {
	t.Helper()
	err := store.EnterAborting(ctx, EnterAbortingParams{StageExecutionID: stageID, ExpectedState: StagePreparing,
		TerminationSchemaVersion: contracts.APIVersion, Termination: StageTermination{
			Outcome: TerminationInterrupted, Code: "test_failure", Message: "test failure", Phase: TerminationPreparing, OccurredAt: time.Now(),
		}, AbortID: stageID + "-abort", Deadline: time.Now().Add(time.Minute), Reason: Reason{Code: "test_failure"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.CompleteStageTermination(ctx, stageID); err != nil {
		t.Fatal(err)
	}
	if err := artifacts.NewService(artifacts.NewPostgresRepository(store.db)).FreezeRunOutputs(ctx, runID); err != nil {
		t.Fatal(err)
	}
	if _, err := store.RecordStageTransitionDecision(ctx, RecordStageTransitionDecisionParams{SourceExecutionID: stageID, RunID: runID, Action: StageTransitionFail}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(ctx, runID, RunRunning, RunFailed, Reason{Code: "test_failure"}); err != nil {
		t.Fatal(err)
	}
}

func TestPostgresResumeFailedStageIsAtomicAndIdempotent(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store, previous := failedResumeFixture(t, ctx, pool, "run-resume")
	before, err := store.GetRun(ctx, "run-resume")
	if err != nil {
		t.Fatal(err)
	}
	successBefore, err := store.GetStageExecution(ctx, before.RunID+"-success")
	if err != nil {
		t.Fatal(err)
	}
	source, err := store.ResumableStage(ctx, "user-1", before.RunID)
	if err != nil || source == nil || *source != previous.StageExecutionID {
		t.Fatalf("source: %v %v", source, err)
	}
	if _, err := store.ResumeFailedRun(ctx, "other-owner", before.RunID, *source, "other-target"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("owner fence: %v", err)
	}
	if _, err := store.ResumeFailedRun(ctx, "user-1", before.RunID, "stale-stage", "stale-target"); !errors.Is(err, ErrConflict) {
		t.Fatalf("source fence: %v", err)
	}
	results := make(chan ResumeRunResult, 8)
	failures := make(chan error, 8)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			result, err := store.ResumeFailedRun(ctx, "user-1", before.RunID, *source, fmt.Sprintf("target-%d", i))
			results <- result
			failures <- err
		}(i)
	}
	wg.Wait()
	close(results)
	close(failures)
	for err := range failures {
		if err != nil {
			t.Fatal(err)
		}
	}
	target := ""
	for result := range results {
		if target == "" {
			target = result.StageExecutionID
		}
		if target != result.StageExecutionID {
			t.Fatal("duplicate attempts")
		}
	}
	after, err := store.GetRun(ctx, before.RunID)
	if err != nil {
		t.Fatal(err)
	}
	if after.State != RunRunning || after.FinishedAt != nil || !after.StartedAt.Equal(*before.StartedAt) || !reflect.DeepEqual(after.RuntimeConfig, before.RuntimeConfig) || !reflect.DeepEqual(after.WorkflowSnapshot, before.WorkflowSnapshot) {
		t.Fatalf("Run changed incorrectly: %+v", after)
	}
	old, err := store.GetStageExecution(ctx, *source)
	if err != nil || !reflect.DeepEqual(old, previous) {
		t.Fatal("old attempt changed", err)
	}
	successAfter, err := store.GetStageExecution(ctx, before.RunID+"-success")
	if err != nil || !reflect.DeepEqual(successBefore, successAfter) {
		t.Fatal("successful upstream stage changed", err)
	}
	var frozen bool
	if err := pool.QueryRow(ctx, `SELECT frozen FROM artifact_bindings WHERE scope_kind='run' AND scope_id=$1 AND namespace='outputs' AND name='upstream'`, before.RunID).Scan(&frozen); err != nil || frozen {
		t.Fatal("output not reopened", err)
	}
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	scoped, _ := service.Run(before.RunID)
	data, err := scoped.Read(ctx, *previous.StageContext.Artifacts["upstream"].Artifact)
	if err != nil || string(data.Payload.Data) != "retained successful output" {
		t.Fatal("upstream artifact changed", err)
	}
	if err := service.FreezeRunOutputs(ctx, before.RunID); err != nil {
		t.Fatal(err)
	}
	_, thawErr := pool.Exec(ctx, `UPDATE artifact_bindings SET frozen=false WHERE scope_kind='run' AND scope_id=$1 AND namespace='outputs'`, before.RunID)
	if persistencepostgres.SQLState(thawErr) != "23514" {
		t.Fatalf("old receipt permitted a later thaw: %v", thawErr)
	}
	next, err := store.GetStageExecution(ctx, target)
	if err != nil {
		t.Fatal(err)
	}
	if next.State != StagePreparing || next.Attempt != 2 || next.PreviousExecutionID == nil || *next.PreviousExecutionID != *source || next.PlannerSessionID != nil || !reflect.DeepEqual(next.StageContext, previous.StageContext) || !reflect.DeepEqual(next.StageSpecSnapshot, previous.StageSpecSnapshot) {
		t.Fatalf("new attempt: %+v", next)
	}
	decision, err := store.GetStageTransitionDecision(ctx, *source)
	if err != nil || decision.Action != StageTransitionFail {
		t.Fatal("original decision rewritten", err)
	}
	failResumeStage(t, ctx, store, before.RunID, target)
	replay, err := store.ResumeFailedRun(ctx, "user-1", before.RunID, *source, "must-not-create")
	if err != nil || replay.StageExecutionID != target {
		t.Fatalf("late response-loss replay: %+v %v", replay, err)
	}
	stillFailed, _ := store.GetRun(ctx, before.RunID)
	if stillFailed.State != RunFailed {
		t.Fatal("stale replay restarted a newer failure")
	}
	if _, err := store.ResumeFailedRun(ctx, "user-1", before.RunID, target, "third-attempt"); err != nil {
		t.Fatal(err)
	}
}

func TestPostgresResumeWaitsForAllocationRelease(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store, previous := failedResumeFixture(t, ctx, pool, "run-release-resume")
	allocation := StageAllocation{
		PerformanceCollectionPolicy: contracts.PerformanceCollectionUnsupported,
		AllocationID:                "pending-release", StageExecutionID: previous.StageExecutionID, LogicalAgentName: "builder", Namespace: "builder",
		AgentTemplateRef: contracts.AgentTemplateRef{TemplateID: "builder", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64)},
		WorkerRuntimeRef: contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"},
		RuntimeAgentID:   strings.Repeat("1", 64), RuntimeAgentInstanceID: "runtime-1", RuntimeAgentLabelRevision: 1,
		RuntimeConfigurationSchemaVersion: AllocationRuntimeConfigurationSchemaVersion, RuntimeConfiguration: testAllocationRuntimeConfiguration(),
	}
	if err := store.RecordStageAllocation(ctx, allocation); err != nil {
		t.Fatal(err)
	}
	source, err := store.ResumableStage(ctx, "user-1", previous.RunID)
	if err != nil || source != nil {
		t.Fatal("advertised before release", err)
	}
	if _, err := store.ResumeFailedRun(ctx, "user-1", previous.RunID, previous.StageExecutionID, "after-release"); !errors.Is(err, ErrConflict) {
		t.Fatalf("resumed before release: %v", err)
	}
	if err := store.MarkStageAllocationReleased(ctx, allocation.AllocationID); err != nil {
		t.Fatal(err)
	}
	if _, err := store.ResumeFailedRun(ctx, "user-1", previous.RunID, previous.StageExecutionID, "after-release"); err != nil {
		t.Fatal(err)
	}
}

func TestPostgresResumeRejectsIneligibleRunsAndRollsBack(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store, previous := failedResumeFixture(t, ctx, pool, "run-invalid-resume")
	// A target ID collision must roll back the Run state change too.
	if _, err := store.ResumeFailedRun(ctx, "user-1", previous.RunID, previous.StageExecutionID, previous.StageExecutionID); !errors.Is(err, ErrConflict) {
		t.Fatalf("collision: %v", err)
	}
	run, _ := store.GetRun(ctx, previous.RunID)
	if run.State != RunFailed {
		t.Fatal("partial transaction committed")
	}
	createTestRun(t, ctx, store, "initialization-failed")
	if _, err := store.TransitionRun(ctx, "initialization-failed", RunInitializing, RunFailed, Reason{Code: "invalid"}); err != nil {
		t.Fatal(err)
	}
	source, err := store.ResumableStage(ctx, "user-1", "initialization-failed")
	if err != nil || source != nil {
		t.Fatal("initialization failure has no stage", err)
	}
	if _, err := pool.Exec(ctx, `UPDATE workflow_runs SET state='succeeded' WHERE run_id=$1`, previous.RunID); err != nil {
		t.Fatal(err)
	}
	if _, err := store.ResumeFailedRun(ctx, "user-1", previous.RunID, previous.StageExecutionID, "cancelled-target"); !errors.Is(err, ErrConflict) {
		t.Fatalf("cancelled: %v", err)
	}
}
