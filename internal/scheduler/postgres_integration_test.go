package scheduler

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresSchedulerRunsPassthroughAndPublishesFrozenOutput(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	workflow := loadSchedulerWorkflow(t)
	store := runstore.NewPostgresStore(pool)
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	runStore := createSchedulerRun(t, ctx, store, artifactService, workflow)
	workerWrite, err := runStore.Write(
		ctx,
		contracts.ArtifactRef{Namespace: "builder", Name: "copied"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("copied\n")},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	candidate := contracts.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded, Summary: "copied",
		Artifacts: map[string]contracts.ArtifactRef{"copied": workerWrite.Ref},
	}

	sessions, err := plannersession.New(store, plannersession.Options{})
	if err != nil {
		t.Fatal(err)
	}
	inspector, err := planner.NewRunArtifactInspector("run-1", runStore)
	if err != nil {
		t.Fatal(err)
	}
	invoker := &postgresTestWorkerInvoker{result: candidate}
	passthrough, err := planner.NewPassthroughFactory(sessions, invoker, inspector)
	if err != nil {
		t.Fatal(err)
	}
	planners, err := planner.NewRegistry(passthrough)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	clock := staticClock{now: now}
	events := &eventRecorder{}
	allocator := &memoryAllocator{
		workflow: workflow, clock: clock, events: events,
		grants: map[string]controlplane.AllocationGrant{},
	}
	workers := &memoryWorkers{allocator: allocator, workflow: workflow, clock: clock, events: events}
	resolver, _ := NewArtifactServiceResolver(artifactService)
	transactions, _ := NewPostgresPersistence(pool)
	workerSelection := workflow.Stages[workflow.EntryStage].ExecutionConfig.Agents["builder"]
	if workerSelection.Credential == nil || workerSelection.LLMGateway == nil {
		t.Fatal("test Worker has no complete credential route")
	}
	credentialProvider, err := credentials.NewStaticProvider([]credentials.StaticEntry{{
		Metadata: workflowconfig.CredentialMetadata{
			Ref:          *workerSelection.Credential,
			LLMGateway:   workerSelection.LLMGateway.Ref,
			Unrestricted: true,
		},
		Token: contracts.NewSecretString("scheduler-test-token"),
	}})
	if err != nil {
		t.Fatal(err)
	}
	sequence := 0
	scheduler, err := New(store, transactions, resolver, allocator, workers, planners, Options{
		PollInterval: time.Second, ClaimDuration: time.Minute, OperationTimeout: 5 * time.Second,
		PlannerTimeout: 20 * time.Second, FinalizationTimeout: 5 * time.Second, AbortTimeout: 5 * time.Second,
		RuntimeSettings: testSchedulerRuntimeSettings(), Credentials: credentialProvider, Clock: clock,
		NewID: func(prefix string) (string, error) {
			sequence++
			return prefix + strings.Repeat("x", sequence), nil
		},
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}

	worked, err := scheduler.RunOnce(ctx)
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	run, err := store.GetRun(ctx, "run-1")
	if err != nil || run.State != runstore.RunSucceeded {
		t.Fatalf("Run = (%+v, %v)", run, err)
	}
	executions, err := store.ListStageExecutions(ctx, "run-1")
	if err != nil || len(executions) != 1 || executions[0].State != runstore.StageSucceeded ||
		executions[0].PlannerSessionID == nil || executions[0].AcceptedResult == nil {
		t.Fatalf("StageExecutions = (%+v, %v)", executions, err)
	}
	releasable, err := store.ListTerminalStageExecutionsWithAllocations(ctx)
	if err != nil || len(releasable) != 0 {
		t.Fatalf("completed allocation remained in recovery query = (%+v, %v)", releasable, err)
	}
	allocations, err := store.ListStageAllocations(ctx, executions[0].StageExecutionID)
	if err != nil || len(allocations) != 1 || allocations[0].ReleaseAttemptedAt == nil ||
		allocations[0].ReleaseCompletedAt == nil {
		t.Fatalf("durable allocation release marker = (%+v, %v)", allocations, err)
	}
	output, err := runStore.Read(ctx, contracts.ArtifactRef{Namespace: "outputs", Name: "result"})
	if err != nil || string(output.Payload.Data) != "copied\n" || output.Payload.MediaType != "text/plain" {
		t.Fatalf("output = (%+v, %v)", output, err)
	}
	if _, err := artifactService.BindOutputExact(
		ctx, "run-1", "result", workerWrite.Ref, output.Ref.Revision,
	); !errors.Is(err, artifacts.ErrArtifactFrozen) {
		t.Fatalf("frozen output update error = %v", err)
	}
	if invoker.calls != 1 || workers.prepareCalls != 1 || workers.finalizeCalls != 1 || workers.releaseCalls != 1 {
		t.Fatalf("semantic/lifecycle calls = invoker:%d workers:(%d,%d,%d)",
			invoker.calls, workers.prepareCalls, workers.finalizeCalls, workers.releaseCalls)
	}
	var contextPins, resultPins, outputPins int64
	if err := pool.QueryRow(ctx, `
SELECT
    count(*) FILTER (WHERE pin_kind = 'stage_context'),
    count(*) FILTER (WHERE pin_kind = 'stage_result'),
    count(*) FILTER (WHERE pin_kind = 'run_output')
FROM artifact_pins`).Scan(&contextPins, &resultPins, &outputPins); err != nil {
		t.Fatal(err)
	}
	if contextPins != 1 || resultPins != 1 || outputPins != 1 {
		t.Fatalf("artifact pins = context:%d result:%d output:%d", contextPins, resultPins, outputPins)
	}
}

func TestPostgresAcceptanceRollsBackStageAndOutputWhenRunCASLoses(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	fixture := createFinalizingFixture(t, ctx, pool)
	if _, err := fixture.store.RequestRunCancellation(ctx, "run-1", runstore.WorkflowRunCancellation{
		Code: runstore.CancellationUserRequested, RequestedAt: time.Now(),
	}); err != nil {
		t.Fatal(err)
	}

	err := fixture.persistence.CommitResultProgression(ctx, ResultProgression{
		RunID: "run-1", StageExecutionID: fixture.executionID, Result: fixture.result,
		WorkflowOutputs: fixture.workflow.Stages["copy"].WorkflowOutputs,
		OutputContracts: fixture.workflow.Outputs,
		Progression:     terminalSuccessProgression("run-1", fixture.executionID),
	})
	if !errors.Is(err, runstore.ErrConflict) {
		t.Fatalf("acceptance error = %v, want Run CAS conflict", err)
	}
	execution, err := fixture.store.GetStageExecution(ctx, fixture.executionID)
	if err != nil || execution.State != runstore.StageFinalizing || execution.AcceptedResult != nil {
		t.Fatalf("rolled back Stage = (%+v, %v)", execution, err)
	}
	runStore, _ := fixture.artifacts.Run("run-1")
	outputs := "outputs"
	listed, err := runStore.List(ctx, &outputs)
	if err != nil || len(listed) != 0 {
		t.Fatalf("rolled back outputs = (%+v, %v)", listed, err)
	}
	run, err := fixture.store.GetRun(ctx, "run-1")
	if err != nil || run.State != runstore.RunCancelling {
		t.Fatalf("winning Run state = (%+v, %v)", run, err)
	}
	if err := fixture.persistence.AcceptResultDuringCancellation(
		ctx, "run-1", fixture.executionID, fixture.result,
	); err != nil {
		t.Fatalf("accept finalizing result for audit: %v", err)
	}
	execution, err = fixture.store.GetStageExecution(ctx, fixture.executionID)
	run, runErr := fixture.store.GetRun(ctx, "run-1")
	listed, listErr := runStore.List(ctx, &outputs)
	if err != nil || runErr != nil || listErr != nil || execution.State != runstore.StageSucceeded ||
		execution.AcceptedResult == nil || run.State != runstore.RunCancelled || len(listed) != 0 {
		t.Fatalf("audit-only acceptance = stage:(%+v,%v) run:(%+v,%v) outputs:(%v,%v)",
			execution, err, run, runErr, listed, listErr)
	}
}

func TestPostgresOutputCommitUsesCandidateRevisionAfterBindingAdvances(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	fixture := createFinalizingFixture(t, ctx, pool)
	runArtifacts, err := fixture.artifacts.Run("run-1")
	if err != nil {
		t.Fatal(err)
	}
	selected := fixture.result.Artifacts["copied"]
	advanced, err := runArtifacts.Write(
		ctx,
		contracts.ArtifactRef{Namespace: selected.Namespace, Name: selected.Name},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("advanced binding\n")},
		selected.Revision,
	)
	if err != nil {
		t.Fatal(err)
	}
	if advanced.Ref.Revision == nil || selected.Revision == nil || *advanced.Ref.Revision == *selected.Revision {
		t.Fatalf("binding did not advance: selected=%+v advanced=%+v", selected, advanced.Ref)
	}

	err = fixture.persistence.CommitResultProgression(ctx, ResultProgression{
		RunID: "run-1", StageExecutionID: fixture.executionID, Result: fixture.result,
		WorkflowOutputs: fixture.workflow.Stages["copy"].WorkflowOutputs,
		OutputContracts: fixture.workflow.Outputs,
		Progression:     terminalSuccessProgression("run-1", fixture.executionID),
	})
	if err != nil {
		t.Fatal(err)
	}
	output, err := runArtifacts.Read(ctx, contracts.ArtifactRef{Namespace: "outputs", Name: "result"})
	if err != nil || string(output.Payload.Data) != "copied\n" {
		t.Fatalf("output substituted current binding: payload=%q error=%v", output.Payload.Data, err)
	}
}

func TestPostgresCancelAndSuccessRaceSerializesOnRunRow(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	fixture := createFinalizingFixture(t, ctx, pool)

	start := make(chan struct{})
	var wait sync.WaitGroup
	wait.Add(2)
	acceptanceResult := make(chan error, 1)
	cancellationResult := make(chan struct {
		run runstore.WorkflowRun
		err error
	}, 1)
	go func() {
		defer wait.Done()
		<-start
		acceptanceResult <- fixture.persistence.CommitResultProgression(ctx, ResultProgression{
			RunID: "run-1", StageExecutionID: fixture.executionID, Result: fixture.result,
			WorkflowOutputs: fixture.workflow.Stages["copy"].WorkflowOutputs,
			OutputContracts: fixture.workflow.Outputs,
			Progression:     terminalSuccessProgression("run-1", fixture.executionID),
		})
	}()
	go func() {
		defer wait.Done()
		<-start
		run, err := fixture.store.RequestRunCancellation(ctx, "run-1", runstore.WorkflowRunCancellation{
			Code: runstore.CancellationUserRequested, RequestedAt: time.Now(),
		})
		cancellationResult <- struct {
			run runstore.WorkflowRun
			err error
		}{run: run, err: err}
	}()
	close(start)
	wait.Wait()
	acceptErr := <-acceptanceResult
	cancelResult := <-cancellationResult
	if cancelResult.err != nil {
		t.Fatalf("cancel race error: %v", cancelResult.err)
	}

	run, err := fixture.store.GetRun(ctx, "run-1")
	if err != nil {
		t.Fatal(err)
	}
	runArtifacts, _ := fixture.artifacts.Run("run-1")
	outputs := "outputs"
	listed, err := runArtifacts.List(ctx, &outputs)
	if err != nil {
		t.Fatal(err)
	}
	switch run.State {
	case runstore.RunSucceeded:
		if acceptErr != nil || cancelResult.run.State != runstore.RunSucceeded || len(listed) != 1 {
			t.Fatalf("success won race: accept=%v cancel=%+v outputs=%v", acceptErr, cancelResult, listed)
		}
	case runstore.RunCancelling:
		if !errors.Is(acceptErr, runstore.ErrConflict) || cancelResult.run.State != runstore.RunCancelling || len(listed) != 0 {
			t.Fatalf("cancel won race: accept=%v cancel=%+v outputs=%v", acceptErr, cancelResult, listed)
		}
		if err := fixture.persistence.AcceptResultDuringCancellation(
			ctx, "run-1", fixture.executionID, fixture.result,
		); err != nil {
			t.Fatalf("finish cancellation winner: %v", err)
		}
		finished, err := fixture.store.GetRun(ctx, "run-1")
		if err != nil || finished.State != runstore.RunCancelled {
			t.Fatalf("finished cancellation winner = (%+v, %v)", finished, err)
		}
	default:
		t.Fatalf("unexpected race winner state %q", run.State)
	}
}

func TestPostgresRetryProgressionAtomicallyCreatesFreshAttempt(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	workflow := loadSchedulerWorkflow(t)
	store := runstore.NewPostgresStore(pool)
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	runArtifacts := createSchedulerRun(t, ctx, store, artifactService, workflow)
	persistence, err := NewPostgresPersistence(pool)
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	stageJSON, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	original, err := runArtifacts.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
	if err != nil {
		t.Fatal(err)
	}
	sourceID := "stage-retry-1"
	_, err = persistence.CreateStageWithContext(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: sourceID, RunID: "run-1", StageName: workflow.EntryStage, Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageJSON,
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{"objective": "copy exactly"},
			Artifacts: map[string]runstore.PinnedContextArtifact{
				"source": {Required: true, Artifact: &original.Ref},
			},
		},
	}, []ContextPin{{Name: "source", Ref: original.Ref}})
	if err != nil {
		t.Fatal(err)
	}
	termination := runstore.StageTermination{
		Outcome: runstore.TerminationInterrupted,
		Code:    "runtime_unavailable", Message: "runtime unavailable", Retryable: true,
		Phase: runstore.TerminationPreparing, OccurredAt: time.Now().UTC(),
	}
	if err := store.EnterAborting(ctx, runstore.EnterAbortingParams{
		StageExecutionID: sourceID, ExpectedState: runstore.StagePreparing,
		TerminationSchemaVersion: contracts.APIVersion, Termination: termination,
		AbortID: "abort-retry-1", Deadline: time.Now().Add(time.Minute),
		Reason: runstore.Reason{Code: "stage_interrupted"},
	}); err != nil {
		t.Fatal(err)
	}
	advanced, err := runArtifacts.Write(
		ctx,
		contracts.ArtifactRef{Namespace: "inputs", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("new source\n")},
		original.Ref.Revision,
	)
	if err != nil {
		t.Fatal(err)
	}
	targetID, targetStage := "stage-retry-2", workflow.EntryStage
	previous := sourceID
	err = persistence.CommitTerminationProgression(ctx, TerminationProgression{
		RunID: "run-1", StageExecutionID: sourceID,
		Progression: StageProgression{
			Decision: runstore.RecordStageTransitionDecisionParams{
				SourceExecutionID: sourceID, RunID: "run-1", Action: runstore.StageTransitionRetry,
				TargetStageName: &targetStage, TargetExecutionID: &targetID,
			},
			NextStage: &NextStageCreation{
				Params: runstore.CreateStageExecutionParams{
					StageExecutionID: targetID, RunID: "run-1", StageName: targetStage,
					Attempt: 2, PreviousExecutionID: &previous,
					ExecutionConfigVariant: runstore.StageExecutionConfigBase,
					StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageJSON,
					StageContextSchemaVersion: contracts.APIVersion,
					StageContext: runstore.StageContextSnapshot{
						Parameters: map[string]string{"objective": "copy exactly"},
						Artifacts: map[string]runstore.PinnedContextArtifact{
							"source": {Required: true, Artifact: &advanced.Ref},
						},
					},
				},
				ContextPins: []ContextPin{{Name: "source", Ref: advanced.Ref}},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	executions, err := store.ListStageExecutions(ctx, "run-1")
	if err != nil || len(executions) != 2 || executions[0].State != runstore.StageInterrupted ||
		executions[1].State != runstore.StagePreparing || executions[1].Attempt != 2 ||
		executions[1].PreviousExecutionID == nil || *executions[1].PreviousExecutionID != sourceID {
		t.Fatalf("retry executions = (%+v, %v)", executions, err)
	}
	first := executions[0].StageContext.Artifacts["source"].Artifact
	second := executions[1].StageContext.Artifacts["source"].Artifact
	if first == nil || second == nil || first.Revision == nil || second.Revision == nil ||
		*first.Revision == *second.Revision || *second.Revision != *advanced.Ref.Revision {
		t.Fatalf("retry context revisions = first:%+v second:%+v", first, second)
	}
	decision, err := store.GetStageTransitionDecision(ctx, sourceID)
	if err != nil || decision.Action != runstore.StageTransitionRetry ||
		decision.TargetExecutionID == nil || *decision.TargetExecutionID != targetID {
		t.Fatalf("retry decision = (%+v, %v)", decision, err)
	}
	run, err := store.GetRun(ctx, "run-1")
	if err != nil || run.State != runstore.RunRunning {
		t.Fatalf("retry Run = (%+v, %v)", run, err)
	}
}

func TestPostgresEscalationProgressionIsAtomicAndOrdinalIsUnique(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	workflow := loadSchedulerWorkflow(t)
	store := runstore.NewPostgresStore(pool)
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	runArtifacts := createSchedulerRun(t, ctx, store, artifactService, workflow)
	persistence, err := NewPostgresPersistence(pool)
	if err != nil {
		t.Fatal(err)
	}
	stageJSON, err := json.Marshal(workflow.Stages[workflow.EntryStage])
	if err != nil {
		t.Fatal(err)
	}
	input, err := runArtifacts.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
	if err != nil {
		t.Fatal(err)
	}
	sourceID := "stage-escalation-1"
	_, err = persistence.CreateStageWithContext(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: sourceID, RunID: "run-1", StageName: workflow.EntryStage, Attempt: 1,
		ExecutionConfigVariant: runstore.StageExecutionConfigBase,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageJSON,
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{"objective": "copy exactly"},
			Artifacts: map[string]runstore.PinnedContextArtifact{
				"source": {Required: true, Artifact: &input.Ref},
			},
		},
	}, []ContextPin{{Name: "source", Ref: input.Ref}})
	if err != nil {
		t.Fatal(err)
	}
	termination := runstore.StageTermination{
		Outcome: runstore.TerminationInterrupted,
		Code:    "permanent", Message: "non-retryable failure", Retryable: false,
		Phase: runstore.TerminationPreparing, OccurredAt: time.Now().UTC(),
	}
	if err := store.EnterAborting(ctx, runstore.EnterAbortingParams{
		StageExecutionID: sourceID, ExpectedState: runstore.StagePreparing,
		TerminationSchemaVersion: contracts.APIVersion, Termination: termination,
		AbortID: "abort-escalation-1", Deadline: time.Now().Add(time.Minute),
		Reason: runstore.Reason{Code: "stage_interrupted"},
	}); err != nil {
		t.Fatal(err)
	}

	targetID, targetStage, ordinal := "stage-escalation-2", workflow.EntryStage, 1
	previous := sourceID
	progression := TerminationProgression{
		RunID: "run-1", StageExecutionID: sourceID,
		Progression: StageProgression{
			Decision: runstore.RecordStageTransitionDecisionParams{
				SourceExecutionID: sourceID, RunID: "run-1", Action: runstore.StageTransitionEscalate,
				TargetStageName: &targetStage, TargetExecutionID: &targetID,
				EscalationOrdinal: &ordinal,
			},
			NextStage: &NextStageCreation{
				Params: runstore.CreateStageExecutionParams{
					StageExecutionID: targetID, RunID: "run-1", StageName: targetStage,
					Attempt: 2, PreviousExecutionID: &previous,
					ExecutionConfigVariant: runstore.StageExecutionConfigInterruptedEscalation,
					EscalationOrdinal:      &ordinal,
					StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageJSON,
					StageContextSchemaVersion: contracts.APIVersion,
					StageContext: runstore.StageContextSnapshot{
						Parameters: map[string]string{"objective": "copy exactly"},
						Artifacts: map[string]runstore.PinnedContextArtifact{
							"source": {Required: true, Artifact: &input.Ref},
						},
					},
				},
				ContextPins: []ContextPin{{Name: "source", Ref: input.Ref}},
			},
		},
	}
	if err := persistence.CommitTerminationProgression(ctx, progression); err != nil {
		t.Fatal(err)
	}
	if err := persistence.CommitTerminationProgression(ctx, progression); !errors.Is(err, runstore.ErrConflict) {
		t.Fatalf("replayed escalation commit error = %v, want conflict", err)
	}
	executions, err := store.ListStageExecutions(ctx, "run-1")
	if err != nil || len(executions) != 2 || executions[0].State != runstore.StageInterrupted ||
		executions[1].State != runstore.StagePreparing ||
		executions[1].ExecutionConfigVariant != runstore.StageExecutionConfigInterruptedEscalation ||
		executions[1].EscalationOrdinal == nil || *executions[1].EscalationOrdinal != 1 ||
		executions[1].PreviousExecutionID == nil || *executions[1].PreviousExecutionID != sourceID {
		t.Fatalf("escalation executions = (%+v, %v)", executions, err)
	}
	decisions, err := store.ListStageTransitionDecisions(ctx, "run-1")
	if err != nil || len(decisions) != 1 || decisions[0].Action != runstore.StageTransitionEscalate ||
		decisions[0].EscalationOrdinal == nil || *decisions[0].EscalationOrdinal != 1 {
		t.Fatalf("escalation decisions = (%+v, %v)", decisions, err)
	}
	if _, err := pool.Exec(ctx, `
UPDATE stage_executions SET escalation_ordinal = 2
WHERE stage_execution_id = 'stage-escalation-2'`); persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("escalation identity rewrite SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}
	duplicateID, duplicatePrevious := "stage-escalation-duplicate", targetID
	_, err = store.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: duplicateID, RunID: "run-1", StageName: targetStage,
		Attempt: 3, PreviousExecutionID: &duplicatePrevious,
		ExecutionConfigVariant: runstore.StageExecutionConfigInterruptedEscalation,
		EscalationOrdinal:      &ordinal,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageJSON,
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{"objective": "copy exactly"},
			Artifacts: map[string]runstore.PinnedContextArtifact{
				"source": {Required: true, Artifact: &input.Ref},
			},
		},
	})
	if !errors.Is(err, runstore.ErrConflict) {
		t.Fatalf("duplicate escalation ordinal error = %v, want conflict", err)
	}
}

func terminalSuccessProgression(runID, sourceExecutionID string) StageProgression {
	return StageProgression{
		Decision: runstore.RecordStageTransitionDecisionParams{
			SourceExecutionID: sourceExecutionID,
			RunID:             runID,
			Action:            runstore.StageTransitionSucceed,
		},
		TerminalRunState: runstore.RunSucceeded,
		RunReason:        runstore.Reason{Code: "workflow_succeeded"},
	}
}

type finalizingFixture struct {
	store       *runstore.PostgresStore
	artifacts   *artifacts.Service
	persistence *PostgresPersistence
	workflow    workflowconfig.ResolvedWorkflow
	executionID string
	result      contracts.StageContentResult
}

func createFinalizingFixture(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool,
) finalizingFixture {
	t.Helper()
	workflow := loadSchedulerWorkflow(t)
	store := runstore.NewPostgresStore(pool)
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	runArtifacts := createSchedulerRun(t, ctx, store, artifactService, workflow)
	input, err := runArtifacts.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
	if err != nil {
		t.Fatal(err)
	}
	written, err := runArtifacts.Write(
		ctx,
		contracts.ArtifactRef{Namespace: "builder", Name: "copied"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("copied\n")},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	persistence, _ := NewPostgresPersistence(pool)
	stage := workflow.Stages["copy"]
	stageJSON, _ := json.Marshal(stage)
	executionID := "stage-finalizing"
	_, err = persistence.CreateStageWithContext(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: executionID, RunID: "run-1", StageName: "copy", Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageJSON,
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{"objective": "copy exactly"},
			Artifacts: map[string]runstore.PinnedContextArtifact{
				"source": {Required: true, Artifact: &input.Ref},
			},
		},
	}, []ContextPin{{Name: "source", Ref: input.Ref}})
	if err != nil {
		t.Fatal(err)
	}
	startedData, err := json.Marshal(map[string]string{
		"stageExecutionId": executionID,
		"sessionId":        "session-finalizing",
		"invocationId":     "invocation-finalizing",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.StartPlanner(ctx, runstore.StartPlannerParams{
		StageExecutionID: executionID, SessionID: "session-finalizing", InvocationID: "invocation-finalizing",
		StateSchemaVersion: contracts.APIVersion, InitialState: json.RawMessage(`{"status":"test"}`),
		EventID: "planner-started-finalizing", EventSchemaVersion: contracts.APIVersion,
		Event: json.RawMessage(`{"kind":"planner_started"}`),
		RunEvent: runstore.RunEventAppend{
			EventID: "planner-started-finalizing", EventSchemaVersion: contracts.APIVersion,
			Kind: runstore.RunEventPlannerStarted, Data: startedData,
		},
		Reason: runstore.Reason{Code: "planner_started"},
	}); err != nil {
		t.Fatal(err)
	}
	result := contracts.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded, Summary: "copied",
		Artifacts: map[string]contracts.ArtifactRef{"copied": written.Ref},
	}
	if err := persistence.EnterFinalizingWithResult(ctx, runstore.EnterFinalizingParams{
		StageExecutionID: executionID, ResultSchemaVersion: contracts.APIVersion,
		Candidate: result, FinalizationID: "finalization-1", Deadline: time.Now().Add(time.Minute),
		Reason: runstore.Reason{Code: "planner_completed"},
	}); err != nil {
		t.Fatal(err)
	}
	return finalizingFixture{
		store: store, artifacts: artifactService, persistence: persistence,
		workflow: workflow, executionID: executionID, result: result,
	}
}

func createSchedulerRun(
	t *testing.T,
	ctx context.Context,
	store *runstore.PostgresStore,
	artifactService *artifacts.Service,
	workflow workflowconfig.ResolvedWorkflow,
) artifacts.ScopedStore {
	t.Helper()
	workflowJSON, _ := json.Marshal(workflow)
	if _, err := store.CreateRun(ctx, runstore.CreateRunParams{
		RunID: "run-1", OwnerID: "user-1", WorkflowName: workflow.Ref.Name,
		WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot: workflowJSON, Parameters: map[string]string{"objective": "copy exactly"},
		RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(
		ctx, "run-1", runstore.RunInitializing, runstore.RunRunning,
		runstore.Reason{Code: "initialized"},
	); err != nil {
		t.Fatal(err)
	}
	runArtifacts, err := artifactService.Run("run-1")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runArtifacts.Write(
		ctx,
		contracts.ArtifactRef{Namespace: "inputs", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source\n")},
		nil,
	); err != nil {
		t.Fatal(err)
	}
	return runArtifacts
}

func loadSchedulerWorkflow(t *testing.T) workflowconfig.ResolvedWorkflow {
	t.Helper()
	snapshot, err := workflowconfig.Load("../../configs", workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	return workflow
}

type postgresTestWorkerInvoker struct {
	result contracts.StageContentResult
	calls  int
}

func (i *postgresTestWorkerInvoker) Invoke(
	_ context.Context,
	_ string,
	_ contracts.WorkerHandle,
	request contracts.StageContentRequest,
) (contracts.WorkerCompletion, error) {
	i.calls++
	result := cloneStageResult(i.result)
	return contracts.WorkerCompletion{
		APIVersion: contracts.APIVersion,
		Result: &contracts.WorkerResult{
			SubtaskID: request.SubtaskID, Result: result.Summary,
			Observations: contracts.WorkerObservations{
				Profile: contracts.WorkerObservationProfileLeanV1,
				Tools:   map[string]contracts.ToolObservationCount{},
			},
			Artifacts: result.Artifacts, Summarized: false,
		},
		InvocationID: "worker-postgres-test", StateRevision: 2,
	}, nil
}

func testSchedulerRuntimeSettings() contracts.RuntimeSettings {
	return contracts.RuntimeSettings{
		LLMGatewayURL:         "https://gateway.test/v1",
		LLMGatewayToken:       contracts.NewSecretString("test-token"),
		ArtifactAPIURL:        "https://control.test/private/v1",
		RequestTimeoutSeconds: 5,
	}
}

func isolatedSchedulerPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	adminConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	admin, err := pgxpool.NewWithConfig(ctx, adminConfig)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := cryptorand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_scheduler_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop Scheduler test schema: %v", err)
		}
		admin.Close()
	})
	return pool
}
