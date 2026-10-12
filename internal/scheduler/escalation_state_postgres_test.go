package scheduler

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgconn"
)

// stageHistoryReads counts the Scheduler's reads of a Run's Stage history,
// which only the escalation budget count (escalationAttempts) makes while a
// Stage finalizes or aborts. The next reads fail with failures, in order.
type stageHistoryReads struct {
	Store
	reads    int
	failures []error
}

func (s *stageHistoryReads) ListStageExecutions(ctx context.Context, runID string) ([]runstore.StageExecution, error) {
	s.reads++
	if len(s.failures) > 0 {
		failure := s.failures[0]
		s.failures = s.failures[1:]
		return nil, failure
	}
	return s.Store.ListStageExecutions(ctx, runID)
}

// escalationFixture is run-1 on PostgreSQL whose copy Stage escalates one
// outcome, failed or interrupted, at most twice and then fails the Run. It
// writes the Stage history through the persistence commits directly, so a
// test can store an escalation history that the Scheduler never derives.
type escalationFixture struct {
	deferredPlacementHarness
	outcome      string
	reads        *stageHistoryReads
	persistence  *PostgresPersistence
	stage        resolved.ResolvedStage
	stageContext runstore.StageContextSnapshot
	pins         []ContextPin
}

func newEscalationFixture(t *testing.T, ctx context.Context, outcome string) escalationFixture {
	t.Helper()
	h := newDeferredPlacementHarness(t, ctx, nil, func(workflow *resolved.ResolvedWorkflow) {
		stage := workflow.Stages[workflow.EntryStage]
		policy := stage.ExecutionConfig.Agents["builder"].ModelPolicy
		escalate := resolved.TransitionAction{
			Kind: resolved.TransitionEscalate,
			Escalate: &resolved.EscalateTransition{
				MaxAttempts: 2,
				ExecutionConfig: resolved.ResolvedEscalationExecutionConfig{
					Override: resolved.ResolvedStageExecutionConfigOverride{
						Agents: map[string]resolved.ResolvedExecutionSelectionOverride{
							"builder": {ModelPolicy: &policy},
						},
					},
				},
				Then: resolved.TransitionAction{Kind: resolved.TransitionFail},
			},
		}
		switch outcome {
		case "failed":
			stage.On.Failed = escalate
		case "interrupted":
			stage.On.Interrupted = escalate
		default:
			t.Fatalf("unknown escalation outcome %q", outcome)
		}
		workflow.Stages[workflow.EntryStage] = stage
		if err := resolved.ResolveEscalationVariants(workflow); err != nil {
			t.Fatal(err)
		}
	})
	if err := resolved.ValidateWorkflowGraph(h.workflow); err != nil {
		t.Fatal(err)
	}
	persistence, err := NewPostgresPersistence(h.pool)
	if err != nil {
		t.Fatal(err)
	}
	runArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(h.pool)).Run("run-1")
	if err != nil {
		t.Fatal(err)
	}
	input, err := runArtifacts.Metadata(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
	if err != nil {
		t.Fatal(err)
	}
	reads := &stageHistoryReads{Store: h.store}
	h.scheduler.store = reads
	return escalationFixture{
		deferredPlacementHarness: h, outcome: outcome, reads: reads, persistence: persistence,
		stage: h.workflow.Stages[h.workflow.EntryStage],
		stageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{"objective": "copy exactly"},
			Artifacts: map[string]runstore.PinnedContextArtifact{
				"source": {Required: true, Artifact: &input.Ref},
			},
		},
		pins: []ContextPin{{Name: "source", Ref: input.Ref}},
	}
}

// attempt returns the creation of Stage id as attempt of the copy Stage: the
// base executionConfig when ordinal is nil, otherwise the escalated one with
// that ordinal.
func (f escalationFixture) attempt(
	t *testing.T, id string, attempt int, previous *string, ordinal *int,
) runstore.CreateStageExecutionParams {
	t.Helper()
	stage, variant := f.stage, runstore.StageExecutionConfigBase
	if ordinal != nil {
		action, escalated := stage.On.Failed, runstore.StageExecutionConfigFailedEscalation
		if f.outcome == "interrupted" {
			action, escalated = stage.On.Interrupted, runstore.StageExecutionConfigInterruptedEscalation
		}
		stage.ExecutionConfig, variant = action.Escalate.ExecutionConfig.Effective, escalated
	}
	spec, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	return runstore.CreateStageExecutionParams{
		StageExecutionID: id, RunID: "run-1", StageName: f.workflow.EntryStage, Attempt: attempt,
		PreviousExecutionID: previous, ExecutionConfigVariant: variant, EscalationOrdinal: ordinal,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: spec,
		StageContextSchemaVersion: contracts.APIVersion, StageContext: f.stageContext,
	}
}

// startBase creates the base attempt 1 of the copy Stage and brings it to the
// outcome its escalation decides.
func (f escalationFixture) startBase(t *testing.T, ctx context.Context) {
	t.Helper()
	if _, err := f.persistence.CreateStageWithContext(ctx, f.attempt(t, "stage-base", 1, nil, nil), f.pins); err != nil {
		t.Fatal(err)
	}
	f.endAttempt(t, ctx, "stage-base")
}

// endAttempt brings the preparing Stage id to the outcome the fixture
// escalates: a failed candidate in finalizing, or an interrupted termination
// in aborting.
func (f escalationFixture) endAttempt(t *testing.T, ctx context.Context, id string) {
	t.Helper()
	if f.outcome == "interrupted" {
		if err := f.persistence.EnterAbortingWithTermination(ctx, "run-1", runstore.EnterAbortingParams{
			StageExecutionID: id, ExpectedState: runstore.StagePreparing,
			TerminationSchemaVersion: contracts.APIVersion,
			Termination: runstore.StageTermination{
				Outcome: runstore.TerminationInterrupted, Code: "runtime_unavailable",
				Message: "Runtime Agent became unavailable", Retryable: true,
				Phase: runstore.TerminationPreparing, OccurredAt: time.Now().UTC(),
			},
			AbortID: "abort-" + id, Deadline: time.Now().Add(time.Minute),
			Reason: runstore.Reason{Code: "stage_interrupted"},
		}); err != nil {
			t.Fatal(err)
		}
		return
	}
	started, err := json.Marshal(map[string]string{
		"stageExecutionId": id, "sessionId": "session-" + id, "invocationId": "invocation-" + id,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := f.store.StartPlanner(ctx, runstore.StartPlannerParams{
		StageExecutionID: id, SessionID: "session-" + id, InvocationID: "invocation-" + id,
		StateSchemaVersion: contracts.APIVersion, InitialState: json.RawMessage(`{"status":"test"}`),
		RunEvent: runstore.RunEventAppend{
			EventID: "planner-started-" + id, EventSchemaVersion: contracts.APIVersion,
			Kind: runstore.RunEventPlannerStarted, Data: started,
		},
		Reason: runstore.Reason{Code: "planner_started"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := f.persistence.EnterFinalizingWithResult(ctx, runstore.EnterFinalizingParams{
		StageExecutionID: id, ResultSchemaVersion: contracts.APIVersion,
		Candidate: failedStageResult("worker_failed", false), FinalizationID: "finalization-" + id,
		Deadline: time.Now().Add(time.Minute), Reason: runstore.Reason{Code: "planner_completed"},
	}); err != nil {
		t.Fatal(err)
	}
}

// escalate commits the escalation of the base attempt, which startBase
// brought to its outcome, to attempt 2 with the given ordinal. The store takes
// any positive ordinal that is unique on its branch: only the Scheduler
// derives the next one from the Stage history.
func (f escalationFixture) escalate(t *testing.T, ctx context.Context, ordinal int) {
	t.Helper()
	source, target, stageName := "stage-base", "stage-escalated", f.workflow.EntryStage
	progression := StageProgression{
		Decision: runstore.RecordStageTransitionDecisionParams{
			SourceExecutionID: source, RunID: "run-1", Action: runstore.StageTransitionEscalate,
			TargetStageName: &stageName, TargetExecutionID: &target, EscalationOrdinal: &ordinal,
		},
		NextStage: &NextStageCreation{Params: f.attempt(t, target, 2, &source, &ordinal), ContextPins: f.pins},
	}
	var err error
	if f.outcome == "interrupted" {
		err = f.persistence.CommitTerminationProgression(ctx, TerminationProgression{
			RunID: "run-1", StageExecutionID: source, Progression: progression,
		})
	} else {
		err = f.persistence.CommitResultProgression(ctx, ResultProgression{
			RunID: "run-1", StageExecutionID: source, Result: failedStageResult("worker_failed", false),
			WorkflowOutputs: f.stage.WorkflowOutputs, OutputContracts: f.workflow.Outputs,
			Progression: progression,
		})
	}
	if err != nil {
		t.Fatal(err)
	}
}

// An escalation history that the Scheduler cannot have written, here an
// escalation numbered 2 without a first one, fails every count of the
// escalation budget. A finalizing or aborting escalated Stage of that Run
// therefore ends the Run once as durable state the Scheduler cannot progress
// (SCH-R102, FAIL-SCH-13), instead of failing the same progression on every
// pass. The ended Stage keeps its own outcome and gets no transition decision.
func TestPostgresInvalidEscalationHistoryEndsRunOnce(t *testing.T) {
	for _, outcome := range []string{"failed", "interrupted"} {
		t.Run(outcome, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			f := newEscalationFixture(t, ctx, outcome)
			f.startBase(t, ctx)
			f.escalate(t, ctx, 2)
			f.endAttempt(t, ctx, "stage-escalated")

			if worked, err := f.scheduler.RunOnce(ctx); err != nil || !worked {
				t.Fatalf("first pass = (%v, %v), want the Run ended", worked, err)
			}
			// The second pass finds the Run terminal: it claims nothing and
			// counts the escalation budget no more.
			if worked, err := f.scheduler.RunOnce(ctx); err != nil || worked {
				t.Fatalf("second pass = (%v, %v), want no work", worked, err)
			}
			if f.reads.reads != 1 {
				t.Fatalf("Stage history reads = %d, want the one count of the first pass", f.reads.reads)
			}
			run, err := f.store.GetRun(ctx, "run-1")
			if err != nil || run.State != runstore.RunFailed || run.StateReason.Code != "scheduler_state_invalid" {
				t.Fatalf("Run = (%s %+v, %v), want failed as scheduler_state_invalid", run.State, run.StateReason, err)
			}
			executions, err := f.store.ListStageExecutions(ctx, "run-1")
			if err != nil || len(executions) != 2 || executions[1].StageExecutionID != "stage-escalated" {
				t.Fatalf("StageExecutions = (%v, %v), want the base and the escalated attempt", stageExecutionStates(executions), err)
			}
			escalated := executions[1]
			if outcome == "failed" {
				// The finalizing Stage accepts its candidate without progression.
				if escalated.State != runstore.StageFailed || escalated.AcceptedResult == nil ||
					escalated.Termination != nil {
					t.Fatalf("escalated Stage = %s (accepted %v, termination %+v), want its failed candidate accepted",
						escalated.State, escalated.AcceptedResult != nil, escalated.Termination)
				}
			} else if escalated.State != runstore.StageInterrupted || escalated.Termination == nil ||
				escalated.Termination.Code != "runtime_unavailable" {
				// The aborting Stage commits its own stored termination.
				t.Fatalf("escalated Stage = %s (termination %+v), want its stored termination committed",
					escalated.State, escalated.Termination)
			}
			decisions, err := f.store.ListStageTransitionDecisions(ctx, "run-1")
			if err != nil || len(decisions) != 1 || decisions[0].SourceExecutionID != "stage-base" {
				t.Fatalf("transition decisions = (%+v, %v), want only the base attempt's", decisions, err)
			}
		})
	}
}

// A transient database failure while counting the escalation budget says
// nothing about the Stage history: it is no durable state the Scheduler
// cannot progress. The Run keeps its finalizing or aborting Stage, and the
// next pass counts again and escalates it.
func TestPostgresTransientEscalationHistoryReadIsRetried(t *testing.T) {
	for _, outcome := range []string{"failed", "interrupted"} {
		t.Run(outcome, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			f := newEscalationFixture(t, ctx, outcome)
			f.startBase(t, ctx)
			f.reads.failures = []error{fmt.Errorf("list StageExecutions for WorkflowRun %q: %w", "run-1", &pgconn.PgError{
				Code: persistencepostgres.SQLStateDeadlockDetected, Message: "injected deadlock",
			})}

			worked, err := f.scheduler.RunOnce(ctx)
			if !worked || persistencepostgres.SQLState(err) != persistencepostgres.SQLStateDeadlockDetected {
				t.Fatalf("first pass = (%v, %v), want the transient database failure", worked, err)
			}
			want := runstore.StageFinalizing
			if outcome == "interrupted" {
				want = runstore.StageAborting
			}
			run, err := f.store.GetRun(ctx, "run-1")
			if err != nil || run.State != runstore.RunRunning {
				t.Fatalf("Run after the transient failure = (%s %+v, %v), want running", run.State, run.StateReason, err)
			}
			executions, err := f.store.ListStageExecutions(ctx, "run-1")
			if err != nil || len(executions) != 1 || executions[0].State != want {
				t.Fatalf("StageExecutions after the transient failure = (%v, %v), want %s", stageExecutionStates(executions), err, want)
			}

			if worked, err := f.scheduler.RunOnce(ctx); err != nil || !worked {
				t.Fatalf("replayed pass = (%v, %v)", worked, err)
			}
			if f.reads.reads != 2 {
				t.Fatalf("Stage history reads = %d, want one per pass", f.reads.reads)
			}
			executions, err = f.store.ListStageExecutions(ctx, "run-1")
			if err != nil || len(executions) != 2 || executions[1].State != runstore.StagePreparing ||
				executions[1].EscalationOrdinal == nil || *executions[1].EscalationOrdinal != 1 ||
				executions[1].PreviousExecutionID == nil || *executions[1].PreviousExecutionID != "stage-base" {
				t.Fatalf("StageExecutions after the replay = (%v, %v), want the first escalation", stageExecutionStates(executions), err)
			}
			decisions, err := f.store.ListStageTransitionDecisions(ctx, "run-1")
			if err != nil || len(decisions) != 1 || decisions[0].Action != runstore.StageTransitionEscalate ||
				decisions[0].TargetExecutionID == nil || *decisions[0].TargetExecutionID != executions[1].StageExecutionID {
				t.Fatalf("transition decisions = (%+v, %v), want the first escalation", decisions, err)
			}
			if run, err := f.store.GetRun(ctx, "run-1"); err != nil || run.State != runstore.RunRunning {
				t.Fatalf("Run after the replay = (%s, %v), want running", run.State, err)
			}
		})
	}
}
