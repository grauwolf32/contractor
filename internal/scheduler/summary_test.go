package scheduler

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner/stageresult"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgconn"
)

func configureSummaryWorkflow(t *testing.T, workflow *resolved.ResolvedWorkflow) {
	t.Helper()
	workflow.MaxSubtasks = 64
	for name, stage := range workflow.Stages {
		stage.MaxSubtasks = workflow.MaxSubtasks
		workflow.Stages[name] = stage
	}
	catalog, err := configload.Load(configtest.CopyWithPolicies(t, "../../testdata/configs"))
	if err != nil {
		t.Fatal(err)
	}
	policy, err := catalog.ModelPolicy("test-planner@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	binding := stage.Agents["builder"]
	worker := stage.ExecutionConfig.Agents["builder"]
	worker.ModelPolicy.ContextWindowTokens = 131072
	stage.ExecutionConfig.Agents["builder"] = worker
	binding.Template.ModelPolicy = worker.ModelPolicy.Clone()
	summaryPolicy := worker.ModelPolicy.Clone()
	summaryPolicy.MaxModelCalls, summaryPolicy.MaxToolCalls, summaryPolicy.MaxWorkerCalls = 1, 0, 0
	binding.Template.Summarizer = &contracts.WorkerSummarizerConfig{ModelPolicy: summaryPolicy, ContextWindowRatio: 0.9}
	stage.Agents["builder"] = binding
	stage.On.Failed = resolved.TransitionAction{Kind: resolved.TransitionFail}
	stage.On.Interrupted = resolved.TransitionAction{Kind: resolved.TransitionFail}
	gateway := *worker.LLMGateway
	stage.On.Summarized = &resolved.TransitionAction{
		Kind: resolved.TransitionEscalate,
		Escalate: &resolved.EscalateTransition{
			MaxAttempts: 1, Planner: &resolved.PlannerRef{PlannerID: "streamline", Version: "1"}, DeadlineSeconds: 10,
			ExecutionConfig: resolved.ResolvedEscalationExecutionConfig{
				Override: resolved.ResolvedStageExecutionConfigOverride{
					Planner: &resolved.ResolvedExecutionSelectionOverride{ModelPolicy: &policy, LLMGateway: &gateway},
				},
			},
		},
	}
	workflow.Stages[workflow.EntryStage] = stage
	if err := resolved.ResolveEscalationVariants(workflow); err != nil {
		t.Fatal(err)
	}
}

func summarizedCandidate() stageresult.StageContentResult {
	result := failedStageResult(stageresult.SummaryRequiredCode, false)
	result.Summarized, result.Summary = true, "Partial analysis; write and verify the report."
	return result
}

func TestSchedulerSummaryContinuationPreservesOriginalContextAndRetriesStrategy(t *testing.T) {
	h := newSchedulerHarness(t)
	configureSummaryWorkflow(t, &h.workflow)
	configureRetryWorkflow(t, h, 3)
	h.planners.results = []stageresult.StageContentResult{summarizedCandidate(), failedStageResult("gateway_unavailable", true), h.planners.result}
	for index := range 3 {
		if worked, err := h.scheduler.RunOnce(t.Context()); !worked || err != nil {
			t.Fatalf("pass %d: %v %v", index, worked, err)
		}
		if index == 0 {
			source, target := h.store.stages[0], h.store.stages[1]
			if source.AcceptedResult == nil || source.AcceptedResult.SummaryArtifact == nil || target.StageContext.Continuation == nil {
				t.Fatalf("summary was not durably handed off: %+v", h.store.stages)
			}
			h.artifacts.current["run-1/inputs/source"] = "advanced-head"
			if !reflect.DeepEqual(source.StageContext.Artifacts, target.StageContext.Artifacts) ||
				target.ExecutionConfigVariant != runstore.StageExecutionConfigSummaryEscalation {
				t.Fatalf("continuation changed original context: %+v", target)
			}
		}
	}
	if h.store.run.State != runstore.RunSucceeded || len(h.store.stages) != 3 || len(h.artifacts.summaries) != 1 {
		t.Fatalf("continuation/retry result: %s stages=%d summaries=%d", h.store.run.State, len(h.store.stages), len(h.artifacts.summaries))
	}
	for _, attempt := range h.store.stages[1:] {
		stage, err := resolved.DecodeResolvedStageSnapshot(attempt.StageSpecSnapshot)
		if err != nil || stage.MaxSubtasks != h.workflow.MaxSubtasks {
			t.Fatalf("continuation/retry lost its Workflow subtask limit: %v", err)
		}
		if attempt.ExecutionConfigVariant != runstore.StageExecutionConfigSummaryEscalation || *attempt.EscalationOrdinal != 1 ||
			!reflect.DeepEqual(attempt.StageContext, h.store.stages[1].StageContext) {
			t.Fatalf("retry changed strategy/context: %+v", attempt)
		}
	}
	if exists, err := h.store.HasSummaryEscalation(t.Context(), "run-1", h.workflow.EntryStage); err != nil || !exists {
		t.Fatalf("retry lost its escalation root: %v %v", exists, err)
	}
}

func TestSchedulerSummaryWithoutPolicyIsSavedAndFailsNormally(t *testing.T) {
	h := newSchedulerHarness(t)
	h.planners.result = summarizedCandidate()
	if worked, err := h.scheduler.RunOnce(t.Context()); !worked || err != nil {
		t.Fatalf("completion: %v %v", worked, err)
	}
	if h.store.run.State != runstore.RunFailed || h.store.run.StateReason.Code != stageresult.SummaryRequiredCode ||
		len(h.store.stages) != 1 || len(h.artifacts.summaries) != 1 || h.store.stages[0].AcceptedResult.SummaryArtifact == nil {
		t.Fatalf("summary without policy was lost or escalated: %+v", h.store.stages)
	}
}

func TestSchedulerSummaryPublicationCannotTurnResultOverflowIntoAnEndlessRetry(t *testing.T) {
	h := newSchedulerHarness(t)
	candidate := summarizedCandidate()
	base, err := contracts.ResultJSONSize(candidate)
	if err != nil {
		t.Fatal(err)
	}
	candidate.Summary += strings.Repeat("\x01", (stageresult.MaxStageResultBytes-base-16)/6)
	if err := candidate.Validate(); err != nil {
		t.Fatalf("source candidate was not within its bound: %v", err)
	}
	h.planners.result = candidate
	if worked, err := h.scheduler.RunOnce(t.Context()); !worked || err != nil {
		t.Fatalf("overflow remained retryable: %v %v", worked, err)
	}
	if h.store.run.State != runstore.RunFailed || len(h.artifacts.summaries) != 1 ||
		h.store.stages[0].Termination == nil || h.store.stages[0].Termination.Code != "result_contract_violation" ||
		h.store.stages[0].Termination.Retryable {
		t.Fatal("enriched result overflow did not fail permanently while retaining the summary")
	}
}

func TestSchedulerSummarySourceAndRetryLineageRejectCorruption(t *testing.T) {
	h := newSchedulerHarness(t)
	configureSummaryWorkflow(t, &h.workflow)
	installHarnessWorkflow(t, h)
	h.planners.result = summarizedCandidate()
	if _, err := h.scheduler.RunOnce(t.Context()); err != nil {
		t.Fatal(err)
	}
	source, target := h.store.stages[0], h.store.stages[1]
	for _, corrupt := range []func(*runstore.StageExecution){
		func(e *runstore.StageExecution) { e.StageContext.Continuation.Summary = "changed" },
		func(e *runstore.StageExecution) { e.StageContext.Parameters["objective"] = "broader scope" },
		func(e *runstore.StageExecution) { *e.StageContext.Artifacts["source"].Artifact.Revision = "new" },
		func(e *runstore.StageExecution) { e.StageContext.Continuation.SourceStageExecutionID = "missing" },
		func(e *runstore.StageExecution) { e.PreviousExecutionID = stringPointer("missing-parent") },
		func(e *runstore.StageExecution) { e.Attempt++ },
	} {
		changed := target
		changed.StageContext = target.StageContext.Clone()
		corrupt(&changed)
		if err := h.scheduler.validateSummarySource(t.Context(), h.store.run, changed); !errors.Is(err, errInvalidRunState) {
			t.Fatalf("corruption accepted: %v", err)
		}
	}
	target.State = runstore.StageFailed
	retry := target
	retry.StageExecutionID, retry.PreviousExecutionID, retry.Attempt = "retry", &target.StageExecutionID, target.Attempt+1
	retry.StageContext = target.StageContext.Clone()
	h.store.stages = []runstore.StageExecution{source, target, retry}
	if err := h.scheduler.validateSummarySource(t.Context(), h.store.run, retry); err != nil {
		t.Fatalf("valid retry rejected: %v", err)
	}
	h.store.stages[2].Attempt++
	if err := h.scheduler.validateSummarySource(t.Context(), h.store.run, h.store.stages[2]); !errors.Is(err, errInvalidRunState) {
		t.Fatalf("corrupt lineage admitted: %v", err)
	}
}

func TestSchedulerSummaryDeadlineStartsAtAdmissionForEveryAttempt(t *testing.T) {
	h := newSchedulerHarness(t)
	configureSummaryWorkflow(t, &h.workflow)
	stage := h.workflow.Stages[h.workflow.EntryStage]
	execution := runstore.StageExecution{StageContext: runstore.StageContextSnapshot{Continuation: &stageresult.SummaryContinuation{}}}
	for _, elapsed := range []time.Duration{0, 24 * time.Hour} {
		now := h.clock.now.Add(elapsed)
		h.scheduler.options.Clock = staticClock{now: now}
		if deadline := h.scheduler.stageDeadline(execution, stage); !deadline.Equal(now.Add(10 * time.Second)) {
			t.Fatalf("queue consumed allowance: %v", deadline)
		}
		admitted := now.Add(-time.Second)
		execution.AdmittedAt = &admitted
		if deadline := h.scheduler.stageDeadline(execution, stage); !deadline.Equal(admitted.Add(10 * time.Second)) {
			t.Fatalf("admission allowance: %v", deadline)
		}
		execution.AdmittedAt = nil
	}
	h.scheduler.options.PlannerTimeout = 5 * time.Second
	if deadline := h.scheduler.stageDeadline(execution, stage); !deadline.Equal(h.scheduler.now().Add(5 * time.Second)) {
		t.Fatalf("enclosing timeout was extended: %v", deadline)
	}
}

func TestSummaryPublicationRetriesOnlyKnownTransientFailures(t *testing.T) {
	for _, err := range []error{context.Canceled, context.DeadlineExceeded, artifacts.ErrBlobIO, &pgconn.PgError{Code: persistencepostgres.SQLStateSerializationFailure}, &pgconn.PgError{Code: persistencepostgres.SQLStateConnectionFailure}} {
		if !summaryPublicationRetryable(err) {
			t.Fatalf("transient error rejected: %v", err)
		}
	}
	for _, err := range []error{errors.New("unclassified"), artifacts.ErrInvalidScope, artifacts.ErrInvalidName, artifacts.ErrInvalidMediaType,
		artifacts.ErrVersionedWriteTarget, artifacts.ErrArtifactIntegrity, artifacts.ErrScopeDeleting, artifacts.ErrBlobBackend,
		artifacts.ErrBlobMissing, artifacts.ErrRunRevisionLimit, errSummaryArtifactConflict} {
		if summaryPublicationRetryable(err) {
			t.Fatalf("permanent error retried: %v", err)
		}
	}
}
