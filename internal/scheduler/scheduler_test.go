package scheduler

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestSchedulerExecutesSingleStageAndFencesBeforeFinalizing(t *testing.T) {
	harness := newSchedulerHarness(t)

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	if harness.store.run.State != runstore.RunSucceeded || len(harness.store.stages) != 1 ||
		harness.store.stages[0].State != runstore.StageSucceeded {
		t.Fatalf("terminal state = run:%s stages:%+v", harness.store.run.State, harness.store.stages)
	}
	output, ok := harness.persistence.outputs["result"]
	if !ok || output.Revision == nil || output.Namespace != "outputs" || output.Name != "result" {
		t.Fatalf("frozen output = %+v", harness.persistence.outputs)
	}
	if harness.planners.createCalls != 1 || harness.planners.runCalls != 1 ||
		harness.workers.prepareCalls != 1 || harness.workers.finalizeCalls != 1 ||
		harness.workers.releaseCalls != 1 {
		t.Fatalf(
			"calls planner=(%d,%d) workers=(%d,%d,%d)",
			harness.planners.createCalls, harness.planners.runCalls,
			harness.workers.prepareCalls, harness.workers.finalizeCalls,
			harness.workers.releaseCalls,
		)
	}
	assertOrderedEvents(t, harness.events.values,
		"create_stage", "reserve", "record_allocation", "prepare", "planner",
		"fence", "enter_finalizing", "finalize", "accept", "release",
	)
	if !reflect.DeepEqual(harness.persistence.stageStates, []runstore.StageExecutionState{
		runstore.StagePreparing, runstore.StageRunning, runstore.StageFinalizing, runstore.StageSucceeded,
	}) {
		t.Fatalf("Stage states = %v", harness.persistence.stageStates)
	}
	if harness.planners.invocation.Context.Artifacts["source"] == nil ||
		harness.planners.invocation.Context.Artifacts["source"].Revision == nil {
		t.Fatalf("Planner did not receive exact StageContext: %+v", harness.planners.invocation.Context)
	}
}

func TestSchedulerMissingRequiredContextInterruptsWithoutAllocation(t *testing.T) {
	harness := newSchedulerHarness(t)
	delete(harness.artifacts.values, "run-1/inputs/source/input-r1")
	delete(harness.artifacts.current, "run-1/inputs/source")

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	execution := harness.store.stages[0]
	if harness.store.run.State != runstore.RunFailed || execution.State != runstore.StageInterrupted ||
		execution.Termination == nil || execution.Termination.Code != "context_artifact_missing" ||
		execution.Termination.Retryable || execution.Termination.Phase != runstore.TerminationPreparing {
		t.Fatalf("missing context lifecycle = run:%s stage:%+v", harness.store.run.State, execution)
	}
	if harness.allocator.reserveCalls != 0 || harness.workers.prepareCalls != 0 ||
		harness.planners.createCalls != 0 {
		t.Fatalf("missing context allocated or planned work")
	}
	assertOrderedEvents(t, harness.events.values, "create_stage", "enter_aborting", "commit_termination")
}

func TestSchedulerLeavesPreparingStageDeferredWhenCapacityIsUnavailable(t *testing.T) {
	harness := newSchedulerHarness(t)
	harness.allocator.reserveError = controlplane.ErrInsufficientCapacity

	worked, err := harness.scheduler.RunOnce(context.Background())
	if !worked || !errors.Is(err, ErrDeferred) {
		t.Fatalf("RunOnce = (%v, %v), want deferred", worked, err)
	}
	if harness.store.run.State != runstore.RunRunning || harness.store.stages[0].State != runstore.StagePreparing ||
		harness.store.stages[0].Termination != nil || harness.planners.createCalls != 0 {
		t.Fatalf("deferred lifecycle = run:%s stage:%+v", harness.store.run.State, harness.store.stages[0])
	}
}

func TestSchedulerPreparationFailureInterruptsWithoutPlanner(t *testing.T) {
	harness := newSchedulerHarness(t)
	harness.workers.prepareError = &controlplane.RuntimeAPIError{
		StatusCode: 503, Code: "worker_prepare_failed", Retryable: true,
	}

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	execution := harness.store.stages[0]
	if execution.State != runstore.StageInterrupted || execution.Termination == nil ||
		execution.Termination.Code != "allocation_preparation_failed" ||
		!execution.Termination.Retryable || execution.Termination.Phase != runstore.TerminationPreparing {
		t.Fatalf("preparation termination = %+v", execution)
	}
	if harness.planners.createCalls != 0 || harness.planners.runCalls != 0 {
		t.Fatal("Planner ran after allocation preparation failed")
	}
	if _, hasFinalizing := eventIndex(harness.events.values, "enter_finalizing"); hasFinalizing {
		t.Fatal("preparation failure entered finalizing")
	}
}

func TestSchedulerInvalidCandidateAbortsWithoutPublishingOutput(t *testing.T) {
	harness := newSchedulerHarness(t)
	invalid := harness.planners.result
	ref := invalid.Artifacts["copied"]
	ref.Revision = nil
	invalid.Artifacts["copied"] = ref
	harness.planners.result = invalid

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	execution := harness.store.stages[0]
	if execution.State != runstore.StageInterrupted || execution.Termination == nil ||
		execution.Termination.Code != "result_contract_violation" || execution.Termination.Retryable {
		t.Fatalf("invalid candidate termination = %+v", execution)
	}
	if len(harness.persistence.outputs) != 0 {
		t.Fatalf("invalid candidate published outputs: %+v", harness.persistence.outputs)
	}
	if _, found := eventIndex(harness.events.values, "enter_finalizing"); found {
		t.Fatal("invalid candidate entered finalizing")
	}
}

func TestSchedulerRecoversDurableFinalizingCandidateWithoutPlannerInvocation(t *testing.T) {
	harness := newSchedulerHarness(t)
	execution := harness.persistedExecution(t, runstore.StageFinalizing)
	result := harness.planners.result
	finalizationID := "finalization-recovery"
	deadline := harness.clock.now.Add(time.Minute)
	execution.CandidateResultSchemaVersion = stringPointer(contracts.APIVersion)
	execution.CandidateResult = &result
	execution.FinalizationID = &finalizationID
	execution.FinalizationDeadline = &deadline
	harness.store.stages = []runstore.StageExecution{execution}
	harness.persistence.stageStates = []runstore.StageExecutionState{runstore.StageFinalizing}
	harness.installRecordedReservation(execution.StageExecutionID)

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	if harness.planners.createCalls != 0 || harness.planners.runCalls != 0 ||
		harness.workers.prepareCalls != 0 || harness.workers.finalizeCalls != 1 {
		t.Fatalf("recovery repeated semantic work: planner=(%d,%d), workers=(%d,%d)",
			harness.planners.createCalls, harness.planners.runCalls,
			harness.workers.prepareCalls, harness.workers.finalizeCalls)
	}
	if harness.store.run.State != runstore.RunSucceeded || harness.store.stages[0].State != runstore.StageSucceeded {
		t.Fatalf("recovered terminal state = run:%s stage:%s", harness.store.run.State, harness.store.stages[0].State)
	}
	assertOrderedEvents(t, harness.events.values, "reserve", "finalize", "accept", "release")
}

func TestSchedulerRestartRetriesReleaseAfterTerminalTransaction(t *testing.T) {
	harness := newSchedulerHarness(t)
	harness.workers.releaseError = errors.New("release acknowledgement lost")

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked || harness.store.run.State != runstore.RunSucceeded {
		t.Fatalf("first RunOnce = (%v, %v), run=%s", worked, err, harness.store.run.State)
	}
	output := harness.persistence.outputs["result"]
	if output.Revision == nil {
		t.Fatal("terminal transaction did not publish output")
	}
	harness.workers.releaseError = nil
	restarted, err := New(
		harness.store, harness.persistence, harness.artifacts, harness.allocator,
		harness.workers, harness.planners, harness.scheduler.options,
	)
	if err != nil {
		t.Fatal(err)
	}
	worked, err = restarted.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("release recovery RunOnce = (%v, %v)", worked, err)
	}
	if harness.workers.releaseCalls != 2 || harness.planners.runCalls != 1 ||
		!reflect.DeepEqual(harness.persistence.outputs["result"], output) {
		t.Fatalf("release recovery changed semantic work: releases=%d planner=%d output=%+v",
			harness.workers.releaseCalls, harness.planners.runCalls, harness.persistence.outputs["result"])
	}
}

func TestSchedulerAbortsRunningStageWhenVolatileControlPlaneStateWasLost(t *testing.T) {
	harness := newSchedulerHarness(t)
	execution := harness.persistedExecution(t, runstore.StageRunning)
	harness.store.stages = []runstore.StageExecution{execution}
	harness.persistence.stageStates = []runstore.StageExecutionState{runstore.StageRunning}
	harness.store.allocations = []runstore.StageAllocation{{
		AllocationID: "lost-allocation", StageExecutionID: execution.StageExecutionID,
		LogicalAgentName: "builder", Namespace: "builder",
		AgentTemplateRef:       harness.workflow.Stages["copy"].Agents["builder"].Template.Ref,
		WorkerRuntimeRef:       harness.workflow.Stages["copy"].Agents["builder"].Template.Runtime,
		RuntimeAgentInstanceID: "lost-runtime",
	}}

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	execution = harness.store.stages[0]
	if execution.State != runstore.StageInterrupted || execution.Termination == nil ||
		execution.Termination.Code != "control_plane_state_lost" ||
		execution.Termination.Phase != runstore.TerminationRunning || !execution.Termination.Retryable {
		t.Fatalf("lost Control Plane termination = %+v", execution)
	}
	if harness.planners.createCalls != 0 || harness.workers.prepareCalls != 0 {
		t.Fatal("lost Control Plane state caused semantic re-execution")
	}
}

type schedulerHarness struct {
	t           *testing.T
	workflow    workflowconfig.ResolvedWorkflow
	clock       staticClock
	events      *eventRecorder
	store       *memorySchedulerStore
	persistence *memoryAtomicPersistence
	artifacts   *memoryArtifactResolver
	allocator   *memoryAllocator
	workers     *memoryWorkers
	planners    *memoryPlannerRegistry
	scheduler   *Scheduler
}

func newSchedulerHarness(t *testing.T) *schedulerHarness {
	t.Helper()
	snapshot, err := workflowconfig.Load("../../configs", workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(workflow)
	clock := staticClock{now: time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC)}
	events := &eventRecorder{}
	store := &memorySchedulerStore{run: runstore.WorkflowRun{
		RunID: "run-1", OwnerID: "user-1", WorkflowName: workflow.Ref.Name,
		WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot: encoded, Parameters: map[string]string{"objective": "copy exactly"},
		State: runstore.RunRunning,
	}, events: events}
	resolver := &memoryArtifactResolver{
		current: map[string]string{"run-1/inputs/source": "input-r1"},
		values: map[string]ResolvedArtifact{
			"run-1/inputs/source/input-r1": {
				Ref: exactRef("inputs", "source", "input-r1"), MediaType: "text/plain",
			},
			"run-1/builder/copied/result-r1": {
				Ref: exactRef("builder", "copied", "result-r1"), MediaType: "text/plain",
			},
		},
	}
	allocator := &memoryAllocator{workflow: workflow, clock: clock, events: events, grants: map[string]controlplane.AllocationGrant{}}
	workers := &memoryWorkers{allocator: allocator, workflow: workflow, clock: clock, events: events}
	result := contracts.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded, Summary: "copied",
		Artifacts: map[string]contracts.ArtifactRef{"copied": exactRef("builder", "copied", "result-r1")},
	}
	planners := &memoryPlannerRegistry{store: store, result: result, events: events}
	persistence := &memoryAtomicPersistence{store: store, allocator: allocator, events: events, outputs: map[string]contracts.ArtifactRef{}}
	planners.onRun = func() {
		persistence.stageStates = append(persistence.stageStates, runstore.StageRunning)
	}
	idSequence := 0
	scheduler, err := New(store, persistence, resolver, allocator, workers, planners, Options{
		PollInterval: time.Second, ClaimDuration: time.Hour, OperationTimeout: time.Second,
		PlannerTimeout: 30 * time.Second, FinalizationTimeout: 10 * time.Second, AbortTimeout: 10 * time.Second,
		RuntimeSettings: contracts.RuntimeSettings{
			LLMGatewayURL:   "https://gateway.test/v1",
			LLMGatewayToken: contracts.NewSecretString("test-token"),
			ArtifactAPIURL:  "https://control.test/private/v1", RequestTimeoutSeconds: 5,
		},
		Clock: clock,
		NewID: func(prefix string) (string, error) {
			idSequence++
			return fmt.Sprintf("%s%d", prefix, idSequence), nil
		},
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	return &schedulerHarness{
		t: t, workflow: workflow, clock: clock, events: events, store: store,
		persistence: persistence, artifacts: resolver, allocator: allocator,
		workers: workers, planners: planners, scheduler: scheduler,
	}
}

func (h *schedulerHarness) persistedExecution(
	t *testing.T,
	state runstore.StageExecutionState,
) runstore.StageExecution {
	t.Helper()
	stage := h.workflow.Stages[h.workflow.EntryStage]
	encodedStage, _ := json.Marshal(stage)
	input := exactRef("inputs", "source", "input-r1")
	execution := runstore.StageExecution{
		StageExecutionID: "stage-recovery", RunID: h.store.run.RunID,
		StageName: h.workflow.EntryStage, Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: encodedStage,
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: cloneParameters(h.store.run.Parameters),
			Artifacts: map[string]runstore.PinnedContextArtifact{
				"source": {Required: true, Artifact: &input},
			},
		},
		State: state,
	}
	if state != runstore.StagePreparing {
		sessionID, invocationID := "session-recovery", "invocation-recovery"
		execution.PlannerSessionID = &sessionID
		execution.PlannerInvocationID = &invocationID
		started := h.clock.now.Add(-time.Second)
		execution.PlannerStartedAt = &started
	}
	return execution
}

func (h *schedulerHarness) installRecordedReservation(stageExecutionID string) {
	reservation := h.allocator.reservation(stageExecutionID)
	h.allocator.cached = []controlplane.Reservation{reservation}
	h.allocator.grants[reservation.Grant.AllocationID] = reservation.Grant
	if h.allocator.fenced == nil {
		h.allocator.fenced = make(map[string]bool)
	}
	h.allocator.fenced[reservation.Grant.AllocationID] = true
	h.store.allocations = []runstore.StageAllocation{{
		AllocationID:           reservation.Grant.AllocationID,
		StageExecutionID:       stageExecutionID,
		LogicalAgentName:       reservation.Grant.LogicalAgentName,
		Namespace:              reservation.Grant.Namespace,
		AgentTemplateRef:       reservation.AgentTemplate.Ref,
		WorkerRuntimeRef:       reservation.AgentTemplate.Runtime,
		RuntimeAgentInstanceID: reservation.Grant.RuntimeInstanceID,
	}}
}

type eventRecorder struct{ values []string }

func (r *eventRecorder) add(value string) { r.values = append(r.values, value) }

type staticClock struct{ now time.Time }

func (c staticClock) Now() time.Time { return c.now }
func (staticClock) After(time.Duration) <-chan time.Time {
	return make(chan time.Time)
}

type memorySchedulerStore struct {
	run         runstore.WorkflowRun
	stages      []runstore.StageExecution
	allocations []runstore.StageAllocation
	claimID     string
	events      *eventRecorder
}

func (s *memorySchedulerStore) ClaimRunnableRun(
	_ context.Context, claimID string, duration time.Duration,
) (runstore.WorkflowRun, error) {
	if s.run.State != runstore.RunRunning || s.claimID != "" || duration <= 0 {
		return runstore.WorkflowRun{}, runstore.ErrNoWork
	}
	s.claimID = claimID
	return s.run, nil
}

func (s *memorySchedulerStore) RenewRunClaim(
	_ context.Context, runID, claimID string, _ time.Duration,
) error {
	if runID != s.run.RunID || claimID != s.claimID {
		return runstore.ErrConflict
	}
	return nil
}

func (s *memorySchedulerStore) ReleaseRunClaim(_ context.Context, runID, claimID string) error {
	if runID != s.run.RunID || claimID != s.claimID {
		return runstore.ErrConflict
	}
	s.claimID = ""
	return nil
}

func (s *memorySchedulerStore) GetRun(_ context.Context, runID string) (runstore.WorkflowRun, error) {
	if runID != s.run.RunID {
		return runstore.WorkflowRun{}, runstore.ErrNotFound
	}
	return s.run, nil
}

func (s *memorySchedulerStore) TransitionRun(
	_ context.Context,
	runID string,
	expected runstore.WorkflowRunState,
	next runstore.WorkflowRunState,
	reason runstore.Reason,
) (runstore.WorkflowRun, error) {
	if runID != s.run.RunID || s.run.State != expected {
		return runstore.WorkflowRun{}, runstore.ErrConflict
	}
	s.run.State, s.run.StateReason = next, reason
	if next == runstore.RunSucceeded || next == runstore.RunFailed || next == runstore.RunCancelled {
		s.claimID = ""
	}
	return s.run, nil
}

func (s *memorySchedulerStore) ListStageExecutions(
	_ context.Context, runID string,
) ([]runstore.StageExecution, error) {
	if runID != s.run.RunID {
		return nil, runstore.ErrNotFound
	}
	return append([]runstore.StageExecution(nil), s.stages...), nil
}

func (s *memorySchedulerStore) ListTerminalStageExecutionsWithAllocations(
	_ context.Context,
) ([]runstore.StageExecution, error) {
	result := make([]runstore.StageExecution, 0)
	for _, execution := range s.stages {
		if execution.State != runstore.StageSucceeded && execution.State != runstore.StageFailed &&
			execution.State != runstore.StageInterrupted && execution.State != runstore.StageCancelled {
			continue
		}
		for _, allocation := range s.allocations {
			if allocation.StageExecutionID == execution.StageExecutionID {
				result = append(result, execution)
				break
			}
		}
	}
	return result, nil
}

func (s *memorySchedulerStore) GetStageExecution(
	_ context.Context, stageExecutionID string,
) (runstore.StageExecution, error) {
	for _, execution := range s.stages {
		if execution.StageExecutionID == stageExecutionID {
			return execution, nil
		}
	}
	return runstore.StageExecution{}, runstore.ErrNotFound
}

func (s *memorySchedulerStore) RecordStageAllocation(
	_ context.Context, allocation runstore.StageAllocation,
) error {
	for _, current := range s.allocations {
		if current.AllocationID == allocation.AllocationID {
			if sameStageAllocation(current, allocation) {
				return nil
			}
			return runstore.ErrConflict
		}
	}
	s.allocations = append(s.allocations, allocation)
	s.events.add("record_allocation")
	return nil
}

func (s *memorySchedulerStore) ListStageAllocations(
	_ context.Context, stageExecutionID string,
) ([]runstore.StageAllocation, error) {
	result := make([]runstore.StageAllocation, 0)
	for _, allocation := range s.allocations {
		if allocation.StageExecutionID == stageExecutionID {
			result = append(result, allocation)
		}
	}
	return result, nil
}

func (s *memorySchedulerStore) EnterAborting(
	_ context.Context, params runstore.EnterAbortingParams,
) error {
	for index := range s.stages {
		if s.stages[index].StageExecutionID == params.StageExecutionID && s.stages[index].State == params.ExpectedState {
			s.stages[index].State = runstore.StageAborting
			s.stages[index].Termination = &params.Termination
			s.stages[index].TerminationSchemaVersion = stringPointer(params.TerminationSchemaVersion)
			s.stages[index].AbortID = stringPointer(params.AbortID)
			s.stages[index].AbortDeadline = &params.Deadline
			s.events.add("enter_aborting")
			return nil
		}
	}
	return runstore.ErrConflict
}

type memoryAtomicPersistence struct {
	store       *memorySchedulerStore
	allocator   *memoryAllocator
	events      *eventRecorder
	outputs     map[string]contracts.ArtifactRef
	stageStates []runstore.StageExecutionState
}

func (p *memoryAtomicPersistence) CreateStageWithContext(
	_ context.Context, params runstore.CreateStageExecutionParams, _ []ContextPin,
) (runstore.StageExecution, error) {
	execution := runstore.StageExecution{
		StageExecutionID: params.StageExecutionID, RunID: params.RunID, StageName: params.StageName,
		Attempt: params.Attempt, PreviousExecutionID: params.PreviousExecutionID,
		StageSpecSchemaVersion: params.StageSpecSchemaVersion, StageSpecSnapshot: params.StageSpecSnapshot,
		StageContextSchemaVersion: params.StageContextSchemaVersion, StageContext: params.StageContext,
		State: runstore.StagePreparing,
	}
	p.store.stages = append(p.store.stages, execution)
	p.stageStates = append(p.stageStates, runstore.StagePreparing)
	p.events.add("create_stage")
	return execution, nil
}

func (p *memoryAtomicPersistence) EnterFinalizingWithResult(
	_ context.Context, params runstore.EnterFinalizingParams,
) error {
	for _, reservation := range p.allocator.cached {
		if !p.allocator.fenced[reservation.Grant.AllocationID] {
			return errors.New("candidate persisted before write fence")
		}
	}
	for index := range p.store.stages {
		if p.store.stages[index].StageExecutionID == params.StageExecutionID &&
			p.store.stages[index].State == runstore.StageRunning {
			candidate := cloneStageResult(params.Candidate)
			p.store.stages[index].State = runstore.StageFinalizing
			p.store.stages[index].CandidateResult = &candidate
			p.store.stages[index].CandidateResultSchemaVersion = stringPointer(params.ResultSchemaVersion)
			p.store.stages[index].FinalizationID = stringPointer(params.FinalizationID)
			p.store.stages[index].FinalizationDeadline = &params.Deadline
			p.stageStates = append(p.stageStates, runstore.StageFinalizing)
			p.events.add("enter_finalizing")
			return nil
		}
	}
	return runstore.ErrConflict
}

func (p *memoryAtomicPersistence) AcceptResultAndFinishRun(
	_ context.Context, acceptance ResultAcceptance,
) error {
	if err := validateAcceptance(acceptance); err != nil {
		return err
	}
	for index := range p.store.stages {
		if p.store.stages[index].StageExecutionID != acceptance.StageExecutionID ||
			p.store.stages[index].State != runstore.StageFinalizing {
			continue
		}
		result := cloneStageResult(acceptance.Result)
		p.store.stages[index].State = runstore.StageExecutionState(result.Outcome)
		p.store.stages[index].AcceptedResult = &result
		p.store.stages[index].AcceptedResultSchemaVersion = stringPointer(contracts.APIVersion)
		p.stageStates = append(p.stageStates, p.store.stages[index].State)
		if result.Outcome == contracts.StageSucceeded {
			for output, resultName := range acceptance.WorkflowOutputs {
				if source, ok := result.Artifacts[resultName]; ok {
					revision := "output-" + *source.Revision
					p.outputs[output] = exactRef("outputs", output, revision)
				}
			}
		}
		p.store.run.State = acceptance.ExpectedRunOutcome
		p.store.claimID = ""
		p.events.add("accept")
		return nil
	}
	return runstore.ErrConflict
}

func (p *memoryAtomicPersistence) CommitTerminationAndFailRun(
	_ context.Context, runID, stageExecutionID string, _ runstore.Reason,
) error {
	if runID != p.store.run.RunID {
		return runstore.ErrNotFound
	}
	for index := range p.store.stages {
		if p.store.stages[index].StageExecutionID == stageExecutionID &&
			p.store.stages[index].State == runstore.StageAborting {
			p.store.stages[index].State = runstore.StageInterrupted
			p.stageStates = append(p.stageStates, runstore.StageAborting, runstore.StageInterrupted)
			p.store.run.State = runstore.RunFailed
			p.store.claimID = ""
			p.events.add("commit_termination")
			return nil
		}
	}
	return runstore.ErrConflict
}

type memoryArtifactResolver struct {
	current map[string]string
	values  map[string]ResolvedArtifact
}

func (r *memoryArtifactResolver) Resolve(
	_ context.Context, runID string, ref contracts.ArtifactRef,
) (ResolvedArtifact, error) {
	revision := ""
	if ref.Revision == nil {
		revision = r.current[runID+"/"+ref.Namespace+"/"+ref.Name]
	} else {
		revision = *ref.Revision
	}
	if revision == "" {
		return ResolvedArtifact{}, artifacts.ErrArtifactNotFound
	}
	result, ok := r.values[runID+"/"+ref.Namespace+"/"+ref.Name+"/"+revision]
	if !ok {
		return ResolvedArtifact{}, artifacts.ErrArtifactNotFound
	}
	return result, nil
}

type memoryAllocator struct {
	workflow     workflowconfig.ResolvedWorkflow
	clock        staticClock
	events       *eventRecorder
	grants       map[string]controlplane.AllocationGrant
	cached       []controlplane.Reservation
	fenced       map[string]bool
	reserveCalls int
	reserveError error
}

func (a *memoryAllocator) ReserveAll(request controlplane.ReservationRequest) ([]controlplane.Reservation, error) {
	a.reserveCalls++
	a.events.add("reserve")
	if a.reserveError != nil {
		return nil, a.reserveError
	}
	if len(a.cached) == 0 {
		a.cached = []controlplane.Reservation{a.reservation(request.StageExecutionID)}
		for _, reservation := range a.cached {
			a.grants[reservation.Grant.AllocationID] = reservation.Grant
		}
	}
	return append([]controlplane.Reservation(nil), a.cached...), nil
}

func (a *memoryAllocator) reservation(stageExecutionID string) controlplane.Reservation {
	binding := a.workflow.Stages[a.workflow.EntryStage].Agents["builder"]
	return controlplane.Reservation{
		Grant: controlplane.AllocationGrant{
			AllocationID: "allocation-1", RuntimeInstanceID: "runtime-1",
			RunID: "run-1", StageExecutionID: stageExecutionID,
			LogicalAgentName: "builder", Namespace: binding.Namespace,
			ReadPolicy: controlplane.ReadCurrentRun, WritePolicy: controlplane.WriteInputsAndIntermediates,
		},
		ControlURL: "https://runtime.test", A2AURL: "https://runtime.test",
		AgentTemplate: binding.Template, LeaseExpiresAt: a.clock.now.Add(time.Minute),
	}
}

func (a *memoryAllocator) GetGrant(allocationID string) (controlplane.AllocationGrant, error) {
	grant, ok := a.grants[allocationID]
	if !ok {
		return controlplane.AllocationGrant{}, controlplane.ErrAllocationNotFound
	}
	return grant, nil
}

func (a *memoryAllocator) SetWriteFence(allocationID string) error {
	grant, ok := a.grants[allocationID]
	if !ok {
		return controlplane.ErrAllocationNotFound
	}
	if a.fenced == nil {
		a.fenced = make(map[string]bool)
	}
	a.fenced[allocationID] = true
	grant.WriteFenced = true
	a.grants[allocationID] = grant
	a.events.add("fence")
	return nil
}

func (a *memoryAllocator) Release(allocationID string) error {
	if _, ok := a.grants[allocationID]; !ok {
		return controlplane.ErrAllocationNotFound
	}
	delete(a.grants, allocationID)
	return nil
}

type memoryWorkers struct {
	allocator     *memoryAllocator
	workflow      workflowconfig.ResolvedWorkflow
	clock         staticClock
	events        *eventRecorder
	prepareCalls  int
	finalizeCalls int
	abortCalls    int
	releaseCalls  int
	prepareError  error
	releaseError  error
}

func (w *memoryWorkers) PrepareAll(
	_ context.Context, reservations []controlplane.Reservation, _ contracts.RuntimeSettings,
) (map[string]contracts.WorkerHandle, error) {
	w.prepareCalls++
	w.events.add("prepare")
	if w.prepareError != nil {
		return nil, w.prepareError
	}
	result := make(map[string]contracts.WorkerHandle, len(reservations))
	for _, reservation := range reservations {
		result[reservation.Grant.LogicalAgentName] = contracts.WorkerHandle{
			AllocationID:     reservation.Grant.AllocationID,
			AgentTemplateRef: reservation.AgentTemplate.Ref,
			WorkerRuntimeRef: reservation.AgentTemplate.Runtime,
			AgentCard:        map[string]any{"name": reservation.Grant.LogicalAgentName},
			LeaseExpiresAt:   reservation.LeaseExpiresAt,
		}
	}
	return result, nil
}

func (w *memoryWorkers) FinalizeAll(
	_ context.Context, reservations []controlplane.Reservation, _ string, _ time.Time,
) (map[string]contracts.ExecutionReport, error) {
	w.finalizeCalls++
	for _, reservation := range reservations {
		if !w.allocator.fenced[reservation.Grant.AllocationID] {
			return nil, errors.New("finalize observed an unfenced allocation")
		}
	}
	w.events.add("finalize")
	return map[string]contracts.ExecutionReport{}, nil
}

func (w *memoryWorkers) AbortAll(
	_ context.Context, _ []controlplane.Reservation, _ string, _ contracts.TerminationError, _ time.Time,
) (map[string]contracts.ExecutionReport, error) {
	w.abortCalls++
	w.events.add("abort")
	return map[string]contracts.ExecutionReport{}, nil
}

func (w *memoryWorkers) ReleaseAll(_ context.Context, reservations []controlplane.Reservation) error {
	w.releaseCalls++
	w.events.add("release")
	if w.releaseError != nil {
		return w.releaseError
	}
	for _, reservation := range reservations {
		delete(w.allocator.grants, reservation.Grant.AllocationID)
	}
	return nil
}

type memoryPlannerRegistry struct {
	store       *memorySchedulerStore
	result      contracts.StageContentResult
	events      *eventRecorder
	createCalls int
	runCalls    int
	invocation  planner.Invocation
	onRun       func()
}

func (r *memoryPlannerRegistry) Create(
	ref string, invocation planner.Invocation,
) (planner.Planner, error) {
	r.createCalls++
	r.invocation = invocation
	if ref != planner.PassthroughRef {
		return nil, errors.New("unexpected Planner ref")
	}
	return plannerFunc(func(context.Context) (contracts.StageContentResult, error) {
		r.runCalls++
		for index := range r.store.stages {
			if r.store.stages[index].StageExecutionID == invocation.StageExecutionID &&
				r.store.stages[index].State == runstore.StagePreparing {
				r.store.stages[index].State = runstore.StageRunning
				sessionID, invocationID := "session-1", "invocation-1"
				r.store.stages[index].PlannerSessionID = &sessionID
				r.store.stages[index].PlannerInvocationID = &invocationID
			}
		}
		if r.onRun != nil {
			r.onRun()
		}
		r.events.add("planner")
		return cloneStageResult(r.result), nil
	}), nil
}

type plannerFunc func(context.Context) (contracts.StageContentResult, error)

func (f plannerFunc) Run(ctx context.Context) (contracts.StageContentResult, error) { return f(ctx) }

func exactRef(namespace, name, revision string) contracts.ArtifactRef {
	return contracts.ArtifactRef{Namespace: namespace, Name: name, Revision: &revision}
}

func sameStageAllocation(left, right runstore.StageAllocation) bool {
	return left.AllocationID == right.AllocationID && left.StageExecutionID == right.StageExecutionID &&
		left.LogicalAgentName == right.LogicalAgentName && left.Namespace == right.Namespace &&
		left.AgentTemplateRef == right.AgentTemplateRef && left.WorkerRuntimeRef == right.WorkerRuntimeRef &&
		left.RuntimeAgentInstanceID == right.RuntimeAgentInstanceID
}

func assertOrderedEvents(t *testing.T, events []string, expected ...string) {
	t.Helper()
	position := 0
	for _, event := range events {
		if position < len(expected) && event == expected[position] {
			position++
		}
	}
	if position != len(expected) {
		t.Fatalf("events = %v, missing ordered suffix %v", events, expected[position:])
	}
}

func eventIndex(events []string, expected string) (int, bool) {
	for index, event := range events {
		if event == expected {
			return index, true
		}
	}
	return 0, false
}

func sortedKeys[T any](values map[string]T) []string {
	result := make([]string, 0, len(values))
	for key := range values {
		result = append(result, key)
	}
	sort.Strings(result)
	return result
}
