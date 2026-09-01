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
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestDecodeExecutableWorkflowAllowsMultiWorkerRouterOnly(t *testing.T) {
	snapshot, err := workflowconfig.Load("../../configs", workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	reviewer := stage.Agents["builder"]
	reviewer.Namespace = "reviewer"
	stage.Agents["reviewer"] = reviewer
	stage.ExecutionConfig.Agents["reviewer"] = stage.ExecutionConfig.Agents["builder"]
	plannerPolicy, err := snapshot.ModelPolicy("planner@1")
	if err != nil {
		t.Fatal(err)
	}
	gateway, err := snapshot.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	plannerConfig := workflowconfig.ResolvedConsumerExecutionConfig{
		ModelPolicy: plannerPolicy,
		LLMGateway:  &gateway,
		Origins: workflowconfig.ExecutionConfigOrigins{
			ModelPolicy: "test",
			LLMGateway:  "test",
		},
	}

	decode := func(value workflowconfig.ResolvedWorkflow) error {
		encoded, marshalErr := json.Marshal(value)
		if marshalErr != nil {
			t.Fatal(marshalErr)
		}
		_, decodeErr := decodeExecutableWorkflow(runstore.WorkflowRun{
			WorkflowName: value.Ref.Name, WorkflowVersion: value.Ref.Version,
			WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: encoded,
		})
		return decodeErr
	}
	for _, plannerID := range []string{"streamline", "passthrough"} {
		stage.Planner = workflowconfig.PlannerRef{PlannerID: plannerID, Version: "1"}
		stage.ExecutionConfig.Planner = &plannerConfig
		if plannerID == "passthrough" {
			stage.ExecutionConfig.Planner = nil
		}
		workflow.Stages[workflow.EntryStage] = stage
		if err := decode(workflow); !errors.Is(err, ErrUnsupportedWorkflow) {
			t.Fatalf("multi-Worker %s error = %v", plannerID, err)
		}
	}
	stage.Planner = workflowconfig.PlannerRef{PlannerID: "router", Version: "1"}
	stage.ExecutionConfig.Planner = &plannerConfig
	workflow.Stages[workflow.EntryStage] = stage
	if err := decode(workflow); err != nil {
		t.Fatalf("multi-Worker router Workflow was rejected: %v", err)
	}
}

func TestDecodeExecutableWorkflowRejectsLegacySnapshotWithoutExecutionConfig(t *testing.T) {
	snapshot, err := workflowconfig.Load("../../configs", workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	stage.ExecutionConfig = workflowconfig.ResolvedStageExecutionConfig{}
	workflow.Stages[workflow.EntryStage] = stage
	encoded, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	_, err = decodeExecutableWorkflow(runstore.WorkflowRun{
		WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: encoded,
	})
	if !errors.Is(err, ErrUnsupportedWorkflow) || !strings.Contains(err.Error(), "executionConfig") {
		t.Fatalf("legacy Workflow snapshot error = %v", err)
	}
}

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
		"fence", "enter_finalizing", "finalize", "record_report", "accept", "release",
	)
	if len(harness.store.reports) != 1 ||
		harness.store.reports[0].Report.Worker.Metrics.ModelCalls == nil ||
		*harness.store.reports[0].Report.Worker.Metrics.ModelCalls != 1 {
		t.Fatalf("persisted execution reports = %+v", harness.store.reports)
	}
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

func TestSchedulerBuildsIndependentPinnedPlannerAndWorkerModelAccess(t *testing.T) {
	harness := newSchedulerHarness(t)
	snapshot, err := workflowconfig.Load("../../configs", workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	plannerPolicy, _ := snapshot.ModelPolicy("planner@1")
	workerPolicy, _ := snapshot.ModelPolicy("domain_worker@1")
	workerGateway, _ := snapshot.LLMGateway("local-litellm@1")
	plannerGateway := contracts.ResolvedLLMGatewayConfig{
		Ref: contracts.LLMGatewayConfigRef{
			GatewayID: "planner-gateway", Version: "1", Digest: "sha256:" + strings.Repeat("b", 64),
		},
		Protocol: contracts.OpenAICompatibleProtocol,
		URL:      "https://planner.example/v1",
	}
	plannerCredential := contracts.LLMCredentialRef{CredentialID: "planner-credential"}
	workerCredential := contracts.LLMCredentialRef{CredentialID: "worker-credential"}
	provider, err := credentials.NewStaticProvider([]credentials.StaticEntry{
		{
			Metadata: workflowconfig.CredentialMetadata{
				Ref: plannerCredential, LLMGateway: plannerGateway.Ref,
			},
			Token: contracts.NewSecretString("planner-secret"),
		},
		{
			Metadata: workflowconfig.CredentialMetadata{
				Ref: workerCredential, LLMGateway: workerGateway.Ref,
			},
			Token: contracts.NewSecretString("worker-secret"),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	harness.scheduler.options.Credentials = provider
	stage := harness.workflow.Stages[harness.workflow.EntryStage]
	stage.ExecutionConfig.Planner = &workflowconfig.ResolvedConsumerExecutionConfig{
		ModelPolicy: plannerPolicy, LLMGateway: &plannerGateway, Credential: &plannerCredential,
		Origins: workflowconfig.ExecutionConfigOrigins{ModelPolicy: "test", LLMGateway: "test", Credential: "test"},
	}
	workerSelection := stage.ExecutionConfig.Agents["builder"]
	workerSelection.ModelPolicy = workerPolicy
	workerSelection.Credential = &workerCredential
	stage.ExecutionConfig.Agents["builder"] = workerSelection

	workerSettings, err := harness.scheduler.workerExecutionSettings(t.Context(), stage)
	if err != nil {
		t.Fatal(err)
	}
	worker := workerSettings["builder"]
	if worker.ModelPolicy.Ref != workerPolicy.Ref ||
		worker.RuntimeSettings.LLMGatewayURL != workerGateway.URL ||
		worker.RuntimeSettings.LLMGatewayToken.Reveal() != "worker-secret" ||
		worker.RuntimeSettings.ArtifactAPIURL != "https://control.test/private/v1" ||
		stage.Agents["builder"].Template.ModelPolicy.Ref == workerPolicy.Ref {
		t.Fatalf("Worker execution settings = %+v", worker)
	}
	plannerAccess, err := harness.scheduler.plannerModelAccess(t.Context(), stage)
	if err != nil {
		t.Fatal(err)
	}
	if plannerAccess == nil || plannerAccess.ModelPolicy.Ref != plannerPolicy.Ref ||
		plannerAccess.LLMGateway.Ref != plannerGateway.Ref ||
		plannerAccess.Token.Reveal() != "planner-secret" ||
		plannerAccess.Credential == nil || *plannerAccess.Credential != plannerCredential {
		t.Fatalf("Planner model access = %+v", plannerAccess)
	}
	harness.scheduler.options.Credentials = credentialResolverFunc(func(
		context.Context, contracts.LLMCredentialRef, contracts.LLMGatewayConfigRef,
	) (contracts.SecretString, error) {
		return contracts.SecretString{}, errors.New("provider leaked worker-secret")
	})
	if _, err := harness.scheduler.workerExecutionSettings(t.Context(), stage); err == nil ||
		strings.Contains(err.Error(), "worker-secret") {
		t.Fatalf("unsafe credential resolution error = %v", err)
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

func TestSchedulerCapacityWaitStopsAtImmutableStageDeadline(t *testing.T) {
	harness := newSchedulerHarness(t)
	harness.allocator.reserveError = controlplane.ErrInsufficientCapacity

	worked, err := harness.scheduler.RunOnce(context.Background())
	if !worked || !errors.Is(err, ErrDeferred) || len(harness.store.stages) != 1 {
		t.Fatalf("initial capacity wait = (%v, %v), stages=%d", worked, err, len(harness.store.stages))
	}
	harness.store.stages[0].CreatedAt = harness.clock.now.Add(-31 * time.Second)

	worked, err = harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("expired capacity wait = (%v, %v)", worked, err)
	}
	execution := harness.store.stages[0]
	if execution.State != runstore.StageInterrupted || execution.Termination == nil ||
		execution.Termination.Code != "stage_deadline_exceeded" ||
		execution.Termination.Phase != runstore.TerminationPreparing ||
		!execution.Termination.Retryable || harness.planners.createCalls != 0 {
		t.Fatalf("expired capacity lifecycle = %+v", execution)
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

func TestSchedulerPlannerBudgetFailureUsesBoundedRunningAbort(t *testing.T) {
	harness := newSchedulerHarness(t)
	harness.planners.runErrors = []error{planner.NewError(
		"planner_model_call_limit", "Planner exhausted its model-call limit", true, nil,
	)}

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	execution := harness.store.stages[0]
	if execution.State != runstore.StageInterrupted || execution.Termination == nil ||
		execution.Termination.Code != "planner_model_call_limit" ||
		execution.Termination.Phase != runstore.TerminationRunning ||
		!execution.Termination.Retryable {
		t.Fatalf("budget termination = %+v", execution)
	}
	if harness.workers.abortCalls != 1 || harness.workers.finalizeCalls != 0 ||
		len(harness.persistence.outputs) != 0 {
		t.Fatalf("budget abort path = abort:%d finalize:%d outputs:%v",
			harness.workers.abortCalls, harness.workers.finalizeCalls, harness.persistence.outputs)
	}
	assertOrderedEvents(t, harness.events.values,
		"planner", "fence", "enter_aborting", "abort", "record_report", "commit_termination", "release",
	)
}

func TestSchedulerMetricsPersistenceFailureDoesNotChangeSemanticResult(t *testing.T) {
	harness := newSchedulerHarness(t)
	harness.store.reportError = errors.New("telemetry database unavailable")

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	if harness.store.run.State != runstore.RunSucceeded ||
		harness.store.stages[0].State != runstore.StageSucceeded ||
		len(harness.persistence.outputs) != 1 {
		t.Fatalf("telemetry failure changed semantic result: run=%s stage=%s outputs=%v",
			harness.store.run.State, harness.store.stages[0].State, harness.persistence.outputs)
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
	assertOrderedEvents(t, harness.events.values, "reserve", "finalize", "record_report", "accept", "release")
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

func TestTerminalReleaseFailureDoesNotBlockClaimPath(t *testing.T) {
	harness := newSchedulerHarness(t)
	execution := harness.persistedExecution(t, runstore.StageSucceeded)
	harness.store.stages = []runstore.StageExecution{execution}
	harness.store.run.State = runstore.RunSucceeded
	harness.installRecordedReservation(execution.StageExecutionID)
	harness.workers.releaseError = errors.New("runtime unavailable")

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || worked {
		t.Fatalf("RunOnce with failed cleanup = (%v, %v)", worked, err)
	}
	if harness.store.claimCalls != 1 || harness.workers.releaseCalls != 1 {
		t.Fatalf("failed cleanup blocked claim path: claims=%d releases=%d",
			harness.store.claimCalls, harness.workers.releaseCalls)
	}
}

func TestTerminalReleaseRecoveryRetriesOnlyRemainingLiveAllocations(t *testing.T) {
	harness := newSchedulerHarness(t)
	execution := harness.persistedExecution(t, runstore.StageSucceeded)
	harness.store.stages = []runstore.StageExecution{execution}
	harness.store.run.State = runstore.RunSucceeded
	first := harness.allocator.reservation(execution.StageExecutionID)
	second := first
	second.Grant.AllocationID = "allocation-second"
	second.Grant.RuntimeInstanceID = "runtime-2"
	second.Grant.LogicalAgentName = "reviewer"
	second.Grant.Namespace = "reviewer"
	harness.allocator.cached = []controlplane.Reservation{first, second}
	harness.allocator.grants[first.Grant.AllocationID] = first.Grant
	harness.allocator.grants[second.Grant.AllocationID] = second.Grant
	harness.store.allocations = []runstore.StageAllocation{
		stageAllocationFromReservation(first), stageAllocationFromReservation(second),
	}
	harness.workers.releaseErrors = map[string]error{
		second.Grant.AllocationID: errors.New("second Runtime unavailable"),
	}

	worked, err := harness.scheduler.recoverTerminalRelease(context.Background())
	if !worked || err == nil {
		t.Fatalf("partial release = (%v, %v)", worked, err)
	}
	if harness.store.allocations[0].ReleaseCompletedAt == nil ||
		harness.store.allocations[1].ReleaseCompletedAt != nil {
		t.Fatalf("partial release markers = %+v", harness.store.allocations)
	}
	if _, ok := harness.allocator.grants[first.Grant.AllocationID]; ok {
		t.Fatal("successfully released allocation remained live")
	}
	delete(harness.workers.releaseErrors, second.Grant.AllocationID)
	worked, err = harness.scheduler.recoverTerminalRelease(context.Background())
	if !worked || err != nil || harness.store.allocations[1].ReleaseCompletedAt == nil {
		t.Fatalf("remaining release retry = (%v, %v), allocations=%+v", worked, err, harness.store.allocations)
	}
	if harness.workers.releasedBatches[0] != 2 || harness.workers.releasedBatches[1] != 1 {
		t.Fatalf("release retry batches = %v", harness.workers.releasedBatches)
	}
}

func stageAllocationFromReservation(reservation controlplane.Reservation) runstore.StageAllocation {
	return runstore.StageAllocation{
		AllocationID: reservation.Grant.AllocationID, StageExecutionID: reservation.Grant.StageExecutionID,
		LogicalAgentName: reservation.Grant.LogicalAgentName, Namespace: reservation.Grant.Namespace,
		AgentTemplateRef: reservation.AgentTemplate.Ref, WorkerRuntimeRef: reservation.AgentTemplate.Runtime,
		RuntimeAgentInstanceID: reservation.Grant.RuntimeInstanceID,
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

func TestSchedulerCancelsRunWithoutCreatingStage(t *testing.T) {
	harness := newSchedulerHarness(t)
	harness.requestCancellation("stop before work starts")

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	if harness.store.run.State != runstore.RunCancelled || len(harness.store.stages) != 0 ||
		harness.planners.createCalls != 0 || harness.allocator.reserveCalls != 0 {
		t.Fatalf("cancel-before-Stage = run:%s stages:%v planners:%d reserves:%d",
			harness.store.run.State, harness.store.stages, harness.planners.createCalls, harness.allocator.reserveCalls)
	}
}

func TestSchedulerCancelInterruptsPlannerAndIgnoresLateCandidate(t *testing.T) {
	harness := newSchedulerHarness(t)
	harness.planners.onRun = func() {
		harness.persistence.stageStates = append(harness.persistence.stageStates, runstore.StageRunning)
		harness.requestCancellation("late candidate must not win")
		harness.scheduler.Cancel(harness.store.run.RunID)
	}

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	execution := harness.store.stages[0]
	if harness.store.run.State != runstore.RunCancelled || execution.State != runstore.StageCancelled ||
		execution.Termination == nil || execution.Termination.Outcome != runstore.TerminationCancelled ||
		execution.Termination.Code != runstore.CancellationUserRequested {
		t.Fatalf("cancelled lifecycle = run:%s stage:%+v", harness.store.run.State, execution)
	}
	if execution.CandidateResult != nil || execution.AcceptedResult != nil || len(harness.persistence.outputs) != 0 {
		t.Fatalf("late candidate became semantic output: stage=%+v outputs=%v", execution, harness.persistence.outputs)
	}
	if harness.workers.abortCalls != 1 || harness.workers.finalizeCalls != 0 ||
		len(harness.store.reports) != 1 || harness.store.reports[0].Report.Worker.Complete {
		t.Fatalf("bounded abort/report calls = abort:%d finalize:%d reports:%+v",
			harness.workers.abortCalls, harness.workers.finalizeCalls, harness.store.reports)
	}
	assertOrderedEvents(t, harness.events.values,
		"planner", "fence", "enter_aborting", "abort", "record_report", "commit_termination", "release",
	)
}

func TestSchedulerMetricsCancelDeadlineTerminatesWithIncompleteReportAndFencedAllocation(t *testing.T) {
	harness := newSchedulerHarness(t)
	harness.requestCancellation("runtime is unreachable")
	execution := harness.persistedExecution(t, runstore.StageAborting)
	abortID := "abort-recovery"
	deadline := harness.clock.now.Add(-time.Second)
	execution.TerminationSchemaVersion = stringPointer(contracts.APIVersion)
	execution.Termination = &runstore.StageTermination{
		Outcome: runstore.TerminationCancelled, Code: runstore.CancellationUserRequested,
		Message: "runtime is unreachable", Retryable: false,
		Phase: runstore.TerminationRunning, OccurredAt: harness.clock.now.Add(-2 * time.Second),
	}
	execution.AbortID = &abortID
	execution.AbortDeadline = &deadline
	harness.store.stages = []runstore.StageExecution{execution}
	harness.installRecordedReservation(execution.StageExecutionID)
	harness.workers.releaseError = errors.New("runtime unreachable")

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	allocationID := harness.store.allocations[0].AllocationID
	if harness.workers.abortCalls != 0 || harness.store.run.State != runstore.RunCancelled ||
		harness.store.stages[0].State != runstore.StageCancelled || !harness.allocator.fenced[allocationID] {
		t.Fatalf("deadline cancel = aborts:%d run:%s stage:%s fenced:%v",
			harness.workers.abortCalls, harness.store.run.State, harness.store.stages[0].State,
			harness.allocator.fenced[allocationID])
	}
	if _, stillUnavailable := harness.allocator.grants[allocationID]; !stillUnavailable ||
		len(harness.store.reports) != 1 || harness.store.reports[0].Report.Worker.Complete ||
		len(harness.store.reports[0].Report.Worker.Errors) != 1 ||
		harness.store.reports[0].Report.Worker.Errors[0].Code != "allocation_report_unavailable" {
		t.Fatalf("lost allocation/report = grants:%v reports:%+v", harness.allocator.grants, harness.store.reports)
	}
}

func TestSchedulerCancelAcceptsAlreadyFinalizingResultForAuditOnly(t *testing.T) {
	harness := newSchedulerHarness(t)
	harness.requestCancellation("result already won Stage race")
	execution := harness.persistedExecution(t, runstore.StageFinalizing)
	result := cloneStageResult(harness.planners.result)
	finalizationID := "finalization-before-cancel"
	deadline := harness.clock.now.Add(time.Minute)
	execution.CandidateResultSchemaVersion = stringPointer(contracts.APIVersion)
	execution.CandidateResult = &result
	execution.FinalizationID = &finalizationID
	execution.FinalizationDeadline = &deadline
	harness.store.stages = []runstore.StageExecution{execution}
	harness.installRecordedReservation(execution.StageExecutionID)

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	execution = harness.store.stages[0]
	if harness.store.run.State != runstore.RunCancelled || execution.State != runstore.StageSucceeded ||
		execution.AcceptedResult == nil || len(harness.persistence.outputs) != 0 {
		t.Fatalf("audit-only result = run:%s stage:%+v outputs:%v",
			harness.store.run.State, execution, harness.persistence.outputs)
	}
	assertOrderedEvents(t, harness.events.values, "finalize", "record_report", "accept_cancelled", "release")
}

func TestSchedulerLeaseLossInterruptsPlannerAndStartsBoundedAbort(t *testing.T) {
	harness := newSchedulerHarness(t)
	harness.planners.onRun = func() {
		harness.persistence.stageStates = append(harness.persistence.stageStates, runstore.StageRunning)
		reservation := harness.allocator.cached[0]
		reservation.Grant.Lost = true
		reservation.Grant.WriteFenced = true
		harness.allocator.cached[0] = reservation
		grant := harness.allocator.grants[reservation.Grant.AllocationID]
		grant.Lost = true
		grant.WriteFenced = true
		harness.allocator.grants[reservation.Grant.AllocationID] = grant
		harness.allocator.losses = append(harness.allocator.losses, controlplane.AllocationLoss{
			AllocationID:      reservation.Grant.AllocationID,
			RuntimeInstanceID: reservation.Grant.RuntimeInstanceID,
			RunID:             harness.store.run.RunID,
			StageExecutionID:  reservation.Grant.StageExecutionID,
			Reason:            controlplane.LossControlLeaseExpired,
		})
		harness.scheduler.pollAllocationLosses()
	}

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	execution := harness.store.stages[0]
	if harness.store.run.State != runstore.RunFailed || execution.State != runstore.StageInterrupted ||
		execution.Termination == nil || execution.Termination.Code != "control_lease_expired" ||
		!execution.Termination.Retryable || execution.CandidateResult != nil {
		t.Fatalf("lease-loss lifecycle = run:%s stage:%+v", harness.store.run.State, execution)
	}
	if harness.workers.abortCalls != 1 || harness.workers.finalizeCalls != 0 ||
		len(harness.persistence.outputs) != 0 {
		t.Fatalf("lease-loss remote path = abort:%d finalize:%d outputs:%v",
			harness.workers.abortCalls, harness.workers.finalizeCalls, harness.persistence.outputs)
	}
}

func TestSchedulerRetryCreatesThreeFreshAttemptsBeforeSuccess(t *testing.T) {
	harness := newSchedulerHarness(t)
	configureRetryWorkflow(t, harness, 3)
	harness.artifacts.current["run-1/inputs/source"] = "input-r1"
	harness.artifacts.values["run-1/inputs/source/input-r2"] = ResolvedArtifact{
		Ref: exactRef("inputs", "source", "input-r2"), MediaType: "text/plain",
	}
	harness.planners.runErrors = []error{
		planner.NewError("transient_one", "first transient failure", true, nil),
		planner.NewError("transient_two", "second transient failure", true, nil),
		nil,
	}
	harness.planners.onRun = func() {
		harness.persistence.stageStates = append(harness.persistence.stageStates, runstore.StageRunning)
		if harness.planners.runCalls == 1 {
			harness.artifacts.current["run-1/inputs/source"] = "input-r2"
		}
	}

	for iteration := 0; iteration < 3; iteration++ {
		worked, err := harness.scheduler.RunOnce(context.Background())
		if err != nil || !worked {
			t.Fatalf("RunOnce %d = (%v, %v)", iteration+1, worked, err)
		}
	}
	if harness.store.run.State != runstore.RunSucceeded || len(harness.store.stages) != 3 {
		t.Fatalf("retry terminal state = run:%s stages:%+v", harness.store.run.State, harness.store.stages)
	}
	for index, execution := range harness.store.stages {
		if execution.Attempt != index+1 {
			t.Fatalf("attempt[%d] = %d", index, execution.Attempt)
		}
		if index == 0 && execution.PreviousExecutionID != nil {
			t.Fatalf("first attempt has previous execution: %v", execution.PreviousExecutionID)
		}
		if index > 0 && (execution.PreviousExecutionID == nil ||
			*execution.PreviousExecutionID != harness.store.stages[index-1].StageExecutionID) {
			t.Fatalf("attempt[%d] lineage = %v", index, execution.PreviousExecutionID)
		}
	}
	firstRef := harness.store.stages[0].StageContext.Artifacts["source"].Artifact
	secondRef := harness.store.stages[1].StageContext.Artifacts["source"].Artifact
	if firstRef == nil || secondRef == nil || firstRef.Revision == nil || secondRef.Revision == nil ||
		*firstRef.Revision != "input-r1" || *secondRef.Revision != "input-r2" {
		t.Fatalf("fresh retry contexts = first:%+v second:%+v", firstRef, secondRef)
	}
	if got := []runstore.StageTransitionAction{
		harness.persistence.decisions[0].Action,
		harness.persistence.decisions[1].Action,
		harness.persistence.decisions[2].Action,
	}; !reflect.DeepEqual(got, []runstore.StageTransitionAction{
		runstore.StageTransitionRetry, runstore.StageTransitionRetry, runstore.StageTransitionSucceed,
	}) {
		t.Fatalf("retry decisions = %v", got)
	}
}

func TestSchedulerRetryNonRetryableFailureExecutesThenImmediately(t *testing.T) {
	harness := newSchedulerHarness(t)
	configureRetryWorkflow(t, harness, 3)
	harness.planners.runErrors = []error{
		planner.NewError("permanent", "permanent failure", false, nil),
	}

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	if harness.store.run.State != runstore.RunFailed || len(harness.store.stages) != 1 ||
		len(harness.persistence.decisions) != 1 ||
		harness.persistence.decisions[0].Action != runstore.StageTransitionFail {
		t.Fatalf("non-retryable progression = run:%s stages:%d decisions:%+v",
			harness.store.run.State, len(harness.store.stages), harness.persistence.decisions)
	}
}

func TestSchedulerRetryExhaustionFailsAfterMaximumAttempts(t *testing.T) {
	harness := newSchedulerHarness(t)
	configureRetryWorkflow(t, harness, 3)
	failure := planner.NewError("still_unavailable", "transient failure persisted", true, nil)
	harness.planners.runErrors = []error{failure, failure, failure}

	for iteration := 0; iteration < 3; iteration++ {
		worked, err := harness.scheduler.RunOnce(context.Background())
		if err != nil || !worked {
			t.Fatalf("RunOnce %d = (%v, %v)", iteration+1, worked, err)
		}
	}
	if harness.store.run.State != runstore.RunFailed || len(harness.store.stages) != 3 ||
		len(harness.persistence.decisions) != 3 ||
		harness.persistence.decisions[2].Action != runstore.StageTransitionFail {
		t.Fatalf("exhausted retry = run:%s stages:%d decisions:%+v",
			harness.store.run.State, len(harness.store.stages), harness.persistence.decisions)
	}
}

func TestSchedulerRetryCancellationPreventsNewAttempt(t *testing.T) {
	harness := newSchedulerHarness(t)
	configureRetryWorkflow(t, harness, 3)
	harness.planners.runErrors = []error{
		planner.NewError("transient", "retry would otherwise be eligible", true, nil),
	}
	harness.planners.onRun = func() {
		harness.persistence.stageStates = append(harness.persistence.stageStates, runstore.StageRunning)
		harness.requestCancellation("cancel before retry decision")
	}

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	if harness.store.run.State != runstore.RunCancelled || len(harness.store.stages) != 1 ||
		len(harness.persistence.decisions) != 0 {
		t.Fatalf("cancel/retry race = run:%s stages:%d decisions:%+v",
			harness.store.run.State, len(harness.store.stages), harness.persistence.decisions)
	}
}

func TestSchedulerEscalatesNonRetryableFailureWithPinnedConfigurationAfterRestart(t *testing.T) {
	harness := newSchedulerHarness(t)
	strongPolicy := configureEscalationWorkflow(t, harness, "failed", 1)
	harness.planners.results = []contracts.StageContentResult{
		failedStageResult("permanent", false),
		harness.planners.result,
	}

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("first RunOnce = (%v, %v)", worked, err)
	}
	if harness.store.run.State != runstore.RunRunning || len(harness.store.stages) != 2 ||
		len(harness.persistence.decisions) != 1 {
		t.Fatalf("escalation commit = run:%s stages:%+v decisions:%+v",
			harness.store.run.State, harness.store.stages, harness.persistence.decisions)
	}
	base, escalated := harness.store.stages[0], harness.store.stages[1]
	if base.State != runstore.StageFailed ||
		escalated.ExecutionConfigVariant != runstore.StageExecutionConfigFailedEscalation ||
		escalated.EscalationOrdinal == nil || *escalated.EscalationOrdinal != 1 ||
		escalated.PreviousExecutionID == nil || *escalated.PreviousExecutionID != base.StageExecutionID {
		t.Fatalf("escalated StageExecution identity = base:%+v escalated:%+v", base, escalated)
	}
	decision := harness.persistence.decisions[0]
	if decision.Action != runstore.StageTransitionEscalate || decision.EscalationOrdinal == nil ||
		*decision.EscalationOrdinal != 1 || decision.EscalationExhausted {
		t.Fatalf("escalation decision = %+v", decision)
	}

	restarted, err := New(
		harness.store, harness.persistence, harness.artifacts, harness.allocator,
		harness.workers, harness.planners, harness.scheduler.options,
	)
	if err != nil {
		t.Fatal(err)
	}
	worked, err = restarted.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("restarted RunOnce = (%v, %v)", worked, err)
	}
	if harness.store.run.State != runstore.RunSucceeded || len(harness.store.stages) != 2 ||
		len(harness.persistence.decisions) != 2 {
		t.Fatalf("recovered escalation = run:%s stages:%+v decisions:%+v",
			harness.store.run.State, harness.store.stages, harness.persistence.decisions)
	}
	if len(harness.workers.preparedSettings) != 2 ||
		harness.workers.preparedSettings[0]["builder"].ModelPolicy.Ref == strongPolicy.Ref ||
		harness.workers.preparedSettings[1]["builder"].ModelPolicy.Ref != strongPolicy.Ref {
		t.Fatalf("effective Worker policies = %+v", harness.workers.preparedSettings)
	}
}

func TestSchedulerEscalationExhaustionAppliesThenExactlyOnce(t *testing.T) {
	harness := newSchedulerHarness(t)
	configureEscalationWorkflow(t, harness, "failed", 1)
	harness.planners.results = []contracts.StageContentResult{
		failedStageResult("permanent_one", false),
		failedStageResult("permanent_two", false),
	}

	for iteration := 0; iteration < 2; iteration++ {
		worked, err := harness.scheduler.RunOnce(context.Background())
		if err != nil || !worked {
			t.Fatalf("RunOnce %d = (%v, %v)", iteration+1, worked, err)
		}
	}
	if harness.store.run.State != runstore.RunFailed || len(harness.store.stages) != 2 ||
		len(harness.persistence.decisions) != 2 {
		t.Fatalf("exhausted escalation = run:%s stages:%+v decisions:%+v",
			harness.store.run.State, harness.store.stages, harness.persistence.decisions)
	}
	first, exhausted := harness.persistence.decisions[0], harness.persistence.decisions[1]
	if first.Action != runstore.StageTransitionEscalate ||
		exhausted.Action != runstore.StageTransitionFail || !exhausted.EscalationExhausted ||
		exhausted.EscalationOrdinal == nil || *exhausted.EscalationOrdinal != 1 ||
		exhausted.TargetExecutionID != nil {
		t.Fatalf("bounded escalation decisions = first:%+v exhausted:%+v", first, exhausted)
	}

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || worked || len(harness.store.stages) != 2 || len(harness.persistence.decisions) != 2 {
		t.Fatalf("terminal re-run = (%v, %v), stages:%d decisions:%d",
			worked, err, len(harness.store.stages), len(harness.persistence.decisions))
	}
}

func TestSchedulerEscalatesNonRetryableInterruption(t *testing.T) {
	harness := newSchedulerHarness(t)
	configureEscalationWorkflow(t, harness, "interrupted", 1)
	harness.planners.runErrors = []error{
		planner.NewError("planner_rejected", "Planner cannot continue", false, nil),
	}

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	if harness.store.run.State != runstore.RunRunning || len(harness.store.stages) != 2 ||
		harness.store.stages[0].State != runstore.StageInterrupted ||
		harness.store.stages[1].ExecutionConfigVariant != runstore.StageExecutionConfigInterruptedEscalation ||
		harness.store.stages[1].EscalationOrdinal == nil || *harness.store.stages[1].EscalationOrdinal != 1 ||
		len(harness.persistence.decisions) != 1 ||
		harness.persistence.decisions[0].Action != runstore.StageTransitionEscalate {
		t.Fatalf("interrupted escalation = run:%s stages:%+v decisions:%+v",
			harness.store.run.State, harness.store.stages, harness.persistence.decisions)
	}
}

func TestSchedulerCancellationWinsOverEscalation(t *testing.T) {
	harness := newSchedulerHarness(t)
	configureEscalationWorkflow(t, harness, "failed", 1)
	harness.planners.result = failedStageResult("would_escalate", false)
	harness.planners.onRun = func() {
		harness.persistence.stageStates = append(harness.persistence.stageStates, runstore.StageRunning)
		harness.requestCancellation("cancel before escalation decision")
		harness.scheduler.Cancel(harness.store.run.RunID)
	}

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	if harness.store.run.State != runstore.RunCancelled || len(harness.store.stages) != 1 ||
		len(harness.persistence.decisions) != 0 {
		t.Fatalf("cancel/escalation race = run:%s stages:%+v decisions:%+v",
			harness.store.run.State, harness.store.stages, harness.persistence.decisions)
	}
}

func TestSchedulerExecutesMultiStageWorkflowSerially(t *testing.T) {
	harness := newSchedulerHarness(t)
	configureMultiStageWorkflow(t, harness)
	harness.artifacts.current["run-1/builder/copied"] = "result-r1"

	for iteration := 0; iteration < 2; iteration++ {
		worked, err := harness.scheduler.RunOnce(context.Background())
		if err != nil || !worked {
			t.Fatalf("RunOnce %d = (%v, %v)", iteration+1, worked, err)
		}
	}
	if harness.store.run.State != runstore.RunSucceeded || len(harness.store.stages) != 2 ||
		harness.store.stages[0].StageName != "build" || harness.store.stages[1].StageName != "review" ||
		harness.store.stages[0].State != runstore.StageSucceeded ||
		harness.store.stages[1].State != runstore.StageSucceeded {
		t.Fatalf("multi-Stage lifecycle = run:%s stages:%+v", harness.store.run.State, harness.store.stages)
	}
	draft := harness.store.stages[1].StageContext.Artifacts["draft"].Artifact
	if draft == nil || draft.Revision == nil || draft.Namespace != "builder" || *draft.Revision != "result-r1" {
		t.Fatalf("later Stage context = %+v", draft)
	}
	if got := []runstore.StageTransitionAction{
		harness.persistence.decisions[0].Action,
		harness.persistence.decisions[1].Action,
	}; !reflect.DeepEqual(got, []runstore.StageTransitionAction{
		runstore.StageTransitionNext, runstore.StageTransitionSucceed,
	}) {
		t.Fatalf("multi-Stage decisions = %v", got)
	}
}

func configureRetryWorkflow(t *testing.T, harness *schedulerHarness, maxAttempts int) {
	t.Helper()
	stage := harness.workflow.Stages[harness.workflow.EntryStage]
	retry := workflowconfig.TransitionAction{
		Kind: workflowconfig.TransitionRetry,
		Retry: &workflowconfig.RetryTransition{
			MaxAttempts: maxAttempts,
			Then:        workflowconfig.TransitionAction{Kind: workflowconfig.TransitionFail},
		},
	}
	stage.On.Failed = retry
	stage.On.Interrupted = retry
	harness.workflow.Stages[harness.workflow.EntryStage] = stage
	installHarnessWorkflow(t, harness)
}

func configureEscalationWorkflow(
	t *testing.T,
	harness *schedulerHarness,
	outcome string,
	maxAttempts int,
) contracts.ResolvedModelPolicy {
	t.Helper()
	snapshot, err := workflowconfig.Load("../../configs", workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	strongPolicy, err := snapshot.ModelPolicy("strong_domain_worker@1")
	if err != nil {
		t.Fatal(err)
	}
	stageName := harness.workflow.EntryStage
	stage := harness.workflow.Stages[stageName]
	effective := stage.ExecutionConfig
	effective.Agents = make(map[string]workflowconfig.ResolvedConsumerExecutionConfig, len(stage.ExecutionConfig.Agents))
	for logicalName, selection := range stage.ExecutionConfig.Agents {
		effective.Agents[logicalName] = selection
	}
	selection := effective.Agents["builder"]
	selection.ModelPolicy = strongPolicy
	selection.Origins.ModelPolicy = fmt.Sprintf(
		"workflow.stages.%s.on.%s.escalate.executionConfig.agents.builder", stageName, outcome,
	)
	effective.Agents["builder"] = selection
	policyOverride := strongPolicy
	escalate := workflowconfig.TransitionAction{
		Kind: workflowconfig.TransitionEscalate,
		Escalate: &workflowconfig.EscalateTransition{
			MaxAttempts: maxAttempts,
			ExecutionConfig: workflowconfig.ResolvedEscalationExecutionConfig{
				Override: workflowconfig.ResolvedStageExecutionConfigOverride{
					Agents: map[string]workflowconfig.ResolvedExecutionSelectionOverride{
						"builder": {ModelPolicy: &policyOverride},
					},
				},
				Effective: effective,
			},
			Then: workflowconfig.TransitionAction{Kind: workflowconfig.TransitionFail},
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
	harness.workflow.Stages[stageName] = stage
	installHarnessWorkflow(t, harness)
	return strongPolicy
}

func failedStageResult(code string, retryable bool) contracts.StageContentResult {
	return contracts.StageContentResult{
		APIVersion: contracts.APIVersion,
		Outcome:    contracts.StageFailed,
		Summary:    "Stage failed",
		Artifacts:  map[string]contracts.ArtifactRef{},
		Error: &contracts.TerminationError{
			Code: code, Message: "Stage could not produce a result", Retryable: retryable,
		},
	}
}

func configureMultiStageWorkflow(t *testing.T, harness *schedulerHarness) {
	t.Helper()
	encodedStage, err := json.Marshal(harness.workflow.Stages[harness.workflow.EntryStage])
	if err != nil {
		t.Fatal(err)
	}
	var build workflowconfig.ResolvedStage
	var review workflowconfig.ResolvedStage
	if err := json.Unmarshal(encodedStage, &build); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(encodedStage, &review); err != nil {
		t.Fatal(err)
	}
	build.WorkflowOutputs = map[string]string{}
	build.On.Succeeded = workflowconfig.TransitionAction{Kind: workflowconfig.TransitionNext, NextStage: "review"}
	review.Context.Artifacts = map[string]workflowconfig.ContextArtifact{
		"draft": {Namespace: "builder", Name: "copied", Required: true},
	}
	harness.workflow.EntryStage = "build"
	harness.workflow.Stages = map[string]workflowconfig.ResolvedStage{"build": build, "review": review}
	installHarnessWorkflow(t, harness)
}

func installHarnessWorkflow(t *testing.T, harness *schedulerHarness) {
	t.Helper()
	if err := workflowconfig.ValidateWorkflowGraph(harness.workflow); err != nil {
		t.Fatalf("test Workflow graph: %v", err)
	}
	encoded, err := json.Marshal(harness.workflow)
	if err != nil {
		t.Fatal(err)
	}
	harness.store.run.WorkflowSnapshot = encoded
	harness.allocator.workflow = harness.workflow
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
		ExecutionConfigVariant: runstore.StageExecutionConfigBase,
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

func (h *schedulerHarness) requestCancellation(reason string) {
	requestedBy := "user-1"
	h.store.run.State = runstore.RunCancelling
	h.store.run.StateReason = runstore.Reason{Code: runstore.CancellationUserRequested}
	h.store.run.Cancellation = &runstore.WorkflowRunCancellation{
		Code: runstore.CancellationUserRequested, RequestedAt: h.clock.now,
		RequestedBy: &requestedBy, Reason: &reason,
	}
	version := contracts.APIVersion
	h.store.run.CancellationSchemaVersion = &version
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
	reports     []runstore.RecordStageExecutionReportParams
	reportError error
	claimID     string
	claimCalls  int
	events      *eventRecorder
}

func (s *memorySchedulerStore) ClaimRunnableRun(
	_ context.Context, claimID string, duration time.Duration,
) (runstore.WorkflowRun, error) {
	s.claimCalls++
	if s.run.State != runstore.RunRunning && s.run.State != runstore.RunCancelling ||
		s.claimID != "" || duration <= 0 {
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
			if allocation.StageExecutionID == execution.StageExecutionID &&
				allocation.ReleaseCompletedAt == nil {
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

func (s *memorySchedulerStore) MarkStageAllocationReleaseAttempt(
	_ context.Context, allocationID string,
) error {
	for index := range s.allocations {
		if s.allocations[index].AllocationID == allocationID {
			now := time.Now().UTC()
			s.allocations[index].ReleaseAttemptedAt = &now
			return nil
		}
	}
	return runstore.ErrNotFound
}

func (s *memorySchedulerStore) MarkStageAllocationReleased(
	_ context.Context, allocationID string,
) error {
	for index := range s.allocations {
		if s.allocations[index].AllocationID == allocationID {
			now := time.Now().UTC()
			s.allocations[index].ReleaseAttemptedAt = &now
			s.allocations[index].ReleaseCompletedAt = &now
			return nil
		}
	}
	return runstore.ErrNotFound
}

func (s *memorySchedulerStore) RecordStageExecutionReport(
	_ context.Context, report runstore.RecordStageExecutionReportParams,
) error {
	if s.reportError != nil {
		return s.reportError
	}
	s.reports = append(s.reports, report)
	s.events.add("record_report")
	return nil
}

func (s *memorySchedulerStore) RecordPlannerExecutionReport(
	_ context.Context, _ runstore.RecordPlannerExecutionReportParams,
) error {
	return nil
}

func (s *memorySchedulerStore) RebuildStageMetrics(
	_ context.Context, _, _ string,
) error {
	return nil
}

func (s *memorySchedulerStore) CleanupExpiredTelemetry(
	_ context.Context, _ time.Time, _ int,
) (int64, error) {
	return 0, nil
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
	decisions   []runstore.StageTransitionDecision
}

func (p *memoryAtomicPersistence) CreateStageWithContext(
	_ context.Context, params runstore.CreateStageExecutionParams, _ []ContextPin,
) (runstore.StageExecution, error) {
	if p.store.run.State != runstore.RunRunning {
		return runstore.StageExecution{}, runstore.ErrConflict
	}
	execution := runstore.StageExecution{
		StageExecutionID: params.StageExecutionID, RunID: params.RunID, StageName: params.StageName,
		Attempt: params.Attempt, PreviousExecutionID: params.PreviousExecutionID,
		ExecutionConfigVariant: params.ExecutionConfigVariant,
		EscalationOrdinal:      params.EscalationOrdinal,
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
	if p.store.run.State != runstore.RunRunning {
		return runstore.ErrConflict
	}
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

func (p *memoryAtomicPersistence) CommitResultProgression(
	_ context.Context, value ResultProgression,
) error {
	if err := validateResultProgression(value); err != nil {
		return err
	}
	if p.store.run.State != runstore.RunRunning {
		return runstore.ErrConflict
	}
	for index := range p.store.stages {
		if p.store.stages[index].StageExecutionID != value.StageExecutionID ||
			p.store.stages[index].State != runstore.StageFinalizing {
			continue
		}
		result := cloneStageResult(value.Result)
		p.store.stages[index].State = runstore.StageExecutionState(result.Outcome)
		p.store.stages[index].AcceptedResult = &result
		p.store.stages[index].AcceptedResultSchemaVersion = stringPointer(contracts.APIVersion)
		p.stageStates = append(p.stageStates, p.store.stages[index].State)
		if result.Outcome == contracts.StageSucceeded {
			for output, resultName := range value.WorkflowOutputs {
				if source, ok := result.Artifacts[resultName]; ok {
					revision := "output-" + *source.Revision
					p.outputs[output] = exactRef("outputs", output, revision)
				}
			}
		}
		p.commitProgression(value.Progression)
		p.events.add("accept")
		return nil
	}
	return runstore.ErrConflict
}

func (p *memoryAtomicPersistence) CommitTerminationProgression(
	_ context.Context,
	value TerminationProgression,
) error {
	if err := validateTerminationProgression(value); err != nil {
		return err
	}
	if value.RunID != p.store.run.RunID || p.store.run.State != runstore.RunRunning {
		return runstore.ErrConflict
	}
	for index := range p.store.stages {
		if p.store.stages[index].StageExecutionID == value.StageExecutionID &&
			p.store.stages[index].State == runstore.StageAborting {
			terminal := runstore.StageExecutionState(p.store.stages[index].Termination.Outcome)
			p.store.stages[index].State = terminal
			p.stageStates = append(p.stageStates, runstore.StageAborting, terminal)
			p.commitProgression(value.Progression)
			p.events.add("commit_termination")
			return nil
		}
	}
	return runstore.ErrConflict
}

func (p *memoryAtomicPersistence) commitProgression(value StageProgression) {
	decision := runstore.StageTransitionDecision{
		SourceExecutionID:   value.Decision.SourceExecutionID,
		RunID:               value.Decision.RunID,
		Action:              value.Decision.Action,
		TargetStageName:     value.Decision.TargetStageName,
		TargetExecutionID:   value.Decision.TargetExecutionID,
		EscalationOrdinal:   value.Decision.EscalationOrdinal,
		EscalationExhausted: value.Decision.EscalationExhausted,
	}
	p.decisions = append(p.decisions, decision)
	if value.NextStage != nil {
		params := value.NextStage.Params
		p.store.stages = append(p.store.stages, runstore.StageExecution{
			StageExecutionID: params.StageExecutionID, RunID: params.RunID,
			StageName: params.StageName, Attempt: params.Attempt,
			PreviousExecutionID:       params.PreviousExecutionID,
			ExecutionConfigVariant:    params.ExecutionConfigVariant,
			EscalationOrdinal:         params.EscalationOrdinal,
			StageSpecSchemaVersion:    params.StageSpecSchemaVersion,
			StageSpecSnapshot:         params.StageSpecSnapshot,
			StageContextSchemaVersion: params.StageContextSchemaVersion,
			StageContext:              params.StageContext, State: runstore.StagePreparing,
		})
		p.stageStates = append(p.stageStates, runstore.StagePreparing)
	}
	if value.TerminalRunState != "" {
		p.store.run.State = value.TerminalRunState
		p.store.run.StateReason = value.RunReason
		p.store.claimID = ""
	}
}

func (p *memoryAtomicPersistence) AcceptResultDuringCancellation(
	_ context.Context,
	runID string,
	stageExecutionID string,
	result contracts.StageContentResult,
) error {
	if runID != p.store.run.RunID || p.store.run.State != runstore.RunCancelling {
		return runstore.ErrConflict
	}
	for index := range p.store.stages {
		if p.store.stages[index].StageExecutionID == stageExecutionID &&
			p.store.stages[index].State == runstore.StageFinalizing {
			accepted := cloneStageResult(result)
			p.store.stages[index].State = runstore.StageExecutionState(result.Outcome)
			p.store.stages[index].AcceptedResult = &accepted
			p.store.stages[index].AcceptedResultSchemaVersion = stringPointer(contracts.APIVersion)
			p.store.run.State = runstore.RunCancelled
			p.store.claimID = ""
			p.events.add("accept_cancelled")
			return nil
		}
	}
	return runstore.ErrConflict
}

func (p *memoryAtomicPersistence) CommitTerminationAndFinishRun(
	_ context.Context,
	runID string,
	stageExecutionID string,
	expectedRunState runstore.WorkflowRunState,
	nextRunState runstore.WorkflowRunState,
	_ runstore.Reason,
) error {
	if runID != p.store.run.RunID || p.store.run.State != expectedRunState {
		return runstore.ErrConflict
	}
	for index := range p.store.stages {
		if p.store.stages[index].StageExecutionID == stageExecutionID &&
			p.store.stages[index].State == runstore.StageAborting {
			terminal := runstore.StageExecutionState(p.store.stages[index].Termination.Outcome)
			p.store.stages[index].State = terminal
			p.stageStates = append(p.stageStates, runstore.StageAborting, terminal)
			p.store.run.State = nextRunState
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
	workflow           workflowconfig.ResolvedWorkflow
	clock              staticClock
	events             *eventRecorder
	grants             map[string]controlplane.AllocationGrant
	cached             []controlplane.Reservation
	fenced             map[string]bool
	reserveCalls       int
	allocationSequence int
	reserveError       error
	losses             []controlplane.AllocationLoss
}

func (a *memoryAllocator) ReserveAll(request controlplane.ReservationRequest) ([]controlplane.Reservation, error) {
	a.reserveCalls++
	a.events.add("reserve")
	if a.reserveError != nil {
		return nil, a.reserveError
	}
	if len(a.cached) == 0 || a.cached[0].Grant.StageExecutionID != request.StageExecutionID {
		a.cached = []controlplane.Reservation{a.reservationForRequest(request)}
		for _, reservation := range a.cached {
			a.grants[reservation.Grant.AllocationID] = reservation.Grant
		}
	}
	return append([]controlplane.Reservation(nil), a.cached...), nil
}

func (a *memoryAllocator) reservationForRequest(request controlplane.ReservationRequest) controlplane.Reservation {
	a.allocationSequence++
	binding := request.Bindings[0]
	reservation := controlplane.Reservation{
		Grant: controlplane.AllocationGrant{
			AllocationID:   fmt.Sprintf("allocation-%d", a.allocationSequence),
			RuntimeAgentID: strings.Repeat("1", 64), RuntimeInstanceID: "runtime-1",
			RunID: request.RunID, StageExecutionID: request.StageExecutionID,
			LogicalAgentName: binding.LogicalAgentName, Namespace: binding.Namespace,
			ReadPolicy: controlplane.ReadCurrentRun, WritePolicy: controlplane.WriteInputsAndIntermediates,
		},
		ControlURL: "https://runtime.test", A2AURL: "https://runtime.test",
		AgentTemplate: binding.AgentTemplate, ExecutionConfig: binding.ExecutionConfig,
		RuntimeAgentLabelRevision: 1, LeaseExpiresAt: a.clock.now.Add(time.Minute),
	}
	if binding.RuntimeSelection != nil && request.RuntimeConfig != nil {
		resolved, err := fallbackResolvedWorkerConfig(*binding.RuntimeSelection)
		if err == nil {
			reservation.ResolvedRuntimeConfig = &resolved
		}
	}
	return reservation
}

func (a *memoryAllocator) reservation(stageExecutionID string) controlplane.Reservation {
	binding := a.workflow.Stages[a.workflow.EntryStage].Agents["builder"]
	return a.reservationForRequest(controlplane.ReservationRequest{
		RunID: "run-1", StageExecutionID: stageExecutionID,
		Bindings: []controlplane.BindingRequirement{{
			LogicalAgentName: "builder", Namespace: binding.Namespace, AgentTemplate: binding.Template,
			ExecutionConfig: bindingRequirements(a.workflow.Stages[a.workflow.EntryStage])[0].ExecutionConfig,
		}},
	})
}

func (a *memoryAllocator) GetGrant(allocationID string) (controlplane.AllocationGrant, error) {
	grant, ok := a.grants[allocationID]
	if !ok {
		return controlplane.AllocationGrant{}, controlplane.ErrAllocationNotFound
	}
	return grant, nil
}

func (a *memoryAllocator) GetReservation(allocationID string) (controlplane.Reservation, error) {
	for _, reservation := range a.cached {
		if reservation.Grant.AllocationID == allocationID {
			if _, ok := a.grants[allocationID]; ok {
				return reservation, nil
			}
		}
	}
	return controlplane.Reservation{}, controlplane.ErrAllocationNotFound
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
	if len(a.cached) > 0 && a.cached[0].Grant.AllocationID == allocationID {
		a.cached = nil
	}
	return nil
}

func (a *memoryAllocator) PollAllocationLosses() []controlplane.AllocationLoss {
	result := append([]controlplane.AllocationLoss(nil), a.losses...)
	a.losses = nil
	return result
}

type memoryWorkers struct {
	allocator        *memoryAllocator
	workflow         workflowconfig.ResolvedWorkflow
	clock            staticClock
	events           *eventRecorder
	prepareCalls     int
	finalizeCalls    int
	abortCalls       int
	releaseCalls     int
	preparedSettings []map[string]contracts.WorkerExecutionSettingsV2
	prepareError     error
	abortError       error
	releaseError     error
	releaseErrors    map[string]error
	releasedBatches  []int
}

func (w *memoryWorkers) PrepareAll(
	_ context.Context, reservations []controlplane.Reservation,
	settings map[string]contracts.WorkerExecutionSettingsV2,
) (map[string]contracts.WorkerHandle, error) {
	w.prepareCalls++
	w.preparedSettings = append(w.preparedSettings, settings)
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
) (map[string]contracts.AllocationFinalReport, error) {
	w.finalizeCalls++
	for _, reservation := range reservations {
		if !w.allocator.fenced[reservation.Grant.AllocationID] {
			return nil, errors.New("finalize observed an unfenced allocation")
		}
	}
	w.events.add("finalize")
	reports := make(map[string]contracts.AllocationFinalReport, len(reservations))
	for _, reservation := range reservations {
		reports[reservation.Grant.LogicalAgentName] = schedulerTestAllocationReport(
			reservation.Grant.AllocationID, w.clock.now,
		)
	}
	return reports, nil
}

func (w *memoryWorkers) AbortAll(
	_ context.Context, _ []controlplane.Reservation, _ string, _ contracts.TerminationError, _ time.Time,
) (map[string]contracts.AllocationFinalReport, error) {
	w.abortCalls++
	w.events.add("abort")
	return map[string]contracts.AllocationFinalReport{}, w.abortError
}

func schedulerTestAllocationReport(
	allocationID string, finishedAt time.Time,
) contracts.AllocationFinalReport {
	modelCalls := int64(1)
	return contracts.AllocationFinalReport{
		ReportID: "allocation-final-" + allocationID, AllocationID: allocationID,
		StartedAt: finishedAt.Add(-time.Second), FinishedAt: finishedAt,
		Worker: contracts.ExecutionReport{
			ReportID: "worker-" + allocationID, Complete: true,
			Metrics: contracts.ExecutionMetrics{
				ModelCalls: &modelCalls, Tools: map[string]contracts.ToolMetrics{},
			},
			ToolCalls: []contracts.ToolCallRecord{}, Errors: []contracts.ExecutionError{},
		},
		Runtime: contracts.RuntimeReport{Complete: true},
	}
}

func (w *memoryWorkers) ReleaseAll(_ context.Context, reservations []controlplane.Reservation) error {
	w.releaseCalls++
	w.releasedBatches = append(w.releasedBatches, len(reservations))
	w.events.add("release")
	if w.releaseError != nil {
		return w.releaseError
	}
	var failures []error
	for _, reservation := range reservations {
		if err := w.releaseErrors[reservation.Grant.AllocationID]; err != nil {
			failures = append(failures, err)
			continue
		}
		delete(w.allocator.grants, reservation.Grant.AllocationID)
	}
	remaining := w.allocator.cached[:0]
	for _, reservation := range w.allocator.cached {
		if _, live := w.allocator.grants[reservation.Grant.AllocationID]; live {
			remaining = append(remaining, reservation)
		}
	}
	w.allocator.cached = remaining
	return errors.Join(failures...)
}

type memoryPlannerRegistry struct {
	store       *memorySchedulerStore
	result      contracts.StageContentResult
	results     []contracts.StageContentResult
	runErrors   []error
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
		callIndex := r.runCalls - 1
		if callIndex < len(r.runErrors) && r.runErrors[callIndex] != nil {
			return contracts.StageContentResult{}, r.runErrors[callIndex]
		}
		if callIndex < len(r.results) {
			return cloneStageResult(r.results[callIndex]), nil
		}
		return cloneStageResult(r.result), nil
	}), nil
}

type plannerFunc func(context.Context) (contracts.StageContentResult, error)

func (f plannerFunc) Run(ctx context.Context) (contracts.StageContentResult, error) { return f(ctx) }

type credentialResolverFunc func(
	context.Context,
	contracts.LLMCredentialRef,
	contracts.LLMGatewayConfigRef,
) (contracts.SecretString, error)

func (f credentialResolverFunc) ResolveLLMCredential(
	ctx context.Context,
	credential contracts.LLMCredentialRef,
	gateway contracts.LLMGatewayConfigRef,
) (contracts.SecretString, error) {
	return f(ctx, credential, gateway)
}

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
