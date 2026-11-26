package streamline

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/clone"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

const (
	addSubtaskToolName            = "add_subtask"
	listSubtasksToolName          = "list_subtasks"
	executeCurrentSubtaskToolName = "execute_current_subtask"
	finishToolName                = "finish"
)

type workerCallArgs struct {
	SubtaskID string `json:"subtask_id"`
}

type routerWorkerCallArgs struct {
	SubtaskID  string `json:"subtask_id"`
	WorkerName string `json:"worker_name"`
}

type workerCallOutput struct {
	OK     bool                    `json:"ok"`
	Result *contracts.WorkerResult `json:"result,omitempty"`
	Error  *toolFailure            `json:"error,omitempty"`
}

type materialSelection struct {
	Namespace string `json:"namespace"`
	Name      string `json:"name"`
}

type finishArgs struct {
	Outcome   stageresult.StageOutcome     `json:"outcome"`
	Summary   string                       `json:"summary"`
	Artifacts map[string]materialSelection `json:"artifacts"`
	Error     *contracts.TerminationError  `json:"error,omitempty"`
}

type addSubtaskArgs struct {
	Objective    string `json:"objective"`
	Instructions string `json:"instructions"`
}

type listSubtasksArgs struct{}

type plannerPlanOutput struct {
	OK    bool                 `json:"ok"`
	Plan  *planner.PlannerPlan `json:"plan,omitempty"`
	Error *toolFailure         `json:"error,omitempty"`
}

type completionToolOutput struct {
	Accepted bool         `json:"accepted"`
	Error    *toolFailure `json:"error,omitempty"`
}

type toolFailure struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
}

func (p *streamlinePlanner) addSubtask(
	ctx context.Context,
	state *executionState,
	identity planner.SessionIdentity,
	args addSubtaskArgs,
) plannerPlanOutput {
	started := time.Now()
	span := planner.InvocationInstrumentation(p.invocation).StartSpan(
		telemetry.PlannerSpanSubtask,
		telemetry.PlannerSpanAttributes{Operation: "subtask.add", ToolName: addSubtaskToolName},
	)
	safeArguments := map[string]any{
		"objectiveBytes": len(args.Objective), "instructionsBytes": len(args.Instructions),
	}
	before := p.plan.Snapshot()
	plan, planErr := p.plan.AddSubtask(args.Objective, args.Instructions)
	if planErr != nil {
		failure := failureFromPlanError(planErr)
		span.End("rejected", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		state.recordTool(addSubtaskToolName, safeArguments, false, time.Since(started), 0, &failure)
		return plannerPlanOutput{Error: toolFailureFrom(failure)}
	}
	if failure := p.persistPlanTransition(
		ctx, state, identity, before, plan, planner.PlannerEventPlanChanged,
	); failure != nil {
		span.End("failed", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		state.recordTool(addSubtaskToolName, safeArguments, false, time.Since(started), 0, failure)
		return plannerPlanOutput{Error: toolFailureFrom(*failure)}
	}
	if before.CurrentSubtaskID != plan.CurrentSubtaskID {
		if failure := p.persistCurrentChanged(ctx, state, identity, plan); failure != nil {
			span.End("failed", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
			state.recordTool(addSubtaskToolName, safeArguments, false, time.Since(started), 0, failure)
			return plannerPlanOutput{Error: toolFailureFrom(*failure)}
		}
	}
	encoded, _ := json.Marshal(plan)
	state.recordTool(addSubtaskToolName, safeArguments, true, time.Since(started), len(encoded), nil)
	span.End("succeeded", telemetry.PlannerSpanAttributes{
		SubtaskID: plan.CurrentSubtaskID, PlanRevision: plan.Revision,
	})
	return plannerPlanOutput{OK: true, Plan: &plan}
}

func (p *streamlinePlanner) listSubtasks(
	_ context.Context, state *executionState,
) plannerPlanOutput {
	started := time.Now()
	plan := p.plan.Snapshot()
	span := planner.InvocationInstrumentation(p.invocation).StartSpan(
		telemetry.PlannerSpanSubtask,
		telemetry.PlannerSpanAttributes{
			Operation: "subtask.list", ToolName: listSubtasksToolName,
			SubtaskID: plan.CurrentSubtaskID, PlanRevision: plan.Revision,
		},
	)
	encoded, _ := json.Marshal(plan)
	state.recordTool(listSubtasksToolName, map[string]any{}, true, time.Since(started), len(encoded), nil)
	span.End("succeeded", telemetry.PlannerSpanAttributes{})
	return plannerPlanOutput{OK: true, Plan: &plan}
}

// callWorker returns whether the Worker budget stopped execution. The SDK
// adapter owns the corresponding summarization action.
func (p *streamlinePlanner) callWorker(
	ctx context.Context,
	state *executionState,
	identity planner.SessionIdentity,
	binding workerBinding,
	subtaskID string,
) (workerCallOutput, bool) {
	started := time.Now()
	safeArguments := map[string]any{
		"binding": binding.logicalName, "subtaskId": safeSubtaskID(subtaskID),
	}
	beforeClaim := p.plan.Snapshot()
	claim, planErr := p.plan.ClaimCurrentSubtask(subtaskID, binding.logicalName)
	if planErr != nil {
		failure := failureFromPlanError(planErr)
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, &failure)
		return workerFailure(failure), false
	}
	claimedPlan := p.plan.Snapshot()
	// Every failure after the claim settles that dispatch before exposing its
	// tool result. A persistence failure keeps precedence over the Worker error.
	reject := func(failure planner.Failure) workerCallOutput {
		if persisted := p.failDispatch(ctx, state, identity, claim.CallID); persisted != nil {
			failure = *persisted
		}
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, &failure)
		return workerFailure(failure)
	}
	if failure := p.persistDispatchSelected(
		ctx, state, identity, beforeClaim.Revision, claim, binding.logicalName,
	); failure != nil {
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, failure)
		return workerFailure(*failure), false
	}
	if failure := p.persistPlanTransition(
		ctx, state, identity, beforeClaim, claimedPlan, planner.PlannerEventDispatchStarted,
	); failure != nil {
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, failure)
		return workerFailure(*failure), false
	}
	request, failure := p.workerRequest(ctx, claim.Subtask)
	if failure != nil {
		return reject(*failure), false
	}
	deadline := p.deadline
	if !deadline.After(time.Now()) {
		failure := &planner.Failure{
			Code: "worker_deadline_exceeded", Message: "Worker invocation deadline expired", Retryable: true,
		}
		return reject(*failure), false
	}
	if limit := state.reserveWorkerCall(); limit != nil {
		return reject(limit.Failure), true
	}
	workerContext, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	workerSpan := planner.InvocationInstrumentation(p.invocation).StartSpan(
		telemetry.PlannerSpanWorker,
		telemetry.PlannerSpanAttributes{
			Operation: "a2a.invoke", ToolName: executeCurrentSubtaskToolName,
			WorkerName: binding.logicalName, SubtaskID: safeSubtaskID(subtaskID),
		},
	)
	completion, err := p.invoker.Invoke(
		workerContext, binding.logicalName, binding.handle.Clone(), request,
	)
	if err != nil {
		failure := planner.FailureFrom(err)
		workerSpan.End("failed", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		return reject(failure), false
	}
	if validation := planner.ValidateWorkerCompletion(completion, claim.Subtask.ID); validation != nil {
		failure := validation.Failure
		workerSpan.End("rejected", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		return reject(failure), false
	}
	// Live State is advisory. Failure to retain its hidden correlation must not
	// rewrite an otherwise valid WorkerCompletion or dispatch outcome.
	_ = p.stateViews.RecordCompletion(binding.logicalName, claim.Subtask.ID, completion)
	if completion.Failure != nil {
		failure := planner.Failure{
			Code: completion.Failure.Code, Message: completion.Failure.Message,
			Retryable: completion.Failure.Retryable,
		}
		workerSpan.End("failed", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		return reject(failure), false
	}
	result := planner.CloneWorkerResult(*completion.Result)
	if validation := p.validateWorkerResult(workerContext, result); validation != nil {
		failure := validation.Failure
		workerSpan.End("rejected", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		return reject(failure), false
	}
	workerSpan.End("succeeded", telemetry.PlannerSpanAttributes{})
	beforeCompletion := p.plan.Snapshot()
	completedPlan, planErr := p.plan.CompleteDispatch(claim.CallID, stageresult.StageSucceeded)
	if planErr != nil {
		failure := failureFromPlanError(planErr)
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, &failure)
		return workerFailure(failure), false
	}
	if failure := p.persistPlanTransition(
		ctx, state, identity, beforeCompletion, completedPlan, planner.PlannerEventDispatchCompleted,
	); failure != nil {
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, failure)
		return workerFailure(*failure), false
	}
	if beforeCompletion.CurrentSubtaskID != completedPlan.CurrentSubtaskID {
		if failure := p.persistCurrentChanged(ctx, state, identity, completedPlan); failure != nil {
			state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, failure)
			return workerFailure(*failure), false
		}
	}
	if p.recoveryArtifacts == nil {
		p.recoveryArtifacts = map[string]bool{}
	}
	for name := range p.resultContract {
		if _, ok := result.Artifacts[name]; ok {
			p.recoveryArtifacts[name] = true
		}
	}
	if p.compaction != nil {
		// The exact result stays readable after compaction; an index over
		// its retained-data ceiling stops the Planner before its next call.
		if failure := p.compaction.recordResult(binding.logicalName, claim.Subtask.ID, result); failure != nil {
			state.setExternalFailure(failure)
		}
	}
	if !state.selectWorkerArtifacts(binding.logicalName, result.Artifacts) {
		failure := planner.NewError("finish_rejected", "Selected result material limit exceeded", false, nil)
		state.setExternalFailure(failure)
		return workerFailure(failure.Failure), false
	}
	encoded, _ := json.Marshal(result)
	state.recordTool(executeCurrentSubtaskToolName, safeArguments, true, time.Since(started), len(encoded), nil)
	return workerCallOutput{OK: true, Result: &result}, false
}

func (p *streamlinePlanner) routeWorker(
	ctx context.Context,
	state *executionState,
	identity planner.SessionIdentity,
	args routerWorkerCallArgs,
) (workerCallOutput, bool) {
	for _, binding := range p.workers {
		if args.WorkerName == binding.logicalName {
			return p.callWorker(ctx, state, identity, binding, args.SubtaskID)
		}
	}
	started := time.Now()
	failure := planner.Failure{
		Code: "planner_worker_unknown", Message: "worker_name does not identify an available logical Worker",
		Retryable: false,
	}
	state.recordTool(executeCurrentSubtaskToolName, map[string]any{
		"binding": "unknown", "subtaskId": safeSubtaskID(args.SubtaskID),
	}, false, time.Since(started), 0, &failure)
	return workerFailure(failure), false
}

func routerExecuteSchema(workers []workerBinding) (*jsonschema.Schema, error) {
	schema, err := jsonschema.For[routerWorkerCallArgs](nil)
	if err != nil {
		return nil, err
	}
	workerName, ok := schema.Properties["worker_name"]
	if !ok {
		return nil, fmt.Errorf("worker_name schema is absent")
	}
	workerName.Enum = make([]any, 0, len(workers))
	for _, binding := range workers {
		workerName.Enum = append(workerName.Enum, binding.logicalName)
	}
	return schema, nil
}

func (p *streamlinePlanner) workerRequest(
	ctx context.Context, subtask planner.PlannerSubtask,
) (contracts.StageContentRequest, *planner.Failure) {
	request := p.request.Clone()
	request.Deadline = p.deadline
	request.SubtaskID = subtask.ID
	request.Objective = subtask.Objective
	request.Instructions = subtask.Instructions
	for _, name := range slices.Sorted(maps.Keys(request.Artifacts)) {
		ref := request.Artifacts[name]
		if err := ref.ValidateExact(); err != nil {
			return contracts.StageContentRequest{}, safeToolFailure(
				"worker_request_invalid", "Worker request contains an invalid exact artifact reference", false,
			)
		}
		if _, err := p.inspector.Inspect(ctx, p.invocation.RunID, ref.Clone()); err != nil {
			if artifacts.IsExactReferenceRejection(err) {
				return contracts.StageContentRequest{}, safeToolFailure(
					"worker_request_invalid",
					"Worker request artifact does not identify an exact revision in the RunScope", false,
				)
			}
			return contracts.StageContentRequest{}, safeToolFailure(
				"worker_artifact_unavailable", "Worker request artifact could not be verified", true,
			)
		}
	}
	if err := request.Validate(); err != nil {
		return contracts.StageContentRequest{}, safeToolFailure(
			"worker_request_invalid",
			"Worker objective, instructions, or structured context is invalid or exceeds its size bound", false,
		)
	}
	return request, nil
}

func (p *streamlinePlanner) validateWorkerResult(
	ctx context.Context, result contracts.WorkerResult,
) *planner.Error {
	size, err := contracts.ResultJSONSize(result)
	if err != nil || size > contracts.MaxWorkerResultPayloadBytes ||
		len(result.Result) > contracts.MaxWorkerResultBytes || len(result.Artifacts) > contracts.MaxWorkerResultArtifacts {
		return planner.NewError(
			"invalid_worker_result", "Worker returned an oversized WorkerResult", false, err,
		)
	}
	for _, name := range slices.Sorted(maps.Keys(result.Artifacts)) {
		if _, err := p.inspector.Inspect(ctx, p.invocation.RunID, result.Artifacts[name].Clone()); err != nil {
			if artifacts.IsExactReferenceRejection(err) {
				return planner.NewError(
					"invalid_worker_result",
					"Worker result artifact does not identify an exact revision in the RunScope", false, err,
				)
			}
			return planner.NewError(
				"result_artifact_unavailable", "Worker result artifact could not be verified", true, err,
			)
		}
	}
	return nil
}

func (p *streamlinePlanner) finish(
	ctx context.Context,
	state *executionState,
	identity planner.SessionIdentity,
	args finishArgs,
) completionToolOutput {
	started := time.Now()
	span := planner.InvocationInstrumentation(p.invocation).StartSpan(
		telemetry.PlannerSpanFinish,
		telemetry.PlannerSpanAttributes{Operation: "finish", ToolName: finishToolName},
	)
	safeOutcome := "invalid"
	if args.Outcome == stageresult.StageSucceeded || args.Outcome == stageresult.StageFailed {
		safeOutcome = string(args.Outcome)
	}
	selected, selectionError := p.finishArtifacts(state, args.Artifacts)
	if selectionError != nil {
		failure := planner.Failure{Code: "finish_rejected", Message: selectionError.Error()}
		span.End("rejected", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		state.recordTool(finishToolName, map[string]any{"outcome": safeOutcome}, false, time.Since(started), 0, &failure)
		return completionToolOutput{Error: toolFailureFrom(failure)}
	}
	result := stageresult.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: args.Outcome,
		Summary: args.Summary, Artifacts: selected, Error: clone.Pointer(args.Error),
	}
	if err := planner.ValidateCandidate(
		ctx, p.invocation.RunID, p.resultContract, result, p.inspector,
	); err != nil {
		// The rejection keeps the class of the candidate verification: a
		// candidate whose artifacts could not be verified may be finished
		// again unchanged (PLN-R118).
		failure := planner.Failure{
			Code: "finish_rejected", Message: "finish candidate does not satisfy the Stage result contract",
			Retryable: false,
		}
		if err.Retryable {
			failure.Message, failure.Retryable = "finish candidate artifacts could not be verified", true
		}
		span.End("rejected", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		state.recordTool(finishToolName, map[string]any{"outcome": safeOutcome}, false, time.Since(started), 0, &failure)
		return completionToolOutput{Error: toolFailureFrom(failure)}
	}
	if planErr := p.plan.CanFinish(args.Outcome); planErr != nil {
		failure := planner.Failure{
			Code: "finish_rejected", Message: planErr.Message, Retryable: false,
		}
		span.End("rejected", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		state.recordTool(finishToolName, map[string]any{"outcome": safeOutcome}, false, time.Since(started), 0, &failure)
		return completionToolOutput{Error: toolFailureFrom(failure)}
	}
	planRevision := p.plan.Snapshot().Revision
	if err := p.sessions.RecordFact(ctx, identity, planner.PlannerFact{
		Kind:         planner.PlannerEventFinishRequested,
		Key:          fmt.Sprintf("finish:%d:%s", planRevision, args.Outcome),
		PlanRevision: planRevision, Outcome: string(args.Outcome),
	}); err != nil {
		failure := p.planPersistenceFailure(state, err)
		span.End("failed", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		state.recordTool(finishToolName, map[string]any{"outcome": safeOutcome}, false, time.Since(started), 0, &failure)
		return completionToolOutput{Error: toolFailureFrom(failure)}
	}
	if !state.setCompletion(result) {
		failure := planner.Failure{
			Code: "finish_rejected", Message: "Planner already has a terminal decision", Retryable: false,
		}
		span.End("rejected", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		state.recordTool(finishToolName, map[string]any{"outcome": safeOutcome}, false, time.Since(started), 0, &failure)
		return completionToolOutput{Error: toolFailureFrom(failure)}
	}
	state.recordTool(finishToolName, map[string]any{"outcome": safeOutcome}, true, time.Since(started), 0, nil)
	span.End("succeeded", telemetry.PlannerSpanAttributes{PlanRevision: planRevision})
	return completionToolOutput{Accepted: true}
}

func workerFailure(failure planner.Failure) workerCallOutput {
	return workerCallOutput{OK: false, Error: toolFailureFrom(failure)}
}

func toolFailureFrom(failure planner.Failure) *toolFailure {
	return &toolFailure{Code: failure.Code, Message: failure.Message, Retryable: failure.Retryable}
}

func safeToolFailure(code, message string, retryable bool) *planner.Failure {
	return &planner.Failure{Code: code, Message: message, Retryable: retryable}
}

func failureFromPlanError(value *planner.PlanError) planner.Failure {
	return planner.Failure{Code: value.Code, Message: value.Message, Retryable: false}
}

func (p *streamlinePlanner) persistPlanTransition(
	ctx context.Context,
	state *executionState,
	identity planner.SessionIdentity,
	before planner.PlannerPlan,
	after planner.PlannerPlan,
	kind planner.PlannerEventKind,
) *planner.Failure {
	span := planner.InvocationInstrumentation(p.invocation).StartSpan(
		telemetry.PlannerSpanSubtask,
		telemetry.PlannerSpanAttributes{
			Operation: string(kind), SubtaskID: after.CurrentSubtaskID, PlanRevision: after.Revision,
		},
	)
	err := p.sessions.RecordPlan(ctx, identity, planner.PlannerPlanTransition{
		Kind: kind, ExpectedRevision: before.Revision, Plan: after.Projection(),
	})
	if err == nil {
		span.End("succeeded", telemetry.PlannerSpanAttributes{})
		return nil
	}
	failure := p.planPersistenceFailure(state, err)
	span.End("failed", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
	return &failure
}

func (p *streamlinePlanner) persistCurrentChanged(
	ctx context.Context,
	state *executionState,
	identity planner.SessionIdentity,
	planSnapshot planner.PlannerPlan,
) *planner.Failure {
	err := p.sessions.RecordFact(ctx, identity, planner.PlannerFact{
		Kind: planner.PlannerEventCurrentChanged,
		Key:  fmt.Sprintf("current:%d", planSnapshot.Revision), PlanRevision: planSnapshot.Revision,
		SubtaskID: planSnapshot.CurrentSubtaskID,
	})
	if err == nil {
		return nil
	}
	failure := p.planPersistenceFailure(state, err)
	return &failure
}

func (p *streamlinePlanner) persistDispatchSelected(
	ctx context.Context,
	state *executionState,
	identity planner.SessionIdentity,
	planRevision uint64,
	claim planner.PlannerDispatchClaim,
	workerName string,
) *planner.Failure {
	err := p.sessions.RecordFact(ctx, identity, planner.PlannerFact{
		Kind: planner.PlannerEventDispatchSelected,
		Key:  "selected:" + claim.CallID, PlanRevision: planRevision,
		SubtaskID: claim.Subtask.ID, CallID: claim.CallID, WorkerName: workerName,
	})
	if err == nil {
		return nil
	}
	failure := p.planPersistenceFailure(state, err)
	return &failure
}

func (p *streamlinePlanner) failDispatch(
	ctx context.Context,
	state *executionState,
	identity planner.SessionIdentity,
	callID string,
) *planner.Failure {
	before := p.plan.Snapshot()
	after, planErr := p.plan.FailDispatch(callID)
	if planErr != nil {
		failure := failureFromPlanError(planErr)
		state.setExternalFailure(planner.NewErrorFromFailure(failure, nil))
		return &failure
	}
	return p.persistPlanTransition(
		ctx, state, identity, before, after, planner.PlannerEventDispatchCompleted,
	)
}

func (p *streamlinePlanner) planPersistenceFailure(
	state *executionState,
	cause error,
) planner.Failure {
	value := planner.NewError(
		"planner_plan_persistence_failed",
		"Planner plan progress could not be persisted",
		true,
		cause,
	)
	if planner.SessionWriteRejected(cause) {
		value = planner.SessionError("record plan progress", cause)
	}
	state.setExternalFailure(value)
	return value.Failure
}

func safeSubtaskID(value string) string {
	if !runstore.ValidPlannerSubtaskID(value) {
		return "invalid"
	}
	return value
}

// finishArtifacts resolves only admitted facts, never a mutable Artifact alias or model history.
func (p *streamlinePlanner) finishArtifacts(state *executionState, selections map[string]materialSelection) (map[string]contracts.ArtifactRef, error) {
	type key struct{ namespace, name string }
	exact := map[key]contracts.ArtifactRef{}
	ambiguous := map[key]bool{}
	add := func(ref contracts.ArtifactRef) {
		k := key{ref.Namespace, ref.Name}
		if old, ok := exact[k]; ok && (old.Revision == nil || ref.Revision == nil || *old.Revision != *ref.Revision) {
			ambiguous[k] = true
		}
		exact[k] = ref
	}
	state.mu.Lock()
	for _, refs := range state.materials {
		for _, ref := range refs {
			add(ref)
		}
	}
	state.mu.Unlock()
	// Accepted Worker outputs supersede inputs at the same address. Distinct
	// trusted input roles at an otherwise unproduced address must agree exactly.
	produced := maps.Clone(exact)
	for _, ref := range p.invocation.Context.Artifacts {
		if ref != nil {
			if _, ok := produced[key{ref.Namespace, ref.Name}]; !ok {
				add(*ref)
			}
		}
	}
	if prior := p.invocation.Context.Continuation; prior != nil {
		for _, ref := range prior.Artifacts {
			if _, ok := produced[key{ref.Namespace, ref.Name}]; !ok {
				add(ref)
			}
		}
	}
	result := make(map[string]contracts.ArtifactRef, len(selections))
	for slot, selection := range selections {
		k := key{selection.Namespace, selection.Name}
		ref, ok := exact[k]
		if !ok || ambiguous[k] || ref.Revision == nil {
			return nil, fmt.Errorf("finish material is unknown or ambiguous; use an accepted Worker output or unambiguous Stage input")
		}
		result[slot] = ref.Clone()
	}
	return result, nil
}
