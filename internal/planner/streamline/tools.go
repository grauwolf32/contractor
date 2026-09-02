package streamline

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

const (
	addSubtaskToolName            = "add_subtask"
	listSubtasksToolName          = "list_subtasks"
	executeCurrentSubtaskToolName = "execute_current_subtask"
	finishToolName                = "finish"
	maxStagePayloadBytes          = 256 * 1024
	maxStageSummaryBytes          = 64 * 1024
	maxStageArtifacts             = 128
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

type finishArgs struct {
	Outcome   contracts.StageOutcome           `json:"outcome"`
	Summary   string                           `json:"summary"`
	Artifacts map[string]contracts.ArtifactRef `json:"artifacts"`
	Error     *contracts.TerminationError      `json:"error,omitempty"`
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

func (p *streamlinePlanner) buildTools(
	state *executionState,
	identity planner.SessionIdentity,
) ([]tool.Tool, map[string]struct{}, error) {
	result := make([]tool.Tool, 0, 4+len(p.workers)*2)
	allowed := make(map[string]struct{}, 4+len(p.workers)*2)
	addSubtask, err := functiontool.New(functiontool.Config{
		Name:        addSubtaskToolName,
		Description: "Append one bounded immutable subtask to the ordered Stage plan. Objective and instructions are stored once; later Worker dispatches identify this exact work only by subtask_id.",
	}, func(ctx agent.ToolContext, args addSubtaskArgs) (plannerPlanOutput, error) {
		return p.addSubtask(ctx, state, identity, args), nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("build add_subtask tool: %w", err)
	}
	listSubtasks, err := functiontool.New(functiontool.Config{
		Name:        listSubtasksToolName,
		Description: "Return the current bounded ordered subtask plan, including adapter-controlled status and the exact current subtask_id.",
		InputSchema: &jsonschema.Schema{
			Type: "object", Properties: map[string]*jsonschema.Schema{},
			AdditionalProperties: &jsonschema.Schema{Not: &jsonschema.Schema{}},
		},
	}, func(ctx agent.ToolContext, _ listSubtasksArgs) (plannerPlanOutput, error) {
		return p.listSubtasks(ctx, state), nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("build list_subtasks tool: %w", err)
	}
	result = append(result, addSubtask, listSubtasks)
	allowed[addSubtaskToolName] = struct{}{}
	allowed[listSubtasksToolName] = struct{}{}
	var executeCurrentSubtask tool.Tool
	if p.profile.routesWorkers {
		schema, schemaErr := routerExecuteSchema(p.workers)
		if schemaErr != nil {
			return nil, nil, fmt.Errorf("build Router execute schema: %w", schemaErr)
		}
		executeCurrentSubtask, err = functiontool.New(functiontool.Config{
			Name:        executeCurrentSubtaskToolName,
			Description: "Execute the exact current stored subtask with one selected immutable logical Worker. Server supplies the complete StageContext; subtask_id and worker_name are the only arguments.",
			InputSchema: schema,
		}, func(ctx agent.ToolContext, args routerWorkerCallArgs) (workerCallOutput, error) {
			return p.routeWorker(ctx, state, identity, args), nil
		})
	} else {
		binding := p.workers[0]
		executeCurrentSubtask, err = functiontool.New(functiontool.Config{
			Name: executeCurrentSubtaskToolName,
			Description: fmt.Sprintf(
				"Execute the exact current stored subtask with the sole prepared logical Worker %q (%s). Server supplies the complete immutable StageContext; subtask_id is the only argument.",
				binding.logicalName, binding.description,
			),
		}, func(ctx agent.ToolContext, args workerCallArgs) (workerCallOutput, error) {
			return p.callWorker(ctx, state, identity, binding, args.SubtaskID), nil
		})
	}
	if err != nil {
		return nil, nil, fmt.Errorf("build execute_current_subtask tool: %w", err)
	}
	result = append(result, executeCurrentSubtask)
	allowed[executeCurrentSubtaskToolName] = struct{}{}
	memoryTools, err := p.buildMemoryTools(state, allowed)
	if err != nil {
		return nil, nil, err
	}
	result = append(result, memoryTools...)
	finish, err := functiontool.New(functiontool.Config{
		Name:        finishToolName,
		Description: "Finish the Stage with a succeeded or failed candidate. A failed candidate requires a safe error; a succeeded candidate forbids one. Every artifact must name a declared result slot and include an exact revision. Workflow Scheduler remains the acceptance and transition owner.",
	}, func(ctx agent.ToolContext, args finishArgs) (completionToolOutput, error) {
		return p.finish(ctx, state, identity, args), nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("build finish tool: %w", err)
	}
	result = append(result, finish)
	allowed[finishToolName] = struct{}{}
	return result, allowed, nil
}

func (p *streamlinePlanner) addSubtask(
	ctx agent.ToolContext,
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
	_ agent.ToolContext, state *executionState,
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

func (p *streamlinePlanner) callWorker(
	ctx agent.ToolContext,
	state *executionState,
	identity planner.SessionIdentity,
	binding workerBinding,
	subtaskID string,
) workerCallOutput {
	started := time.Now()
	safeArguments := map[string]any{
		"binding": binding.logicalName, "subtaskId": safeSubtaskID(subtaskID),
	}
	beforeClaim := p.plan.Snapshot()
	claim, planErr := p.plan.ClaimCurrentSubtask(subtaskID, binding.logicalName)
	if planErr != nil {
		failure := failureFromPlanError(planErr)
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, &failure)
		return workerFailure(failure)
	}
	claimedPlan := p.plan.Snapshot()
	if failure := p.persistDispatchSelected(
		ctx, state, identity, beforeClaim.Revision, claim, binding.logicalName,
	); failure != nil {
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, failure)
		return workerFailure(*failure)
	}
	if failure := p.persistPlanTransition(
		ctx, state, identity, beforeClaim, claimedPlan, planner.PlannerEventDispatchStarted,
	); failure != nil {
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, failure)
		return workerFailure(*failure)
	}
	request, failure := p.workerRequest(ctx, claim.Subtask)
	if failure != nil {
		if persisted := p.failDispatch(ctx, state, identity, claim.CallID); persisted != nil {
			failure = persisted
		}
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, failure)
		return workerFailure(*failure)
	}
	deadline := p.deadline
	if !deadline.After(time.Now()) {
		failure := &planner.Failure{
			Code: "worker_deadline_exceeded", Message: "Worker invocation deadline expired", Retryable: true,
		}
		if persisted := p.failDispatch(ctx, state, identity, claim.CallID); persisted != nil {
			failure = persisted
		}
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, failure)
		return workerFailure(*failure)
	}
	if limit := state.reserveWorkerCall(); limit != nil {
		ctx.Actions().SkipSummarization = true
		failure := &limit.Failure
		if persisted := p.failDispatch(ctx, state, identity, claim.CallID); persisted != nil {
			failure = persisted
		}
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, failure)
		return workerFailure(*failure)
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
		workerContext, binding.logicalName, planner.CloneWorkerHandle(binding.handle), request,
	)
	if err != nil {
		failure := planner.FailureFrom(err)
		workerSpan.End("failed", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		if persisted := p.failDispatch(ctx, state, identity, claim.CallID); persisted != nil {
			failure = *persisted
		}
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, &failure)
		return workerFailure(failure)
	}
	if validation := planner.ValidateWorkerCompletion(completion, claim.Subtask.ID); validation != nil {
		failure := validation.Failure
		workerSpan.End("rejected", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		if persisted := p.failDispatch(ctx, state, identity, claim.CallID); persisted != nil {
			failure = *persisted
		}
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, &failure)
		return workerFailure(failure)
	}
	if completion.Failure != nil {
		failure := planner.Failure{
			Code: completion.Failure.Code, Message: completion.Failure.Message,
			Retryable: completion.Failure.Retryable,
		}
		workerSpan.End("failed", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		if persisted := p.failDispatch(ctx, state, identity, claim.CallID); persisted != nil {
			failure = *persisted
		}
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, &failure)
		return workerFailure(failure)
	}
	result := planner.CloneWorkerResult(*completion.Result)
	if validation := p.validateWorkerResult(workerContext, result); validation != nil {
		failure := validation.Failure
		workerSpan.End("rejected", telemetry.PlannerSpanAttributes{ErrorCode: failure.Code})
		if persisted := p.failDispatch(ctx, state, identity, claim.CallID); persisted != nil {
			failure = *persisted
		}
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, &failure)
		return workerFailure(failure)
	}
	workerSpan.End("succeeded", telemetry.PlannerSpanAttributes{})
	beforeCompletion := p.plan.Snapshot()
	completedPlan, planErr := p.plan.CompleteDispatch(claim.CallID, contracts.StageSucceeded)
	if planErr != nil {
		failure := failureFromPlanError(planErr)
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, &failure)
		return workerFailure(failure)
	}
	if failure := p.persistPlanTransition(
		ctx, state, identity, beforeCompletion, completedPlan, planner.PlannerEventDispatchCompleted,
	); failure != nil {
		state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, failure)
		return workerFailure(*failure)
	}
	if beforeCompletion.CurrentSubtaskID != completedPlan.CurrentSubtaskID {
		if failure := p.persistCurrentChanged(ctx, state, identity, completedPlan); failure != nil {
			state.recordTool(executeCurrentSubtaskToolName, safeArguments, false, time.Since(started), 0, failure)
			return workerFailure(*failure)
		}
	}
	cloned := planner.CloneWorkerResult(result)
	encoded, _ := json.Marshal(cloned)
	state.recordTool(executeCurrentSubtaskToolName, safeArguments, true, time.Since(started), len(encoded), nil)
	return workerCallOutput{OK: true, Result: &cloned}
}

func (p *streamlinePlanner) routeWorker(
	ctx agent.ToolContext,
	state *executionState,
	identity planner.SessionIdentity,
	args routerWorkerCallArgs,
) workerCallOutput {
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
	return workerFailure(failure)
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
	request := planner.CloneStageRequest(p.request)
	request.SubtaskID = subtask.ID
	request.Objective = subtask.Objective
	request.Instructions = subtask.Instructions
	for _, ref := range request.Artifacts {
		if err := ref.ValidateExact(); err != nil {
			return contracts.StageContentRequest{}, safeToolFailure(
				"worker_request_invalid", "Worker request contains an invalid exact artifact reference", false,
			)
		}
		if _, err := p.inspector.Inspect(ctx, p.invocation.RunID, planner.CloneArtifactRef(ref)); err != nil {
			return contracts.StageContentRequest{}, safeToolFailure(
				"worker_artifact_unavailable", "Worker request artifact could not be verified", true,
			)
		}
	}
	if err := request.Validate(); err != nil {
		return contracts.StageContentRequest{}, safeToolFailure(
			"worker_request_invalid", "Worker objective, instructions, or structured context is invalid", false,
		)
	}
	encoded, err := json.Marshal(request)
	if err != nil || len(encoded) > maxStagePayloadBytes {
		return contracts.StageContentRequest{}, safeToolFailure(
			"worker_request_invalid", "Worker request exceeds its bounded contract", false,
		)
	}
	return request, nil
}

func (p *streamlinePlanner) validateWorkerResult(
	ctx context.Context, result contracts.WorkerResult,
) *planner.Error {
	encoded, err := json.Marshal(result)
	if err != nil || len(encoded) > maxStagePayloadBytes ||
		len(result.Result) > maxStageSummaryBytes || len(result.Artifacts) > maxStageArtifacts {
		return planner.NewError(
			"invalid_worker_result", "Worker returned an oversized WorkerResult", false, err,
		)
	}
	for _, ref := range result.Artifacts {
		if _, err := p.inspector.Inspect(ctx, p.invocation.RunID, planner.CloneArtifactRef(ref)); err != nil {
			return planner.NewError(
				"result_artifact_unavailable", "Worker result artifact could not be verified", true, err,
			)
		}
	}
	return nil
}

func (p *streamlinePlanner) finish(
	ctx agent.ToolContext,
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
	if args.Outcome == contracts.StageSucceeded || args.Outcome == contracts.StageFailed {
		safeOutcome = string(args.Outcome)
	}
	result := contracts.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: args.Outcome,
		Summary: args.Summary, Artifacts: cloneArtifactMap(args.Artifacts), Error: cloneTerminationError(args.Error),
	}
	if err := planner.ValidateCandidate(
		ctx, p.invocation.RunID, p.resultContract, result, p.inspector,
	); err != nil {
		failure := planner.Failure{
			Code: "finish_rejected", Message: "finish candidate does not satisfy the Stage result contract",
			Retryable: false,
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
	ctx.Actions().SkipSummarization = true
	ctx.Actions().Escalate = true
	state.recordTool(finishToolName, map[string]any{"outcome": safeOutcome}, true, time.Since(started), 0, nil)
	span.End("succeeded", telemetry.PlannerSpanAttributes{PlanRevision: planRevision})
	return completionToolOutput{Accepted: true}
}

func cloneTerminationError(input *contracts.TerminationError) *contracts.TerminationError {
	if input == nil {
		return nil
	}
	cloned := *input
	return &cloned
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
	state.setExternalFailure(value)
	return value.Failure
}

func safeSubtaskID(value string) string {
	if len(value) == 0 || len(value) > 2 {
		return "invalid"
	}
	for _, current := range value {
		if current < '0' || current > '9' {
			return "invalid"
		}
	}
	return value
}

func cloneArtifactMap(input map[string]contracts.ArtifactRef) map[string]contracts.ArtifactRef {
	result := make(map[string]contracts.ArtifactRef, len(input))
	for name, ref := range input {
		result[name] = planner.CloneArtifactRef(ref)
	}
	return result
}
