package streamline

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

const (
	finishToolName       = "finish"
	maxStagePayloadBytes = 256 * 1024
	maxStageSummaryBytes = 64 * 1024
	maxStageArtifacts    = 128
)

type workerCallArgs struct {
	Objective    string                           `json:"objective"`
	Instructions string                           `json:"instructions"`
	Parameters   map[string]string                `json:"parameters,omitempty"`
	Artifacts    map[string]contracts.ArtifactRef `json:"artifacts,omitempty"`
}

type workerCallOutput struct {
	OK     bool                          `json:"ok"`
	Result *contracts.StageContentResult `json:"result,omitempty"`
	Error  *toolFailure                  `json:"error,omitempty"`
}

type finishArgs struct {
	Outcome   contracts.StageOutcome           `json:"outcome"`
	Summary   string                           `json:"summary"`
	Artifacts map[string]contracts.ArtifactRef `json:"artifacts"`
	Error     *contracts.TerminationError      `json:"error,omitempty"`
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

func (p *streamlinePlanner) buildTools(state *executionState) ([]tool.Tool, map[string]struct{}, error) {
	result := make([]tool.Tool, 0, len(p.workers)+1)
	allowed := make(map[string]struct{}, len(p.workers)+1)
	for _, current := range p.workers {
		binding := current
		adapter, err := functiontool.New(functiontool.Config{
			Name: binding.toolName,
			Description: fmt.Sprintf(
				"Invoke the fixed prepared Worker %q. Capability: %s. Calls are sequential; pass only exact artifact revisions.",
				binding.logicalName, binding.description,
			),
		}, func(ctx agent.ToolContext, args workerCallArgs) (workerCallOutput, error) {
			return p.callWorker(ctx, state, binding, args), nil
		})
		if err != nil {
			return nil, nil, fmt.Errorf("build Worker tool %q: %w", binding.logicalName, err)
		}
		result = append(result, adapter)
		allowed[binding.toolName] = struct{}{}
	}
	finish, err := functiontool.New(functiontool.Config{
		Name:        finishToolName,
		Description: "Finish the Stage with a succeeded or failed candidate. A failed candidate requires a safe error; a succeeded candidate forbids one. Every artifact must name a declared result slot and include an exact revision. Workflow Scheduler remains the acceptance and transition owner.",
	}, func(ctx agent.ToolContext, args finishArgs) (completionToolOutput, error) {
		return p.finish(ctx, state, args), nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("build finish tool: %w", err)
	}
	result = append(result, finish)
	allowed[finishToolName] = struct{}{}
	return result, allowed, nil
}

func (p *streamlinePlanner) callWorker(
	ctx agent.ToolContext,
	state *executionState,
	binding workerBinding,
	args workerCallArgs,
) workerCallOutput {
	started := time.Now()
	safeArguments := map[string]any{
		"binding": binding.logicalName, "artifactNames": sortedArtifactNames(args.Artifacts),
	}
	if limit := state.reserveWorkerCall(); limit != nil {
		ctx.Actions().SkipSummarization = true
		failure := limit.Failure
		state.recordTool(binding.toolName, safeArguments, false, time.Since(started), 0, &failure)
		return workerFailure(failure)
	}
	request, failure := p.workerRequest(ctx, args)
	if failure != nil {
		state.recordTool(binding.toolName, safeArguments, false, time.Since(started), 0, failure)
		return workerFailure(*failure)
	}
	deadline := p.deadline
	if !deadline.After(time.Now()) {
		failure := planner.Failure{
			Code: "worker_deadline_exceeded", Message: "Worker invocation deadline expired", Retryable: true,
		}
		state.recordTool(binding.toolName, safeArguments, false, time.Since(started), 0, &failure)
		return workerFailure(failure)
	}
	workerContext, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	result, err := p.invoker.Invoke(
		workerContext, binding.logicalName, planner.CloneWorkerHandle(binding.handle), request,
	)
	if err != nil {
		failure := planner.FailureFrom(err)
		state.recordTool(binding.toolName, safeArguments, false, time.Since(started), 0, &failure)
		return workerFailure(failure)
	}
	if validation := p.validateWorkerResult(workerContext, result); validation != nil {
		failure := validation.Failure
		state.recordTool(binding.toolName, safeArguments, false, time.Since(started), 0, &failure)
		return workerFailure(failure)
	}
	cloned := planner.CloneStageResult(result)
	encoded, _ := json.Marshal(cloned)
	state.recordTool(binding.toolName, safeArguments, true, time.Since(started), len(encoded), nil)
	return workerCallOutput{OK: true, Result: &cloned}
}

func (p *streamlinePlanner) workerRequest(
	ctx context.Context, args workerCallArgs,
) (contracts.StageContentRequest, *planner.Failure) {
	request := planner.CloneStageRequest(p.request)
	request.Objective = args.Objective
	request.Instructions = args.Instructions
	request.Parameters = make(map[string]string, len(args.Parameters))
	for name, value := range args.Parameters {
		request.Parameters[name] = value
	}
	request.Artifacts = make(map[string]contracts.ArtifactRef, len(args.Artifacts))
	for name, ref := range args.Artifacts {
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
		request.Artifacts[name] = planner.CloneArtifactRef(ref)
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
	ctx context.Context, result contracts.StageContentResult,
) *planner.Error {
	if err := result.Validate(); err != nil {
		return planner.NewError(
			"invalid_worker_result", "Worker returned an invalid StageContentResult", false, err,
		)
	}
	encoded, err := json.Marshal(result)
	if err != nil || len(encoded) > maxStagePayloadBytes ||
		len(result.Summary) > maxStageSummaryBytes || len(result.Artifacts) > maxStageArtifacts {
		return planner.NewError(
			"invalid_worker_result", "Worker returned an oversized StageContentResult", false, err,
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
	ctx agent.ToolContext, state *executionState, args finishArgs,
) completionToolOutput {
	started := time.Now()
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
		state.recordTool(finishToolName, map[string]any{"outcome": safeOutcome}, false, time.Since(started), 0, &failure)
		return completionToolOutput{Error: toolFailureFrom(failure)}
	}
	if !state.setCompletion(result) {
		failure := planner.Failure{
			Code: "finish_rejected", Message: "Planner already has a terminal decision", Retryable: false,
		}
		state.recordTool(finishToolName, map[string]any{"outcome": safeOutcome}, false, time.Since(started), 0, &failure)
		return completionToolOutput{Error: toolFailureFrom(failure)}
	}
	ctx.Actions().SkipSummarization = true
	ctx.Actions().Escalate = true
	state.recordTool(finishToolName, map[string]any{"outcome": safeOutcome}, true, time.Since(started), 0, nil)
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

func cloneArtifactMap(input map[string]contracts.ArtifactRef) map[string]contracts.ArtifactRef {
	result := make(map[string]contracts.ArtifactRef, len(input))
	for name, ref := range input {
		result[name] = planner.CloneArtifactRef(ref)
	}
	return result
}

func sortedArtifactNames(input map[string]contracts.ArtifactRef) []string {
	result := make([]string, 0, len(input))
	for name := range input {
		result = append(result, name)
	}
	sort.Strings(result)
	return result
}
