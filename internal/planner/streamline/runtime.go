package streamline

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// runtimeHost is owned by the current Planner.Run. The last ordered response is
// retained only until the next operation, to reconcile a lost transport reply.
// Server process loss uses ordinary Stage interruption; this is not a journal.
type runtimeHost struct {
	ctx             context.Context
	planner         *streamlinePlanner
	state           *executionState
	identity        planner.SessionIdentity
	allowed         map[string]struct{}
	mu              sync.Mutex
	sequence        int
	last            []byte
	reply           contracts.PlannerOperationResult
	failure         *planner.Error
	modelPhase      string
	nextTool        string
	checkpointReady bool
	modelSpan       telemetry.PlannerSpan
	inputCaptured   bool
}

func (f *Factory) ExecuteOperation(ctx context.Context, stageID, allocationID string, request contracts.PlannerOperation) (contracts.PlannerOperationResult, error) {
	value, ok := f.runtimeHosts.Load(stageID)
	if !ok {
		return contracts.PlannerOperationResult{}, errors.New("Planner execution is unavailable")
	}
	h := value.(*runtimeHost)
	if h.planner.invocation.RuntimeExecutor == nil || h.planner.invocation.RuntimeExecutor.AllocationID != allocationID {
		return contracts.PlannerOperationResult{}, errors.New("Planner allocation is unavailable")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if err := h.ctx.Err(); err != nil {
		return contracts.PlannerOperationResult{}, err
	}
	if err := request.Validate(); err != nil {
		return contracts.PlannerOperationResult{}, err
	}
	encoded, _ := json.Marshal(request)
	if request.Sequence == h.sequence && bytes.Equal(encoded, h.last) {
		return h.reply, nil
	}
	if request.Sequence != h.sequence+1 || h.failure != nil {
		return contracts.PlannerOperationResult{}, errors.New("Planner operation order is invalid")
	}
	if err := ctx.Err(); err != nil {
		return contracts.PlannerOperationResult{}, err
	}
	// A disconnected reply does not revoke an admitted operation. Run owns its
	// cancellation and joins this lock before release; a retry replays its result.
	result, failure := h.execute(h.ctx, request)
	h.sequence, h.last = request.Sequence, encoded
	h.reply = result
	if failure != nil {
		h.endModel("failed", failure.Code, nil)
		h.failure = failure
		h.reply.Failure = &contracts.TerminationError{Code: failure.Code, Message: failure.Message, Retryable: failure.Retryable}
	}
	if err := h.reply.Validate(); err != nil {
		h.failure = invalidRuntimeOperation()
		h.reply = contracts.PlannerOperationResult{APIVersion: contracts.APIVersion, Result: json.RawMessage(`{}`), Failure: &contracts.TerminationError{Code: h.failure.Code, Message: h.failure.Message}}
	}
	return h.reply, nil
}

func (h *runtimeHost) execute(ctx context.Context, request contracts.PlannerOperation) (contracts.PlannerOperationResult, *planner.Error) {
	reply := contracts.PlannerOperationResult{APIVersion: contracts.APIVersion, Result: json.RawMessage(`{}`)}
	var value any
	var event *planner.Activity
	switch request.Name {
	case "before_model":
		var args struct {
			Retained int64 `json:"retainedBytes" strictjson:"required"`
		}
		if strictjson.DecodeNonNull(request.Arguments, &args) != nil || h.modelPhase != "" || h.nextTool != "" || h.checkpointReady {
			return reply, invalidRuntimeOperation()
		}
		if failure := h.checkRuntimeRetained(args.Retained); failure != nil {
			return reply, failure
		}
		if h.planner.finishRecoveryCallStarted {
			return reply, planner.NewError("planner_gateway_invalid_response", "Document recovery did not produce a valid finish", true, nil)
		}
		if h.planner.finishRecovery {
			h.planner.finishRecoveryCallStarted = true
		}
		if failure := h.state.beforeModel(); failure != nil {
			return reply, failure
		}
		h.modelPhase = "main"
		guidance, err := h.planner.prepareGuidance(h.state)
		if err != nil {
			return reply, invalidRuntimeOperation()
		}
		capture := h.startModel("model.generate", h.planner.invocation.ModelAccess.ModelPolicy.Model)
		value = map[string]any{"guidance": guidance, "plan": planView(h.planner.plan.Snapshot()), "captureContent": capture}
	case "after_model":
		var args struct {
			Input   int32              `json:"inputTokens" strictjson:"required"`
			Output  int32              `json:"outputTokens" strictjson:"required"`
			Cached  int32              `json:"cachedTokens" strictjson:"required"`
			Calls   []string           `json:"calls" strictjson:"required"`
			Content string             `json:"content,omitempty"`
			Cost    *reporting.LLMCost `json:"cost,omitempty"`
		}
		if strictjson.DecodeNonNull(request.Arguments, &args) != nil || args.Input < 0 || args.Output < 0 || args.Cached < 0 || h.modelPhase != "main" || !h.acceptCost(args.Cost) {
			return reply, invalidRuntimeOperation()
		}
		usage := &modelUsage{PromptTokenCount: args.Input, CandidatesTokenCount: args.Output, CachedContentTokenCount: args.Cached}
		rejected, err := h.state.afterModel(usage, args.Calls, h.allowed)
		if len(args.Content) <= telemetry.MaxContentBytes && len(args.Content) > 0 {
			telemetry.CapturePlannerOutput(h.modelSpan, func() any { return json.RawMessage(args.Content) })
		}
		if err != nil {
			failure := planner.FailureFrom(err)
			h.endModel("failed", failure.Code, usage)
			return reply, planner.NewError(failure.Code, failure.Message, failure.Retryable, err)
		}
		h.modelPhase = ""
		if h.planner.finishRecovery && (len(args.Calls) != 1 || args.Calls[0] != finishToolName) {
			h.endModel("failed", "planner_tool_selection_invalid", usage)
			return reply, planner.NewError("planner_tool_selection_invalid", "Document recovery requires exactly one finish", true, nil)
		}
		h.endModel("succeeded", "", usage)
		if !rejected && len(args.Calls) == 1 {
			h.nextTool = args.Calls[0]
		}
		value = map[string]bool{"rejected": rejected}
		event = h.activity()
		event.InputTokens, event.OutputTokens = int64(args.Input), int64(args.Output)
		if !rejected {
			event.FunctionCalls = append(event.FunctionCalls, args.Calls...)
		}
	case "model_failed":
		var args struct {
			Cost  *reporting.LLMCost `json:"cost,omitempty"`
			Kind  string             `json:"kind" strictjson:"required"`
			Code  string             `json:"code" strictjson:"required"`
			Usage *struct {
				Input  int32 `json:"inputTokens" strictjson:"required"`
				Output int32 `json:"outputTokens" strictjson:"required"`
			} `json:"usage" strictjson:"required,nullable"`
			EmptyStop     bool `json:"emptyStop" strictjson:"required"`
			OutputLimited bool `json:"outputLimited" strictjson:"required"`
		}
		if strictjson.DecodeNonNull(request.Arguments, &args) != nil || h.modelPhase == "" || !h.acceptCost(args.Cost) || (args.Usage != nil && (args.Usage.Input < 0 || args.Usage.Output < 0)) {
			return reply, invalidRuntimeOperation()
		}
		var usage *modelUsage
		if args.Usage != nil {
			usage = &modelUsage{PromptTokenCount: args.Usage.Input, CandidatesTokenCount: args.Usage.Output}
		}
		switch args.Kind {
		case "invalid_response":
			h.endModel("failed", "planner_gateway_invalid_response", usage)

			if h.modelPhase == "main" && args.Usage != nil && (args.EmptyStop || args.OutputLimited) && h.planner.readyForDocumentRecovery() {
				if failure := h.state.chargeUsage(usage); failure != nil {
					return reply, failure
				}
				h.modelPhase, h.planner.finishRecovery = "", true
				reply.Result = json.RawMessage(`{"recoverFinish":true}`)
				return reply, nil
			}
			return reply, h.state.rejectedGatewayResponse(usage)
		case "rejected":
			if !contracts.ValidName(args.Code) || args.Usage != nil {
				return reply, invalidRuntimeOperation()
			}
			h.endModel("failed", "planner_gateway_rejected", nil)
			return reply, h.state.rejectedGatewayRequest(args.Code)
		case "unavailable":
			if args.Usage != nil {
				return reply, invalidRuntimeOperation()
			}
			h.endModel("failed", "planner_gateway_unavailable", nil)
			return reply, h.state.providerFailure()
		default:
			return reply, invalidRuntimeOperation()
		}
	case "model_content":
		var args struct {
			Content string `json:"content" strictjson:"required"`
		}
		if strictjson.DecodeNonNull(request.Arguments, &args) != nil || h.modelPhase == "" || h.inputCaptured {
			return reply, invalidRuntimeOperation()
		}
		if len(args.Content) <= telemetry.MaxContentBytes {
			telemetry.CapturePlannerInput(h.modelSpan, func() any { return json.RawMessage(args.Content) })
		}
		h.inputCaptured = true
		value = map[string]any{}
	case "context", "before_compactor", "after_compactor", "checkpoint":
		var failure *planner.Error
		value, event, failure = h.contextOperation(ctx, request)
		if failure != nil {
			return reply, failure
		}
	default:
		if _, ok := h.allowed[request.Name]; !ok || h.modelPhase != "" || h.nextTool != request.Name || h.checkpointReady {
			return reply, invalidRuntimeOperation()
		}
		h.nextTool = ""
		p, state, identity := h.planner, h.state, h.identity
		switch request.Name {
		case addSubtaskToolName:
			var args addSubtaskArgs
			if strictjson.DecodeNonNull(request.Arguments, &args) != nil {
				return reply, invalidRuntimeOperation()
			}
			value = p.addSubtask(ctx, state, identity, args)
		case listSubtasksToolName:
			var args struct{}
			if strictjson.DecodeNonNull(request.Arguments, &args) != nil {
				return reply, invalidRuntimeOperation()
			}
			value = p.listSubtasks(ctx, state)
		case executeCurrentSubtaskToolName:
			if p.profile.routesWorkers {
				var args routerWorkerCallArgs
				if strictjson.DecodeNonNull(request.Arguments, &args) != nil {
					return reply, invalidRuntimeOperation()
				}
				value, reply.Stop = p.routeWorker(ctx, state, identity, args)
			} else {
				var args workerCallArgs
				if strictjson.DecodeNonNull(request.Arguments, &args) != nil {
					return reply, invalidRuntimeOperation()
				}
				value, reply.Stop = p.callWorker(ctx, state, identity, p.workers[0], args.SubtaskID)
			}
		case finishToolName:
			var args finishArgs
			if strictjson.DecodeNonNull(request.Arguments, &args) != nil {
				return reply, invalidRuntimeOperation()
			}
			result := p.finish(ctx, state, identity, args)
			value, reply.Stop, reply.Escalate = result, result.Accepted, result.Accepted
		case readContextHistoryToolName:
			if p.profile.routesWorkers {
				var args routerHistoryArgs
				if strictjson.DecodeNonNull(request.Arguments, &args) != nil {
					return reply, invalidRuntimeOperation()
				}
				value = p.routeHistory(state, identity.SessionID, args)
			} else {
				var args historyArgs
				if strictjson.DecodeNonNull(request.Arguments, &args) != nil {
					return reply, invalidRuntimeOperation()
				}
				value = p.readHistory(state, identity.SessionID, p.workers[0].logicalName, args.Cursor)
			}
		default:
			var args map[string]any
			if strictjson.DecodeNonNull(request.Arguments, &args) != nil {
				return reply, invalidRuntimeOperation()
			}
			if bindings := p.memoryBindings(request.Name); len(bindings) > 0 {
				value, _ = (&plannerMemoryTool{planner: p, state: state, operation: request.Name, bindings: bindings, router: p.profile.routesWorkers}).run(ctx, args)
			} else {
				value, _ = (&plannerStateTool{planner: p, state: state, operation: request.Name, bindings: p.stateBindings(request.Name), router: p.profile.routesWorkers}).run(ctx, args)
			}
		}
		event = h.activity()
		event.FunctionResults = append(event.FunctionResults, request.Name)
	}
	if event != nil {
		event.SkipSummarization, event.Escalate = reply.Stop, reply.Escalate
		if err := h.planner.sessions.RecordActivity(ctx, h.identity, *event); err != nil {
			if result, _ := h.state.terminal(); result != nil {
				return reply, planner.CompletionWriteError(planner.Completion{Result: result}, err)
			}
			return reply, planner.SessionError("record Runtime Planner event", err)
		}
	}

	encoded, err := json.Marshal(value)
	if err != nil {
		return reply, invalidRuntimeOperation()
	}
	reply.Result = encoded
	return reply, nil
}

func invalidRuntimeOperation() *planner.Error {
	return planner.NewError("planner_operation_invalid", "Runtime Planner operation is invalid", false, nil)
}

func (p *streamlinePlanner) runRuntime(ctx context.Context, identity planner.SessionIdentity, state *executionState, tools []contracts.PlannerTool, allowed map[string]struct{}) (stageresult.StageContentResult, error) {
	deadline := minTime(p.invocation.Deadline, time.Now().Add(p.limits.MaxWallTime))
	p.deadline = deadline
	runContext, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	host := &runtimeHost{ctx: runContext, planner: p, state: state, identity: identity, allowed: allowed}
	if _, exists := p.runtimeHosts.LoadOrStore(p.invocation.StageExecutionID, host); exists {
		return stageresult.StageContentResult{}, invalidRuntimeOperation()
	}
	defer p.runtimeHosts.Delete(p.invocation.StageExecutionID)
	message, err := p.initialMessage()
	if err != nil {
		return stageresult.StageContentResult{}, p.fail(ctx, identity, state, planner.NewError("planner_context_invalid", "Planner context could not be encoded", false, err))
	}
	// The canonical Stage context is already in Input. Do not duplicate its
	// parameters and exact refs in the outer execution envelope.
	request := contracts.StageContentRequest{
		APIVersion: contracts.APIVersion, SubtaskID: "0", Objective: "Execute the prepared Stage Planner",
		Instructions: p.systemInstruction(), Deadline: deadline,
		Parameters: map[string]string{}, Artifacts: map[string]contracts.ArtifactRef{},
	}
	request.Planner = &contracts.PlannerInvocation{Input: message, Tools: tools}
	if c := p.compactionSettings; c != nil {
		request.Planner.Compaction = &contracts.PlannerCompaction{Instructions: c.instructions, Threshold: c.threshold, KeepRecentGroups: c.keepRecent, MaxCompactions: c.maxCompactions, Policy: c.compactor.Clone()}
	}
	if err := request.Validate(); err != nil {
		return stageresult.StageContentResult{}, p.fail(ctx, identity, state, planner.NewError("invalid_stage_content", "Planner execution request violates its contract", false, err))
	}
	completion, callErr := p.invoker.Invoke(runContext, "planner", p.invocation.RuntimeExecutor.Clone(), request)
	stopped := stoppedPlannerError(ctx, runContext)
	cancel()
	host.mu.Lock()
	failure := host.failure
	if stopped != nil {
		host.endModel("cancelled", stopped.Code, nil)
	} else {
		host.endModel("failed", "planner_execution_failed", nil)
	}
	host.mu.Unlock()
	// A confirmed terminal turn survives late transport failure or cancellation.
	// A required append failure still prevents candidate completion (PLN-R106).
	if result, _ := state.terminal(); failure == nil && result != nil {
		outcome := planner.Completion{Result: result}
		if err := planner.CompleteSession(ctx, p.sessions, identity, outcome); err != nil {
			return stageresult.StageContentResult{}, planner.CompletionWriteError(outcome, err)
		}
		return result.Clone(), nil
	}
	if stopped != nil {
		if failure != nil {
			stopped = planner.NewError(stopped.Code, stopped.Message, stopped.Retryable && failure.Retryable, errors.Join(stopped, failure))
		}
		// Cancellation owns the outcome after the admitted callback has joined;
		// a write interrupted by that cancellation must not replace its cause.
		failure := stopped.Failure
		completion := planner.Completion{Failure: &failure}
		if err := planner.CompleteSession(ctx, p.sessions, identity, completion); err != nil {
			return stageresult.StageContentResult{}, planner.CompletionWriteError(completion, err)
		}
		return stageresult.StageContentResult{}, stopped
	}
	if failure != nil {
		return stageresult.StageContentResult{}, p.fail(ctx, identity, state, failure)
	}
	if callErr != nil {
		var classified *planner.Error
		if !errors.As(callErr, &classified) {
			classified = planner.NewError("planner_execution_failed", "Runtime Planner invocation failed", true, callErr)
		}
		return stageresult.StageContentResult{}, p.fail(ctx, identity, state, classified)
	}
	if _, failure := state.terminal(); failure != nil {
		return stageresult.StageContentResult{}, p.fail(ctx, identity, state, failure)
	}
	if completion.Failure != nil {
		return stageresult.StageContentResult{}, p.fail(ctx, identity, state, planner.NewError(completion.Failure.Code, completion.Failure.Message, completion.Failure.Retryable, nil))
	}
	return stageresult.StageContentResult{}, p.fail(ctx, identity, state, invalidRuntimeOperation())
}

func (h *runtimeHost) startModel(operation, alias string) bool {
	h.modelSpan = planner.InvocationInstrumentation(h.planner.invocation).StartSpan(telemetry.PlannerSpanModel, telemetry.PlannerSpanAttributes{Operation: operation, ModelAlias: alias})
	h.inputCaptured = false
	return telemetry.PlannerSpanCapturesContent(h.modelSpan)
}

func (h *runtimeHost) endModel(outcome, code string, usage *modelUsage) {
	if h.modelSpan == nil {
		return
	}
	attributes := telemetry.PlannerSpanAttributes{ErrorCode: code}
	if usage != nil {
		attributes.ModelUsage = &telemetry.PlannerModelUsage{InputTokens: int64(usage.PromptTokenCount), OutputTokens: int64(usage.CandidatesTokenCount), CachedInputTokens: int64(usage.CachedContentTokenCount)}
	}
	h.modelSpan.End(outcome, attributes)
	h.modelSpan = nil
}

func minTime(a, b time.Time) time.Time {
	if a.Before(b) {
		return a
	}
	return b
}

// The Runtime owns native cost observation; the existing Server Planner report
// remains its single durable contribution. Replay replaces a cumulative snapshot.
func (h *runtimeHost) acceptCost(cost *reporting.LLMCost) bool {
	if cost == nil {
		return true
	}
	if cost.Validate() != nil {
		return false
	}
	h.state.mu.Lock()
	h.state.runtimeCost = cost
	h.state.mu.Unlock()
	return true
}

func (h *runtimeHost) activity() *planner.Activity {
	return &planner.Activity{Kind: "adk_event", Author: h.planner.profile.agentName, FunctionCalls: []string{}, FunctionResults: []string{}}
}
