package streamline

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"google.golang.org/adk/v2/model"
	adksession "google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"
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
	events          adksession.Service
	session         adksession.Session
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
	var event *adksession.Event
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
		projected := &model.LLMRequest{}
		if err := h.planner.prepareGuidance(h.state, projected); err != nil {
			return reply, invalidRuntimeOperation()
		}
		capture := h.startModel("model.generate", h.planner.invocation.ModelAccess.ModelPolicy.Model)
		value = map[string]any{"guidance": h.planner.guidance.Parts[0].Text, "plan": planView(h.planner.plan.Snapshot()), "captureContent": capture}
	case "after_model":
		var args struct {
			Input   int32              `json:"inputTokens" strictjson:"required"`
			Output  int32              `json:"outputTokens" strictjson:"required"`
			Cached  int32              `json:"cachedTokens" strictjson:"required"`
			Calls   []string           `json:"calls" strictjson:"required"`
			Content string             `json:"content,omitempty"`
			Cost    *reporting.LLMCost `json:"cost,omitempty"`
		}
		if strictjson.DecodeNonNull(request.Arguments, &args) != nil || args.Input < 0 || args.Output < 0 || args.Cached < 0 || len(args.Calls) > contracts.MaxPlannerTools || h.modelPhase != "main" || !h.acceptCost(args.Cost) {
			return reply, invalidRuntimeOperation()
		}
		response := &model.LLMResponse{UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: args.Input, CandidatesTokenCount: args.Output, CachedContentTokenCount: args.Cached}, Content: &genai.Content{Role: "model"}}
		for _, name := range args.Calls {
			response.Content.Parts = append(response.Content.Parts, &genai.Part{FunctionCall: &genai.FunctionCall{Name: name}})
		}
		replacement, err := h.state.afterModel(response, h.allowed)
		if len(args.Content) <= telemetry.MaxContentBytes && len(args.Content) > 0 {
			telemetry.CapturePlannerOutput(h.modelSpan, func() any { return json.RawMessage(args.Content) })
		}
		if err != nil {
			failure := planner.FailureFrom(err)
			h.endModel("failed", failure.Code, response.UsageMetadata)
			return reply, planner.NewError(failure.Code, failure.Message, failure.Retryable, err)
		}
		h.modelPhase = ""
		if h.planner.finishRecovery && !singleFinish(response) {
			h.endModel("failed", "planner_tool_selection_invalid", response.UsageMetadata)
			return reply, planner.NewError("planner_tool_selection_invalid", "Document recovery requires exactly one finish", true, nil)
		}
		h.endModel("succeeded", "", response.UsageMetadata)
		if replacement == nil && len(args.Calls) == 1 {
			h.nextTool = args.Calls[0]
		}
		value = map[string]bool{"rejected": replacement != nil}
		event = adksession.NewEvent(ctx, h.identity.InvocationID)
		event.LLMResponse = *response
		if replacement != nil {
			event.LLMResponse = *replacement
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
		var usage *genai.GenerateContentResponseUsageMetadata
		if args.Usage != nil {
			usage = &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: args.Usage.Input, CandidatesTokenCount: args.Usage.Output}
		}
		switch args.Kind {
		case "invalid_response":
			h.endModel("failed", "planner_gateway_invalid_response", usage)
			rejected := &gatewayResponseRejected{}
			if args.Usage != nil {
				rejected.usage = &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: args.Usage.Input, CandidatesTokenCount: args.Usage.Output}
			}
			if h.modelPhase == "main" && args.Usage != nil && (args.EmptyStop || args.OutputLimited) && h.planner.readyForDocumentRecovery() {
				if failure := h.state.chargeUsage(rejected.usage); failure != nil {
					return reply, failure
				}
				h.modelPhase, h.planner.finishRecovery = "", true
				reply.Result = json.RawMessage(`{"recoverFinish":true}`)
				return reply, nil
			}
			return reply, h.state.rejectedGatewayResponse(rejected)
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
			var args workerCallArgs
			if strictjson.DecodeNonNull(request.Arguments, &args) != nil {
				return reply, invalidRuntimeOperation()
			}
			value, reply.Stop = p.callWorker(ctx, state, identity, p.workers[0], args.SubtaskID)
		case finishToolName:
			var args finishArgs
			if strictjson.DecodeNonNull(request.Arguments, &args) != nil {
				return reply, invalidRuntimeOperation()
			}
			result := p.finish(ctx, state, identity, args)
			value, reply.Stop, reply.Escalate = result, result.Accepted, result.Accepted
		case readContextHistoryToolName:
			var args historyArgs
			if strictjson.DecodeNonNull(request.Arguments, &args) != nil {
				return reply, invalidRuntimeOperation()
			}
			value = p.readHistory(state, identity.SessionID, p.workers[0].logicalName, args.Cursor)
		default:
			var args map[string]any
			if strictjson.DecodeNonNull(request.Arguments, &args) != nil {
				return reply, invalidRuntimeOperation()
			}
			if bindings := p.memoryBindings(request.Name); len(bindings) > 0 {
				value, _ = (&plannerMemoryTool{planner: p, state: state, operation: request.Name, bindings: bindings}).run(ctx, args)
			} else {
				value, _ = (&plannerStateTool{planner: p, state: state, operation: request.Name, bindings: p.stateBindings(request.Name)}).run(ctx, args)
			}
		}
		event = adksession.NewEvent(ctx, h.identity.InvocationID)
		event.Content = &genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{Name: request.Name, Response: map[string]any{}}}}}
	}
	if event != nil {
		event.Author = h.planner.profile.agentName
		event.Actions.SkipSummarization = reply.Stop
		event.Actions.Escalate = reply.Escalate
		if err := h.events.AppendEvent(ctx, h.session, event); err != nil {
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

func (p *streamlinePlanner) runRuntime(ctx context.Context, identity planner.SessionIdentity, state *executionState, tools []tool.Tool, allowed map[string]struct{}) (stageresult.StageContentResult, error) {
	deadline := minTime(p.invocation.Deadline, time.Now().Add(p.limits.MaxWallTime))
	p.deadline = deadline
	runContext, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	names := make([]string, 0, len(allowed))
	for name := range allowed {
		names = append(names, name)
	}
	sort.Strings(names)
	events, err := p.adkSessions.NewADKSession(runContext, identity, plannersession.ADKOptions{AppName: p.profile.adkAppName, UserID: p.invocation.StageExecutionID, AllowedTools: names})
	if err != nil {
		return stageresult.StageContentResult{}, p.fail(ctx, identity, state, planner.SessionError("open Runtime Planner events", err))
	}
	session, err := events.Create(runContext, &adksession.CreateRequest{AppName: p.profile.adkAppName, UserID: p.invocation.StageExecutionID, SessionID: identity.SessionID})
	if err != nil {
		return stageresult.StageContentResult{}, p.fail(ctx, identity, state, planner.SessionError("create Runtime Planner events", err))
	}
	host := &runtimeHost{ctx: runContext, planner: p, state: state, identity: identity, allowed: allowed, events: events, session: session.Session}
	if _, exists := p.runtimeHosts.LoadOrStore(p.invocation.StageExecutionID, host); exists {
		return stageresult.StageContentResult{}, invalidRuntimeOperation()
	}
	defer p.runtimeHosts.Delete(p.invocation.StageExecutionID)
	message, err := p.initialMessage()
	if err != nil {
		return stageresult.StageContentResult{}, invalidRuntimeOperation()
	}
	// The canonical Stage context is already in Input. Do not duplicate its
	// parameters and exact refs in the outer execution envelope.
	request := contracts.StageContentRequest{
		APIVersion: contracts.APIVersion, SubtaskID: "0", Objective: "Execute the prepared Stage Planner",
		Instructions: p.systemInstruction(), Deadline: deadline,
		Parameters: map[string]string{}, Artifacts: map[string]contracts.ArtifactRef{},
	}
	request.Planner = &contracts.PlannerInvocation{Input: message.Parts[0].Text, Tools: []contracts.PlannerTool{}}
	if c := p.compactionSettings; c != nil {
		request.Planner.Compaction = &contracts.PlannerCompaction{Instructions: c.instructions, Threshold: c.threshold, KeepRecentGroups: c.keepRecent, MaxCompactions: c.maxCompactions, Policy: c.compactor.Clone()}
	}
	for _, current := range tools {
		declaring, ok := current.(interface {
			Declaration() *genai.FunctionDeclaration
		})
		if !ok {
			return stageresult.StageContentResult{}, fmt.Errorf("Planner tool has no declaration")
		}
		declaration := declaring.Declaration()
		parameters, err := json.Marshal(declaration.ParametersJsonSchema)
		if err != nil {
			return stageresult.StageContentResult{}, err
		}
		request.Planner.Tools = append(request.Planner.Tools, contracts.PlannerTool{Name: declaration.Name, Description: declaration.Description, Parameters: parameters})
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
		return stageresult.StageContentResult{}, p.fail(ctx, identity, state, planner.NewError("planner_execution_failed", "Runtime Planner invocation failed", true, callErr))
	}
	if result, failure := state.terminal(); result != nil {
		outcome := planner.Completion{Result: result}
		if err := planner.CompleteSession(ctx, p.sessions, identity, outcome); err != nil {
			return stageresult.StageContentResult{}, planner.CompletionWriteError(outcome, err)
		}
		return result.Clone(), nil
	} else if failure != nil {
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

func (h *runtimeHost) endModel(outcome, code string, usage *genai.GenerateContentResponseUsageMetadata) {
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
