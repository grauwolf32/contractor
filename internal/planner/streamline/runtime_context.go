package streamline

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/telemetry"
	adksession "google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

func (h *runtimeHost) checkRuntimeRetained(retained int64) *planner.Error {
	if retained < 0 {
		return invalidRuntimeOperation()
	}
	if c := h.planner.compaction; c != nil {
		if retained+c.retainedLocked() > maxCompactionRetainedBytes {
			return compactionLimit("retained")
		}
	} else if retained != 0 {
		return invalidRuntimeOperation()
	}
	return nil
}

// Conversation selection and model execution stay in Runtime. Server reuses
// compactor acceptance, budget, exact-result history and the required marker write.
func (h *runtimeHost) contextOperation(ctx context.Context, request contracts.PlannerOperation) (any, *adksession.Event, *planner.Error) {
	c := h.planner.compaction
	if c == nil || h.nextTool != "" {
		return nil, nil, invalidRuntimeOperation()
	}
	switch request.Name {
	case "context":
		var args struct{}
		if strictjson.DecodeNonNull(request.Arguments, &args) != nil || h.modelPhase != "" || h.checkpointReady {
			return nil, nil, invalidRuntimeOperation()
		}
		return map[string]any{"plan": planView(h.planner.plan.Snapshot()), "retainedBytes": c.retainedLocked()}, nil, nil
	case "before_compactor":
		var args struct {
			Retained int64 `json:"retainedBytes" strictjson:"required"`
			Input    int64 `json:"inputBytes" strictjson:"required"`
			Messages int   `json:"messages" strictjson:"required"`
		}
		if strictjson.DecodeNonNull(request.Arguments, &args) != nil || h.modelPhase != "" || h.checkpointReady || args.Input < 0 || args.Messages < 1 || args.Messages > maxCompactionGroups {
			return nil, nil, invalidRuntimeOperation()
		}
		if failure := h.checkRuntimeRetained(args.Retained); failure != nil {
			return nil, nil, failure
		}
		if c.generation >= c.settings.maxCompactions {
			return nil, nil, compactionLimit("cycle")
		}
		policy := c.settings.compactor
		framing := int64(compactorFramingTokens)
		if args.Messages > compactorFramingMessages {
			framing += int64(compactorTokensPerMessage * args.Messages)
		}
		required := args.Input + framing + int64(policy.MaxOutputTokens)
		if args.Input > maxGatewayRequestBytes || required > int64(policy.ContextWindowTokens) || required > int64(policy.MaxTotalTokens) {
			return nil, nil, compactionFailed(compactionReasonInputCapacity)
		}
		if args.Retained+c.retainedLocked()+2*args.Input > maxCompactionPeakBytes {
			return nil, nil, compactionLimit("peak")
		}
		if failure := h.state.beforeModel(); failure != nil {
			return nil, nil, failure
		}
		h.modelPhase = "compaction"
		return map[string]any{"captureContent": h.startModel("model.compact", policy.Model)}, nil, nil
	case "after_compactor":
		var args struct {
			Input   int32              `json:"inputTokens" strictjson:"required"`
			Output  int32              `json:"outputTokens" strictjson:"required"`
			Cached  int32              `json:"cachedTokens" strictjson:"required"`
			Text    string             `json:"text" strictjson:"required"`
			Finish  string             `json:"finishReason" strictjson:"required"`
			Tools   int                `json:"toolCalls" strictjson:"required"`
			Source  int64              `json:"sourceBytes" strictjson:"required"`
			Content string             `json:"content,omitempty"`
			Cost    *reporting.LLMCost `json:"cost,omitempty"`
		}
		if strictjson.DecodeNonNull(request.Arguments, &args) != nil || h.modelPhase != "compaction" || args.Input < 0 || args.Output < 0 || args.Cached < 0 || args.Tools < 0 || args.Tools > contracts.MaxPlannerTools || args.Source < 1 || args.Source > maxCompactionRetainedBytes || !h.acceptCost(args.Cost) {
			return nil, nil, invalidRuntimeOperation()
		}
		usage := &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: args.Input, CandidatesTokenCount: args.Output, CachedContentTokenCount: args.Cached}
		if len(args.Content) <= telemetry.MaxContentBytes && len(args.Content) > 0 {
			telemetry.CapturePlannerOutput(h.modelSpan, func() any { return json.RawMessage(args.Content) })
		}
		if failure := h.state.chargeUsage(usage); failure != nil {
			h.endModel("failed", failure.Code, usage)
			return nil, nil, failure
		}
		usage.TotalTokenCount = args.Input + args.Output
		summary, failure := validateCompactorReply(compactorReply{text: args.Text, toolCalls: args.Tools, finishReason: args.Finish, usage: usage}, c.settings.compactor, args.Source)
		if failure != nil {
			h.endModel("failed", failure.Code, usage)
			return nil, nil, failure
		}
		h.endModel("succeeded", "", usage)
		h.modelPhase, h.checkpointReady = "", true
		return map[string]string{"summary": summary}, nil, nil
	case "checkpoint":
		var args struct {
			Generation int   `json:"generation" strictjson:"required"`
			Retained   int64 `json:"retainedBytes" strictjson:"required"`
		}
		if strictjson.DecodeNonNull(request.Arguments, &args) != nil || h.modelPhase != "" || !h.checkpointReady || args.Generation != c.generation+1 || h.planner.plan.Snapshot().ActiveDispatch != nil {
			return nil, nil, invalidRuntimeOperation()
		}
		if failure := h.checkRuntimeRetained(args.Retained); failure != nil {
			return nil, nil, failure
		}
		marker := adksession.NewEvent(ctx, "planner-context-checkpoint")
		marker.Author = h.planner.profile.agentName
		if err := h.events.AppendEvent(ctx, h.session, marker); err != nil {
			return nil, nil, planner.SessionError("record Runtime context checkpoint", err)
		}
		c.generation++
		c.history.invalidate()
		h.checkpointReady = false
		return map[string]any{}, nil, nil
	}
	return nil, nil, invalidRuntimeOperation()
}
