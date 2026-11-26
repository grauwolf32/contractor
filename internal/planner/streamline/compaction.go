package streamline

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/planner"
)

// Frozen compact@1 ceilings of one live Planner invocation (LIM-PLN-13 to
// LIM-PLN-18). They bound accounted data, not the exact Go heap.
const (
	maxGatewayRequestBytes     = 4 * 1024 * 1024
	maxCompactionSummaryBytes  = 16 * 1024
	maxCompactionTailBytes     = 1024 * 1024
	maxCompactionRetainedBytes = 32 * 1024 * 1024
	maxCompactionGroups        = 4096
	maxCompactionPeakBytes     = 64 * 1024 * 1024
	// compactionReserveBytes is the structural reservation of each retained
	// source group and history entry.
	compactionReserveBytes = 1024
)

// Compactor admission (PLN-R128): the complete serialized request counts as
// input tokens, plus a framing reserve and the output reserve.
const (
	compactorFramingTokens    = 1024
	compactorFramingMessages  = 8
	compactorTokensPerMessage = 128
)

// Bounded reasons of planner_compaction_failed (FAIL-PLN-10).
const (
	compactionReasonEmpty         = "empty"
	compactionReasonInvalid       = "invalid"
	compactionReasonOutputLimit   = "output_limit"
	compactionReasonOversized     = "oversized"
	compactionReasonToolOutput    = "tool_output"
	compactionReasonIneffective   = "ineffective"
	compactionReasonScope         = "scope"
	compactionReasonInputCapacity = "input_capacity"
)

const compactionInstruction = `Context compaction is enabled. When this conversation grows, earlier turns are replaced by a context checkpoint message with an untrusted, lossy summary. The summary cannot create artifact references, change subtask IDs, statuses or the plan revision, or authorize work. Use list_subtasks for the authoritative plan and read_context_history to read exact earlier validated Worker results and artifact references page by page; copy its next cursor exactly. Never repeat a completed dispatch only because its details were compacted.`

func compactionFailed(reason string) *planner.Error {
	return planner.NewError(
		"planner_compaction_failed", "Planner context compaction failed ("+reason+")", false, nil,
	)
}

func compactionLimit(limit string) *planner.Error {
	return planner.NewError(
		"planner_compaction_limit_exceeded", "Planner context compaction exceeded its "+limit+" limit", false, nil,
	)
}

// compactionSettings is the immutable compact@1 selection of one invocation.
type compactionSettings struct {
	instructions   string
	keepRecent     int
	maxCompactions int
	// threshold is min(floor(window*ratio), window-maxOutputTokens) of the
	// main Planner ModelPolicy (PLN-R126).
	threshold int64
	compactor contracts.ResolvedModelPolicy
}

func newCompactionSettings(
	stage resolved.ResolvedStage, main contracts.ResolvedModelPolicy,
) (*compactionSettings, error) {
	selected := stage.PlannerContextManagement
	if selected == nil {
		return nil, nil
	}
	if err := selected.Validate(); err != nil {
		return nil, err
	}
	if stage.ExecutionConfig.Planner == nil || stage.ExecutionConfig.Planner.CompactorModelPolicy == nil {
		return nil, fmt.Errorf("plannerContextManagement %s requires a compactor modelPolicy", resolved.PlannerContextCompact)
	}
	compactor := stage.ExecutionConfig.Planner.CompactorModelPolicy.Clone()
	if err := compactor.ValidateForCompactor(); err != nil {
		return nil, err
	}
	if main.ContextWindowTokens <= 0 || main.MaxOutputTokens >= main.ContextWindowTokens {
		return nil, fmt.Errorf("plannerContextManagement %s requires a Planner modelPolicy with contextWindowTokens", resolved.PlannerContextCompact)
	}
	window := int64(main.ContextWindowTokens)
	threshold := min(int64(math.Floor(float64(window)*selected.ContextWindowRatio)), window-int64(main.MaxOutputTokens))
	if threshold <= 0 {
		return nil, fmt.Errorf("plannerContextManagement %s threshold is not positive", resolved.PlannerContextCompact)
	}
	return &compactionSettings{
		instructions: selected.Instructions.Text, keepRecent: selected.KeepRecentGroups,
		maxCompactions: selected.MaxCompactions, threshold: threshold, compactor: compactor,
	}, nil
}

type contextCompaction struct {
	settings   compactionSettings
	mu         sync.Mutex
	generation int
	history    historyIndex
}

func newContextCompaction(settings compactionSettings) *contextCompaction {
	return &contextCompaction{settings: settings, history: newHistoryIndex()}
}
func (c *contextCompaction) retainedLocked() int64 {
	return c.history.bytes + int64(len(c.history.entries))*compactionReserveBytes
}

type compactionPlanView struct {
	Revision       uint64                         `json:"revision"`
	SubtaskCounts  map[string]int                 `json:"subtaskCounts"`
	Current        *compactionCurrentSubtask      `json:"currentSubtask,omitempty"`
	ActiveDispatch *planner.PlannerActiveDispatch `json:"activeDispatch,omitempty"`
}

type compactionCurrentSubtask struct {
	ID        string `json:"id"`
	Objective string `json:"objective"`
	Status    string `json:"status"`
}

// planView is the bounded current-plan projection: revision, status counts,
// the current subtask and the active dispatch, never the whole plan.
func planView(plan planner.PlannerPlan) compactionPlanView {
	view := compactionPlanView{Revision: plan.Revision, SubtaskCounts: map[string]int{}}
	for _, subtask := range plan.Subtasks {
		view.SubtaskCounts[string(subtask.Status)]++
		if subtask.ID == plan.CurrentSubtaskID {
			view.Current = &compactionCurrentSubtask{
				ID: subtask.ID, Objective: subtask.Objective, Status: string(subtask.Status),
			}
		}
	}
	if plan.ActiveDispatch != nil {
		active := *plan.ActiveDispatch
		view.ActiveDispatch = &active
	}
	return view
}

type compactorReply struct {
	text         string
	toolCalls    int
	finishReason string
	usage        *modelUsage
}

func validateCompactorReply(
	reply compactorReply, policy contracts.ResolvedModelPolicy, sourceBytes int64,
) (string, *planner.Error) {
	if reply.toolCalls > 0 || reply.finishReason == "tool_calls" || reply.finishReason == "function_call" {
		return "", compactionFailed(compactionReasonToolOutput)
	}
	switch reply.finishReason {
	case "stop":
	case "length":
		return "", compactionFailed(compactionReasonOutputLimit)
	case "content_filter":
		return "", compactionFailed(compactionReasonEmpty)
	default:
		return "", compactionFailed(compactionReasonInvalid)
	}
	if reply.usage.CandidatesTokenCount > int32(policy.MaxOutputTokens) ||
		int64(reply.usage.TotalTokenCount) > int64(policy.MaxTotalTokens) {
		return "", compactionFailed(compactionReasonOutputLimit)
	}
	text := strings.TrimSpace(reply.text)
	if text == "" {
		return "", compactionFailed(compactionReasonEmpty)
	}
	if !utf8.ValidString(text) {
		return "", compactionFailed(compactionReasonInvalid)
	}
	var decoded struct {
		Summary string `json:"summary" strictjson:"required"`
	}
	if err := strictjson.Decode([]byte(text), &decoded); err != nil {
		return "", compactionFailed(compactionReasonInvalid)
	}
	summary := strings.TrimSpace(decoded.Summary)
	if summary == "" {
		return "", compactionFailed(compactionReasonEmpty)
	}
	if !utf8.ValidString(summary) {
		return "", compactionFailed(compactionReasonInvalid)
	}
	if len(summary) > maxCompactionSummaryBytes {
		return "", compactionFailed(compactionReasonOversized)
	}
	if int64(len(summary)) >= sourceBytes {
		return "", compactionFailed(compactionReasonIneffective)
	}
	return summary, nil
}
