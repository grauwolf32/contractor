package streamline

import (
	"encoding/json"
	"fmt"

	"github.com/grauwolf32/contractor/internal/stageresult"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// LIM-WRK-39; cross-language request-guidance.json pins both consumers.
const maxGuidanceBytes = 2048
const guidancePreamble = "Current execution facts (authoritative for this request):\n"

func (p *streamlinePlanner) prepareGuidance(state *executionState, request *model.LLMRequest) error {
	plan := p.plan.Snapshot()
	view := planView(plan)
	state.mu.Lock()
	modelRemaining := int64(state.limits.MaxModelCalls) - state.modelCalls
	workerRemaining := int64(state.limits.MaxWorkerCalls) - state.workerCalls
	state.mu.Unlock()
	facts := map[string]any{
		"planRevision": view.Revision, "subtaskCounts": view.SubtaskCounts,
		"usedPlanSlots": len(plan.Subtasks), "maxPlanSlots": p.invocation.Stage.MaxSubtasks.Effective(),
		"freePlanSlots":                     p.invocation.Stage.MaxSubtasks.Effective() - len(plan.Subtasks),
		"additionalModelCallsAfterThisCall": modelRemaining, "remainingWorkerCalls": workerRemaining,
		"successfulFinishPlanBlocker": "none; candidate artifacts still require validation",
	}
	if view.Current != nil {
		facts["currentSubtask"] = map[string]string{"id": view.Current.ID, "status": view.Current.Status}
	}
	if rejection := p.plan.CanFinish(stageresult.StageSucceeded); rejection != nil {
		facts["successfulFinishPlanBlocker"] = rejection.Message
	}
	encoded, err := json.Marshal(facts)
	if err != nil {
		return err
	}
	text := guidancePreamble + string(encoded)
	if len(text) > maxGuidanceBytes {
		return fmt.Errorf("Planner guidance exceeds its byte limit")
	}
	// Remove only our own request object, never text supplied by a user/tool.
	contents := make([]*genai.Content, 0, len(request.Contents)+1)
	for _, content := range request.Contents {
		if content != p.guidance {
			contents = append(contents, content)
		}
	}
	p.guidance = genai.NewContentFromText(text, genai.RoleUser)
	request.Contents = append(contents, p.guidance)
	return nil
}
