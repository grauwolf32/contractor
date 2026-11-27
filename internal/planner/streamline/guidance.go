package streamline

import (
	"encoding/json"
	"fmt"

	"github.com/grauwolf32/contractor/internal/stageresult"
)

// LIM-WRK-39; cross-language request-guidance.json pins both consumers.
const maxGuidanceBytes = 2048
const guidancePreamble = "Current execution facts (authoritative for this request):\n"

func (p *streamlinePlanner) prepareGuidance(state *executionState) (string, error) {
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
		return "", err
	}
	text := guidancePreamble + string(encoded)
	if len(text) > maxGuidanceBytes {
		return "", fmt.Errorf("Planner guidance exceeds its byte limit")
	}
	return text, nil
}
