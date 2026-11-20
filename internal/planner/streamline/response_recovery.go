package streamline

import (
	"errors"

	"github.com/grauwolf32/contractor/internal/stageresult"
	"google.golang.org/adk/v2/model"
)

var errDocumentResponseRecovery = errors.New("document response recovery")

func (p *streamlinePlanner) readyForDocumentRecovery() bool {
	if p.finishRecovery || p.plan.CanFinish(stageresult.StageSucceeded) != nil {
		return false
	}
	for _, binding := range p.invocation.Stage.Agents {
		if binding.Template.ResponseRecovery != "document@1" {
			return false
		}
	}
	for name, slot := range p.resultContract {
		if slot.Required && !p.recoveryArtifacts[name] {
			return false
		}
	}
	return len(p.invocation.Stage.Agents) > 0
}

func singleFinish(response *model.LLMResponse) bool {
	if response == nil || response.Content == nil {
		return false
	}
	count := 0
	for _, part := range response.Content.Parts {
		if part != nil && part.FunctionCall != nil {
			if part.FunctionCall.Name != finishToolName {
				return false
			}
			count++
		}
	}
	return count == 1
}
