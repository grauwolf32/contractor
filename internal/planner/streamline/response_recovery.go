package streamline

import "github.com/grauwolf32/contractor/internal/stageresult"

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
