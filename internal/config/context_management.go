package config

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// resolveContextManagement normalizes one binding's authored selection
// (WFL-R91, WFL-R92). Omission selects template@1; compact@1 requires an
// explicit includeReasoning and instructions refs, and
// its other members take the documented defaults. Stage and execution
// compatibility is checked with the resolved Stage (WFL-R93).
func (l *loader) resolveContextManagement(source *contextManagementSource) (contracts.WorkerContextManagement, error) {
	if source == nil {
		return contracts.WorkerContextManagement{Strategy: contracts.ContextStrategyTemplate}, nil
	}
	switch contracts.WorkerContextStrategy(source.Strategy) {
	case contracts.ContextStrategyTemplate:
		if source.Instructions != nil || source.IncludeReasoning.Kind != 0 ||
			source.ContextWindowRatio != nil || source.KeepRecentGroups != nil || source.MaxCompactions != nil {
			return contracts.WorkerContextManagement{}, fmt.Errorf("template@1 accepts only strategy")
		}
		return contracts.WorkerContextManagement{Strategy: contracts.ContextStrategyTemplate}, nil
	case contracts.ContextStrategyCompact:
	default:
		return contracts.WorkerContextManagement{}, fmt.Errorf("strategy must be template@1 or compact@1")
	}
	// includeReasoning has no default: only an explicit YAML boolean selects
	// whether provider reasoning reaches the compactor.
	var includeReasoning bool
	if source.IncludeReasoning.Tag != "!!bool" || source.IncludeReasoning.Decode(&includeReasoning) != nil {
		return contracts.WorkerContextManagement{}, fmt.Errorf("compact@1 requires an explicit boolean includeReasoning")
	}
	if source.Instructions == nil {
		return contracts.WorkerContextManagement{}, fmt.Errorf("compact@1 requires instructions")
	}
	instructions, err := l.resolveInstructions(source.Instructions)
	if err != nil {
		return contracts.WorkerContextManagement{}, fmt.Errorf("instructions: %w", err)
	}
	compaction := contracts.WorkerCompaction{
		Instructions:       instructions,
		IncludeReasoning:   includeReasoning,
		ContextWindowRatio: contracts.DefaultWorkerCompactionContextRatio,
		KeepRecentGroups:   contracts.DefaultWorkerCompactionRecentGroups,
		MaxCompactions:     contracts.DefaultWorkerCompactions,
	}
	if source.ContextWindowRatio != nil {
		compaction.ContextWindowRatio = *source.ContextWindowRatio
	}
	if source.KeepRecentGroups != nil {
		compaction.KeepRecentGroups = *source.KeepRecentGroups
	}
	if source.MaxCompactions != nil {
		compaction.MaxCompactions = *source.MaxCompactions
	}
	result := contracts.WorkerContextManagement{Strategy: contracts.ContextStrategyCompact, WorkerCompaction: &compaction}
	if err := result.ValidateBinding(); err != nil {
		return contracts.WorkerContextManagement{}, err
	}
	return result, nil
}
