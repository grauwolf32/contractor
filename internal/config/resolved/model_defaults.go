package resolved

import (
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// ApplyStageModelDefaults materializes the captured deployment choices for
// canonical role policies. It also runs after a transition override so a
// Passthrough-to-Streamline continuation gets the pinned Planner choice.
// Explicit model-specific policies and explicit Run selections are preserved.
func ApplyStageModelDefaults(stage *ResolvedStage) {
	replace := func(selection *ResolvedConsumerExecutionConfig, role string) {
		policy, ok := stage.ModelDefaults[role]
		if !ok || strings.HasPrefix(selection.Origins.ModelPolicy, "run.") {
			return
		}
		if selection.ModelPolicy.Ref.PolicyID == role && selection.ModelPolicy.Ref.Version == policy.Ref.Version {
			selection.ModelPolicy = policy.Clone()
			selection.Origins.ModelPolicy = "server.modelSelection." + role
		}
	}
	if stage.ExecutionConfig.Planner != nil {
		replace(stage.ExecutionConfig.Planner, "planner")
	}
	for name, binding := range stage.Agents {
		if binding.Template.IsToolWorker() {
			continue
		}
		selection := stage.ExecutionConfig.Agents[name]
		replace(&selection, "worker")
		replace(&selection, "audit_completion_worker")
		if binding.Template.Summarizer != nil && selection.SummarizerModelPolicy == nil {
			source := binding.Template.Summarizer.ModelPolicy.Ref
			if policy, ok := stage.ModelDefaults["summarizer"]; ok && source.PolicyID == "summarizer" && source.Version == policy.Ref.Version {
				copy := policy.Clone()
				selection.SummarizerModelPolicy = &copy
				selection.Origins.SummarizerModelPolicy = "server.modelSelection.summarizer"
			}
		}
		stage.ExecutionConfig.Agents[name] = selection
	}
}

func validateModelDefaults(defaults map[string]contracts.ResolvedModelPolicy) error {
	if len(defaults) > 4 {
		return fmt.Errorf("too many captured role model defaults")
	}
	for role, policy := range defaults {
		var err error
		switch role {
		case "planner":
			err = ValidatePlannerModelPolicy(policy)
		case "summarizer":
			err = policy.ValidateForWorkerSummarizer()
		case "worker", "audit_completion_worker":
			err = policy.ValidateForWorker(true)
		default:
			return fmt.Errorf("unknown captured model role")
		}
		if err != nil {
			return err
		}
	}
	return nil
}
