package config

import "github.com/grauwolf32/contractor/internal/contracts"

// ValidatePlannerModelPolicy checks the budgets a Planner model policy must
// carry. A Planner runs in the Server, so its policy never reaches the
// Runtime; the wire checks of a Worker policy are
// contracts.ResolvedModelPolicy.ValidateForWorker.
func ValidatePlannerModelPolicy(policy contracts.ResolvedModelPolicy) error {
	if err := policy.Validate(); err != nil {
		return err
	}
	if policy.MaxOutputTokens <= 0 {
		return contracts.Invalidf("Planner modelPolicy requires maxOutputTokens")
	}
	if policy.MaxModelCalls <= 0 {
		return contracts.Invalidf("Planner modelPolicy requires maxModelCalls")
	}
	if policy.MaxWorkerCalls <= 0 {
		return contracts.Invalidf("Planner modelPolicy requires maxWorkerCalls")
	}
	if policy.MaxTotalTokens <= 0 {
		return contracts.Invalidf("Planner modelPolicy requires maxTotalTokens")
	}
	if policy.MaxToolCalls != 0 {
		return contracts.Invalidf("Planner modelPolicy must omit maxToolCalls")
	}
	return nil
}
