package resolved

import "github.com/grauwolf32/contractor/internal/contracts"

// ValidatePlannerModelPolicy uses the common Planner consumer contract.
func ValidatePlannerModelPolicy(policy contracts.ResolvedModelPolicy) error {
	return policy.ValidateForPlanner()
}
