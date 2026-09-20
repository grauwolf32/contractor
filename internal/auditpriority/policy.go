package auditpriority

// ResolveTopN pins one effective value. Nil means omitted; zero is not omission.
// It deliberately does not reduce topN to available candidates or execution
// capacity. The selector handles all-if-fewer; admission must check full capacity.
func ResolveTopN(policy Policy, override *int) (int, error) {
	count, ceiling := DefaultTopN, MaxTopN
	if policy.DefaultTopN != nil {
		count = *policy.DefaultTopN
	}
	if policy.MaxTopN != nil {
		ceiling = *policy.MaxTopN
	}
	if !validTopN(count) || !validTopN(ceiling) || count > ceiling {
		return 0, invalid(CodeInvalidPolicy)
	}
	if override != nil {
		if !validTopN(*override) || *override > ceiling {
			return 0, invalid(CodeInvalidPolicy)
		}
		count = *override
	}
	return count, nil
}

func validTopN(value int) bool { return value >= MinTopN && value <= MaxTopN }

func (c CycleBinding) Validate() error {
	if !validIdentifier(c.CycleID) || !validTopN(c.TopN) {
		return invalid(CodeInvalidBinding)
	}
	for _, digest := range []string{c.InventoryDigest, c.PoolDigest, c.ContextDigest, c.PolicyDigest, c.PromptDigest, c.ModelConfigDigest} {
		if !digestPattern.MatchString(digest) {
			return invalid(CodeInvalidBinding)
		}
	}
	return nil
}
