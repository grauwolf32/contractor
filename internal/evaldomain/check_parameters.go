package evaldomain

// ValidateHumanReviewParameters keeps the registered human-review contract
// separate from producer-owned parameters of other external evaluators.
func (c Check) ValidateHumanReviewParameters() error {
	if c.Evaluator == "human-review@1" && len(c.Parameters) != 0 {
		return Failure("eval_invalid")
	}
	return nil
}
