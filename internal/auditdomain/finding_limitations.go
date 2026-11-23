package auditdomain

// ValidateFindingLimitations is the admitted finding/coverage gap contract.
// Order is source-owned; repeated gaps and values over512 bytes are invalid.
func ValidateFindingLimitations(values []string) error {
	if values == nil || len(values) > MaximumCoverageValues {
		return invalid(CodeInvalid, "limitations")
	}
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if len(value) > MaximumCoverageValueBytes || validateText(value, "limitations", true) != nil {
			return invalid(CodeInvalid, "limitations")
		}
		if _, duplicate := seen[value]; duplicate {
			return invalid(CodeInvalid, "limitations")
		}
		seen[value] = struct{}{}
	}
	return nil
}
