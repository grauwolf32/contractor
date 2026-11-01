package auditdomain

// ValidateFindingAssessment is the common closed current-field shape for
// canonical records and immutable captures. Source conclusions never enter it.
func ValidateFindingAssessment(state *string, verdict, severity, duplicateTarget string) error {
	if state == nil {
		if verdict != "" || severity != "" || duplicateTarget != "" {
			return invalid(CodeInvalid, "finding.assessment")
		}
		return nil
	}
	expected := ""
	switch verdict {
	case "true_positive":
		expected = "confirmed"
	case "false_positive":
		expected = "rejected"
	case "duplicate":
		expected = "duplicate"
	case "reopen":
		expected = "proposed"
	case "needs_evidence":
		expected = "needs-evidence"
	}
	if expected == "" || *state != expected || (verdict == "true_positive") != (severity != "") || !validSeverity(severity) || (verdict == "duplicate") != (duplicateTarget != "") {
		return invalid(CodeInvalid, "finding.assessment")
	}
	return nil
}
