package auditdomain

const MaximumAcceptedPlanBytes = 64 << 10

type CheckPlan struct {
	Schema    string `json:"schema"`
	Method    string `json:"method"`
	Phase     string `json:"phase"`
	Objective string `json:"objective"`
}

// BuildFindingCheckPlan projects one already validated source ordinal. The
// caller still proves ownership and accepted source membership before admission.
func BuildFindingCheckPlan(document FindingProposal, ordinal int) (CheckPlan, error) {
	if ordinal < 0 || ordinal >= len(document.ProposedChecks) {
		return CheckPlan{}, invalid(CodeInvalid, "plan.ordinal")
	}
	check := document.ProposedChecks[ordinal]
	key, ok := LookupCheckPlanMethod(FindingCheckPlanSchema, check.Method)
	if !ok {
		return CheckPlan{}, invalid(CodeInvalid, "plan.method")
	}
	plan := CheckPlan{Schema: key.Schema, Method: key.Method, Phase: key.Phase, Objective: check.Objective}
	return plan, ValidateCheckPlan(plan)
}

func ValidateCheckPlan(plan CheckPlan) error {
	key, ok := LookupCheckPlanMethod(plan.Schema, plan.Method)
	if !ok || key.Phase != plan.Phase || validateText(plan.Objective, "plan.objective", true) != nil {
		return invalid(CodeInvalid, "plan")
	}
	return nil
}

// EncodeCheckPlan uses RFC 8785 bytes, including for a plan too large to admit.
func EncodeCheckPlan(plan CheckPlan) ([]byte, error) {
	return encodeDocument(plan, ValidateCheckPlan)
}
