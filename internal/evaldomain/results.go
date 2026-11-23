package evaldomain

type Collection struct {
	Status string   `json:"status"`
	Gaps   []string `json:"gaps"`
}
type Evidence struct {
	ID       string   `json:"id"`
	Artifact Artifact `json:"artifact"`
	Location string   `json:"location,omitempty"`
}
type ResultInput struct {
	SchemaVersion        string              `json:"schemaVersion"`
	PlanSHA256           string              `json:"planSha256"`
	MemberID             string              `json:"memberId"`
	Source               Source              `json:"source"`
	Execution            ExecutionView       `json:"execution"`
	Collection           Collection          `json:"collection"`
	Outputs              map[string]Artifact `json:"outputs"`
	Evidence             []Evidence          `json:"evidence"`
	Usage                Usage               `json:"usage"`
	PreviousResultSHA256 *string             `json:"previousResultSha256"`
}
type CheckResult struct {
	ID                   string   `json:"id"`
	Evaluator            string   `json:"evaluator"`
	ImplementationSHA256 string   `json:"implementationSha256"`
	Status               string   `json:"status"`
	Reason               string   `json:"reason"`
	EvidenceRefs         []string `json:"evidenceRefs"`
}
type AssessmentSource struct {
	Kind         string `json:"kind"`
	ProducerID   string `json:"producerId,omitempty"`
	RecordSHA256 string `json:"recordSha256,omitempty"`
}
type AssessmentInput struct {
	SchemaVersion            string           `json:"schemaVersion"`
	Source                   AssessmentSource `json:"source"`
	ResultSHA256             string           `json:"resultSha256"`
	Checks                   []CheckResult    `json:"checks"`
	PreviousAssessmentSHA256 *string          `json:"previousAssessmentSha256"`
}
type SelectionEntry struct {
	MemberID         string  `json:"memberId"`
	ResultSHA256     string  `json:"resultSha256"`
	AssessmentSHA256 *string `json:"assessmentSha256"`
}
type SelectionInput struct {
	PlanSHA256 string           `json:"planSha256"`
	Selections []SelectionEntry `json:"selections"`
}
type CheckRequest struct {
	SchemaVersion string           `json:"schemaVersion"`
	Source        AssessmentSource `json:"source"`
	ResultSHA256  string           `json:"resultSha256"`
	CheckIDs      []string         `json:"checkIds"`
}
type Pair struct {
	ID         string     `json:"pairId"`
	SuiteID    string     `json:"suiteId"`
	CaseID     string     `json:"caseId"`
	Sample     int        `json:"sample"`
	A          MemberView `json:"a"`
	B          MemberView `json:"b"`
	Exclusions []string   `json:"exclusions"`
	Regression bool       `json:"regression"`
}

// AssessmentDecision applies only the pinned required checks. An absent record
// is unscored; an attempted assessment with missing checks is incomplete.
// Missing evidence cannot turn a known fail/error into a pass.
func AssessmentDecision(required []Check, assessment *AssessmentInput, collectionComplete bool) string {
	if assessment == nil {
		return "unscored"
	}
	seen := map[string]CheckResult{}
	for _, c := range assessment.Checks {
		if _, exists := seen[c.ID]; exists {
			return "error"
		}
		seen[c.ID] = c
	}
	decision := "pass"
	rank := map[string]int{"pass": 0, "incomplete": 1, "fail": 2, "error": 3}
	apply := func(s string) {
		if rank[s] > rank[decision] {
			decision = s
		}
	}
	if !collectionComplete {
		apply("incomplete")
	}
	n := 0
	for _, c := range required {
		if !c.Required {
			continue
		}
		n++
		r, ok := seen[c.ID]
		if !ok {
			apply("incomplete")
			continue
		}
		if r.Evaluator != c.Evaluator || r.ImplementationSHA256 != c.ImplementationSHA256 {
			apply("error")
			continue
		}
		if r.Status == "not_applicable" {
			if !c.AllowNotApplicable {
				apply("incomplete")
			}
			continue
		}
		if _, ok := rank[r.Status]; !ok {
			apply("error")
		} else {
			apply(r.Status)
		}
	}
	if n == 0 {
		apply("incomplete")
	}
	return decision
}
