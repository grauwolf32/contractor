package evaldomain

import "time"

type MemberIdentity struct {
	ID            string  `json:"memberId"`
	SuiteID       string  `json:"suiteId"`
	CaseID        string  `json:"caseId"`
	Sample        int     `json:"sample"`
	VariantID     string  `json:"variantId"`
	CaseSHA256    string  `json:"caseSha256"`
	BindingSHA256 string  `json:"bindingSha256"`
	Eligibility   string  `json:"eligibility"`
	Reason        *string `json:"reason"`
}

type ExecutionView struct {
	Ref        *ExecutionRef `json:"ref"`
	State      string        `json:"state"`
	StartedAt  *time.Time    `json:"startedAt"`
	FinishedAt *time.Time    `json:"finishedAt"`
	Reason     *string       `json:"reason"`
}

type MemberView struct {
	Member           MemberIdentity `json:"member"`
	Execution        *ExecutionView `json:"execution"`
	ResultSHA256     *string        `json:"resultSha256"`
	AssessmentSHA256 *string        `json:"assessmentSha256"`
	Conflicting      bool           `json:"conflicting"`
	Assessment       string         `json:"assessment"`
	Usage            *Usage         `json:"usage"`
}
