package evaldomain

// These types use the frozen portable format's field names. Managed API DTOs
// retain their separate camelCase contract.
type PortableComparisonPolicy struct {
	Baseline           string   `json:"baseline"`
	Candidate          string   `json:"candidate"`
	RequiredEqual      []string `json:"required_equal"`
	AllowedDifferences []string `json:"allowed_differences"`
}
type PortableComparisonGates struct {
	MinCandidateEndToEndPass float64  `json:"min_candidate_end_to_end_pass"`
	MaxQualityDrop           float64  `json:"max_quality_drop"`
	MaxTotalTokensRatio      *float64 `json:"max_total_tokens_ratio,omitempty"`
}
type PortableExperimentExtensions struct {
	ComparisonGates PortableComparisonGates `json:"playground:comparison-gates"`
}
type ExperimentSetup struct {
	Dataset     DatasetRef `json:"dataset"`
	CaseIDs     []string   `json:"caseIds"`
	Repetitions int        `json:"repetitions"`
	Variants    []Variant  `json:"variants"`
	Checks      []Check    `json:"checks"`
	Comparison  Comparison `json:"comparison"`
	Budgets     Budgets    `json:"budgets"`
}

func (d Draft) Setup() ExperimentSetup {
	return ExperimentSetup{
		Dataset:     d.Dataset,
		CaseIDs:     d.CaseIDs,
		Repetitions: d.Repetitions,
		Variants:    d.Variants,
		Checks:      d.Checks,
		Comparison:  d.Comparison,
		Budgets:     d.Budgets,
	}
}
