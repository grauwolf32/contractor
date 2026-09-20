package evaldomain

import "github.com/grauwolf32/contractor/internal/config"

type Source struct {
	System       string  `json:"system"`
	ID           string  `json:"id"`
	Revision     *string `json:"revision"`
	SourceSHA256 *string `json:"sourceSha256"`
}

type Artifact struct {
	Scope     string `json:"scope"`
	ScopeID   string `json:"scopeId"`
	Namespace string `json:"namespace"`
	Name      string `json:"name"`
	Revision  string `json:"revision"`
	SHA256    string `json:"sha256"`
	MediaType string `json:"mediaType"`
	SizeBytes int64  `json:"sizeBytes"`
}

type Task struct {
	Kind       string            `json:"kind"`
	Objective  string            `json:"objective"`
	Parameters map[string]string `json:"parameters"`
}

type Output struct {
	MediaTypes []string `json:"mediaTypes"`
	Required   bool     `json:"required"`
}

type Case struct {
	ID       string              `json:"id"`
	Task     Task                `json:"task"`
	Inputs   map[string]Artifact `json:"inputs"`
	Requires []string            `json:"requires"`
	Outputs  map[string]Output   `json:"outputs"`
	Source   *Source             `json:"source,omitempty"`
}

type PrivateCheck struct {
	ID       string            `json:"id"`
	Revision string            `json:"revision"`
	Rubric   string            `json:"rubric"`
	Expected map[string]string `json:"expected"`
	Schema   string            `json:"schema,omitempty"`
}

type DatasetInput struct {
	DatasetID     string         `json:"datasetId"`
	Name          string         `json:"name"`
	Source        *Source        `json:"source,omitempty"`
	Cases         []Case         `json:"cases"`
	PrivateChecks []PrivateCheck `json:"privateChecks"`
}

type Dataset struct {
	DatasetID     string  `json:"datasetId"`
	Name          string  `json:"name"`
	ProjectID     string  `json:"projectId"`
	Revision      string  `json:"revision"`
	VisibleSHA256 string  `json:"visibleSha256"`
	CaseCount     int     `json:"caseCount"`
	Source        *Source `json:"source,omitempty"`
}

type DatasetRef struct {
	ID       string `json:"id"`
	Revision string `json:"revision"`
}

type Variant struct {
	ID              string                      `json:"id"`
	Kind            string                      `json:"kind"`
	Selector        string                      `json:"selector"`
	ExecutionConfig config.ExecutionConfigPatch `json:"executionConfig"`
	RuntimeLabels   []string                    `json:"runtimeLabels,omitempty"`
	InputMapping    map[string]string           `json:"inputMapping,omitempty"`
	OutputMapping   map[string]string           `json:"outputMapping,omitempty"`
	Parameters      map[string]string           `json:"parameters,omitempty"`
}

type Check struct {
	ID                   string            `json:"id"`
	Evaluator            string            `json:"evaluator"`
	Required             bool              `json:"required"`
	RubricRevision       string            `json:"rubricRevision,omitempty"`
	ImplementationSHA256 string            `json:"implementationSha256,omitempty"`
	Parameters           map[string]string `json:"parameters,omitempty"`
	AllowNotApplicable   bool              `json:"allowNotApplicable,omitempty"`
}

type Budgets struct {
	MaxMembers             int    `json:"maxMembers"`
	MaxInFlight            int    `json:"maxInFlight"`
	WallMS                 int64  `json:"wallMs"`
	MaxObservedTotalTokens *int64 `json:"maxObservedTotalTokens"`
}

type Order struct {
	Kind string `json:"kind"`
	Seed *int64 `json:"seed,omitempty"`
}

type Gates struct {
	MinCandidateEndToEndPass float64  `json:"minCandidateEndToEndPass"`
	MaxQualityDrop           float64  `json:"maxQualityDrop"`
	MaxTotalTokensRatio      *float64 `json:"maxTotalTokensRatio,omitempty"`
}

type Comparison struct {
	Baseline           string   `json:"baseline"`
	Candidate          string   `json:"candidate"`
	Gates              Gates    `json:"gates"`
	RequiredEqual      []string `json:"requiredEqual"`
	AllowedDifferences []string `json:"allowedDifferences"`
}

type Draft struct {
	Dataset     DatasetRef `json:"dataset"`
	CaseIDs     []string   `json:"caseIds"`
	Variants    []Variant  `json:"variants"`
	Repetitions int        `json:"repetitions"`
	Order       Order      `json:"order"`
	Checks      []Check    `json:"checks"`
	Comparison  Comparison `json:"comparison"`
	Budgets     Budgets    `json:"budgets"`
}

type PublicMember struct {
	MemberID      string `json:"member_id"`
	SuiteID       string `json:"suite_id"`
	CaseID        string `json:"case_id"`
	VariantID     string `json:"variant_id"`
	CaseSHA256    string `json:"case_sha256"`
	BindingSHA256 string `json:"binding_sha256"`
	Sample        int    `json:"sample"`
	Eligibility   string `json:"eligibility"`
}

type PublicPlan struct {
	SchemaVersion       string         `json:"schema_version"`
	SourceSchemaVersion string         `json:"source_schema_version"`
	SourceRecordSHA256  string         `json:"source_record_sha256"`
	ExperimentID        string         `json:"experiment_id"`
	CreatedAt           string         `json:"created_at"`
	Members             []PublicMember `json:"members"`
}

type MemberRecipe struct {
	MemberID string `json:"memberId"`
	Case     Case   `json:"case"`
}

type ExternalRegistration struct {
	SchemaVersion    string         `json:"schemaVersion"`
	SourcePlanSHA256 string         `json:"sourcePlanSha256"`
	Manifest         PublicPlan     `json:"manifest"`
	Source           Source         `json:"source"`
	Variants         []Variant      `json:"variants"`
	Recipes          []MemberRecipe `json:"recipes"`
	Checks           []Check        `json:"checks"`
	Comparison       Comparison     `json:"comparison"`
	Budgets          Budgets        `json:"budgets"`
}

type CreateExperiment struct {
	Name         string                `json:"name"`
	ControlMode  ControlMode           `json:"controlMode"`
	Draft        *Draft                `json:"draft,omitempty"`
	Registration *ExternalRegistration `json:"registration,omitempty"`
}

type DraftUpdate struct {
	Name  string `json:"name"`
	Draft Draft  `json:"draft"`
}

type Command struct {
	Kind       CommandKind `json:"kind"`
	PlanSHA256 string      `json:"planSha256,omitempty"`
}

type Submission struct {
	PlanSHA256 string `json:"planSha256"`
}

type ExecutionRef struct {
	Kind string `json:"kind"`
	ID   string `json:"id"`
}

type Interval struct {
	Start string `json:"start"`
	End   string `json:"end"`
}

type MeasureScope struct {
	MemberID   string         `json:"memberId"`
	Kind       string         `json:"kind"`
	Executions []ExecutionRef `json:"executions"`
	Missing    []string       `json:"missing"`
	Interval   *Interval      `json:"interval"`
}

type Measure struct {
	Value        *float64     `json:"value"`
	Unit         string       `json:"unit"`
	Completeness string       `json:"completeness"`
	SourceRefs   []string     `json:"sourceRefs"`
	Scope        MeasureScope `json:"scope"`
}

type Usage struct {
	InputTokens       Measure `json:"inputTokens"`
	OutputTokens      Measure `json:"outputTokens"`
	TotalTokens       Measure `json:"totalTokens"`
	CachedInputTokens Measure `json:"cachedInputTokens"`
	ModelCalls        Measure `json:"modelCalls"`
	ToolCalls         Measure `json:"toolCalls"`
	ToolFailures      Measure `json:"toolFailures"`
	WallMS            Measure `json:"wallMs"`
}
