package scanplan

import "github.com/grauwolf32/contractor/internal/contracts"

const (
	PlanMediaType       = "application/vnd.contractor.scan-plan+json"
	TargetListMediaType = "text/vnd.contractor.target-list"
	MaxPlanBytes        = 4 * 1024 * 1024
	MaxPlanCandidates   = 4000
)

type PlanInput struct {
	Artifact  contracts.ArtifactRef
	MediaType string
	Data      []byte
}

type ToolBinding struct {
	Namespace string
	Template  contracts.ResolvedAgentTemplate
}

type PlanSource struct {
	Artifact      contracts.ArtifactRef `json:"artifact"`
	MediaType     string                `json:"mediaType"`
	ContentDigest string                `json:"contentDigest"`
}

type ScanPlan struct {
	PreparationCoverage *contracts.RequestSetCoverage `json:"preparationCoverage,omitempty"`
	SchemaVersion       int                           `json:"schemaVersion"`
	ID                  string                        `json:"id"`
	Source              PlanSource                    `json:"source"`
	Policy              contracts.ScanPlanPolicy      `json:"policy"`
	PreparationGaps     []contracts.PreparationGap    `json:"preparationGaps"`
	Candidates          []ScanCandidate               `json:"candidates"`
	Jobs                []ScanJob                     `json:"jobs"`
}

type ScanCandidate struct {
	ID        string   `json:"id"`
	Worker    string   `json:"worker"`
	Tool      string   `json:"tool"`
	SourceIDs []string `json:"sourceIds"`
	Selection string   `json:"selection"`
	Code      string   `json:"code"`
}

type SQLMapRequest struct {
	SchemaVersion  int                           `json:"schemaVersion"`
	Method         string                        `json:"method"`
	URL            string                        `json:"url"`
	Headers        []contracts.HTTPRequestHeader `json:"headers"`
	Body           string                        `json:"body"`
	TestParameters []string                      `json:"testParameters"`
}

// ScanJob describes semantic input. Generated artifact revisions and physical
// allocation identities belong to execution state, never this immutable plan.
type ScanJob struct {
	TemplateDigest string                           `json:"templateDigest"`
	ID             string                           `json:"id"`
	CandidateID    string                           `json:"candidateId"`
	Worker         string                           `json:"worker"`
	Tool           string                           `json:"tool"`
	Namespace      string                           `json:"namespace"`
	TemplateRef    string                           `json:"templateRef"`
	Execution      contracts.ToolExecutionConfig    `json:"execution"`
	TimeoutSeconds int                              `json:"timeoutSeconds"`
	Parameters     map[string]string                `json:"parameters"`
	Artifacts      map[string]contracts.ArtifactRef `json:"artifacts"`
	Request        *SQLMapRequest                   `json:"request,omitempty"`
}
