// Package findingexport owns immutable Audit-local finding export projections.
// Source authorization and publication belong to the application service.
package findingexport

import (
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/findinganalysis"
)

const (
	SnapshotSchema      = "contractor.findings.export-snapshot.v2"
	DiagnosticsSchema   = "contractor.findings.export-diagnostics.v1"
	ExporterVersion     = "2.0.0"
	PolicyVersion       = "generic/1"
	MaxBytes            = 16 << 20
	MaxDiagnosticsBytes = 4 << 20
)

type Request struct {
	IncludeDuplicates bool        `json:"includeDuplicates"`
	ClientKey         string      `json:"clientKey" strictjson:"required"`
	Format            string      `json:"format" strictjson:"required"`
	Profile           string      `json:"profile" strictjson:"required"`
	Findings          []Selection `json:"findings" strictjson:"required"`
}

type Selection struct {
	FindingID             string `json:"findingId" strictjson:"required"`
	Revision              uint64 `json:"revision" strictjson:"required"`
	PresentationReceiptID string `json:"presentationReceiptId,omitempty"`
}

type Snapshot struct {
	Schema          string    `json:"schema" strictjson:"required"`
	AuditID         string    `json:"audit_id" strictjson:"required"`
	SnapshotAt      string    `json:"snapshot_at" strictjson:"required"`
	ExporterVersion string    `json:"exporter_version" strictjson:"required"`
	PolicyVersion   string    `json:"policy_version" strictjson:"required"`
	Request         Request   `json:"request" strictjson:"required"`
	RequestDigest   string    `json:"request_digest" strictjson:"required"`
	Receipts        []Receipt `json:"receipts" strictjson:"required"`
	Results         []Result  `json:"results" strictjson:"required"`
	size            int
}

type Receipt struct {
	ProposalArtifact Evidence                      `json:"proposal_artifact" strictjson:"required"`
	ReceiptID        string                        `json:"receipt_id" strictjson:"required"`
	ProposalID       string                        `json:"proposal_id" strictjson:"required"`
	Origin           Origin                        `json:"origin" strictjson:"required"`
	Proposal         auditdomain.FindingProposal   `json:"proposal" strictjson:"required"`
	Evidence         []Evidence                    `json:"evidence" strictjson:"required"`
	Analysis         *findinganalysis.Resolved     `json:"analysis,omitempty"`
	HTTP             *findinganalysis.ResolvedHTTP `json:"http,omitempty"`
}

type Origin struct {
	RunID                 string `json:"run_id" strictjson:"required"`
	InvocationID          string `json:"invocation_id" strictjson:"required"`
	StageExecutionID      string `json:"stage_execution_id" strictjson:"required"`
	AllocationID          string `json:"allocation_id" strictjson:"required"`
	LogicalAgentName      string `json:"logical_agent_name" strictjson:"required"`
	WorkflowName          string `json:"workflow_name" strictjson:"required"`
	WorkflowVersion       string `json:"workflow_version" strictjson:"required"`
	WorkflowClosureDigest string `json:"workflow_closure_digest" strictjson:"required"`
	AuditID               string `json:"audit_id,omitempty"`
	ExecutionID           string `json:"execution_id,omitempty"`
	Role                  string `json:"role,omitempty"`
}

type Evidence struct {
	Original auditdomain.FindingCollectionDocument `json:"original" strictjson:"required"`
	Retained auditdomain.FindingCollectionDocument `json:"retained" strictjson:"required"`
	Git      *GitSource                            `json:"git,omitempty"`
}

type GitSource struct {
	RepositoryURI string `json:"repository_uri" strictjson:"required"`
	Commit        string `json:"commit" strictjson:"required"`
}

type Result struct {
	FindingID             string      `json:"finding_id" strictjson:"required"`
	Revision              uint64      `json:"revision" strictjson:"required"`
	PresentationReceiptID string      `json:"presentation_receipt_id" strictjson:"required"`
	ReceiptIDs            []string    `json:"receipt_ids" strictjson:"required"`
	Review                Review      `json:"review" strictjson:"required"`
	PrimaryKey            *PrimaryKey `json:"primary_key,omitempty"`
}

type Review struct {
	State             *string `json:"state" strictjson:"required"`
	Verdict           string  `json:"verdict,omitempty"`
	Severity          string  `json:"severity,omitempty"`
	Rationale         string  `json:"rationale,omitempty"`
	ActorID           string  `json:"actor_id,omitempty"`
	DecisionAt        string  `json:"decision_at,omitempty"`
	DuplicateTargetID string  `json:"duplicate_target_id,omitempty"`
}

type PrimaryKey struct {
	ProjectID string `json:"project_id" strictjson:"required"`
	Algorithm string `json:"algorithm" strictjson:"required"`
	Value     string `json:"value" strictjson:"required"`
	GroupKey  string `json:"group_key" strictjson:"required"`
	Policy    string `json:"policy" strictjson:"required"`
}

// ResultGroup retains the chosen finding and every selected member identity.
type ResultGroup struct {
	Result     Result
	FindingIDs []string
}

type Diagnostic struct {
	Code      string `json:"code" strictjson:"required"`
	Severity  string `json:"severity" strictjson:"required"`
	FindingID string `json:"finding_id" strictjson:"required"`
	Field     string `json:"field" strictjson:"required"`
}

type Diagnostics struct {
	Schema  string       `json:"schema" strictjson:"required"`
	AuditID string       `json:"audit_id" strictjson:"required"`
	Items   []Diagnostic `json:"items" strictjson:"required"`
}

// IsState tests an explicitly set current state. Unassessed Findings stay null.
func (r Review) IsState(state string) bool { return r.State != nil && *r.State == state }
