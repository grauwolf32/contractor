package auditdomain

import (
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	PackageSchema           = "contractor.audit.package.v1"
	ChecklistSchema         = "contractor.audit.checklist.v1"
	WorklistSchema          = "contractor.audit.worklist.v1"
	TaskSchema              = "contractor.audit.item-task.v1"
	ExecutionManifestSchema = "contractor.audit.execution-manifest.v1"
	CheckResultsSchema      = "contractor.audit.check-results.v1"
	FindingProposalSchema   = "contractor.audit.finding-proposal.v1"
	EvidenceSchema          = "contractor.audit.evidence.v1"
	CoverageSchema          = "contractor.audit.coverage.v1"
	InventoryBasisSchema    = "contractor.audit.inventory-basis.v1"
)

type PackageKind string

const (
	PackageKindWorklist        PackageKind = "worklist"
	PackageKindTask            PackageKind = "item-task"
	PackageKindExecution       PackageKind = "execution-manifest"
	PackageKindCheckResults    PackageKind = "check-results"
	PackageKindFindingProposal PackageKind = "finding-proposal"
	PackageKindEvidence        PackageKind = "evidence"
	PackageKindCoverage        PackageKind = "coverage"
	PackageKindOpenAPISource   PackageKind = "openapi-source"
)

type PackageManifest struct {
	Schema     string                  `json:"schema"`
	PackageID  string                  `json:"package_id"`
	Kind       PackageKind             `json:"kind"`
	EntryPoint string                  `json:"entrypoint,omitempty"`
	Members    []PackageMemberManifest `json:"members"`
}

type PackageMemberManifest struct {
	ID        string `json:"id"`
	Path      string `json:"path"`
	MediaType string `json:"media_type"`
	Size      int64  `json:"size"`
	Digest    string `json:"digest"`
}

type PackageInput struct {
	ID        string
	Path      string
	MediaType string
	Data      []byte
}

type Member struct {
	metadata PackageMemberManifest
	data     []byte
}

func (m Member) Metadata() PackageMemberManifest { return m.metadata }
func (m Member) Data() []byte                    { return append([]byte(nil), m.data...) }

type Package struct {
	Manifest      PackageManifest `json:"manifest"`
	Digest        string          `json:"digest"`
	StoredBytes   int64           `json:"stored_bytes"`
	ExpandedBytes int64           `json:"expanded_bytes"`
	members       []Member
}

func (p *Package) Members() []Member {
	result := make([]Member, len(p.members))
	copy(result, p.members)
	return result
}

func (p *Package) Member(path string) (Member, bool) {
	for _, member := range p.members {
		if member.metadata.Path == path {
			return member, true
		}
	}
	return Member{}, false
}

func (p *Package) MemberByID(id string) (Member, bool) {
	for _, member := range p.members {
		if member.metadata.ID == id {
			return member, true
		}
	}
	return Member{}, false
}

type ApprovalRequirement string

const (
	ApprovalNone        ApprovalRequirement = "none"
	ApprovalActiveCheck ApprovalRequirement = "active-check"
	ApprovalHumanReview ApprovalRequirement = "human-review"
)

type WorklistManifest struct {
	Schema string         `json:"schema"`
	Round  int            `json:"round"`
	Items  []WorklistItem `json:"items"`
}

type WorklistItem struct {
	ItemKey             string              `json:"item_key"`
	Ordinal             int                 `json:"ordinal"`
	Kind                string              `json:"kind"`
	SubjectKey          string              `json:"subject_key"`
	WorkflowRole        string              `json:"workflow_role"`
	TaskPackageID       string              `json:"task_package_id"`
	ApprovalRequirement ApprovalRequirement `json:"approval_requirement"`
}

type ItemTask struct {
	Schema                   string                `json:"schema"`
	ItemKey                  string                `json:"item_key"`
	Kind                     string                `json:"kind"`
	SubjectKey               string                `json:"subject_key"`
	WorkflowRole             string                `json:"workflow_role"`
	SourceContentDigest      string                `json:"source_content_digest"`
	SourceMediaType          string                `json:"source_media_type"`
	SourceRef                contracts.ArtifactRef `json:"source_ref"`
	CanonicalInventoryDigest string                `json:"canonical_inventory_digest"`
	Scope                    map[string]string     `json:"scope,omitempty"`
	Checklist                *ChecklistTask        `json:"checklist,omitempty"`
	Operation                *OperationTask        `json:"operation,omitempty"`
}

type ChecklistTask struct {
	Version          string   `json:"version"`
	Statement        string   `json:"statement"`
	Applicability    string   `json:"applicability"`
	AllowedMethods   []string `json:"allowed_methods"`
	RequiredEvidence []string `json:"required_evidence"`
	ReviewPolicy     string   `json:"review_policy"`
}

type OperationTask struct {
	Path        string         `json:"path"`
	Method      string         `json:"method"`
	OperationID string         `json:"operation_id,omitempty"`
	Resolved    map[string]any `json:"resolved"`
	Gaps        []string       `json:"gaps"`
}

type ExecutionManifest struct {
	Schema string          `json:"schema"`
	Items  []ExecutionItem `json:"items"`
}

type ExecutionItem struct {
	ItemKey           string                 `json:"item_key"`
	Ordinal           int                    `json:"ordinal"`
	SubjectKey        string                 `json:"subject_key"`
	TaskPackageID     string                 `json:"task_package_id"`
	TaskPackageDigest string                 `json:"task_package_digest"`
	TaskRef           *contracts.ArtifactRef `json:"task_ref,omitempty"`
	Inputs            []ExactInput           `json:"inputs"`
}

type ExactInput struct {
	Name   string                `json:"name"`
	Ref    contracts.ArtifactRef `json:"ref"`
	Digest string                `json:"digest"`
}

type CheckResultSet struct {
	Schema                  string        `json:"schema"`
	ExecutionManifestDigest string        `json:"execution_manifest_digest"`
	Results                 []CheckResult `json:"results"`
}

type CheckResult struct {
	ItemKey     string              `json:"item_key"`
	SubjectKey  string              `json:"subject_key"`
	Assessment  string              `json:"assessment"`
	Summary     string              `json:"summary"`
	EvidenceIDs []string            `json:"evidence_ids"`
	Coverage    ResultCoverage      `json:"coverage"`
	Proposals   []ProposalSelection `json:"proposals"`
}

// ProposalSelection is emitted by trusted Runtime tooling. The model chooses
// only the client key; Runtime injects the current invocation identity so the
// importer never guesses a receipt from shared Run, subject, or batch state.
type ProposalSelection struct {
	InvocationID string `json:"invocation_id"`
	ClientKey    string `json:"client_key"`
}

type ResultCoverage struct {
	Requested []string `json:"requested"`
	Completed []string `json:"completed"`
	Gaps      []string `json:"gaps"`
}

type FindingProposal struct {
	Schema             string              `json:"schema"`
	ClientKey          string              `json:"client_key"`
	Title              string              `json:"title"`
	Description        string              `json:"description"`
	Subject            FindingSubject      `json:"subject"`
	Hypothesis         string              `json:"hypothesis,omitempty"`
	Preconditions      []string            `json:"preconditions"`
	StandardRefs       []StandardReference `json:"standard_refs"`
	EvidenceIDs        []string            `json:"evidence_ids"`
	ProposedChecks     []ProposedCheck     `json:"proposed_checks"`
	SeveritySuggestion string              `json:"severity_suggestion"`
	Limitations        []string            `json:"limitations"`
}

type FindingSubject struct {
	Kind string `json:"kind"`
	Key  string `json:"key"`
}

type StandardReference struct {
	Scheme        string `json:"scheme"`
	Version       string `json:"version"`
	RequirementID string `json:"requirement_id"`
}

type ProposedCheck struct {
	Objective string `json:"objective"`
	Method    string `json:"method"`
}

type EvidenceEnvelope struct {
	Schema   string     `json:"schema"`
	Evidence []Evidence `json:"evidence"`
}

type Evidence struct {
	ID               string                 `json:"id"`
	Kind             string                 `json:"kind"`
	Summary          string                 `json:"summary"`
	Artifact         *contracts.ArtifactRef `json:"artifact,omitempty"`
	ContentMemberID  string                 `json:"content_member_id,omitempty"`
	ObservedAt       string                 `json:"observed_at,omitempty"`
	DeploymentMarker string                 `json:"deployment_marker,omitempty"`
}

type CoverageEnvelope struct {
	Schema string        `json:"schema"`
	Rows   []CoverageRow `json:"rows"`
}

type CoverageRow struct {
	ItemKey    string   `json:"item_key"`
	SubjectKey string   `json:"subject_key"`
	Status     string   `json:"status"`
	Requested  []string `json:"requested"`
	Completed  []string `json:"completed"`
	Gaps       []string `json:"gaps"`
	Rationale  string   `json:"rationale,omitempty"`
}

type ChecklistDocument struct {
	Schema string           `json:"schema"`
	Items  []ChecklistEntry `json:"items"`
}

type ChecklistEntry struct {
	Key              string   `json:"key"`
	Version          string   `json:"version"`
	Statement        string   `json:"statement"`
	Applicability    string   `json:"applicability"`
	AllowedMethods   []string `json:"allowed_methods"`
	RequiredEvidence []string `json:"required_evidence"`
	ReviewPolicy     string   `json:"review_policy"`
}

type InventoryOptions struct {
	Round               int
	WorkflowRole        string
	SourceInputName     string
	SourceRef           contracts.ArtifactRef
	Scope               map[string]string
	ApprovalRequirement ApprovalRequirement
}

type GeneratedTask struct {
	Item          WorklistItem `json:"item"`
	Document      ItemTask     `json:"document"`
	Package       []byte       `json:"-"`
	PackageDigest string       `json:"package_digest"`
}

type Inventory struct {
	SourceContentDigest      string            `json:"source_content_digest"`
	CanonicalInventoryDigest string            `json:"canonical_inventory_digest"`
	CanonicalInventory       []byte            `json:"-"`
	Worklist                 WorklistManifest  `json:"worklist"`
	ExecutionManifest        ExecutionManifest `json:"execution_manifest"`
	Coverage                 CoverageEnvelope  `json:"coverage"`
	Tasks                    []GeneratedTask   `json:"tasks"`
	Gaps                     []string          `json:"gaps"`
}

type inventoryBasis struct {
	Schema   string           `json:"schema"`
	Kind     string           `json:"kind"`
	Subjects []map[string]any `json:"subjects"`
	Gaps     []string         `json:"gaps"`
}

// rawDocument is used only while applying strict codecs to dynamic, trusted
// normalized content. It intentionally never crosses a package boundary.
type rawDocument = json.RawMessage
