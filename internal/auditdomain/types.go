package auditdomain

import (
	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	PackageSchema             = "contractor.audit.package.v1"
	JSONMediaType             = "application/json"
	DirectVerificationsSchema = "contractor.audit.direct-verifications.v1"
	FindingProposalSchema     = "contractor.audit.finding-proposal.v1"
	InventoryBasisSchema      = "contractor.audit.inventory-basis.v1"
)

// DirectVerificationsMediaType opts one exact primary Workflow output into
// trusted direct-finding verification during Audit import. Merely succeeding
// or producing another primary output never implies verification.
const DirectVerificationsMediaType = "application/vnd.contractor.audit.direct-verifications+json"

type PackageKind string

const (
	PackageKindOpenAPISource PackageKind = "openapi-source"
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

func (p *Package) MemberByID(id string) (Member, bool) {
	for _, member := range p.members {
		if member.metadata.ID == id {
			return member, true
		}
	}
	return Member{}, false
}

type OperationTask struct {
	Path        string         `json:"path"`
	Method      string         `json:"method"`
	OperationID string         `json:"operation_id,omitempty"`
	Resolved    map[string]any `json:"resolved"`
	Gaps        []string       `json:"gaps"`
}

// ExactInput pins scan settings selected by deterministic preparation.
type ExactInput struct {
	Name   string                `json:"name"`
	Ref    contracts.ArtifactRef `json:"ref"`
	Digest string                `json:"digest"`
}

// DirectVerificationSet describes findings that were both discovered and
// checked by the same Workflow invocation. The Runtime-generated invocation
// identity and proposal-local key bind each entry to a trusted finding receipt;
// evidence identifiers must match that receipt before the Server accepts it.
type DirectVerificationSet struct {
	Schema        string                     `json:"schema"`
	Verifications []DirectVerificationResult `json:"verifications"`
}

type DirectVerificationResult struct {
	InvocationID string   `json:"invocation_id"`
	ClientKey    string   `json:"client_key"`
	Assessment   string   `json:"assessment"`
	Summary      string   `json:"summary"`
	EvidenceIDs  []string `json:"evidence_ids"`
}

type FindingProposal struct {
	Schema             string               `json:"schema"`
	ClientKey          string               `json:"client_key"`
	Title              string               `json:"title"`
	Description        string               `json:"description"`
	Subject            *FindingSubject      `json:"subject"`
	Hypothesis         string               `json:"hypothesis,omitempty"`
	Preconditions      []string             `json:"preconditions"`
	StandardRefs       []StandardReference  `json:"standard_refs"`
	EvidenceIDs        []string             `json:"evidence_ids"`
	ProposedChecks     []ProposedCheck      `json:"proposed_checks"`
	SeveritySuggestion string               `json:"severity_suggestion"`
	Limitations        []string             `json:"limitations"`
	Locations          []FindingLocation    `json:"locations,omitempty"`
	HTTPExchange       *FindingHTTPExchange `json:"http_exchange,omitempty"`
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

type InventoryOptions struct {
	SourceInputName string
	SourceRef       contracts.ArtifactRef
}

type inventoryBasis struct {
	Schema   string           `json:"schema"`
	Kind     string           `json:"kind"`
	Subjects []map[string]any `json:"subjects"`
	Gaps     []string         `json:"gaps"`
}
