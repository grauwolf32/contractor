// Package findingintake owns allocation-bound candidate finding receipts.
// It deliberately grants no Audit mutation, review, or scheduling authority.
package findingintake

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
)

const (
	APIVersion        = contracts.APIVersion
	MaxRequestBytes   = 1 << 20
	MaxEvidenceRefs   = auditdomain.MaximumEvidencePerItem
	proposalMediaType = "application/json"
)

var (
	ErrInvalid         = errors.New("finding proposal request is invalid")
	ErrNotFound        = errors.New("finding proposal receipt was not found")
	ErrConflict        = errors.New("finding proposal submission conflicts with its receipt")
	ErrAccessDenied    = errors.New("finding proposal access is denied")
	ErrToolNotSelected = errors.New("security finding tool is not selected")
)

var identityPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)

type Submission struct {
	APIVersion   string                      `json:"apiVersion"`
	InvocationID string                      `json:"invocationId"`
	SubmissionID string                      `json:"submissionId"`
	Proposal     auditdomain.FindingProposal `json:"proposal"`
	EvidenceRefs []contracts.ArtifactRef     `json:"evidenceRefs"`
}

type ExactArtifact struct {
	Ref       contracts.ArtifactRef `json:"ref"`
	Digest    string                `json:"digest"`
	MediaType string                `json:"mediaType"`
	SizeBytes int64                 `json:"sizeBytes"`
}

type WorkflowOrigin struct {
	Name             string `json:"name"`
	Version          string `json:"version"`
	SchemaVersion    string `json:"schemaVersion"`
	ConfigurationRef struct {
		Name    string `json:"name"`
		Version string `json:"version"`
	} `json:"configurationRef"`
	ClosureDigest string `json:"closureDigest"`
}

type AuditOrigin struct {
	AuditID     string `json:"auditId"`
	ExecutionID string `json:"executionId"`
	Role        string `json:"role"`
}

type Origin struct {
	RunID            string         `json:"runId"`
	StageExecutionID string         `json:"stageExecutionId"`
	AllocationID     string         `json:"allocationId"`
	InvocationID     string         `json:"invocationId"`
	LogicalAgentName string         `json:"logicalAgentName"`
	Workflow         WorkflowOrigin `json:"workflow"`
	Audit            *AuditOrigin   `json:"audit,omitempty"`
	RunDeleted       bool           `json:"runDeleted"`
}

type RetentionState string

const (
	RetentionSourceHeld RetentionState = "source-held"
	RetentionAuditHeld  RetentionState = "audit-held"
	RetentionDiscarded  RetentionState = "discarded"
)

type AuditHold struct {
	AuditID   string          `json:"auditId"`
	ProjectID string          `json:"projectId"`
	Proposal  ExactArtifact   `json:"proposal"`
	Evidence  []ExactArtifact `json:"evidence"`
	CreatedAt time.Time       `json:"createdAt"`
}

type Receipt struct {
	ReceiptID     string                      `json:"receiptId"`
	ProposalID    string                      `json:"proposalId"`
	RequestDigest string                      `json:"requestDigest"`
	ClientKey     string                      `json:"clientKey"`
	Proposal      ExactArtifact               `json:"proposal"`
	Document      auditdomain.FindingProposal `json:"document"`
	Evidence      []ExactArtifact             `json:"evidence"`
	Origin        Origin                      `json:"origin"`
	Retention     RetentionState              `json:"retention"`
	AuditHolds    []AuditHold                 `json:"auditHolds"`
	CreatedAt     time.Time                   `json:"createdAt"`
}

type SubmissionResponse struct {
	APIVersion string        `json:"apiVersion"`
	ProposalID string        `json:"proposalId"`
	ReceiptID  string        `json:"receiptId"`
	Proposal   ExactArtifact `json:"proposal"`
	Replayed   bool          `json:"replayed"`
}

type ListQuery struct {
	AfterCreatedAt *time.Time
	AfterReceiptID string
	Limit          int
}

type ImportRequest struct {
	OwnerID  string
	AuditID  string
	RunID    string
	Proposal contracts.ArtifactRef
}

type canonicalSubmission struct {
	request       Submission
	proposalBytes []byte
	digest        string
}

func canonicalize(input Submission) (canonicalSubmission, error) {
	if input.APIVersion != APIVersion || !validIdentity(input.InvocationID) ||
		!validIdentity(input.SubmissionID) || len(input.EvidenceRefs) > MaxEvidenceRefs {
		return canonicalSubmission{}, ErrInvalid
	}
	proposalBytes, err := auditdomain.EncodeFindingProposal(input.Proposal)
	if err != nil {
		return canonicalSubmission{}, fmt.Errorf("%w: proposal document", ErrInvalid)
	}
	refs := append([]contracts.ArtifactRef(nil), input.EvidenceRefs...)
	for _, ref := range refs {
		if err := ref.ValidateExact(); err != nil || ref.Namespace == "finding-proposals" {
			return canonicalSubmission{}, fmt.Errorf("%w: evidence reference", ErrInvalid)
		}
	}
	sort.Slice(refs, func(i, j int) bool {
		if refs[i].Namespace != refs[j].Namespace {
			return refs[i].Namespace < refs[j].Namespace
		}
		if refs[i].Name != refs[j].Name {
			return refs[i].Name < refs[j].Name
		}
		return *refs[i].Revision < *refs[j].Revision
	})
	for index, ref := range refs {
		if index > 0 && sameRef(refs[index-1], ref) {
			return canonicalSubmission{}, fmt.Errorf("%w: duplicate evidence reference", ErrInvalid)
		}
	}
	if len(input.Proposal.EvidenceIDs) != len(refs) {
		return canonicalSubmission{}, fmt.Errorf("%w: evidence identity count", ErrInvalid)
	}
	for index, value := range input.Proposal.EvidenceIDs {
		if value != fmt.Sprintf("evidence-%d", index+1) {
			return canonicalSubmission{}, fmt.Errorf("%w: evidence identity", ErrInvalid)
		}
	}
	wantSubmission := stableSubmissionID(input.InvocationID, input.Proposal.ClientKey)
	if input.SubmissionID != wantSubmission {
		return canonicalSubmission{}, fmt.Errorf("%w: submission correlation", ErrInvalid)
	}
	input.EvidenceRefs = refs
	digestValue := struct {
		Proposal     json.RawMessage         `json:"proposal"`
		EvidenceRefs []contracts.ArtifactRef `json:"evidenceRefs"`
	}{Proposal: proposalBytes, EvidenceRefs: refs}
	encoded, err := json.Marshal(digestValue)
	if err != nil {
		return canonicalSubmission{}, fmt.Errorf("canonicalize finding submission: %w", err)
	}
	return canonicalSubmission{
		request: input, proposalBytes: proposalBytes, digest: digestBytes(encoded),
	}, nil
}

func StableSubmissionID(invocationID, clientKey string) string {
	return stableSubmissionID(invocationID, clientKey)
}

func stableSubmissionID(invocationID, clientKey string) string {
	digest := sha256.Sum256([]byte("contractor.finding.submission.v1\x00" + invocationID + "\x00" + clientKey))
	return "finding-" + hex.EncodeToString(digest[:])
}

func deterministicID(prefix string, values ...string) string {
	input := "contractor.finding.identity.v1\x00" + prefix
	for _, value := range values {
		input += "\x00" + value
	}
	digest := sha256.Sum256([]byte(input))
	return prefix + "-" + hex.EncodeToString(digest[:])
}

func digestBytes(value []byte) string {
	digest := sha256.Sum256(value)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func validIdentity(value string) bool {
	return value == strings.TrimSpace(value) && identityPattern.MatchString(value)
}

func sameRef(left, right contracts.ArtifactRef) bool {
	return left.Namespace == right.Namespace && left.Name == right.Name &&
		left.Revision != nil && right.Revision != nil && *left.Revision == *right.Revision
}

func trustedGrantMatches(left, right controlplane.AllocationGrant) bool {
	return left.AllocationID == right.AllocationID && left.RunID == right.RunID &&
		left.StageExecutionID == right.StageExecutionID &&
		left.RuntimeAgentID == right.RuntimeAgentID &&
		left.RuntimeInstanceID == right.RuntimeInstanceID &&
		left.LogicalAgentName == right.LogicalAgentName
}
