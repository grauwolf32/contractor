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
	"strconv"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

const (
	APIVersion = contracts.APIVersion
	// The proposal keeps its 8 MiB document ceiling. The complete private
	// submission can additionally carry selected HTTP evidence (LIM-FND-32).
	MaxRequestBytes   = artifacts.MaxPayloadSize
	MaxEvidenceRefs   = auditdomain.MaximumEvidencePerItem
	proposalMediaType = "application/json"
)

// The public API answers these errors with the Kind each one reports
// (apierror); the private API keeps its own finding codes for the Runtime.
var (
	ErrInvalid         = apierror.New(apierror.InvalidRequest, "finding proposal request is invalid")
	ErrNotFound        = apierror.New(apierror.NotFound, "finding proposal receipt was not found")
	ErrConflict        = apierror.New(apierror.Conflict, "finding proposal submission conflicts with its receipt")
	ErrAccessDenied    = errors.New("finding proposal access is denied")
	ErrToolNotSelected = errors.New("security finding tool is not selected")
	ErrAuditClosed     = errors.New("finding proposal destination Audit is closed")
	// ErrCollectionKeyConflict reports a finding collection clientKey that is
	// already bound to another selection. Like a reused Idempotency-Key it is
	// permanent: only a new clientKey can publish the changed selection.
	ErrCollectionKeyConflict = apierror.Narrow(ErrConflict, apierror.IdempotencyConflict,
		"finding collection clientKey is bound to another selection")
)

var identityPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)

type Submission struct {
	APIVersion   string                            `json:"apiVersion"`
	InvocationID string                            `json:"invocationId"`
	SubmissionID string                            `json:"submissionId"`
	Proposal     auditdomain.FindingProposal       `json:"proposal"`
	EvidenceRefs []contracts.ArtifactRef           `json:"evidenceRefs"`
	HTTPEvidence *auditdomain.CapturedHTTPExchange `json:"httpEvidence,omitempty"`
}

func (input *Submission) UnmarshalJSON(data []byte) error {
	type plain Submission
	if err := strictjson.Decode(data, (*plain)(input)); err != nil {
		return err
	}
	// A missing selection is omitted, never an explicitly unknown capture.
	// Decode only its raw token; all fields were strictly checked above.
	var fields struct {
		HTTPEvidence json.RawMessage `json:"httpEvidence"`
	}
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	if string(fields.HTTPEvidence) == "null" {
		return ErrInvalid
	}
	return nil
}

// ExactArtifact is the shared exact-content descriptor. Receipts always
// record media type and size, so their JSON carries all four fields.
type ExactArtifact = contracts.ExactArtifact

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

// RetentionState is what keeps a stored receipt's proposal: its source Run,
// an Audit hold after that Run's deletion, or nothing.
type RetentionState string

const (
	RetentionSourceHeld RetentionState = "source-held"
	RetentionAuditHeld  RetentionState = "audit-held"
	RetentionDiscarded  RetentionState = "discarded"
)

func (r RetentionState) valid() bool {
	return r == RetentionSourceHeld || r == RetentionAuditHeld || r == RetentionDiscarded
}

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
	APIVersion string          `json:"apiVersion"`
	ProposalID string          `json:"proposalId"`
	ReceiptID  string          `json:"receiptId"`
	Proposal   ExactArtifact   `json:"proposal"`
	Evidence   []ExactArtifact `json:"evidence"`
	Replayed   bool            `json:"replayed"`
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

type ProposalKey struct {
	InvocationID string `json:"invocationId"`
	ClientKey    string `json:"clientKey"`
}

type ResolvedProposal struct {
	ReceiptID string
	Proposal  ExactArtifact
	Origin    Origin
	Document  auditdomain.FindingProposal
}

type canonicalSubmission struct {
	request       Submission
	proposalBytes []byte
	// reportProjection is the canonical report projection of the proposal,
	// stored with the receipt so Audit reports never read the proposal back.
	reportProjection []byte
	digest           string
}

func canonicalize(input Submission) (canonicalSubmission, error) {
	if input.APIVersion != APIVersion || !validIdentity(input.InvocationID) ||
		!validIdentity(input.SubmissionID) || len(input.EvidenceRefs) > MaxEvidenceRefs {
		return canonicalSubmission{}, ErrInvalid
	}
	// Keep the v1 proposal codec able to read already retained documents while
	// refusing new values that cannot become unique Audit coverage gaps.
	seenLimitations := make(map[string]struct{}, len(input.Proposal.Limitations))
	for _, limitation := range input.Proposal.Limitations {
		if len(limitation) > auditdomain.MaximumCoverageValueBytes {
			return canonicalSubmission{}, fmt.Errorf("%w: proposal limitation exceeds Audit coverage bound", ErrInvalid)
		}
		if _, duplicate := seenLimitations[limitation]; duplicate {
			return canonicalSubmission{}, fmt.Errorf("%w: duplicate proposal limitation", ErrInvalid)
		}
		seenLimitations[limitation] = struct{}{}
	}
	proposalBytes, err := auditdomain.EncodeFindingProposal(input.Proposal)
	if err != nil {
		return canonicalSubmission{}, fmt.Errorf("%w: proposal document", ErrInvalid)
	}
	reportProjection, err := auditdomain.EncodeFindingReportProjection(auditdomain.ProjectFindingReport(input.Proposal))
	if err != nil {
		return canonicalSubmission{}, fmt.Errorf("%w: proposal report projection", ErrInvalid)
	}
	refs := append([]contracts.ArtifactRef(nil), input.EvidenceRefs...)
	for _, ref := range refs {
		if err := ref.ValidateExact(); err != nil ||
			ref.Namespace == "finding-proposals" && !strings.HasPrefix(ref.Name, "http.exchange.") {
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
		if index > 0 && refs[index-1].SameExact(ref) {
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
	if input.HTTPEvidence != nil {
		if len(refs) >= MaxEvidenceRefs || auditdomain.ValidateCapturedHTTPExchange(*input.HTTPEvidence) != nil {
			return canonicalSubmission{}, fmt.Errorf("%w: HTTP evidence", ErrInvalid)
		}
		input.Proposal.EvidenceIDs = append(append([]string(nil), input.Proposal.EvidenceIDs...), fmt.Sprintf("evidence-%d", len(refs)+1))
		if snapshot := findingHTTPSnapshot(*input.HTTPEvidence, fmt.Sprintf("evidence-%d", len(refs)+1)); snapshot != nil {
			input.Proposal.HTTPExchange = snapshot
		}
		proposalBytes, err = auditdomain.EncodeFindingProposal(input.Proposal)
		if err != nil {
			return canonicalSubmission{}, fmt.Errorf("%w: HTTP proposal document", ErrInvalid)
		}
		reportProjection, err = auditdomain.EncodeFindingReportProjection(auditdomain.ProjectFindingReport(input.Proposal))
		if err != nil {
			return canonicalSubmission{}, fmt.Errorf("%w: HTTP report projection", ErrInvalid)
		}
	}
	digestValue := struct {
		Proposal     json.RawMessage                   `json:"proposal"`
		EvidenceRefs []contracts.ArtifactRef           `json:"evidenceRefs"`
		HTTPEvidence *auditdomain.CapturedHTTPExchange `json:"httpEvidence,omitempty"`
	}{Proposal: proposalBytes, EvidenceRefs: refs, HTTPEvidence: input.HTTPEvidence}
	encoded, err := json.Marshal(digestValue)
	if err != nil {
		return canonicalSubmission{}, fmt.Errorf("canonicalize finding submission: %w", err)
	}
	return canonicalSubmission{
		request: input, proposalBytes: proposalBytes, reportProjection: reportProjection,
		digest: auditdomain.DigestBytes(encoded),
	}, nil
}

func findingHTTPSnapshot(capture auditdomain.CapturedHTTPExchange, evidenceID string) *auditdomain.FindingHTTPExchange {
	if capture.Source != "http" {
		return nil
	}
	id, _ := strconv.ParseInt(capture.RequestID, 10, 64)
	result := &auditdomain.FindingHTTPExchange{RequestID: id, RequestTag: capture.RequestTag}
	for _, attempt := range capture.Attempts {
		if attempt.Request.Message.Kind != "structured" {
			return nil
		}
		value := auditdomain.FindingHTTPAttempt{Method: attempt.Request.Method, URL: attempt.Request.URL,
			Headers: *attempt.Request.Message.Headers, BodyBase64: *attempt.Request.Message.BodyBase64}
		if response := attempt.Response; response != nil {
			status := response.Status
			value.Status = &status
			if response.Message.Headers != nil {
				value.ResponseHeaders = *response.Message.Headers
			}
			if response.Message.BodyBase64 != nil && *response.Message.BodyBase64 != "" {
				result.ResponseBodyEvidenceID = evidenceID
			}
		} else {
			if attempt.Error != "transport_error" && attempt.Error != "timeout" && attempt.Error != "cancelled" {
				return nil
			}
			value.Error = "transport_error"
			if attempt.Error == "cancelled" {
				value.Error = "cancelled"
			}
		}
		result.Attempts = append(result.Attempts, value)
	}
	return result
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

func validIdentity(value string) bool {
	return value == strings.TrimSpace(value) && identityPattern.MatchString(value)
}
