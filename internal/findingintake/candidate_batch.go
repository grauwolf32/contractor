package findingintake

import (
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

const (
	CandidateSchema            = "contractor.finding.candidates.v1"
	CandidateMediaType         = "application/vnd.contractor.finding-candidates+json"
	CandidateEvidenceMediaType = "application/vnd.contractor.finding-evidence+json"
	MaxCandidateBatch          = 100
	MaxCandidateBatchBytes     = 512 << 10
)

var candidateKey = regexp.MustCompile(`^[0-9a-f]{64}$`)
var candidateMethod = regexp.MustCompile(`^[A-Z]{1,16}$`)

// CandidateBatch is provider-neutral. Its content asserts a problem but cannot
// name a destination owner, Project, Run, allocation or publication authority.
type CandidateBatch struct {
	Schema     string                  `json:"schema" strictjson:"required"`
	Complete   bool                    `json:"complete" strictjson:"required"`
	Candidates []NormalizedCandidate   `json:"candidates" strictjson:"required"`
	Skipped    map[string]int          `json:"skipped" strictjson:"required"`
	Evidence   contracts.ExactArtifact `json:"evidence" strictjson:"required"`
}

type NormalizedCandidate struct {
	Key          string                      `json:"key" strictjson:"required"`
	Producer     string                      `json:"producer" strictjson:"required"`
	Rule         string                      `json:"rule" strictjson:"required"`
	Title        string                      `json:"title" strictjson:"required"`
	Assertion    string                      `json:"assertion" strictjson:"required"`
	Location     auditdomain.FindingLocation `json:"location" strictjson:"required"`
	Severity     string                      `json:"severity" strictjson:"required"`
	QualifiedKey []string                    `json:"qualifiedKey" strictjson:"required,nullable"`
	ItemKey      *string                     `json:"itemKey,omitempty"`
	Incidental   bool                        `json:"incidental,omitempty"`
}

func decodeCandidateBatch(data []byte) (CandidateBatch, error) {
	var batch CandidateBatch
	if len(data) == 0 || len(data) > MaxCandidateBatchBytes || strictjson.Decode(data, &batch) != nil || batch.Schema != CandidateSchema || batch.Candidates == nil || len(batch.Candidates) > MaxCandidateBatch || batch.Skipped == nil || len(batch.Skipped) > 16 || batch.Evidence.Validate() != nil || !batch.Evidence.HasMetadata() || batch.Evidence.MediaType != CandidateEvidenceMediaType || batch.Evidence.SizeBytes > MaxCandidateBatchBytes {
		return batch, ErrInvalid
	}
	for reason, count := range batch.Skipped {
		if !validIdentity(reason) || count < 1 || count > MaxCandidateBatch {
			return batch, ErrInvalid
		}
	}
	// Validate the envelope first. Each well-formed assertion is handled
	// independently; an individual permanent publication rejection keeps a receipt.
	seen := map[string]bool{}
	for _, candidate := range batch.Candidates {
		if seen[candidate.Key] {
			return batch, ErrInvalid
		}
		seen[candidate.Key] = true
		if !candidateKey.MatchString(candidate.Key) || !boundedCandidateText(candidate.Producer, 160) || !boundedCandidateText(candidate.Rule, 160) || !boundedCandidateText(candidate.Title, 512) || !boundedCandidateText(candidate.Assertion, 8192) || len(candidate.QualifiedKey) > 8 || candidate.Location.Method != "" && !candidateMethod.MatchString(candidate.Location.Method) {
			return batch, ErrInvalid
		}
		if _, err := auditdomain.EncodeFindingReportProjection(auditdomain.ProjectFindingReport(candidateSubmission(candidate, nil, batch.Complete).Proposal)); err != nil {
			return batch, ErrInvalid
		}
		for _, part := range candidate.QualifiedKey {
			if !utf8.ValidString(part) || len(part) > 8192 || strings.ContainsRune(part, 0) {
				return batch, ErrInvalid
			}
		}
		switch candidate.Severity {
		case "low", "medium", "high", "critical":
		default:
			return batch, ErrInvalid
		}
	}
	return batch, nil
}

func boundedCandidateText(value string, maximum int) bool {
	return utf8.ValidString(value) && strings.TrimSpace(value) != "" && len(value) <= maximum && !strings.ContainsRune(value, 0)
}

func candidateSubmission(candidate NormalizedCandidate, refs []contracts.ArtifactRef, complete bool) Submission {
	limitations := []string{}
	if !complete {
		limitations = append(limitations, "scanner_incomplete")
	}
	kind := "endpoint"
	if candidate.Location.File != "" {
		kind = "source-file"
	}
	subjectDigest, _ := contentdigest.CanonicalJSON(candidate.Location)
	proposal := auditdomain.FindingProposal{
		Schema: "contractor.audit.finding-proposal.v1", ClientKey: candidate.Key,
		Title: candidate.Title, Description: candidate.Assertion,
		Subject:       &auditdomain.FindingSubject{Kind: kind, Key: kind + "-" + strings.TrimPrefix(subjectDigest, "sha256:")},
		Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
		EvidenceIDs: []string{"evidence-1", "evidence-2"}, ProposedChecks: []auditdomain.ProposedCheck{},
		SeveritySuggestion: candidate.Severity, Limitations: limitations,
		Locations: []auditdomain.FindingLocation{candidate.Location},
	}
	return Submission{APIVersion: contracts.APIVersion, InvocationID: "candidate-" + candidate.Key,
		SubmissionID: stableSubmissionID("candidate-"+candidate.Key, candidate.Key), Proposal: proposal, EvidenceRefs: refs,
		ItemKey: candidate.ItemKey, Incidental: candidate.Incidental}
}

func candidateFingerprint(candidate NormalizedCandidate, exact contracts.ExactArtifact, ordinal int) auditdomain.FindingFingerprint {
	if len(candidate.QualifiedKey) == 0 {
		return auditdomain.UnavailableFindingFingerprint("producer_key_unavailable")
	}
	key, _ := contentdigest.CanonicalJSON([]any{"contractor/normalized-producer-issue/v1", candidate.Producer, candidate.Rule, candidate.QualifiedKey})
	return auditdomain.FindingFingerprint{Schema: auditdomain.FindingFingerprintSchema, Availability: "available", Algorithm: "contractor/producer-issue/v1", Value: strings.TrimPrefix(key, "sha256:"), Adapter: "normalized-candidate/v1", RuleID: candidate.Rule, Report: &exact, RecordLocator: fmt.Sprintf("/candidates/%d", ordinal)}
}

// CandidatePublicationSummary is bounded execution status; canonical Findings
// and complete receipt detail remain in their existing paginated domain readers.
type CandidatePublicationSummary struct {
	SourcesTruncated bool                         `json:"sourcesTruncated,omitempty"`
	Published        int                          `json:"published"`
	Replayed         int                          `json:"replayed"`
	Rejected         int                          `json:"rejected"`
	Skipped          int                          `json:"skipped"`
	Complete         bool                         `json:"complete"`
	Sources          []CandidatePublicationSource `json:"sources"`
}
type CandidatePublicationSource struct {
	Key       string  `json:"key"`
	ReceiptID string  `json:"receiptId"`
	FindingID *string `json:"findingId,omitempty"`
	Error     string  `json:"error,omitempty"`
}
