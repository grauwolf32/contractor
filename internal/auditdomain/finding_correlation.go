package auditdomain

import (
	"slices"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
)

const (
	MaximumFactEvidence               = 4
	factEvidenceAlgorithm             = "contractor/fact-evidence/v1"
	FindingCorrelationPolicy          = "qualified-snapshot/v1"
	MaximumFindingCorrelationRevision = contracts.MaxExactInteger
)

type FindingCorrelationBasis struct {
	Algorithm string `json:"algorithm" strictjson:"required"`
	Value     string `json:"value" strictjson:"required"`
}

// FindingGroupKey hashes the explicit Project scope and one qualified primary
// key. No secondary hint, result rating, prose or transitive union participates.
func FindingGroupKey(projectID, algorithm, value string) (string, error) {
	if len(projectID) > 256 || !identifierPattern.MatchString(projectID) || !fingerprintHex.MatchString(value) || !validCorrelationAlgorithm(algorithm) {
		return "", invalid(CodeInvalid, "correlation.primary")
	}
	key, err := contentdigest.CanonicalJSON([]string{"contractor/finding-group/v1", projectID, algorithm, value})
	return strings.TrimPrefix(key, "sha256:"), err
}

func validCorrelationAlgorithm(algorithm string) bool {
	return algorithm == factEvidenceAlgorithm || algorithm == "contractor/producer-issue/v1" || algorithm == "contractor/snapshot-issue-anchor/v1"
}

// FactEvidenceFingerprint derives one closed informational key from authorized
// exact metadata. It never reads payloads or compares titles, refs or providers.
func FactEvidenceFingerprint(evidence []contracts.ExactArtifact) (FindingFingerprint, error) {
	if len(evidence) == 0 || len(evidence) > MaximumFactEvidence {
		return FindingFingerprint{}, invalid(CodeInvalid, "fact.evidence")
	}
	digests := make([]string, 0, len(evidence))
	for _, source := range evidence {
		if source.Validate() != nil || !source.HasMetadata() {
			return FindingFingerprint{}, invalid(CodeInvalid, "fact.evidence")
		}
		digests = append(digests, source.Digest)
	}
	slices.Sort(digests)
	digests = slices.Compact(digests)
	digest, err := contentdigest.CanonicalJSON([]any{factEvidenceAlgorithm, digests})
	if err != nil {
		return FindingFingerprint{}, err
	}
	return FindingFingerprint{Schema: FindingFingerprintSchema, Availability: "available", Algorithm: factEvidenceAlgorithm, Value: strings.TrimPrefix(digest, "sha256:")}, nil
}
