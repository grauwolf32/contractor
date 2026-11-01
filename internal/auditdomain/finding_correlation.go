package auditdomain

import (
	"strings"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	FindingCorrelationSchema          = "contractor.finding.correlation.v1"
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
	return algorithm == "contractor/producer-issue/v1" || algorithm == "contractor/snapshot-issue-anchor/v1"
}
