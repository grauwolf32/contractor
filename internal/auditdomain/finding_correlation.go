package auditdomain

import (
	"strings"

	"github.com/grauwolf32/contractor/internal/contentdigest"
)

const (
	FindingCorrelationSchema          = "contractor.finding.correlation.v1"
	FindingCorrelationPolicy          = "qualified-snapshot/v1"
	MaximumFindingCorrelationRevision = 1<<53 - 1
)

type FindingCorrelationScope struct {
	Kind      string `json:"kind" strictjson:"required"`
	ProjectID string `json:"projectId" strictjson:"required"`
}

type FindingCorrelationBasis struct {
	Algorithm string `json:"algorithm" strictjson:"required"`
	Value     string `json:"value" strictjson:"required"`
}

type FindingCorrelationReference struct {
	AuditID   string `json:"auditId" strictjson:"required"`
	FindingID string `json:"findingId" strictjson:"required"`
}

// FindingCorrelation is system-owned equality evidence, never analyst triage.
// Its representative and boolean describe retained members at ProjectRevision.
type FindingCorrelation struct {
	Schema               string                       `json:"schema" strictjson:"required"`
	PolicyVersion        string                       `json:"policyVersion" strictjson:"required"`
	Scope                FindingCorrelationScope      `json:"scope" strictjson:"required"`
	ProjectRevision      uint64                       `json:"projectRevision" strictjson:"required"`
	Availability         string                       `json:"availability" strictjson:"required"`
	Reason               string                       `json:"reason,omitempty"`
	GroupKey             string                       `json:"groupKey,omitempty"`
	Basis                *FindingCorrelationBasis     `json:"basis,omitempty"`
	MatchedDomain        string                       `json:"matchedDomain,omitempty"`
	Representative       *FindingCorrelationReference `json:"representative,omitempty"`
	AutomaticDuplicate   bool                         `json:"automaticDuplicate" strictjson:"required"`
	HasOtherObservations bool                         `json:"hasOtherObservations" strictjson:"required"`
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

// ValidateFindingCorrelation checks both the stored primary key and the
// authorized single-statement presentation; callers provide the owning row.
func ValidateFindingCorrelation(value FindingCorrelation, auditID, findingID string) error {
	if value.Schema != FindingCorrelationSchema || value.PolicyVersion != FindingCorrelationPolicy || value.Scope.Kind != "project" || len(value.Scope.ProjectID) > 256 || !identifierPattern.MatchString(value.Scope.ProjectID) || value.ProjectRevision == 0 || value.ProjectRevision > MaximumFindingCorrelationRevision {
		return invalid(CodeInvalid, "correlation")
	}
	if value.Availability == "unavailable" {
		if value.Reason != "not_recorded" && value.Reason != "no_qualified_key" && value.Reason != "primary_key_ambiguous" {
			return invalid(CodeInvalid, "correlation.reason")
		}
		if value.GroupKey != "" || value.Basis != nil || value.Representative != nil || value.MatchedDomain != "" || value.AutomaticDuplicate || value.HasOtherObservations {
			return invalid(CodeInvalid, "correlation.unavailable")
		}
		return nil
	}
	if value.Availability != "available" || value.Reason != "" || value.Basis == nil || value.Representative == nil || value.MatchedDomain != "snapshot" {
		return invalid(CodeInvalid, "correlation")
	}
	key, err := FindingGroupKey(value.Scope.ProjectID, value.Basis.Algorithm, value.Basis.Value)
	if err != nil || key != value.GroupKey || !identifierPattern.MatchString(value.Representative.AuditID) || len(value.Representative.AuditID) > 256 || !identifierPattern.MatchString(value.Representative.FindingID) || len(value.Representative.FindingID) > 256 {
		return invalid(CodeInvalid, "correlation.group")
	}
	self := value.Representative.AuditID == auditID && value.Representative.FindingID == findingID
	if value.AutomaticDuplicate != !self || !self && !value.HasOtherObservations {
		return invalid(CodeInvalid, "correlation.representative")
	}
	return nil
}
