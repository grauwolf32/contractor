package auditcontract

import (
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"strings"
	"unicode/utf8"
)

// EvidenceSubmission contains normalized Runtime selections, not model identity.
// The allocation and accepted execution supply all persisted origin fields.
type EvidenceSubmission struct {
	APIVersion       string                  `json:"apiVersion"`
	InvocationID     string                  `json:"invocationId"`
	SubmissionID     string                  `json:"submissionId"`
	ItemKey          string                  `json:"itemKey"`
	Sources          []contracts.ArtifactRef `json:"sources"`
	Interpretation   string                  `json:"interpretation"`
	TestedConditions []string                `json:"testedConditions"`
	Limitations      []string                `json:"limitations"`
}

type EvidenceReceipt struct {
	APIVersion      string     `json:"apiVersion"`
	SelectedSources []Material `json:"selectedSources"`
	Material        Material   `json:"material"`
	Sources         []Material `json:"sources"`
	ItemKey         string     `json:"itemKey"`
	Replayed        bool       `json:"replayed"`
}

func (v EvidenceSubmission) Validate() error {
	if v.APIVersion != contracts.APIVersion || !contracts.ValidResourceID(v.InvocationID) || len(v.InvocationID) > 128 || !contracts.ValidResourceID(v.SubmissionID) || len(v.SubmissionID) > 128 {
		return errInvalid
	}
	if _, ok := ExecutionItemOrdinal(v.ItemKey); !ok {
		return errInvalid
	}
	return validateEvidenceContent(v.Sources, v.Interpretation, v.TestedConditions, v.Limitations)
}

func validateEvidenceContent(sources []contracts.ArtifactRef, interpretation string, testedConditions, limitations []string) error {
	if testedConditions == nil || limitations == nil || len(sources) == 0 || len(sources) > 256 || len(testedConditions) > 512 || len(limitations) > 512 {
		return errInvalid
	}
	text := func(s string, n int) bool {
		return utf8.ValidString(s) && strings.TrimSpace(s) != "" && !strings.ContainsRune(s, 0) && len(s) <= n
	}
	if !text(interpretation, 65536) {
		return errInvalid
	}
	seen := map[string]bool{}
	for _, ref := range sources {
		if ref.ValidateExact() != nil {
			return errInvalid
		}
		key := ref.Namespace + "/" + ref.Name + "/" + *ref.Revision
		if seen[key] {
			return errInvalid
		}
		seen[key] = true
	}
	for _, values := range [][]string{testedConditions, limitations} {
		seen = map[string]bool{}
		for _, s := range values {
			if !text(s, 512) || seen[s] {
				return errInvalid
			}
			seen[s] = true
		}
	}
	return nil
}

// EvidenceSnapshotRef is an ordinary frozen Run input alias for a selected
// exact source. It keeps model namespace/name selection exact after the
// source's original mutable binding advances; it is not another handle store.
func EvidenceSnapshotRef(source contracts.ArtifactRef) (contracts.ArtifactRef, error) {
	if err := source.ValidateExact(); err != nil {
		return contracts.ArtifactRef{}, err
	}
	digest, err := contentdigest.CanonicalJSON(source)
	if err != nil {
		return contracts.ArtifactRef{}, err
	}
	return contracts.ArtifactRef{Namespace: "inputs", Name: "selected-" + strings.TrimPrefix(digest, "sha256:")}, nil
}

type EvidenceOrigin struct {
	AllocationID     string         `json:"allocation_id"`
	StageExecutionID string         `json:"stage_execution_id"`
	LogicalAgentName string         `json:"logical_agent_name"`
	InvocationID     string         `json:"invocation_id"`
	SubmissionID     string         `json:"submission_id"`
	AuditID          string         `json:"audit_id"`
	TaskID           string         `json:"task_id"`
	RunID            string         `json:"run_id"`
	ExecutionID      string         `json:"execution_id"`
	Hypothesis       *HypothesisRef `json:"hypothesis,omitempty"`
}
type EvidenceRecord struct {
	SelectedSources  []Material     `json:"selected_sources"`
	Schema           string         `json:"schema"`
	RequestDigest    string         `json:"request_digest"`
	Origin           EvidenceOrigin `json:"origin"`
	Sources          []Material     `json:"sources"`
	Interpretation   string         `json:"interpretation"`
	TestedConditions []string       `json:"tested_conditions"`
	Limitations      []string       `json:"limitations"`
}

// Validate checks stored annotations as strictly as normalized submissions.
// Origin authority and assignment membership remain with the Audit owner.
func (v EvidenceRecord) Validate() error {
	if len(v.Origin.InvocationID) > 128 || len(v.Origin.SubmissionID) > 128 || v.Schema != "contractor.audit.evidence.v1" || !contentdigest.Valid(v.RequestDigest) || len(v.Sources) != len(v.SelectedSources) {
		return errInvalid
	}
	for _, id := range []string{v.Origin.AuditID, v.Origin.TaskID, v.Origin.RunID, v.Origin.ExecutionID, v.Origin.AllocationID, v.Origin.StageExecutionID, v.Origin.LogicalAgentName, v.Origin.InvocationID, v.Origin.SubmissionID} {
		if !contracts.ValidResourceID(id) {
			return errInvalid
		}
	}
	if v.Origin.Hypothesis != nil && (!contracts.ValidResourceID(v.Origin.Hypothesis.ID) || v.Origin.Hypothesis.Revision == 0 || v.Origin.Hypothesis.Revision > contracts.MaxExactInteger) {
		return errInvalid
	}
	refs := make([]contracts.ArtifactRef, 0, len(v.Sources))
	for index, source := range v.Sources {
		selected := v.SelectedSources[index]
		if source.Validate() != nil || selected.Validate() != nil || source.ScopeKind != "audit" || source.ScopeID != v.Origin.AuditID || selected.ScopeKind != "run" || selected.ScopeID != v.Origin.RunID || source.Digest != selected.Digest || source.MediaType != selected.MediaType || source.SizeBytes != selected.SizeBytes {
			return errInvalid
		}
		refs = append(refs, source.Ref)
	}
	return validateEvidenceContent(refs, v.Interpretation, v.TestedConditions, v.Limitations)
}
