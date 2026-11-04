package auditservice

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
)

// The request digests below are what an Idempotency-Key binds for each Audit
// mutation (API-R69, PER-R24): the service replays a repeated key with the
// same digest and reports the idempotency conflict for another one. They live
// next to the request types they canonicalize, so every caller binds the same
// form; their values are pinned because stored receipts compare them.

// CreateDraftRequestDigest binds an Audit creation request: its Project,
// profile selector, inputs, normalized Runtime labels and scope, and its
// normalized metadata labels, omitted when empty. Invalid Runtime labels
// report the RuntimeConfig error and invalid metadata labels ErrInvalid.
func CreateDraftRequestDigest(params CreateDraftParams) (string, error) {
	labels, err := normalizeLabels(params.RuntimeLabels)
	if err != nil {
		return "", err
	}
	metadata, err := runlabels.NormalizeAuditLabels(params.Labels)
	if err != nil {
		return "", fmt.Errorf("%w: invalid Audit labels: %v", ErrInvalid, err)
	}
	digest, err := contentdigest.JSON(struct {
		Labels        runlabels.RunMetadataLabels      `json:"labels,omitempty"`
		ProjectID     string                           `json:"projectId"`
		Profile       ProfileSelector                  `json:"profile"`
		Inputs        map[string]contracts.ArtifactRef `json:"inputs"`
		RuntimeLabels []string                         `json:"runtimeLabels"`
		Scope         Scope                            `json:"scope"`
	}{metadata, params.ProjectID, params.Profile, params.Inputs, labels, params.Scope})
	if err != nil {
		return "", fmt.Errorf("encode Audit creation request: %w", err)
	}
	return digest, nil
}

// LifecycleRequestDigest binds one Audit lifecycle request: its action
// (start, pause, resume, cancel or delete), the Audit, the expected revision
// and the optional time limit. Every action is bound, a start included, so no
// action is encoded by omission.
func LifecycleRequestDigest(action, auditID string, revision uint64, deadlineSeconds *int) (string, error) {
	digest, err := contentdigest.JSON(struct {
		Action          string `json:"action"`
		AuditID         string `json:"auditId"`
		Revision        uint64 `json:"revision"`
		DeadlineSeconds *int   `json:"deadlineSeconds,omitempty"`
	}{action, auditID, revision, deadlineSeconds})
	if err != nil {
		return "", fmt.Errorf("encode Audit %s request: %w", action, err)
	}
	return digest, nil
}

// ProjectFindingEditRequestDigest binds the ordinary Project owner command.
func ProjectFindingEditRequestDigest(params ProjectFindingEditParams) (string, error) {
	return contentdigest.CanonicalJSON([]any{"contractor/project-finding-edit/v1", params.ProjectID,
		params.FindingID, params.ExpectedRevision, params.Verdict, params.Severity, params.Rationale, params.DuplicateTargetID})
}
