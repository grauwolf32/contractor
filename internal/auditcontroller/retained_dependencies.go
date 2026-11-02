package auditcontroller

import (
	"bytes"
	"context"
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
)

func (b *PinnedSubmissionBuilder) validateItemDependencies(ctx context.Context, snapshot auditstore.ReconcileSnapshot, binding auditprofile.ResolvedAuditWorkflowBinding, baseline auditservice.BaselineSnapshot, item auditstore.Item) error {
	if item.Dependencies == nil {
		return nil
	}
	d := item.Dependencies
	if auditdomain.ValidateRetainedDependencies(*d) != nil || d.OwnerID != snapshot.Audit.OwnerID || d.ProjectID != snapshot.Audit.ProjectID || d.AuditID != snapshot.Audit.AuditID || d.ItemKey != item.ItemKey || d.BaselineDigest != auditdomain.DigestBytes(snapshot.Audit.BaselineSnapshot) || d.ProfileDigest != snapshot.Audit.Profile.Digest {
		return invalidSubmission("accepted dependency owner or baseline differs")
	}
	workflow, _ := json.Marshal(binding.Workflow)
	skills, err := selectSkills(binding.Workflow, baseline.Skills)
	if err != nil {
		return err
	}
	skillBytes, _ := json.Marshal(skills)
	if d.WorkflowDigest != auditdomain.DigestBytes(workflow) || d.SkillsDigest != auditdomain.DigestBytes(skillBytes) {
		return invalidSubmission("accepted dependency configuration closure differs")
	}
	// Resolve the complete closure, including leaves with no Workflow slot,
	// before creating an intent. Embedded JSON refs are never traversed.
	for _, dependency := range d.Dependencies {
		var resolved auditstore.ExactArtifact
		var err error
		if dependency.Kind == "prepare-output" {
			resolved, err = b.artifacts.ResolveAuditExact(ctx, snapshot.Audit.AuditID, dependency.Retained)
		} else {
			resolved, err = b.artifacts.ResolveProjectExact(ctx, snapshot.Audit.ProjectID, dependency.Retained)
		}
		if err != nil {
			return err
		}
		if dependency.Retained.Verify(resolved) != nil {
			return invalidSubmission("accepted dependency exact metadata differs")
		}
	}
	return nil
}
func sameDependencyManifest(a, b *auditdomain.RetainedDependencyManifest) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	left, err := auditdomain.EncodeRetainedDependencies(*a)
	if err != nil {
		return false
	}
	right, err := auditdomain.EncodeRetainedDependencies(*b)
	return err == nil && bytes.Equal(left, right)
}

func uniqueDependencyInputs(inputs []auditstore.ExactArtifact) ([]auditstore.ExactArtifact, error) {
	result := make([]auditstore.ExactArtifact, 0, len(inputs))
	seen := map[string]auditstore.ExactArtifact{}
	for _, input := range inputs {
		if previous, exists := seen[input.Ref.Key()]; exists {
			if previous.Verify(input) != nil {
				return nil, invalidSubmission("one exact dependency revision has conflicting content")
			}
			continue
		}
		seen[input.Ref.Key()] = input
		result = append(result, input)
	}
	return result, nil
}
