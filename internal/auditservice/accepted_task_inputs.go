package auditservice

import (
	"sort"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
)

// acceptedTaskInputs is the exact shared input computation used by typed
// candidate budgeting and final execution-manifest construction. Declared
// task/manifest slots and same-Round outputs are materialized by dispatch.
func acceptedTaskInputs(binding auditprofile.ResolvedAuditWorkflowBinding, selection DraftSelection, deps *auditdomain.RetainedDependencyManifest, prepared ...map[string]auditstore.ExactArtifact) []auditdomain.ExactInput {
	inputs := workflowInputs(binding, selection, prepared...)
	if deps != nil {
		for slot, mapping := range binding.Inputs {
			if mapping.Source != auditprofile.AuditInputFromRetainedDependency {
				continue
			}
			for _, dependency := range deps.Dependencies {
				if dependency.Name == mapping.Name {
					inputs = append(inputs, auditdomain.ExactInput{Name: slot, Ref: dependency.Retained.Ref.Clone(), Digest: dependency.Retained.Digest})
				}
			}
		}
	}
	sort.Slice(inputs, func(i, j int) bool { return inputs[i].Name < inputs[j].Name })
	return inputs
}
