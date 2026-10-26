package auditbaseline

import (
	"encoding/json"
	"fmt"
	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
)

// ConsumerContentPins preserves the opaque Workflow/Skill JSON bytes used by
// accepted retained dependencies, independently of configuration identity.
func ConsumerContentPins(binding auditprofile.ResolvedAuditWorkflowBinding, baseline BaselineSnapshot) (string, string, error) {
	workflow, err := json.Marshal(binding.Workflow)
	if err != nil {
		return "", "", err
	}
	refs, err := resolved.WorkflowSkillRefs(binding.Workflow)
	if err != nil {
		return "", "", err
	}
	selected := make([]agentskills.RunSkillSnapshot, 0, len(refs))
	for _, ref := range refs {
		found := false
		for _, skill := range baseline.Skills {
			if skill.Name == ref.Name {
				selected = append(selected, skill)
				found = true
				break
			}
		}
		if !found {
			return "", "", fmt.Errorf("%w: retained_dependency_skill_unavailable", auditstore.ErrInvalid)
		}
	}
	skills, err := json.Marshal(selected)
	if err != nil {
		return "", "", err
	}
	return auditdomain.DigestBytes(workflow), auditdomain.DigestBytes(skills), nil
}
