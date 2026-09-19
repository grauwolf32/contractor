package config

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
)

type AgentTemplateWorkflowBinding struct {
	Workflow      WorkflowRef `json:"workflow"`
	Stage         string      `json:"stage"`
	LogicalWorker string      `json:"logicalWorker"`
}

func (left AgentTemplateWorkflowBinding) Less(right AgentTemplateWorkflowBinding) bool {
	if left.Workflow.Name != right.Workflow.Name {
		return left.Workflow.Name < right.Workflow.Name
	}
	if left.Workflow.Version != right.Workflow.Version {
		return left.Workflow.Version < right.Workflow.Version
	}
	if left.Stage != right.Stage {
		return left.Stage < right.Stage
	}
	return left.LogicalWorker < right.LogicalWorker
}

// AgentTemplateWorkflowBindings holds one immutable snapshot's sorted uses and
// cursor identity. Page returns only a detached, bounded portion of the index.
type AgentTemplateWorkflowBindings struct {
	SourceFingerprint string
	items             []AgentTemplateWorkflowBinding
	err               error
}

func (index AgentTemplateWorkflowBindings) Page(after *AgentTemplateWorkflowBinding, limit int) []AgentTemplateWorkflowBinding {
	start := 0
	if after != nil {
		start = sort.Search(len(index.items), func(i int) bool { return after.Less(index.items[i]) })
	}
	count := min(max(limit, 0), len(index.items)-start)
	result := make([]AgentTemplateWorkflowBinding, count)
	copy(result, index.items[start:start+count])
	return result
}

type templateBindingIdentity struct{ name, version, digest string }

func (s *Snapshot) buildWorkflowBindingIndex() {
	uses := make(map[templateBindingIdentity][]AgentTemplateWorkflowBinding)
	for _, workflow := range s.workflows {
		for stageName, stage := range workflow.Stages {
			for worker, binding := range stage.Agents {
				ref := binding.Template.Ref
				key := templateBindingIdentity{ref.TemplateID, ref.Version, ref.Digest}
				uses[key] = append(uses[key], AgentTemplateWorkflowBinding{Workflow: workflow.Ref, Stage: stageName, LogicalWorker: worker})
			}
		}
	}
	// Preserve the previous cursor fingerprint byte for byte. Encode the catalog
	// once while constructing the snapshot, never on a binding-page request.
	workflows := make([]ResolvedWorkflow, 0, len(s.workflows))
	for _, key := range sortedMapKeys(s.workflows) {
		workflows = append(workflows, s.workflows[key])
	}
	workflowJSON, encodeErr := json.Marshal(workflows)
	s.workflowBindings = make(map[string]AgentTemplateWorkflowBindings, len(s.templates))
	for selector, template := range s.templates {
		ref := template.Ref
		items := uses[templateBindingIdentity{ref.TemplateID, ref.Version, ref.Digest}]
		sort.Slice(items, func(i, j int) bool { return items[i].Less(items[j]) })
		index := AgentTemplateWorkflowBindings{items: items, err: encodeErr}
		if index.err == nil {
			refJSON, err := json.Marshal(ConfigurationRef{Kind: ConfigurationAgentTemplates, Name: ref.TemplateID, Version: ref.Version, Digest: ref.Digest})
			index.err = err
			if err == nil {
				digest := sha256.New()
				digest.Write([]byte(`{"Template":`))
				digest.Write(refJSON)
				digest.Write([]byte(`,"Workflows":`))
				digest.Write(workflowJSON)
				digest.Write([]byte(`}`))
				index.SourceFingerprint = hex.EncodeToString(digest.Sum(nil))
			}
		}
		s.workflowBindings[selector] = index
	}
}

func (s *Snapshot) AgentTemplateWorkflowBindings(raw string) (AgentTemplateWorkflowBindings, error) {
	selector, err := ParseSelector(raw)
	if err != nil {
		return AgentTemplateWorkflowBindings{}, fmt.Errorf("%w: invalid selector", ErrInvalidPublication)
	}
	index, ok := s.workflowBindings[selector.String()]
	if !ok {
		return AgentTemplateWorkflowBindings{}, ErrConfigurationNotFound
	}
	if index.err != nil {
		return AgentTemplateWorkflowBindings{}, fmt.Errorf("fingerprint catalog source: %w", index.err)
	}
	return index, nil
}
func (m *Manager) AgentTemplateWorkflowBindings(raw string) (AgentTemplateWorkflowBindings, error) {
	return m.Snapshot().AgentTemplateWorkflowBindings(raw)
}
