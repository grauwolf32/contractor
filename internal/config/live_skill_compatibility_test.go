package config

import (
	"slices"
	"testing"
)

func TestRepositoryLiveSkillCompatibilityBoundary(t *testing.T) {
	t.Parallel()

	targetSkills := map[string]bool{
		"auth": true, "caido": true, "code-exec": true, "exploit": true,
	}
	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	for selector, template := range snapshot.templates {
		for _, skill := range template.Skills {
			if targetSkills[skill.Name] {
				t.Errorf("current AgentTemplate %s prematurely selects skills/%s", selector, skill.Name)
			}
		}
	}

	descriptors := MVPDescriptors()
	futureOperations := map[string]bool{
		"http_request": true, "http_session_set": true,
		"get_vulnerability": true, "submit_verdict": true,
		"run_python": true, "execute_bash": true,
		"caido_replay": true, "caido_automate_run": true,
		"caido_history": true, "caido_request_detail": true,
		"caido_workflow_list": true, "caido_workflow_run": true,
		"caido_workflow_findings": true,
	}
	for ref, descriptor := range descriptors.Toolsets {
		for _, tool := range descriptor.Tools {
			if futureOperations[tool] {
				t.Errorf("current Toolset %s unexpectedly exports future live-testing operation %s", ref, tool)
			}
		}
	}

	memory, ok := descriptors.Toolsets["memory-tools@1"]
	if !ok {
		t.Fatal("memory-tools@1 descriptor is missing")
	}
	wantMemory := []string{
		"append_memory", "list_memories", "list_memory_tags",
		"read_memory", "search_memory", "write_memory",
	}
	if !slices.Equal(memory.Tools, wantMemory) {
		t.Fatalf("memory-tools@1 tools = %v, want %v", memory.Tools, wantMemory)
	}
}
