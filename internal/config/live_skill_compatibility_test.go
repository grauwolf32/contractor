package config

import (
	"slices"
	"testing"
)

func TestRepositoryLiveSkillCompatibilityBoundary(t *testing.T) {
	t.Parallel()

	targetSkills := map[string]bool{
		"auth": true, "code-exec": true, "exploit": true,
	}
	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	caidoSelectors := []string{}
	for selector, template := range snapshot.templates {
		for _, skill := range template.Skills {
			if targetSkills[skill.Name] {
				t.Errorf("current AgentTemplate %s prematurely selects skills/%s", selector, skill.Name)
			}
			if skill.Name == "caido" {
				caidoSelectors = append(caidoSelectors, selector)
			}
		}
	}
	slices.Sort(caidoSelectors)
	wantCaido := []string{"caido_analyst@1"}
	for _, entry := range repositoryMemoryCatalog(t).Templates {
		if entry.Legacy == "caido_analyst@1" {
			wantCaido = append(wantCaido, entry.Active)
		}
	}
	slices.Sort(wantCaido)
	if !slices.Equal(caidoSelectors, wantCaido) {
		t.Fatalf("skills/caido selectors = %v, want %v", caidoSelectors, wantCaido)
	}

	descriptors := MVPDescriptors()
	futureOperations := map[string]bool{
		"get_vulnerability": true, "submit_verdict": true,
		"run_python": true, "execute_bash": true,
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
