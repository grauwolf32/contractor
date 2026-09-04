package config

import (
	"reflect"
	"sort"
	"testing"
)

func TestRepositoryDefaultCatalogContainsOnlyCurrentWorkflows(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	got := make([]string, 0, len(snapshot.workflows))
	for selector := range snapshot.workflows {
		got = append(got, selector)
	}
	sort.Strings(got)
	want := []string{
		"artifact-copy@1",
		"likec4-from-analysis@2",
		"likec4-from-workspace-streamline@1",
		"likec4-from-workspace@3",
		"openapi-from-analysis@1",
		"openapi-from-workspace@3",
		"security-analysis@1",
		"taint-trace-from-workspace@1",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("default Workflow selectors = %v, want %v", got, want)
	}
}

func TestRepositoryDefaultCatalogHasNoUnintentionallyOrphanedWorkerConfig(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	reachableTemplates := make(map[string]bool)
	reachableInstructions := make(map[string]bool)
	for _, workflow := range snapshot.workflows {
		for _, stage := range workflow.Stages {
			reachableInstructions[stage.Instructions.Ref] = true
			for _, binding := range stage.Agents {
				selector := binding.Template.Ref.TemplateID + "@" + binding.Template.Ref.Version
				reachableTemplates[selector] = true
				reachableInstructions[binding.Template.Instructions.Ref] = true
			}
		}
	}

	// http_explorer@1 is intentionally reusable from operator-authored
	// Workflows even though no built-in Workflow currently selects it.
	reachableTemplates["http_explorer@1"] = true
	reachableInstructions["instructions/http-explorer-worker.md"] = true
	for selector := range snapshot.templates {
		if !reachableTemplates[selector] {
			t.Errorf("default AgentTemplate %s is unreachable", selector)
		}
	}
	for ref := range snapshot.instructions {
		if !reachableInstructions[ref] {
			t.Errorf("default instructions %s are unreachable", ref)
		}
	}
}
