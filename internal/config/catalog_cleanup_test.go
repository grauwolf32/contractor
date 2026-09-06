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
		"audit-asvs-source-verification@1",
		"audit-openapi-operation-trace@2",
		"audit-source-check@1",
		"audit-top10-source-risk@1",
		"findings-review@1",
		"likec4-from-analysis@3",
		"likec4-from-workspace-streamline@2",
		"likec4-from-workspace@5",
		"openapi-from-analysis@2",
		"openapi-from-workspace@5",
		"podman-python-check@1",
		"security-analysis@2",
		"taint-trace-from-workspace@2",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("default Workflow selectors = %v, want %v", got, want)
	}
	for _, superseded := range []string{
		"likec4-from-analysis@2", "likec4-from-workspace-streamline@1",
		"likec4-from-workspace@3", "likec4-from-workspace@4",
		"openapi-from-analysis@1", "openapi-from-workspace@3",
		"openapi-from-workspace@4", "security-analysis@1",
		"taint-trace-from-workspace@1",
	} {
		if _, err := snapshot.Workflow(superseded); err == nil {
			t.Errorf("superseded Workflow %s remains reachable", superseded)
		}
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
