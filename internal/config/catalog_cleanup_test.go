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
	want := []string{"audit-source-check@4"}
	for _, entry := range repositoryMemoryCatalog(t).Workflows {
		want = append(want, entry.Active)
	}
	sort.Strings(want)
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
				if summary := binding.Template.Summarizer; summary != nil && summary.Instructions != nil {
					reachableInstructions[summary.Instructions.Ref] = true
				}
			}
		}
	}

	// The HTTP explorer is intentionally reusable by operator-authored Workflows.
	reachableTemplates["http_explorer@3"] = true
	template, err := snapshot.AgentTemplate("http_explorer@3")
	if err != nil {
		t.Fatal(err)
	}
	reachableInstructions[template.Instructions.Ref] = true

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
