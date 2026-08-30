package config

import (
	"sort"
	"testing"
)

func TestSnapshotWorkflowsAreSortedDeepCopies(t *testing.T) {
	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	workflows := snapshot.Workflows()
	if len(workflows) < 2 {
		t.Fatalf("repository snapshot has only %d Workflow", len(workflows))
	}
	selectors := make([]string, len(workflows))
	for index, workflow := range workflows {
		selectors[index] = workflow.Ref.Name + "@" + workflow.Ref.Version
	}
	if !sort.StringsAreSorted(selectors) {
		t.Fatalf("Workflow refs are not sorted: %v", selectors)
	}

	selected := workflows[0]
	selected.Parameters["tampered"] = ParameterSlot{Required: true}
	selected.Stages[selected.EntryStage] = ResolvedStage{Objective: "tampered"}
	loaded, err := snapshot.Workflow(selectors[0])
	if err != nil {
		t.Fatal(err)
	}
	if _, exists := loaded.Parameters["tampered"]; exists || loaded.Stages[loaded.EntryStage].Objective == "tampered" {
		t.Fatal("caller mutation changed the immutable Workflow snapshot")
	}
}
