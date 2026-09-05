package auditservice

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestCloneBaselineDoesNotShareExecutionManifestRefs(t *testing.T) {
	taskRevision := "task-r1"
	inputRevision := "input-r1"
	source := BaselineSnapshot{Inventory: BaselineInventory{
		ExecutionManifest: auditdomain.ExecutionManifest{Items: []auditdomain.ExecutionItem{{
			TaskRef: &contracts.ArtifactRef{Namespace: "audit", Name: "task", Revision: &taskRevision},
			Inputs: []auditdomain.ExactInput{{
				Name: "source",
				Ref:  contracts.ArtifactRef{Namespace: "inputs", Name: "source", Revision: &inputRevision},
			}},
		}}},
	}}

	cloned := cloneBaseline(source)
	*cloned.Inventory.ExecutionManifest.Items[0].TaskRef.Revision = "changed-task"
	*cloned.Inventory.ExecutionManifest.Items[0].Inputs[0].Ref.Revision = "changed-input"
	cloned.Inventory.ExecutionManifest.Items[0].Inputs[0].Name = "changed"

	item := source.Inventory.ExecutionManifest.Items[0]
	if *item.TaskRef.Revision != "task-r1" || *item.Inputs[0].Ref.Revision != "input-r1" ||
		item.Inputs[0].Name != "source" {
		t.Fatalf("clone mutation leaked into source: %+v", item)
	}
}
