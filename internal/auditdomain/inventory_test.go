package auditdomain

import "github.com/grauwolf32/contractor/internal/contracts"

func testInventoryOptions(role string) InventoryOptions {
	revision := "source-revision-1"
	return InventoryOptions{
		Round: 1, WorkflowRole: role, SourceInputName: "source",
		SourceRef:           contracts.ArtifactRef{Namespace: "inputs", Name: "source", Revision: &revision},
		ApprovalRequirement: ApprovalNone,
	}
}
