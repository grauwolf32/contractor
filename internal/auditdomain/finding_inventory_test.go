package auditdomain

import (
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestBuildFindingInventoryPinsProposalAndProposedCheckIdentity(t *testing.T) {
	document := testFindingInventoryDocument(t)
	source, err := EncodeFindingInventory(document)
	if err != nil {
		t.Fatal(err)
	}
	revision := "inventory-r1"
	options := InventoryOptions{
		Round: 2, WorkflowRole: "verify", SourceInputName: "proposal_inventory",
		SourceRef: contracts.ArtifactRef{Namespace: "audit-a", Name: "proposal-inventory", Revision: &revision},
		Scope:     map[string]string{"target": "service-a"}, ApprovalRequirement: ApprovalNone,
	}
	inventory, err := BuildFindingInventory(source, options)
	if err != nil {
		t.Fatal(err)
	}
	if inventory.Worklist.Round != 2 || len(inventory.Tasks) != 2 || ValidateInventory(inventory) != nil {
		t.Fatalf("finding inventory = %+v", inventory)
	}
	first, second := inventory.Tasks[0].Document, inventory.Tasks[1].Document
	if first.Finding == nil || second.Finding == nil || first.Checklist != nil || first.Operation != nil ||
		first.Finding.ReceiptID != "receipt-a" || first.Finding.ProposedCheckOrdinal != 0 ||
		second.Finding.ProposedCheckOrdinal != 1 || first.ItemKey == second.ItemKey ||
		!reflect.DeepEqual(first.Finding.Limitations, []string{"needs-live-confirmation", "partial-trace"}) ||
		!reflect.DeepEqual(inventory.Coverage.Rows[0].Requested, []string{"static-trace"}) {
		t.Fatalf("finding tasks = (%+v, %+v)", first, second)
	}
	again, err := BuildFindingInventory(source, options)
	if err != nil || again.CanonicalInventoryDigest != inventory.CanonicalInventoryDigest ||
		again.Tasks[0].PackageDigest != inventory.Tasks[0].PackageDigest {
		t.Fatalf("repeat finding inventory = (%+v, %v)", again, err)
	}
}

func TestFindingInventoryRejectsOrderAndArtifactDrift(t *testing.T) {
	document := testFindingInventoryDocument(t)
	second := document.Proposals[0]
	second.ReceiptID = "receipt-b"
	document.Proposals = []FindingInventoryProposal{second, document.Proposals[0]}
	if _, err := EncodeFindingInventory(document); err == nil {
		t.Fatal("descending proposal identity was accepted")
	}
	document = testFindingInventoryDocument(t)
	document.Proposals[0].Proposal.Digest = testDigest('f')
	if _, err := EncodeFindingInventory(document); err == nil {
		t.Fatal("proposal content digest drift was accepted")
	}
}

func testFindingInventoryDocument(t *testing.T) FindingInventoryDocument {
	t.Helper()
	document := FindingProposal{
		Schema: FindingProposalSchema, ClientKey: "candidate-a", Title: "Authorization gap",
		Description:   "Ownership validation may be missing.",
		Subject:       FindingSubject{Kind: "openapi-operation", Key: "get-widget"},
		Hypothesis:    "A caller may access another owner's widget.",
		Preconditions: []string{}, StandardRefs: []StandardReference{}, EvidenceIDs: []string{},
		ProposedChecks: []ProposedCheck{
			{Objective: "Trace the ownership predicate.", Method: "static-trace"},
			{Objective: "Confirm behavior against the authorized target.", Method: "http-probe"},
		},
		SeveritySuggestion: "medium",
		Limitations:        []string{"partial-trace", "needs-live-confirmation"},
	}
	encoded, err := EncodeFindingProposal(document)
	if err != nil {
		t.Fatal(err)
	}
	revision := "proposal-r1"
	return FindingInventoryDocument{
		Schema: FindingInventorySchema,
		Proposals: []FindingInventoryProposal{{
			ReceiptID: "receipt-a",
			Proposal: FindingInventoryArtifact{
				Ref:    contracts.ArtifactRef{Namespace: "audit-a", Name: "proposal-a", Revision: &revision},
				Digest: digestBytes(encoded), MediaType: JSONMediaType, SizeBytes: int64(len(encoded)),
			},
			Document: document, SelectedCheckOrdinals: []int{0, 1},
		}},
	}
}
