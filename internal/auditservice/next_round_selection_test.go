package auditservice

import (
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"testing"
)

func TestProposalPageSelectionBoundaries(t *testing.T) {
	audit := auditstore.Audit{AuditID: "audit", ProjectID: "project"}
	receipt := findingintake.Receipt{ReceiptID: "receipt", Document: auditdomain.FindingProposal{ProposedChecks: make([]auditdomain.ProposedCheck, 3)}, AuditHolds: []findingintake.AuditHold{{AuditID: "audit", ProjectID: "project"}}}
	for _, capacity := range []int{0, 1, 2} {
		selection := proposalCheckAccumulator{selected: make(map[string]auditdomain.FindingInventoryProposal)}
		used := map[string]bool{auditdomain.ReceiptCheckIdentity("receipt", 0): true}
		if err := selection.addPage(audit, []findingintake.Receipt{receipt}, used, capacity); err != nil {
			t.Fatal(err)
		}
		if !selection.eligible || selection.selectedCount != capacity {
			t.Fatalf("capacity %d: %+v", capacity, selection)
		}
		for _, ordinal := range selection.selected["receipt"].SelectedCheckOrdinals {
			if ordinal == 0 {
				t.Fatal("scheduled check selected again")
			}
		}
	}
	selection := proposalCheckAccumulator{selected: make(map[string]auditdomain.FindingInventoryProposal), scanned: maxNextRoundInboxScan - 1}
	invalid := findingintake.Receipt{ReceiptID: "missing-hold"}
	if err := selection.addPage(audit, []findingintake.Receipt{receipt, invalid}, nil, 1); err != nil {
		t.Fatal(err)
	}
	if selection.scanned != maxNextRoundInboxScan {
		t.Fatal("scan exceeded budget")
	}
	empty := proposalCheckAccumulator{selected: make(map[string]auditdomain.FindingInventoryProposal)}
	if err := empty.addPage(audit, []findingintake.Receipt{invalid}, nil, 0); err == nil {
		t.Fatal("zero capacity bypassed exact hold check")
	}
}
