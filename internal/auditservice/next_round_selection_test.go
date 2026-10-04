package auditservice

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
)

func TestNextRoundStopReasons(t *testing.T) {
	for _, test := range []struct {
		name         string
		selection    proposalCheckSelection
		roundOrdinal int
		maxRounds    int
		capacity     int
		wantCode     string
	}{
		{"round complete", proposalCheckSelection{}, 1, 2, 1, ""},
		{"round budget with selected work", proposalCheckSelection{Proposals: []auditdomain.FindingInventoryProposal{{}}}, 2, 2, 1, "round_budget_exhausted"},
		{"round budget with pending work", proposalCheckSelection{Eligible: true}, 2, 2, 0, "round_budget_exhausted"},
		{"item budget", proposalCheckSelection{Eligible: true}, 1, 2, 0, "item_budget_exhausted"},
		{"scan budget", proposalCheckSelection{ScanExhausted: true}, 1, 2, 1, "proposal_scan_budget_exhausted"},
		{"available work", proposalCheckSelection{Proposals: []auditdomain.FindingInventoryProposal{{}}, Eligible: true}, 1, 2, 1, ""},
		{"unschedulable proposals only", proposalCheckSelection{Unschedulable: []unschedulableProposal{{ReceiptID: "receipt-a", Limit: "proposal_inventory.bytes"}}}, 1, 2, 1, "proposal_inventory_limit_exceeded"},
		{"unschedulable beside work", proposalCheckSelection{Eligible: true, Proposals: []auditdomain.FindingInventoryProposal{{}}, Unschedulable: []unschedulableProposal{{ReceiptID: "receipt-a", Limit: "proposal_inventory.bytes"}}}, 1, 2, 1, ""},
		{"scan budget before unschedulable", proposalCheckSelection{ScanExhausted: true, Unschedulable: []unschedulableProposal{{ReceiptID: "receipt-a", Limit: "proposal_inventory.bytes"}}}, 1, 2, 1, "proposal_scan_budget_exhausted"},
	} {
		t.Run(test.name, func(t *testing.T) {
			reason := nextRoundStopReason(test.selection, test.roundOrdinal, test.maxRounds, test.capacity)
			if test.wantCode == "" {
				if reason != nil {
					t.Fatalf("unexpected stop reason: %+v", reason)
				}
			} else if reason == nil || reason.Code != test.wantCode || reason.Message == "" {
				t.Fatalf("stop reason = %+v, want %s", reason, test.wantCode)
			}
		})
	}
}

func TestProposalPageSelectionBoundaries(t *testing.T) {
	audit := auditstore.Audit{AuditID: "audit", ProjectID: "project"}
	receipt := findingintake.Receipt{ReceiptID: "receipt", Document: auditdomain.FindingProposal{ProposedChecks: make([]auditdomain.ProposedCheck, 3)}, AuditHolds: []findingintake.AuditHold{{AuditID: "audit", ProjectID: "project"}}}
	for _, capacity := range []int{0, 1, 2} {
		selection := proposalCheckAccumulator{budget: testNextRoundBudget(t, capacity)}
		used := map[string]bool{auditdomain.ReceiptCheckIdentity("receipt", 0): true}
		if err := selection.addPage(audit, []findingintake.Receipt{receipt}, used); err != nil {
			t.Fatal(err)
		}
		if !selection.eligible || selection.budget.Checks() != capacity {
			t.Fatalf("capacity %d: %+v", capacity, selection)
		}
		for _, proposal := range selection.selected {
			for _, ordinal := range proposal.SelectedCheckOrdinals {
				if ordinal == 0 {
					t.Fatal("scheduled check selected again")
				}
			}
		}
	}
	selection := proposalCheckAccumulator{budget: testNextRoundBudget(t, 1), scanned: maxNextRoundInboxScan - 1}
	invalid := findingintake.Receipt{ReceiptID: "missing-hold"}
	if err := selection.addPage(audit, []findingintake.Receipt{receipt, invalid}, nil); err != nil {
		t.Fatal(err)
	}
	if selection.scanned != maxNextRoundInboxScan {
		t.Fatal("scan exceeded budget")
	}
	empty := proposalCheckAccumulator{budget: testNextRoundBudget(t, 0)}
	if err := empty.addPage(audit, []findingintake.Receipt{invalid}, nil); !errors.Is(err, ErrRoundPreparationInconsistent) {
		t.Fatalf("zero capacity bypassed exact hold check: %v", err)
	}
}

// TestNextRoundSelectionSplitsChecksAcrossRounds covers a profile whose
// per-Round limit exceeds the inventory item maximum: 4608 eligible checks
// form one full Round within the inventory limits and a later Round.
func TestNextRoundSelectionSplitsChecksAcrossRounds(t *testing.T) {
	const receipts, checksPerReceipt = 9, auditdomain.MaximumCoverageValues
	inbox := make([]findingintake.Receipt, receipts)
	for index := range inbox {
		checks := make([]auditdomain.ProposedCheck, checksPerReceipt)
		for ordinal := range checks {
			checks[ordinal] = auditdomain.ProposedCheck{
				Objective: fmt.Sprintf("Verify condition %d.", ordinal), Method: "static-trace",
			}
		}
		inbox[index] = testInboxReceipt(t, fmt.Sprintf("receipt-%d", index), time.Unix(int64(index+1), 0), checks)
	}
	scheduled := make(map[string]bool)
	list := func(_ context.Context, _, _ string, query findingintake.ListQuery) ([]findingintake.Receipt, error) {
		start := 0
		for start < len(inbox) && query.AfterReceiptID != "" && inbox[start].ReceiptID <= query.AfterReceiptID {
			start++
		}
		return inbox[start:min(start+query.Limit, len(inbox))], nil
	}
	used := func(_ context.Context, _ string, _ []string) (map[string]bool, error) { return scheduled, nil }
	audit := auditstore.Audit{AuditID: "audit", OwnerID: "owner", ProjectID: "project"}
	capacity := roundItemCapacity(10_000, 100_000, 0)
	var rounds []int
	for range 3 {
		selection, err := selectNextProposalChecksFromInbox(t.Context(), audit, testNextRoundBudget(t, capacity), list, used)
		if err != nil {
			t.Fatal(err)
		}
		if len(selection.Proposals) == 0 {
			if selection.Eligible || nextRoundStopReason(selection, 2, 5, capacity) != nil {
				t.Fatalf("consumed inbox = %+v", selection)
			}
			break
		}
		if _, err := auditdomain.EncodeFindingInventory(auditdomain.FindingInventoryDocument{
			Schema: auditdomain.FindingInventorySchema, Proposals: selection.Proposals,
		}); err != nil {
			t.Fatalf("Round %d inventory: %v", len(rounds)+2, err)
		}
		checks := 0
		for _, proposal := range selection.Proposals {
			for _, ordinal := range proposal.SelectedCheckOrdinals {
				scheduled[auditdomain.ReceiptCheckIdentity(proposal.ReceiptID, ordinal)] = true
				checks++
			}
		}
		rounds = append(rounds, checks)
	}
	if fmt.Sprint(rounds) != "[4096 512]" {
		t.Fatalf("Round sizes = %v, want [4096 512]", rounds)
	}
}

func TestNextRoundScanBoundaryDistinguishesCompleteInboxFromTruncation(t *testing.T) {
	for _, test := range []struct {
		receipts      int
		wantExhausted bool
		wantPages     int
	}{
		{receipts: 9_999, wantPages: 50},
		{receipts: 10_000, wantPages: 51},
		{receipts: 10_001, wantExhausted: true, wantPages: 51},
	} {
		t.Run(fmt.Sprint(test.receipts), func(t *testing.T) {
			pages := 0
			list := func(_ context.Context, ownerID, auditID string, query findingintake.ListQuery) ([]findingintake.Receipt, error) {
				pages++
				if ownerID != "owner" || auditID != "audit" || query.Limit < 1 || query.Limit > nextRoundInboxPage {
					t.Fatalf("unexpected inbox page query: owner=%s audit=%s query=%+v", ownerID, auditID, query)
				}
				start := 0
				if query.AfterReceiptID != "" {
					if query.AfterCreatedAt == nil {
						t.Fatal("receipt cursor is missing its creation time")
					}
					index, err := strconv.Atoi(strings.TrimPrefix(query.AfterReceiptID, "receipt-"))
					if err != nil {
						t.Fatal(err)
					}
					start = index + 1
				}
				end := min(start+query.Limit, test.receipts)
				result := make([]findingintake.Receipt, 0, max(0, end-start))
				for index := start; index < end; index++ {
					result = append(result, findingintake.Receipt{
						ReceiptID: fmt.Sprintf("receipt-%05d", index), CreatedAt: time.Unix(int64(index+1), 0),
						Document:   auditdomain.FindingProposal{ProposedChecks: []auditdomain.ProposedCheck{{}}},
						AuditHolds: []findingintake.AuditHold{{AuditID: "audit", ProjectID: "project"}},
					})
				}
				return result, nil
			}
			scheduled := func(_ context.Context, auditID string, ids []string) (map[string]bool, error) {
				if auditID != "audit" {
					t.Fatalf("unexpected scheduled-check Audit %s", auditID)
				}
				used := make(map[string]bool, len(ids))
				for _, id := range ids {
					used[auditdomain.ReceiptCheckIdentity(id, 0)] = true
				}
				return used, nil
			}
			selection, err := selectNextProposalChecksFromInbox(t.Context(), auditstore.Audit{
				AuditID: "audit", OwnerID: "owner", ProjectID: "project",
			}, testNextRoundBudget(t, 1), list, scheduled)
			if err != nil || selection.Eligible || len(selection.Proposals) != 0 ||
				selection.ScanExhausted != test.wantExhausted || pages != test.wantPages {
				t.Fatalf("fully consumed inbox size %d: selection=%+v pages=%d error=%v",
					test.receipts, selection, pages, err)
			}
			reason := nextRoundStopReason(selection, 1, 2, 1)
			if test.wantExhausted && (reason == nil || reason.Code != "proposal_scan_budget_exhausted") ||
				!test.wantExhausted && reason != nil {
				t.Fatalf("scan boundary stop reason = %+v", reason)
			}
		})
	}
}

func TestUnschedulableStopReasonNamesBoundedProposals(t *testing.T) {
	proposals := make([]unschedulableProposal, maximumReportedUnschedulable+2)
	for index := range proposals {
		proposals[index] = unschedulableProposal{ReceiptID: fmt.Sprintf("receipt-%d", index), Limit: "proposal_inventory.bytes"}
	}
	reason := unschedulableStopReason(proposals)
	if reason.Code != "proposal_inventory_limit_exceeded" ||
		!strings.Contains(reason.Message, "receipt-0 (proposal_inventory.bytes)") ||
		strings.Contains(reason.Message, fmt.Sprintf("receipt-%d ", maximumReportedUnschedulable)) ||
		!strings.HasSuffix(reason.Message, ", and 2 more.") {
		t.Fatalf("stop reason = %+v", reason)
	}
	if one := unschedulableStopReason(proposals[:1]); !strings.HasSuffix(one.Message, ": receipt-0 (proposal_inventory.bytes).") {
		t.Fatalf("single stop reason = %+v", one)
	}
}

func testNextRoundBudget(t *testing.T, capacity int) *auditdomain.FindingRoundBudget {
	t.Helper()
	namespace := auditdomain.ArtifactNamespace("audit")
	budget, err := auditdomain.NewFindingRoundBudget(capacity, auditdomain.FindingRoundShape{
		Inventory: nextRoundInventoryOptions(2, config.ResolvedAuditProfile{
			Mode:      config.AuditModeFindingVerification,
			Inventory: config.AuditInventory{ItemWorkflowRole: "verify"},
		}, BaselineSnapshot{}, auditdomain.ApprovalNone, contracts.ArtifactRef{
			Namespace: namespace, Name: auditdomain.DeterministicID("proposal-inventory", "audit"),
			Revision: &provisionalRevision,
		}),
		TaskNamespace: namespace, TaskRevision: provisionalRevision,
	})
	if err != nil {
		t.Fatal(err)
	}
	return budget
}

func testInboxReceipt(
	t *testing.T, receiptID string, createdAt time.Time, checks []auditdomain.ProposedCheck,
) findingintake.Receipt {
	t.Helper()
	document := auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate-" + receiptID,
		Title: "Candidate " + receiptID, Description: "A retained candidate for review.",
		Subject:       &auditdomain.FindingSubject{Kind: "component", Key: "component-" + receiptID},
		Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
		EvidenceIDs: []string{}, ProposedChecks: checks,
		SeveritySuggestion: "medium", Limitations: []string{},
	}
	encoded, err := auditdomain.EncodeFindingProposal(document)
	if err != nil {
		t.Fatal(err)
	}
	revision := "proposal-r1"
	return findingintake.Receipt{
		ReceiptID: receiptID, CreatedAt: createdAt, Document: document,
		AuditHolds: []findingintake.AuditHold{{
			AuditID: "audit", ProjectID: "project",
			Proposal: findingintake.ExactArtifact{
				Ref:    contracts.ArtifactRef{Namespace: "audit-finding-proposals", Name: receiptID, Revision: &revision},
				Digest: auditdomain.DigestBytes(encoded), MediaType: auditdomain.JSONMediaType, SizeBytes: int64(len(encoded)),
			},
		}},
	}
}
