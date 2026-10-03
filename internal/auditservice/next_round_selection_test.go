package auditservice

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
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
			}, 1, list, scheduled)
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
