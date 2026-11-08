package reporting

import (
	"fmt"
	"testing"
	"time"
)

func costTestContribution(id, amount string, complete bool) *CostContribution {
	return &CostContribution{ReportID: id, ReceivedAt: time.Date(2026, 11, 8, 0, 0, 0, 0, time.UTC), Complete: complete, ModelCalls: 1, Cost: &LLMCost{Currency: "USD", Calls: 1, Groups: []LLMCostGroup{{Model: "model", Phase: "main", Source: "provider", Calls: 1, Amount: amount}}}}
}

func TestLLMCostTotalsReplaceAndPreparedBreakdowns(t *testing.T) {
	total := NewLLMCostTotals(false)
	first := costTestContribution("partial", "0.1", false)
	if err := total.ReplaceCostContribution(nil, first, "worker", true, first.ReceivedAt); err != nil {
		t.Fatal(err)
	}
	if total.Amount != "0.1" || total.UnknownReports != "1" {
		t.Fatalf("partial: %+v", total)
	}
	next := costTestContribution("complete", "0.2", true)
	if err := total.ReplaceCostContribution(first, next, "worker", true, next.ReceivedAt); err != nil {
		t.Fatal(err)
	}
	if total.Amount != "0.2" || total.Reports != "1" || total.UnknownReports != "0" || total.ChecksAmount != "0.2" {
		t.Fatalf("replacement: %+v", total)
	}
	if changed, err := CostContributionChanged(next, *next); err != nil || changed {
		t.Fatalf("replay: %v %v", changed, err)
	}
	stale := *first
	stale.ReportID = next.ReportID
	if changed, err := CostContributionChanged(next, stale); err != nil || changed {
		t.Fatalf("placeholder overwrote complete: %v %v", changed, err)
	}
	standalone := costTestContribution("other", "0.00000625", true)
	if err := total.ReplaceCostContribution(nil, standalone, "planner", false, next.ReceivedAt); err != nil {
		t.Fatal(err)
	}
	response := BuildLLMCostSummary("project", "project", total, false, true, next.ReceivedAt)
	if response.Amount == nil || *response.Amount != "0.20000625" || response.Coverage != "complete" || response.Basis != "reported" || len(response.Activity) != 2 || len(response.Models) != 1 || response.Models[0].Amount != "0.20000625" {
		t.Fatalf("summary: %+v", response)
	}
	total.HistoricalGap = true
	if got := BuildLLMCostSummary("project", "project", total, false, true, next.ReceivedAt); got.Coverage != "partial" {
		t.Fatalf("historical data invented: %+v", got)
	}
}

func TestLLMCostAggregateRemainderIsBoundedAndReversible(t *testing.T) {
	total := NewLLMCostTotals(false)
	for i := 0; i < 1000; i++ {
		c := costTestContribution(fmt.Sprint(i), "0.1", true)
		c.Cost.Groups[0].Model = fmt.Sprintf("model-%04d", i)
		if err := total.ReplaceCostContribution(nil, c, "worker", false, c.ReceivedAt); err != nil {
			t.Fatal(err)
		}
	}
	if len(total.Groups) != 49 || total.Amount != "100" || total.Calls != "1000" {
		t.Fatalf("unbounded/inexact aggregate: %+v", total)
	}
	old := costTestContribution("old", "0.1", true)
	old.Cost.Groups[0].Model = "model-0999"
	next := costTestContribution("new", "0.2", true)
	next.Cost.Groups[0].Model = "model-0999"
	if err := total.ReplaceCostContribution(old, next, "worker", false, next.ReceivedAt); err != nil {
		t.Fatal(err)
	}
	if total.Amount != "100.1" || len(total.Groups) != 49 {
		t.Fatalf("remainder replacement: %+v", total)
	}
}

func TestLLMCostSummaryDoesNotFabricateMissingHistoryOrRoundSmallCosts(t *testing.T) {
	now := time.Date(2026, 11, 8, 0, 0, 0, 0, time.UTC)
	for _, test := range []struct {
		name             string
		total            LLMCostTotals
		pending, hasWork bool
		coverage         string
		known            bool
	}{
		{"new empty", NewLLMCostTotals(false), false, false, "complete", true},
		{"unreported work", NewLLMCostTotals(false), false, true, "unavailable", false},
		{"historical", NewLLMCostTotals(true), false, false, "unavailable", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			got := BuildLLMCostSummary("project", "project", test.total, test.pending, test.hasWork, now)
			if got.Coverage != test.coverage || (got.Amount != nil) != test.known {
				t.Fatalf("summary %+v", got)
			}
		})
	}
	total := NewLLMCostTotals(false)
	c := costTestContribution("report", "0.000000000000000001", true)
	if err := total.ReplaceCostContribution(nil, c, "planner", true, now); err != nil {
		t.Fatal(err)
	}
	got := BuildLLMCostSummary("audit", "audit", total, true, true, now)
	if got.Coverage != "pending" || got.Amount == nil || *got.Amount != "0.000000000000000001" {
		t.Fatalf("tiny cost lost: %+v", got)
	}
	total.Amount = "0"
	if err := total.Validate(); err == nil {
		t.Fatal("inconsistent aggregate accepted")
	}
}
