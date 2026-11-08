package reporting

import (
	"fmt"
	"testing"
)

func TestLLMCostExactAccumulationBoundsAndIsolation(t *testing.T) {
	var c LLMCostCollector
	c.Observe("model", "main", "provider", "0.1")
	c.Observe("model", "main", "provider", "0.2")
	c.Observe("model", "main", "provider", "invalid")
	got := c.Snapshot(0)
	if got.Calls != 3 || len(got.Groups) != 1 || got.Groups[0].Amount != "0.3" || got.Groups[0].Calls != 2 {
		t.Fatalf("exact cost: %+v", got)
	}
	got.Groups[0].Amount = "99"
	if c.Snapshot(0).Groups[0].Amount != "0.3" {
		t.Fatal("snapshot aliases reducer")
	}
	for i := 0; i < 65; i++ {
		c.Observe(fmt.Sprintf("model-%d", i), "main", "gateway", "1e-18")
	}
	got = c.Snapshot(0)
	if got.Calls != 68 || len(got.Groups) != 64 {
		t.Fatalf("unbounded cost: %+v", got)
	}
	if err := got.Validate(); err != nil {
		t.Fatal(err)
	}
	var max LLMCostCollector
	max.Observe("model", "main", "provider", "999999999999999999")
	max.Observe("model", "main", "provider", "1")
	if got := max.Snapshot(0); got.Calls != 2 || got.Groups[0].Calls != 1 || got.Groups[0].Amount != "999999999999999999" {
		t.Fatalf("overflow fabricated a charge: %+v", got)
	}
}
