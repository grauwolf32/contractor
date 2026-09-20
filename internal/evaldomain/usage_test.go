package evaldomain

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestEvalAuditAccountingFixture(t *testing.T) {
	data, err := os.ReadFile(filepath.Join(fixtureDir, "audit-accounting.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Observation UsageObservation                                             `json:"observation"`
		Expected    struct{ TotalTokens, ModelCalls, WallMS, MemberCount int64 } `json:"expected"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	out, err := NormalizeUsage(fixture.Observation)
	if err != nil {
		t.Fatal(err)
	}
	if out.TotalTokens.Value == nil || *out.TotalTokens.Value != float64(fixture.Expected.TotalTokens) || *out.ModelCalls.Value != float64(fixture.Expected.ModelCalls) || *out.WallMS.Value != float64(fixture.Expected.WallMS) {
		t.Fatalf("wrong parent/child accounting: %+v", out)
	}
	if out.TotalTokens.Completeness != "complete" || out.CachedInputTokens.Completeness != "unavailable" {
		t.Fatal("lost per-metric completeness")
	}
	if err := validateUsage(out, fixture.Observation.MemberID); err != nil {
		t.Fatal(err)
	}
	// Missing role metrics make tokens partial without losing known values or
	// replacing the parent's known wall interval with the sum of child times.
	partial := fixture.Observation
	partial.Attempts = partial.Attempts[:3]
	out, err = NormalizeUsage(partial)
	if err != nil {
		t.Fatal(err)
	}
	if out.TotalTokens.Completeness != "partial" || *out.TotalTokens.Value != 60 || out.WallMS.Completeness != "complete" || *out.WallMS.Value != 2000 {
		t.Fatal("missing role scope was hidden")
	}
	conflict := fixture.Observation
	conflict.Attempts = append([]AttemptObservation{}, conflict.Attempts...)
	last := conflict.Attempts[len(conflict.Attempts)-1]
	last.Metrics = map[string]int64{"totalTokens": 21}
	conflict.Attempts[len(conflict.Attempts)-1] = last
	if _, err := NormalizeUsage(conflict); err == nil {
		t.Fatal("conflicting cumulative snapshots accepted")
	}
	foreign := fixture.Observation
	foreign.Attempts = append([]AttemptObservation{}, foreign.Attempts...)
	foreign.Attempts[0].RunID = "other-owner-run"
	if _, err := NormalizeUsage(foreign); err == nil {
		t.Fatal("unassociated execution counted")
	}
}

func TestEvalTraceExampleAgreesWithAllSelectedMembers(t *testing.T) {
	v := objectFixture(t, "member-page")
	summary := asObject(v["experimentSummary"])
	type counts struct {
		expected, submitted, terminal, scored, passed, succeeded int
		tokens                                                   float64
		measured                                                 int
	}
	arms := map[string]*counts{"a": {}, "b": {}}
	pairs := map[string]map[string]map[string]any{}
	for _, raw := range asRows(v["items"]) {
		row := asObject(raw)
		member := asObject(row["member"])
		arm := member["variantId"].(string)
		c := arms[arm]
		c.expected++
		ex := asObject(row["execution"])
		if ex["ref"] != nil {
			c.submitted++
		}
		if terminal(ex["state"]) {
			c.terminal++
		}
		if ex["state"] == "succeeded" {
			c.succeeded++
		}
		if row["assessment"] == "pass" || row["assessment"] == "fail" {
			c.scored++
		}
		if row["assessment"] == "pass" {
			c.passed++
		}
		if row["usage"] != nil {
			m := asObject(asObject(row["usage"])["totalTokens"])
			if m["completeness"] == "complete" {
				c.tokens += numeric(m["value"])
				c.measured++
			}
		}
		id, err := PairID("trace-1", member["suiteId"].(string), member["caseId"].(string), int(numeric(member["sample"])))
		if err != nil {
			t.Fatal(err)
		}
		if pairs[id] == nil {
			pairs[id] = map[string]map[string]any{}
		}
		pairs[id][arm] = row
	}
	for arm, c := range arms {
		want := asObject(asObject(summary["counts"])[arm])
		for key, actual := range map[string]int{"expected": c.expected, "submitted": c.submitted, "terminal": c.terminal, "scored": c.scored, "qualityPassed": c.passed, "executionSucceeded": c.succeeded} {
			if numeric(want[key]) != float64(actual) {
				t.Fatalf("%s/%s: %d", arm, key, actual)
			}
		}
	}
	terminalPairs, qualityPairs, tokenPairs := 0, 0, 0
	for _, p := range pairs {
		a, b := p["a"], p["b"]
		if terminal(asObject(a["execution"])["state"]) && terminal(asObject(b["execution"])["state"]) {
			terminalPairs++
		}
		if a["assessment"] != "unscored" && b["assessment"] != "unscored" {
			qualityPairs++
		}
		if a["usage"] != nil && b["usage"] != nil && asObject(asObject(a["usage"])["totalTokens"])["completeness"] == "complete" && asObject(asObject(b["usage"])["totalTokens"])["completeness"] == "complete" {
			tokenPairs++
		}
	}
	if terminalPairs != 3 || qualityPairs != 2 || tokenPairs != 2 || arms["b"].tokens != 130 || arms["b"].measured != 2 || arms["a"].tokens != 275 {
		t.Fatal("worked-example coverage changed")
	}
	chart := objectFixture(t, "tokens-chart")
	for _, raw := range asRows(chart["bins"]) {
		bin := asObject(raw)
		expected := map[string]int{"a": 0, "b": 0}
		for _, pair := range pairs {
			if pair["a"]["usage"] == nil || pair["b"]["usage"] == nil {
				continue
			}
			ma, mb := asObject(asObject(pair["a"]["usage"])["totalTokens"]), asObject(asObject(pair["b"]["usage"])["totalTokens"])
			if ma["completeness"] != "complete" || mb["completeness"] != "complete" {
				continue
			}
			for arm, m := range map[string]map[string]any{"a": ma, "b": mb} {
				x := numeric(m["value"])
				if x >= numeric(bin["lower"]) && (x < numeric(bin["upper"]) || bin["upperInclusive"] == true && x == numeric(bin["upper"])) {
					expected[arm]++
				}
			}
		}
		for arm, n := range expected {
			if numeric(asObject(bin["counts"])[arm]) != float64(n) {
				t.Fatal("histogram does not match complete pairs")
			}
		}
	}
}
