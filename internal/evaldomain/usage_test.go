package evaldomain

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strconv"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedEvalAuditAccountingFixture(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "evals", "audit-accounting.json")
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
	// Inventory and output diagnostics can jointly exceed the wire bound.
	overflowing := fixture.Observation
	overflowing.Missing = make([]string, MaxEvidenceGaps+128)
	for i := range overflowing.Missing {
		overflowing.Missing[i] = fmt.Sprintf("missing child %04d", i)
	}
	out, err = NormalizeUsage(overflowing)
	if err != nil || out.TotalTokens.Completeness != "partial" || len(out.TotalTokens.Scope.Missing) != MaxEvidenceGaps {
		t.Fatal("overflowing diagnostics blocked incomplete observation", err)
	}
	if err = validateUsage(out, overflowing.MemberID); err != nil {
		t.Fatal("bounded observation violates the public contract", err)
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

func TestSharedEvalTraceExampleAgreesWithAllSelectedMembers(t *testing.T) {
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

// permutations returns every order of items, generated deterministically.
func permutations[T any](items []T) [][]T {
	if len(items) <= 1 {
		return [][]T{append([]T{}, items...)}
	}
	var out [][]T
	for i := range items {
		rest := append(append([]T{}, items[:i]...), items[i+1:]...)
		for _, tail := range permutations(rest) {
			out = append(out, append([]T{items[i]}, tail...))
		}
	}
	return out
}

type wantMeasure struct {
	value        *float64
	completeness string
	missing      []string
	sources      []string
}

func checkMeasure(t *testing.T, name string, got Measure, want wantMeasure) {
	t.Helper()
	show := func(v *float64) string {
		if v == nil {
			return "null"
		}
		return strconv.FormatFloat(*v, 'g', -1, 64)
	}
	if show(got.Value) != show(want.value) || got.Completeness != want.completeness ||
		!slices.Equal(got.Scope.Missing, want.missing) || !slices.Equal(got.SourceRefs, want.sources) {
		t.Fatalf("%s = (%s, %s, missing %v, sources %v), want (%s, %s, missing %v, sources %v)",
			name, show(got.Value), got.Completeness, got.Scope.Missing, got.SourceRefs,
			show(want.value), want.completeness, want.missing, want.sources)
	}
}

func float(v int64) *float64 { f := float64(v); return &f }

// A counter's usage does not depend on the order of its snapshots: one whose
// exact sum exceeds the int64 range is unavailable with an overflow gap in
// every order, instead of the subtotal of whichever snapshots were read first,
// and the other counters keep their own values and sources (EVM-R240).
func TestNormalizeUsageCounterOverflowIsOrderIndependent(t *testing.T) {
	const top = math.MaxInt64
	source := func(stage string) string { return Digest([]byte(stage)) }
	attempt := func(stage string, tokens *int64) AttemptObservation {
		a := AttemptObservation{
			RunID: "run-1", StageExecutionID: stage, ReportsComplete: true, SourceSHA256: source(stage),
			Metrics: map[string]int64{"modelCalls": 1},
		}
		if tokens != nil {
			a.Metrics["totalTokens"] = *tokens
		}
		return a
	}
	tokens := func(v int64) *int64 { return &v }
	incomplete := attempt("c", tokens(20))
	incomplete.ReportsComplete = false
	sorted := func(stages ...string) []string {
		out := []string{}
		for _, stage := range stages {
			out = append(out, source(stage))
		}
		slices.Sort(out)
		return out
	}
	for _, tc := range []struct {
		name     string
		attempts []AttemptObservation
		want     wantMeasure
	}{
		{
			name:     "sum beyond range",
			attempts: []AttemptObservation{attempt("a", tokens(top-100)), attempt("b", tokens(200)), attempt("c", tokens(0))},
			want:     wantMeasure{completeness: "unavailable", missing: []string{"counter overflow"}, sources: sorted("a", "b", "c")},
		},
		{
			name:     "every contributor at the range",
			attempts: []AttemptObservation{attempt("a", tokens(top)), attempt("b", tokens(top)), attempt("c", tokens(top))},
			want:     wantMeasure{completeness: "unavailable", missing: []string{"counter overflow"}, sources: sorted("a", "b", "c")},
		},
		{
			name:     "overflow and a missing counter",
			attempts: []AttemptObservation{attempt("a", tokens(top)), attempt("b", tokens(1)), attempt("c", nil)},
			want:     wantMeasure{completeness: "unavailable", missing: []string{"counter overflow", "counter unavailable"}, sources: sorted("a", "b")},
		},
		{
			name:     "sum exactly at the range",
			attempts: []AttemptObservation{attempt("a", tokens(top-100)), attempt("b", tokens(100)), attempt("c", tokens(0))},
			want:     wantMeasure{value: float(top), completeness: "complete", missing: []string{}, sources: sorted("a", "b", "c")},
		},
		{
			name:     "normal totals",
			attempts: []AttemptObservation{attempt("a", tokens(10)), attempt("b", tokens(20)), attempt("c", tokens(30))},
			want:     wantMeasure{value: float(60), completeness: "complete", missing: []string{}, sources: sorted("a", "b", "c")},
		},
		{
			name:     "known zero",
			attempts: []AttemptObservation{attempt("a", tokens(0)), attempt("b", tokens(0))},
			want:     wantMeasure{value: float(0), completeness: "complete", missing: []string{}, sources: sorted("a", "b")},
		},
		{
			name:     "missing counter",
			attempts: []AttemptObservation{attempt("a", tokens(10)), attempt("b", nil), attempt("c", tokens(20))},
			want:     wantMeasure{value: float(30), completeness: "partial", missing: []string{"counter unavailable"}, sources: sorted("a", "c")},
		},
		{
			name:     "incomplete report",
			attempts: []AttemptObservation{attempt("a", tokens(10)), incomplete},
			want:     wantMeasure{value: float(30), completeness: "partial", missing: []string{"incomplete or truncated execution reports"}, sources: sorted("a", "c")},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var first *Usage
			for _, order := range permutations(tc.attempts) {
				in := UsageObservation{
					MemberID: Digest([]byte("member"))[7:], Parent: ExecutionRef{Kind: "run", ID: "run-1"},
					Executions: []ExecutionRef{{Kind: "run", ID: "run-1"}}, InventoryComplete: true, Terminal: true,
					Missing: []string{}, Attempts: order,
				}
				out, err := NormalizeUsage(in)
				if err != nil {
					t.Fatal(err)
				}
				if err := validateUsage(out, in.MemberID); err != nil {
					t.Fatalf("usage violates the public contract: %v", err)
				}
				checkMeasure(t, "totalTokens", out.TotalTokens, tc.want)
				// modelCalls stays in range and keeps its own value and sources.
				stages := []string{}
				for _, a := range order {
					stages = append(stages, a.StageExecutionID)
				}
				calls := wantMeasure{value: float(int64(len(order))), completeness: "complete", missing: []string{}, sources: sorted(stages...)}
				if slices.ContainsFunc(order, func(a AttemptObservation) bool { return !a.ReportsComplete }) {
					calls.completeness, calls.missing = "partial", []string{"incomplete or truncated execution reports"}
				}
				checkMeasure(t, "modelCalls", out.ModelCalls, calls)
				if first == nil {
					first = &out
				} else if !reflect.DeepEqual(*first, out) {
					t.Fatalf("usage depends on snapshot order:\n%+v\n%+v", *first, out)
				}
			}
		})
	}
}

// An overflow reported inside one snapshot makes the member's counter
// unavailable too, and a snapshot cannot both overflow and report a value.
func TestNormalizeUsageCarriesSnapshotOverflow(t *testing.T) {
	member := Digest([]byte("member"))[7:]
	run := ExecutionRef{Kind: "run", ID: "run-1"}
	observation := func(attempts ...AttemptObservation) UsageObservation {
		return UsageObservation{
			MemberID: member, Parent: run, Executions: []ExecutionRef{run}, InventoryComplete: true,
			Terminal: true, Missing: []string{}, Attempts: attempts,
		}
	}
	overflowed := AttemptObservation{
		RunID: "run-1", StageExecutionID: "a", ReportsComplete: true, SourceSHA256: Digest([]byte("a")),
		Metrics: map[string]int64{"modelCalls": 1}, OverflowMetrics: []string{"toolCalls"},
	}
	other := AttemptObservation{
		RunID: "run-1", StageExecutionID: "b", ReportsComplete: true, SourceSHA256: Digest([]byte("b")),
		Metrics: map[string]int64{"modelCalls": 1, "toolCalls": 3},
	}
	for _, order := range permutations([]AttemptObservation{overflowed, other}) {
		out, err := NormalizeUsage(observation(order...))
		if err != nil {
			t.Fatal(err)
		}
		sources := []string{overflowed.SourceSHA256, other.SourceSHA256}
		slices.Sort(sources)
		checkMeasure(t, "toolCalls", out.ToolCalls, wantMeasure{completeness: "unavailable", missing: []string{"counter overflow"}, sources: sources})
		checkMeasure(t, "modelCalls", out.ModelCalls, wantMeasure{value: float(2), completeness: "complete", missing: []string{}, sources: sources})
	}
	contradictory := overflowed
	contradictory.Metrics = map[string]int64{"modelCalls": 1, "toolCalls": 1}
	if _, err := NormalizeUsage(observation(contradictory)); !IsCode(err, "eval_invalid") {
		t.Fatalf("overflowed counter with a value = %v, want eval_invalid", err)
	}
}

// CounterSum reports overflow exactly when the exact sum exceeds the int64
// range, in every order of the same contributions.
func TestCounterSumIsOrderIndependent(t *testing.T) {
	const top = math.MaxInt64
	for _, tc := range []struct {
		values []int64
		want   int64
		fits   bool
	}{
		{values: []int64{top - 100, 200, 0}, fits: false},
		{values: []int64{top, 1}, fits: false},
		{values: []int64{top - 100, 100, 0}, want: top, fits: true},
		{values: []int64{top, 0, 0}, want: top, fits: true},
		{values: []int64{1, 2, 3}, want: 6, fits: true},
		{values: []int64{}, want: 0, fits: true},
	} {
		for _, order := range permutations(tc.values) {
			var sum CounterSum
			for _, v := range order {
				sum.Add(v)
			}
			got, fits := sum.Value()
			if fits != tc.fits || fits && got != tc.want {
				t.Fatalf("sum of %v = (%d, %t), want (%d, %t)", order, got, fits, tc.want, tc.fits)
			}
		}
	}
}
