package evaldomain

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"testing"
	"time"
)

func traceComparison(t *testing.T) ([]SelectedMember, Summary) {
	t.Helper()
	raw, err := os.ReadFile("../../api/testdata/evals/valid/pair-page.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Items   []Pair  `json:"items"`
		Summary Summary `json:"experimentSummary"`
	}
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	rows := []SelectedMember{}
	for _, p := range f.Items {
		for _, m := range []MemberView{p.A, p.B} {
			rows = append(rows, SelectedMember{View: m, PairID: p.ID, CollectionComplete: QualityComplete(m)})
		}
	}
	return rows, f.Summary
}
func TestEvalComparisonTraceSmallTruthfulCoverage(t *testing.T) {
	rows, want := traceComparison(t)
	c := Comparison{Baseline: "a", Candidate: "b", Gates: Gates{MinCandidateEndToEndPass: 1}}
	v, err := BuildComparison(rows, c, true)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(v.Summary, want) {
		t.Fatalf("summary got %+v want %+v", v.Summary, want)
	}
	if !v.Pairs[2].Regression {
		t.Fatal("known quality regression lost under incomplete overall evidence")
	}
	cohort := MeasurementCohort(v.Pairs, "", "tokens", "workflow", true)
	if cohort.Coverage.Included != 2 || cohort.Coverage.Excluded != 2 || *cohort.Distributions["a"].Total != 125 || *cohort.Distributions["b"].Total != 130 {
		t.Fatalf("failure-inclusive paired tokens: %+v", cohort)
	}
	if *cohort.Distributions["a"].P50 != 45 || *cohort.Distributions["a"].P90 != 80 {
		t.Fatal("percentiles must use nearest-rank raw samples")
	}
	if MeasurementCohort(v.Pairs, "", "tokens", "workflow", false).Coverage.Included != 0 {
		t.Fatal("unverified pins cannot establish comparable cost")
	}
	for _, d := range cohort.Deltas {
		if d.Difference != d.B-d.A {
			t.Fatal("wrong raw unit delta")
		}
	}
	for arm, dist := range cohort.Distributions {
		n := 0
		for _, b := range cohort.Bins {
			n += b.Counts[arm]
		}
		if n != dist.Count {
			t.Fatal("bin membership differs from raw cohort")
		}
	}
}
func TestEvalAssessmentRequiredPrecedenceAndMissingEvidence(t *testing.T) {
	checks := []Check{{ID: "one", Evaluator: "required-artifact@1", ImplementationSHA256: "digest", Required: true}, {ID: "two", Evaluator: "human-review@1", ImplementationSHA256: "digest", Required: true}}
	a := AssessmentInput{Checks: []CheckResult{{ID: "one", Evaluator: "required-artifact@1", ImplementationSHA256: "digest", Status: "pass"}, {ID: "two", Evaluator: "human-review@1", ImplementationSHA256: "digest", Status: "pass"}}}
	if AssessmentDecision(checks, nil, true) != "unscored" {
		t.Fatal("absent review fabricated")
	}
	if AssessmentDecision(checks, &a, true) != "pass" || AssessmentDecision(checks, &a, false) != "incomplete" {
		t.Fatal("missing evidence permitted pass")
	}
	a.Checks[1].Status = "fail"
	if AssessmentDecision(checks, &a, false) != "fail" {
		t.Fatal("known failure hidden")
	}
	a.Checks[0].Status = "error"
	if AssessmentDecision(checks, &a, true) != "error" {
		t.Fatal("check error conflated with fail")
	}
	a.Checks = a.Checks[1:]
	if AssessmentDecision(checks, &a, true) != "fail" {
		t.Fatal("required missing check masked known fail")
	}
	a.Checks[0].Status = "pass"
	if AssessmentDecision(checks, &a, true) != "incomplete" {
		t.Fatal("omitted required check passed")
	}
	a.Checks[0].ImplementationSHA256 = "other"
	if AssessmentDecision(checks, &a, true) != "error" {
		t.Fatal("unpinned evaluator passed")
	}
}
func TestEvalComparisonDeclaredGatesAndEvidenceInvalidation(t *testing.T) {
	rows, _ := traceComparison(t)
	rows = rows[:2]
	c := Comparison{Baseline: "a", Candidate: "b", Gates: Gates{MinCandidateEndToEndPass: 1}}
	v, err := BuildComparison(rows, c, true)
	if err != nil || v.Summary.Conclusion != "pass" {
		t.Fatalf("declared gates: %+v %v", v, err)
	}
	rows[1].CollectionComplete = false
	v, err = BuildComparison(rows, c, true)
	if err != nil || v.Summary.Conclusion != "inconclusive" {
		t.Fatal("deleted evidence left pass")
	}
	rows[1].CollectionComplete = true
	rows[1].View.Assessment = "fail"
	v, err = BuildComparison(rows, c, true)
	if err != nil || v.Summary.Conclusion != "regressions" {
		t.Fatal("gate regression missing")
	}
	rows[1].View.Assessment = "pass"
	zero := 0.0
	rows[0].View.Usage.TotalTokens.Value = &zero
	gate := 2.0
	c.Gates.MaxTotalTokensRatio = &gate
	v, err = BuildComparison(rows, c, true)
	if err != nil || v.Summary.Conclusion != "inconclusive" {
		t.Fatal("zero baseline token ratio invented")
	}
}

func TestEvalQualityDropGateUsesExactDecimalBoundary(t *testing.T) {
	for _, test := range []struct {
		denominator, baselinePassed, candidatePassed int
		gate                                         float64
		want                                         string
	}{
		{10, 4, 3, 0.1, "pass"},
		{10, 5, 4, 0.1, "pass"},
		{5, 4, 3, 0.2, "pass"},
		{20, 4, 3, 0.05, "pass"},
		{10, 4, 1, 0.3, "pass"},
		{10, 4, 2, 0.1, "regressions"},
	} {
		name := fmt.Sprintf("%d/%d to %d/%d at %g", test.baselinePassed, test.denominator, test.candidatePassed, test.denominator, test.gate)
		t.Run(name, func(t *testing.T) {
			view, err := BuildComparison(
				qualityComparisonMembers(test.denominator, test.baselinePassed, test.candidatePassed),
				Comparison{Baseline: "a", Candidate: "b", Gates: Gates{MaxQualityDrop: test.gate}},
				true,
			)
			if err != nil {
				t.Fatal(err)
			}
			if view.Summary.Conclusion != test.want || view.Suites["suite-a"].Conclusion != test.want {
				t.Fatalf("experiment=%s suite=%s, want %s", view.Summary.Conclusion, view.Suites["suite-a"].Conclusion, test.want)
			}
		})
	}
}

func TestEvalQualityDropGateMatchesExactFractions(t *testing.T) {
	for _, gate := range []struct {
		value       float64
		numerator   int
		denominator int
	}{
		{0.05, 1, 20}, {0.1, 1, 10}, {0.2, 1, 5}, {0.3, 3, 10},
	} {
		for scored := 1; scored <= 20; scored++ {
			for baselinePassed := 0; baselinePassed <= scored; baselinePassed++ {
				for candidatePassed := 0; candidatePassed <= scored; candidatePassed++ {
					want := (baselinePassed-candidatePassed)*gate.denominator > gate.numerator*scored
					got := qualityDropExceedsGate(
						Counts{Scored: scored, QualityPassed: baselinePassed},
						Counts{Scored: scored, QualityPassed: candidatePassed}, gate.value,
					)
					if got != want {
						t.Fatalf("%d/%d to %d/%d at %g: got regression=%t, want %t", baselinePassed, scored, candidatePassed, scored, gate.value, got, want)
					}
				}
			}
		}
	}
}

func qualityComparisonMembers(scored, baselinePassed, candidatePassed int) []SelectedMember {
	rows := make([]SelectedMember, 0, scored*2)
	finishedAt := time.Date(2026, 9, 19, 0, 0, 0, 0, time.UTC)
	for index := 0; index < scored; index++ {
		for _, arm := range []struct {
			id     string
			passed int
		}{{"a", baselinePassed}, {"b", candidatePassed}} {
			assessment := "fail"
			if index < arm.passed {
				assessment = "pass"
			}
			rows = append(rows, SelectedMember{
				PairID: fmt.Sprintf("pair-%d", index), CollectionComplete: true,
				View: MemberView{
					Member: MemberIdentity{
						ID: fmt.Sprintf("%s-%d", arm.id, index), SuiteID: "suite-a",
						CaseID: fmt.Sprintf("case-%d", index), VariantID: arm.id,
						Eligibility: "eligible",
					},
					Execution: &ExecutionView{
						Ref:   &ExecutionRef{Kind: "run", ID: fmt.Sprintf("run-%s-%d", arm.id, index)},
						State: "succeeded", FinishedAt: &finishedAt,
					},
					Assessment: assessment,
				},
			})
		}
	}
	return rows
}

func TestEvalChartsEmptySingletonEqualAndProgressGaps(t *testing.T) {
	rows, _ := traceComparison(t)
	v, err := BuildComparison(rows[:2], Comparison{Baseline: "a", Candidate: "b"}, true)
	if err != nil {
		t.Fatal(err)
	}
	x := 7.0
	v.Pairs[0].A.Usage.TotalTokens.Value = &x
	v.Pairs[0].B.Usage.TotalTokens.Value = &x
	c := MeasurementCohort(v.Pairs, "", "tokens", "workflow", true)
	if len(c.Bins) != 1 || c.Bins[0].Lower != 7 || c.Bins[0].Upper != 7 || c.Bins[0].Counts["a"] != 1 || *c.Distributions["a"].P90 != 7 {
		t.Fatal("equal single-point cohort smoothed")
	}
	c = MeasurementCohort(v.Pairs, "absent", "tokens", "workflow", true)
	if len(c.Bins) != 0 || c.Distributions["a"].Total != nil {
		t.Fatal("empty cohort converted to zero")
	}
	start := time.Date(2026, 9, 19, 0, 0, 0, 0, time.UTC)
	observations := []ProgressObservation{{start.Add(time.Minute), 1, 0}, {start.Add(20 * time.Minute), 2, 1}}
	points, width := ProgressBuckets(start, observations)
	if len(points) != 2 || !points[0].GapBefore || !points[1].GapBefore || width < 1 {
		t.Fatal("unknown observed history interpolated")
	}
	for i := 0; i < 10000; i++ {
		observations = append(observations, ProgressObservation{start.Add(time.Duration(i) * time.Second), i / 2, i / 2})
	}
	points, _ = ProgressBuckets(start, observations)
	if len(points) > 200 {
		t.Fatalf("unbounded progress %d", len(points))
	}
}
