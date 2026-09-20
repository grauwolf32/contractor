package evaldomain

import (
	"encoding/json"
	"slices"
)

type SelectedMember struct {
	View               MemberView
	PairID             string
	CollectionComplete bool
}
type ComparisonView struct {
	Summary Summary
	Suites  map[string]Summary
	Pairs   []Pair
}

func Fraction(n, d int) Ratio {
	r := Ratio{Numerator: n, Denominator: d}
	if d > 0 {
		v := float64(n) / float64(d)
		r.Value = &v
	}
	return r
}
func qualityFor(c Counts) Quality {
	return Quality{Fraction(c.ExecutionSucceeded, c.Expected), Fraction(c.EndToEndPassed, c.Expected), Fraction(c.QualityPassed, c.Scored)}
}
func QualityComplete(v MemberView) bool   { return v.Assessment == "pass" || v.Assessment == "fail" }
func TerminalExecution(v MemberView) bool { return v.Execution != nil && terminal(v.Execution.State) }
func EndToEnd(v MemberView, complete bool) bool {
	return v.Member.Eligibility == "eligible" && v.Execution != nil && v.Execution.State == "succeeded" && v.Assessment == "pass" && complete && !v.Conflicting
}

// ComparableMeasure uses a dimension's complete member scope, including known
// execution failures. Run IDs naturally differ between arms; scope semantics
// must agree, while every scope must name its own exact member and parent.
func ComparableMeasure(a, b MemberView, metric string, pinsVerified bool) (Measure, Measure, string) {
	if !pinsVerified {
		return Measure{}, Measure{}, "pins-unverified"
	}
	if a.Conflicting || b.Conflicting {
		return Measure{}, Measure{}, "conflicting"
	}
	if !TerminalExecution(a) || !TerminalExecution(b) {
		return Measure{}, Measure{}, "nonterminal"
	}
	if a.Usage == nil || b.Usage == nil {
		return Measure{}, Measure{}, "usage-unavailable"
	}
	x, y := a.Usage.TotalTokens, b.Usage.TotalTokens
	if metric == "duration" {
		x, y = a.Usage.WallMS, b.Usage.WallMS
	} else if metric != "tokens" {
		return Measure{}, Measure{}, "unsupported-metric"
	}
	if x.Completeness != "complete" || y.Completeness != "complete" || x.Value == nil || y.Value == nil {
		return x, y, "usage-incomplete"
	}
	if x.Unit != y.Unit || x.Scope.Kind != y.Scope.Kind || x.Scope.MemberID != a.Member.ID || y.Scope.MemberID != b.Member.ID || (x.Scope.Interval == nil) != (y.Scope.Interval == nil) {
		return x, y, "scope-mismatch"
	}
	for i, m := range []Measure{x, y} {
		v := a
		if i == 1 {
			v = b
		}
		if v.Execution.Ref == nil || !slices.Contains(m.Scope.Executions, *v.Execution.Ref) || len(m.Scope.Missing) > 0 || len(m.SourceRefs) == 0 {
			return x, y, "scope-incomplete"
		}
	}
	return x, y, ""
}
func countMember(c Counts, m SelectedMember) Counts {
	v := m.View
	c.Expected++
	switch v.Member.Eligibility {
	case "eligible":
		c.Eligible++
	case "unsupported":
		c.Unsupported++
	case "blocked":
		c.Blocked++
	}
	if v.Execution != nil && v.Execution.Ref != nil {
		c.Submitted++
	} else {
		c.Missing++
	}
	if TerminalExecution(v) {
		c.Terminal++
	}
	if v.Conflicting {
		c.Conflicting++
	}
	if m.CollectionComplete {
		c.CollectionComplete++
	}
	if QualityComplete(v) {
		c.Scored++
	}
	if v.Assessment == "pass" {
		c.QualityPassed++
	}
	if v.Execution != nil && v.Execution.State == "succeeded" {
		c.ExecutionSucceeded++
	}
	if EndToEnd(v, m.CollectionComplete) {
		c.EndToEndPassed++
	}
	return c
}
func newSummary(a, b string) Summary {
	return Summary{Counts: map[string]Counts{a: {}, b: {}}, Quality: map[string]Quality{}, Conclusion: "inconclusive"}
}

// BuildComparison consumes the entire frozen matrix in frozen ordinal order.
// Filtering/pagination is a later operation and cannot change its denominators.
func BuildComparison(members []SelectedMember, comparison Comparison, pinsVerified bool) (ComparisonView, error) {
	a, b := comparison.Baseline, comparison.Candidate
	out := ComparisonView{Summary: newSummary(a, b), Suites: map[string]Summary{}, Pairs: []Pair{}}
	if a == "" || b == "" || a == b || len(members) == 0 || len(members) > MaxMembers {
		return out, Failure("eval_invalid")
	}
	pairIndex := map[string]int{}
	seen := map[string]bool{}
	completeness := map[string]bool{}
	for _, m := range members {
		v := m.View
		if seen[v.Member.ID] || v.Member.VariantID != a && v.Member.VariantID != b {
			return out, Failure("eval_member_conflict")
		}
		seen[v.Member.ID] = true
		completeness[v.Member.ID] = m.CollectionComplete
		out.Summary.Counts[v.Member.VariantID] = countMember(out.Summary.Counts[v.Member.VariantID], m)
		suite, ok := out.Suites[v.Member.SuiteID]
		if !ok {
			suite = newSummary(a, b)
		}
		suite.Counts[v.Member.VariantID] = countMember(suite.Counts[v.Member.VariantID], m)
		out.Suites[v.Member.SuiteID] = suite
		if _, exists := pairIndex[m.PairID]; !exists {
			pairIndex[m.PairID] = len(out.Pairs)
			out.Pairs = append(out.Pairs, Pair{ID: m.PairID, SuiteID: v.Member.SuiteID, CaseID: v.Member.CaseID, Sample: v.Member.Sample, Exclusions: []string{}})
		}
		p := &out.Pairs[pairIndex[m.PairID]]
		if m.View.Member.VariantID == a {
			if p.A.Member.ID != "" {
				return out, Failure("eval_member_conflict")
			}
			p.A = m.View
		} else {
			if p.B.Member.ID != "" {
				return out, Failure("eval_member_conflict")
			}
			p.B = m.View
		}
	}
	var tokensA, tokensB float64
	suiteTokens := map[string][2]float64{}
	for i := range out.Pairs {
		p := &out.Pairs[i]
		if p.A.Member.ID == "" || p.B.Member.ID == "" || p.A.Member.CaseID != p.B.Member.CaseID || p.A.Member.Sample != p.B.Member.Sample || p.A.Member.SuiteID != p.B.Member.SuiteID {
			return out, Failure("eval_member_conflict")
		}
		suite := out.Suites[p.SuiteID]
		if TerminalExecution(p.A) && TerminalExecution(p.B) {
			out.Summary.TerminalPairs++
			suite.TerminalPairs++
		} else {
			p.Exclusions = append(p.Exclusions, "nonterminal")
		}
		if QualityComplete(p.A) && QualityComplete(p.B) && completeness[p.A.Member.ID] && completeness[p.B.Member.ID] && !p.A.Conflicting && !p.B.Conflicting {
			out.Summary.CompleteQualityPairs++
			suite.CompleteQualityPairs++
		} else {
			p.Exclusions = append(p.Exclusions, "quality-incomplete")
		}
		x, y, why := ComparableMeasure(p.A, p.B, "tokens", pinsVerified)
		if why == "" {
			out.Summary.CompleteTokenPairs++
			suite.CompleteTokenPairs++
			tokensA += *x.Value
			tokensB += *y.Value
			totals := suiteTokens[p.SuiteID]
			totals[0] += *x.Value
			totals[1] += *y.Value
			suiteTokens[p.SuiteID] = totals
		} else {
			p.Exclusions = append(p.Exclusions, why)
		}
		p.Regression = p.A.Assessment == "pass" && (p.B.Assessment == "fail" || p.B.Execution != nil && (p.B.Execution.State == "failed" || p.B.Execution.State == "cancelled"))
		out.Suites[p.SuiteID] = suite
	}
	conclude(&out.Summary, comparison, pinsVerified, tokensA, tokensB, out.Summary.CompleteTokenPairs == len(out.Pairs))
	for id, s := range out.Suites {
		tokens := suiteTokens[id]
		conclude(&s, comparison, pinsVerified, tokens[0], tokens[1], s.CompleteTokenPairs == s.Counts[a].Expected)
		out.Suites[id] = s
	}
	raw, err := json.Marshal(out.Summary)
	if err == nil {
		err = Validate("Summary", raw)
	}
	return out, err
}

func conclude(s *Summary, comparison Comparison, pinsVerified bool, ta, tb float64, tokenCoverage bool) {
	a, b := comparison.Baseline, comparison.Candidate
	for arm, c := range s.Counts {
		s.Quality[arm] = qualityFor(c)
	}
	if !pinsVerified {
		return
	}
	for _, c := range s.Counts {
		if c.Expected == 0 || c.Terminal != c.Expected || c.Scored != c.Expected || c.CollectionComplete != c.Expected || c.Conflicting > 0 {
			return
		}
	}
	x, y := s.Quality[a], s.Quality[b]
	if y.EndToEndPass.Value == nil || x.ConditionalQuality.Value == nil || y.ConditionalQuality.Value == nil {
		return
	}
	regression := *y.EndToEndPass.Value < comparison.Gates.MinCandidateEndToEndPass || *x.ConditionalQuality.Value-*y.ConditionalQuality.Value > comparison.Gates.MaxQualityDrop
	if gate := comparison.Gates.MaxTotalTokensRatio; gate != nil {
		if !tokenCoverage || ta <= 0 {
			return
		}
		regression = regression || tb/ta > *gate
	}
	s.Conclusion = "pass"
	if regression {
		s.Conclusion = "regressions"
	}
}
