package evaldomain

import (
	"math"
	"reflect"
	"time"
)

func terminal(state any) bool {
	return state == "succeeded" || state == "failed" || state == "cancelled"
}
func validateMeasure(m Measure, memberID string) error {
	if memberID != "" && m.Scope.MemberID != memberID {
		return Failure("eval_member_conflict")
	}
	if (m.Completeness == "unavailable") != (m.Value == nil) {
		return Failure("eval_invalid")
	}
	if m.Value != nil && (math.IsInf(*m.Value, 0) || math.IsNaN(*m.Value) || *m.Value < 0) {
		return Failure("eval_invalid")
	}
	if m.Completeness == "complete" && (len(m.Scope.Missing) > 0 || len(m.Scope.Executions) == 0 || len(m.SourceRefs) == 0) {
		return Failure("eval_invalid")
	}
	if m.Completeness == "partial" && (len(m.Scope.Missing) == 0 || m.Value == nil || *m.Value == 0) {
		return Failure("eval_invalid")
	}
	if m.Scope.Interval != nil {
		s, err := time.Parse(time.RFC3339Nano, m.Scope.Interval.Start)
		if err != nil {
			return Failure("eval_invalid")
		}
		e, err := time.Parse(time.RFC3339Nano, m.Scope.Interval.End)
		if err != nil || e.Before(s) {
			return Failure("eval_invalid")
		}
		if m.Unit == "milliseconds" && m.Value != nil && *m.Value != float64(e.Sub(s).Milliseconds()) {
			return Failure("eval_invalid")
		}
	}
	return nil
}
func validateUsage(u Usage, memberID string) error {
	units := []struct {
		m    Measure
		unit string
	}{{u.InputTokens, "tokens"}, {u.OutputTokens, "tokens"}, {u.TotalTokens, "tokens"}, {u.CachedInputTokens, "tokens"}, {u.ModelCalls, "calls"}, {u.ToolCalls, "calls"}, {u.ToolFailures, "calls"}, {u.WallMS, "milliseconds"}}
	for _, item := range units {
		if item.m.Unit != item.unit {
			return Failure("eval_invalid")
		}
		if err := validateMeasure(item.m, memberID); err != nil {
			return err
		}
	}
	if u.CachedInputTokens.Completeness == "complete" && u.InputTokens.Completeness == "complete" && *u.CachedInputTokens.Value > *u.InputTokens.Value {
		return Failure("eval_invalid")
	}
	if u.WallMS.Completeness == "complete" && u.WallMS.Scope.Interval == nil {
		return Failure("eval_invalid")
	}
	return nil
}
func validateResult(v map[string]any) error {
	ex := asObject(v["execution"])
	if err := validateExecution(ex); err != nil {
		return err
	}
	if !uniqueRows(asRows(v["evidence"]), "id") {
		return Failure("eval_invalid")
	}
	member := v["memberId"].(string)
	return typedCheck(v["usage"], func(u Usage) error {
		if err := validateUsage(u, member); err != nil {
			return err
		}
		for _, m := range []Measure{u.InputTokens, u.OutputTokens, u.TotalTokens, u.CachedInputTokens, u.ModelCalls, u.ToolCalls, u.ToolFailures, u.WallMS} {
			if ex["ref"] != nil {
				parent := asObject(ex["ref"])
				found := false
				wantKind := "workflow"
				if parent["kind"] == "audit" {
					wantKind = "audit"
				}
				if m.Scope.Kind != wantKind {
					return Failure("eval_member_conflict")
				}
				for _, ref := range m.Scope.Executions {
					if ref.Kind == parent["kind"] && ref.ID == parent["id"] {
						found = true
					}
					if ref.Kind == "audit" && (ref.Kind != parent["kind"] || ref.ID != parent["id"]) {
						return Failure("eval_member_conflict")
					}
				}
				if !found || wantKind == "workflow" && len(m.Scope.Executions) != 1 {
					return Failure("eval_member_conflict")
				}
			}

			if !terminal(ex["state"]) && m.Completeness == "complete" {
				return Failure("eval_invalid")
			}
		}
		return nil
	})
}
func validateExecution(v map[string]any) error { return typedCheck(v, checkExecution) }
func checkExecution(ex ExecutionView) error {
	if ex.State == "not_submitted" && ex.Ref != nil || ex.State != "not_submitted" && ex.State != "unknown" && ex.Ref == nil {
		return Failure("eval_member_conflict")
	}
	if terminal(ex.State) != (ex.FinishedAt != nil) {
		return Failure("eval_invalid")
	}
	if ex.StartedAt != nil && ex.FinishedAt != nil && ex.FinishedAt.Before(*ex.StartedAt) {
		return Failure("eval_invalid")
	}
	return nil
}
func validateCounts(v map[string]any) error { return typedCheck(v, checkCounts) }
func checkCounts(c Counts) error {
	for _, n := range []int{c.Eligible, c.Unsupported, c.Blocked, c.Submitted, c.Terminal, c.Missing, c.Conflicting, c.CollectionComplete, c.Scored, c.QualityPassed, c.ExecutionSucceeded, c.EndToEndPassed} {
		if n > c.Expected {
			return Failure("eval_invalid")
		}
	}
	if c.Eligible+c.Unsupported+c.Blocked != c.Expected || c.Terminal > c.Submitted || c.QualityPassed > c.Scored || c.EndToEndPassed > c.ExecutionSucceeded || c.EndToEndPassed > c.QualityPassed || c.EndToEndPassed > c.CollectionComplete || c.EndToEndPassed > c.Eligible {
		return Failure("eval_invalid")
	}
	return nil
}
func validateRatio(v map[string]any) error { return typedCheck(v, checkRatio) }
func checkRatio(r Ratio) error {
	if r.Numerator > r.Denominator || r.Denominator == 0 && r.Value != nil {
		return Failure("eval_invalid")
	}
	if r.Denominator > 0 && (r.Value == nil || math.Abs(float64(r.Numerator)/float64(r.Denominator)-*r.Value) > 1e-12) {
		return Failure("eval_invalid")
	}
	return nil
}
func validateSummary(v map[string]any) error { return typedCheck(v, checkSummary) }
func checkSummary(s Summary) error {
	if len(s.Counts) != 2 || len(s.Quality) != 2 {
		return Failure("eval_invalid")
	}
	for arm, c := range s.Counts {
		if err := checkCounts(c); err != nil {
			return err
		}
		q, ok := s.Quality[arm]
		if !ok {
			return Failure("eval_invalid")
		}
		ratios := []struct {
			ratio                  Ratio
			numerator, denominator int
		}{
			{q.ExecutionSuccess, c.ExecutionSucceeded, c.Expected},
			{q.EndToEndPass, c.EndToEndPassed, c.Expected},
			{q.ConditionalQuality, c.QualityPassed, c.Scored},
		}
		for _, item := range ratios {
			if err := checkRatio(item.ratio); err != nil {
				return err
			}
			if item.ratio.Numerator != item.numerator || item.ratio.Denominator != item.denominator {
				return Failure("eval_invalid")
			}
		}
		if s.TerminalPairs > c.Terminal || s.CompleteQualityPairs > c.Scored || s.CompleteTokenPairs > c.Expected {
			return Failure("eval_invalid")
		}
		if s.Conclusion != "inconclusive" && (c.Expected != c.Scored || c.Expected != c.Terminal || c.Expected != c.CollectionComplete || c.Conflicting > 0) {
			return Failure("eval_invalid")
		}
	}
	return nil
}

type pairValidation struct {
	SuiteID string     `json:"suiteId"`
	CaseID  string     `json:"caseId"`
	Sample  int        `json:"sample"`
	A       MemberView `json:"a"`
	B       MemberView `json:"b"`
}

func validatePair(v map[string]any) error {
	return typedCheck(v, func(p pairValidation) error {
		a, b := p.A.Member, p.B.Member
		if a.ID == b.ID || a.VariantID == b.VariantID || a.SuiteID != p.SuiteID || b.SuiteID != p.SuiteID || a.CaseID != p.CaseID || b.CaseID != p.CaseID || a.Sample != p.Sample || b.Sample != p.Sample {
			return Failure("eval_member_conflict")
		}
		if err := checkMemberView(p.A); err != nil {
			return err
		}
		return checkMemberView(p.B)
	})
}

// Nested semantics follow the declared DTO and its known child fields. Case
// role names, producer data and arbitrary maps never select a validator.
func validateChildren(kind string, value map[string]any) error {
	switch kind {
	case "Page":
		return typedCheck(value, validatePage)
	case "Ratio":
		return validateRatio(value)
	case "Counts":
		return validateCounts(value)
	case "Quality":
		return validateQuality(value)
	case "Execution":
		return validateExecution(value)
	case "MemberView":
		return validateMemberView(value)
	case "Experiment":
		if summary := value["summary"]; summary != nil {
			return validateSummary(asObject(summary))
		}
	case "Report":
		return validateSummary(asObject(value["summary"]))
	case "MemberPage", "PairPage":
		if err := validateSummary(asObject(value["experimentSummary"])); err != nil {
			return err
		}
		for _, row := range asRows(value["items"]) {
			var err error
			if kind == "MemberPage" {
				err = validateMemberView(asObject(row))
			} else {
				err = validatePair(asObject(row))
			}
			if err != nil {
				return err
			}
		}
		return typedCheck(value["page"], validatePage)
	case "DatasetPage", "CasePage", "ExperimentPage", "ExecutionPage", "Capabilities":
		if page, exists := value["page"]; exists {
			return typedCheck(page, validatePage)
		}
	}
	return nil
}

type pageValidation struct {
	HasMore    bool    `json:"hasMore"`
	NextCursor *string `json:"nextCursor"`
}

func validatePage(page pageValidation) error {
	if page.HasMore != (page.NextCursor != nil) {
		return Failure("eval_invalid")
	}
	return nil
}

func validateQuality(v map[string]any) error {
	return typedCheck(v, func(q Quality) error {
		for _, r := range []Ratio{q.ExecutionSuccess, q.EndToEndPass, q.ConditionalQuality} {
			if err := checkRatio(r); err != nil {
				return err
			}
		}
		return nil
	})
}
func validateMemberView(v map[string]any) error { return typedCheck(v, checkMemberView) }
func checkMemberView(v MemberView) error {
	if v.Execution != nil {
		if err := checkExecution(*v.Execution); err != nil {
			return err
		}
	}
	if v.Usage != nil {
		return validateUsage(*v.Usage, v.Member.ID)
	}
	return nil
}

// CheckResultContext is shared by collection and ingestion. Reported execution
// facts may not overwrite the authoritative execution observed by the service.
func CheckResultContext(memberID string, authoritative map[string]any, result map[string]any) error {
	if result["memberId"] != memberID {
		return Failure("eval_member_conflict")
	}
	if !reflect.DeepEqual(authoritative, result["execution"]) {
		return Failure("eval_member_conflict")
	}
	return nil
}

// CheckOwner deliberately returns the same response for unknown/foreign owners.
func CheckOwner(principal, owner string) error {
	if principal == "" || owner == "" || principal != owner {
		return Failure("eval_not_found")
	}
	return nil
}
