package evaldomain

import (
	"encoding/json"
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
func validateExecution(ex map[string]any) error {
	if ex["state"] == "not_submitted" && ex["ref"] != nil || ex["state"] != "not_submitted" && ex["state"] != "unknown" && ex["ref"] == nil {
		return Failure("eval_member_conflict")
	}
	if terminal(ex["state"]) != (ex["finishedAt"] != nil) {
		return Failure("eval_invalid")
	}
	if ex["startedAt"] != nil && ex["finishedAt"] != nil {
		s, _ := time.Parse(time.RFC3339Nano, ex["startedAt"].(string))
		e, _ := time.Parse(time.RFC3339Nano, ex["finishedAt"].(string))
		if e.Before(s) {
			return Failure("eval_invalid")
		}
	}
	return nil
}
func validateCounts(c map[string]any) error {
	expected := numeric(c["expected"])
	for _, value := range c {
		if numeric(value) > expected {
			return Failure("eval_invalid")
		}
	}
	if numeric(c["eligible"])+numeric(c["unsupported"])+numeric(c["blocked"]) != expected || numeric(c["terminal"]) > numeric(c["submitted"]) || numeric(c["qualityPassed"]) > numeric(c["scored"]) || numeric(c["endToEndPassed"]) > numeric(c["executionSucceeded"]) || numeric(c["endToEndPassed"]) > numeric(c["qualityPassed"]) || numeric(c["endToEndPassed"]) > numeric(c["collectionComplete"]) || numeric(c["endToEndPassed"]) > numeric(c["eligible"]) {
		return Failure("eval_invalid")
	}
	return nil
}
func validateRatio(v map[string]any) error {
	n, d := numeric(v["numerator"]), numeric(v["denominator"])
	if n > d || d == 0 && v["value"] != nil || d > 0 && (v["value"] == nil || math.Abs(n/d-numeric(v["value"])) > 1e-12) {
		return Failure("eval_invalid")
	}
	return nil
}
func validateSummary(v map[string]any) error {
	counts, quality := asObject(v["counts"]), asObject(v["quality"])
	if len(counts) != 2 || len(quality) != 2 {
		return Failure("eval_invalid")
	}
	for arm, raw := range counts {
		c := asObject(raw)
		if err := validateCounts(c); err != nil {
			return err
		}
		qraw, ok := quality[arm]
		if !ok {
			return Failure("eval_invalid")
		}
		q := asObject(qraw)
		for name, pair := range map[string][2]string{"executionSuccess": {"executionSucceeded", "expected"}, "endToEndPass": {"endToEndPassed", "expected"}, "conditionalQuality": {"qualityPassed", "scored"}} {
			r := asObject(q[name])
			if err := validateRatio(r); err != nil {
				return err
			}
			if numeric(r["numerator"]) != numeric(c[pair[0]]) || numeric(r["denominator"]) != numeric(c[pair[1]]) {
				return Failure("eval_invalid")
			}
		}
		if numeric(v["terminalPairs"]) > numeric(c["terminal"]) || numeric(v["completeQualityPairs"]) > numeric(c["scored"]) || numeric(v["completeTokenPairs"]) > numeric(c["expected"]) {
			return Failure("eval_invalid")
		}
		if v["conclusion"] != "inconclusive" && (numeric(c["expected"]) != numeric(c["scored"]) || numeric(c["expected"]) != numeric(c["terminal"]) || numeric(c["expected"]) != numeric(c["collectionComplete"]) || numeric(c["conflicting"]) > 0) {
			return Failure("eval_invalid")
		}
	}
	return nil
}
func validatePair(v map[string]any) error {
	a, b := asObject(asObject(v["a"])["member"]), asObject(asObject(v["b"])["member"])
	if a["memberId"] == b["memberId"] || a["variantId"] == b["variantId"] {
		return Failure("eval_member_conflict")
	}
	for _, key := range []string{"suiteId", "caseId", "sample"} {
		if a[key] != v[key] || b[key] != v[key] {
			return Failure("eval_member_conflict")
		}
	}
	return validateNested(v)
}
func validateNested(value any) error {
	switch v := value.(type) {
	case map[string]any:
		if _, ok := v["hasMore"]; ok {
			if v["hasMore"] == true && v["nextCursor"] == nil || v["hasMore"] == false && v["nextCursor"] != nil {
				return Failure("eval_invalid")
			}
		}
		if _, ok := v["denominator"].(json.Number); ok {
			if err := validateRatio(v); err != nil {
				return err
			}
		}
		if _, ok := v["counts"].(map[string]any); ok {
			if _, ok = v["conclusion"]; ok {
				if err := validateSummary(v); err != nil {
					return err
				}
			}
		}
		for key, child := range v {
			if key == "parameters" || key == "expected" || key == "extensions" {
				continue
			}
			if err := validateNested(child); err != nil {
				return err
			}
		}
	case []any:
		for _, child := range v {
			if err := validateNested(child); err != nil {
				return err
			}
		}

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
