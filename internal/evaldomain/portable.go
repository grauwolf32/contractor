package evaldomain

import (
	"encoding/json"
	"strings"
)

func validatePortablePlan(v map[string]any) error {
	id := v["experiment_id"].(string)
	seen := map[string]bool{}
	variants := map[string]string{}
	suites := map[string]bool{}
	for _, raw := range asRows(v["variants"]) {
		row := asObject(raw)
		key := row["id"].(string)
		if _, ok := variants[key]; ok {
			return Failure("eval_invalid")
		}
		variants[key] = asObject(row["binding"])["sha256"].(string)
	}
	for _, raw := range asRows(v["suites"]) {
		row := asObject(raw)
		key := row["id"].(string)
		if suites[key] {
			return Failure("eval_invalid")
		}
		suites[key] = true
	}
	for _, raw := range asRows(v["members"]) {
		m := asObject(raw)
		n, err := m["sample"].(json.Number).Int64()
		if err != nil {
			return Failure("eval_invalid")
		}
		mid, err := MemberID(id, m["suite_id"].(string), m["case_id"].(string), int(n), m["variant_id"].(string))
		if err != nil || mid != m["member_id"] || seen[mid] {
			return Failure("eval_member_conflict")
		}
		seen[mid] = true
		if !suites[m["suite_id"].(string)] || variants[m["variant_id"].(string)] != m["binding_sha256"] {
			return Failure("eval_pin_mismatch")
		}
		if m["eligibility"] != "eligible" && (m["reason"] == nil || m["reason"] == "") {
			return Failure("eval_invalid")
		}
	}
	order := asRows(v["execution_order"])
	if len(order) != len(seen) {
		return Failure("eval_invalid")
	}
	for _, mid := range order {
		if !seen[mid.(string)] {
			return Failure("eval_invalid")
		}
	}
	budget := asObject(v["budgets"])
	if numeric(budget["max_members"]) < float64(len(seen)) || numeric(budget["max_in_flight"]) > numeric(budget["max_members"]) {
		return Failure("eval_limit_exceeded")
	}
	cmp := asObject(v["comparison"])
	if cmp["baseline"] == cmp["candidate"] || variants[cmp["baseline"].(string)] == "" || variants[cmp["candidate"].(string)] == "" {
		return Failure("eval_invalid")
	}
	return nil
}

// PortableComparison keeps gates in the established experiment extension, not
// in plan/v1's closed comparison object. It never rewrites existing plan bytes.
func PortableComparison(c Comparison) (PortableComparisonPolicy, PortableExperimentExtensions) {
	return PortableComparisonPolicy{
			Baseline:           c.Baseline,
			Candidate:          c.Candidate,
			RequiredEqual:      c.RequiredEqual,
			AllowedDifferences: c.AllowedDifferences,
		}, PortableExperimentExtensions{ComparisonGates: PortableComparisonGates{
			MinCandidateEndToEndPass: c.Gates.MinCandidateEndToEndPass,
			MaxQualityDrop:           c.Gates.MaxQualityDrop,
			MaxTotalTokensRatio:      c.Gates.MaxTotalTokensRatio,
		}}
}

func PublicPlanProjection(plan Frozen) (PublicPlan, error) {
	if plan.Kind() != "playground.plan/v1" {
		return PublicPlan{}, Failure("eval_invalid")
	}
	var value struct {
		ExperimentID string         `json:"experiment_id"`
		CreatedAt    string         `json:"created_at"`
		Members      []PublicMember `json:"members"`
	}
	if err := json.Unmarshal(plan.Bytes(), &value); err != nil {
		return PublicPlan{}, Failure("eval_invalid")
	}
	out := PublicPlan{
		SchemaVersion:       "playground.public-projection/v1",
		SourceSchemaVersion: plan.Kind(),
		SourceRecordSHA256:  plan.Digest(),
		ExperimentID:        value.ExperimentID,
		CreatedAt:           value.CreatedAt,
		Members:             value.Members,
	}
	data, err := json.Marshal(out)
	if err != nil {
		return PublicPlan{}, Failure("eval_invalid")
	}
	if err := Validate("PublicPlan", data); err != nil {
		return PublicPlan{}, err
	}
	return out, nil
}

func validatePortableResources(value any) error {
	switch v := value.(type) {
	case map[string]any:
		if resource, ok := v["resource"].(string); ok {
			if _, ref := v["sha256"]; ref {
				for _, part := range strings.Split(resource, "/") {
					if part == "." || part == ".." || part == "" {
						return Failure("eval_invalid")
					}
				}
			}
		}
		for key, child := range v {
			if key == "extensions" || key == "parameters" || key == "settings" {
				continue
			}
			if err := validatePortableResources(child); err != nil {
				return err
			}
		}
	case []any:
		for _, child := range v {
			if err := validatePortableResources(child); err != nil {
				return err
			}
		}
	}
	return nil
}
