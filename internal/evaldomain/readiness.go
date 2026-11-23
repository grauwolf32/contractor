package evaldomain

import (
	"encoding/json"
	"slices"
)

// Readiness is the preparation summary of a frozen native plan: the equality
// status and origin of each pinned dimension and the member eligibility
// counts of each arm. It never carries pin values, binding snapshots or
// private check content. The plan freeze stores it with the plan (EVM-R242);
// strictjson tags make a stored copy that lacks a member fail to decode.
type Readiness struct {
	Pins []ReadinessPin `json:"pins" strictjson:"required"`
	Arms []ReadinessArm `json:"arms" strictjson:"required"`
}

type ReadinessPin struct {
	Dimension       string `json:"dimension" strictjson:"required"`
	RequiredEqual   bool   `json:"requiredEqual" strictjson:"required"`
	BaselineOrigin  string `json:"baselineOrigin" strictjson:"required"`
	CandidateOrigin string `json:"candidateOrigin" strictjson:"required"`
	Status          string `json:"status" strictjson:"required"`
}

type ReadinessArm struct {
	VariantID   string `json:"variantId" strictjson:"required"`
	Expected    int    `json:"expected" strictjson:"required"`
	Eligible    int    `json:"eligible" strictjson:"required"`
	Unsupported int    `json:"unsupported" strictjson:"required"`
	Blocked     int    `json:"blocked" strictjson:"required"`
}

// NativeReadiness derives the readiness of a playground.plan/v1 document
// under its setup comparison. Pins list every dimension any variant pins, in
// byte order; arms list the baseline, then the candidate. Its size depends on
// the pinned dimensions, at most two arms of 1,024 each, never on the member
// count (LIM-EVM-19).
func NativeReadiness(plan Frozen, comparison Comparison) (Readiness, error) {
	if plan.Kind() != "playground.plan/v1" {
		return Readiness{}, Failure("eval_invalid")
	}
	type pin struct {
		Value  *string `json:"value"`
		Origin string  `json:"origin"`
	}
	var body struct {
		Pins    map[string]map[string]pin `json:"pins"`
		Members []struct {
			VariantID   string `json:"variant_id"`
			Eligibility string `json:"eligibility"`
		} `json:"members"`
	}
	if err := json.Unmarshal(plan.Bytes(), &body); err != nil {
		return Readiness{}, Failure("eval_invalid")
	}
	origin := func(p pin) string {
		if p.Origin == "observed" || p.Origin == "producer-supplied" {
			return p.Origin
		}
		return "unavailable"
	}
	result := Readiness{Pins: []ReadinessPin{}, Arms: []ReadinessArm{}}
	var dimensions []string
	for _, pins := range body.Pins {
		for dimension := range pins {
			dimensions = append(dimensions, dimension)
		}
	}
	slices.Sort(dimensions)
	for _, dimension := range slices.Compact(dimensions) {
		a, b := body.Pins[comparison.Baseline][dimension], body.Pins[comparison.Candidate][dimension]
		status := "unavailable"
		if a.Origin == "observed" && b.Origin == "observed" && a.Value != nil && b.Value != nil {
			status = "different"
			if *a.Value == *b.Value {
				status = "equal"
			}
		}
		result.Pins = append(result.Pins, ReadinessPin{dimension, slices.Contains(comparison.RequiredEqual, dimension), origin(a), origin(b), status})
	}
	for _, arm := range []string{comparison.Baseline, comparison.Candidate} {
		counts := ReadinessArm{VariantID: arm}
		for _, member := range body.Members {
			if member.VariantID != arm {
				continue
			}
			counts.Expected++
			switch member.Eligibility {
			case "eligible":
				counts.Eligible++
			case "unsupported":
				counts.Unsupported++
			case "blocked":
				counts.Blocked++
			}
		}
		result.Arms = append(result.Arms, counts)
	}
	return result, nil
}
