package evalservice

import (
	"encoding/json"
	"slices"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

// Readiness exposes only preparation facts. Binding snapshots and private check
// content remain in their existing owner-scoped stores.
type Readiness struct {
	Pins []ReadinessPin `json:"pins"`
	Arms []ReadinessArm `json:"arms"`
}

type ReadinessPin struct {
	Dimension       string `json:"dimension"`
	RequiredEqual   bool   `json:"requiredEqual"`
	BaselineOrigin  string `json:"baselineOrigin"`
	CandidateOrigin string `json:"candidateOrigin"`
	Status          string `json:"status"`
}

type ReadinessArm struct {
	VariantID   string `json:"variantId"`
	Expected    int    `json:"expected"`
	Eligible    int    `json:"eligible"`
	Unsupported int    `json:"unsupported"`
	Blocked     int    `json:"blocked"`
}

func nativeReadiness(plan evalstore.Plan) (*Readiness, error) {
	if plan.Document.Kind() != portablePlanSchema {
		return nil, nil
	}
	var body portablePlan
	var setup struct {
		Comparison evaldomain.Comparison `json:"comparison"`
	}
	if err := json.Unmarshal(plan.Document.Bytes(), &body); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(plan.Setup, &setup); err != nil {
		return nil, err
	}
	result := &Readiness{Pins: []ReadinessPin{}, Arms: []ReadinessArm{}}
	comparison := setup.Comparison
	dimensions := make([]string, 0)
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
		result.Pins = append(result.Pins, ReadinessPin{dimension, slices.Contains(comparison.RequiredEqual, dimension), pinOrigin(a), pinOrigin(b), status})
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

func pinOrigin(pin Pin) string {
	if pin.Origin == "observed" || pin.Origin == "producer-supplied" {
		return pin.Origin
	}
	return "unavailable"
}
