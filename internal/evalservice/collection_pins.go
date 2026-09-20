package evalservice

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

func verifiedComparisonPins(ctx context.Context, st *evalstore.Store, e evalstore.Experiment, plan evalstore.Plan, comparison evaldomain.Comparison) (bool, error) {
	pins := map[string]map[string]Pin{}
	if plan.Document.Kind() == "playground.plan/v1" {
		var body struct {
			Pins map[string]map[string]Pin `json:"pins"`
		}
		if err := json.Unmarshal(plan.Document.Bytes(), &body); err != nil {
			return false, err
		}
		pins = body.Pins
	} else {
		for _, arm := range []string{comparison.Baseline, comparison.Candidate} {
			r, err := st.PlanResource(ctx, e.OwnerID, e.ID, "bindings/"+arm+".json")
			if notFound(err) {
				return false, nil
			}
			if err != nil {
				return false, err
			}
			var binding struct {
				Settings struct {
					Pins map[string]Pin `json:"observedPins"`
				} `json:"settings"`
			}
			if err = json.Unmarshal(r.Document.Bytes(), &binding); err != nil {
				return false, err
			}
			pins[arm] = binding.Settings.Pins
		}
	}
	for _, dimension := range comparison.RequiredEqual {
		a, b := pins[comparison.Baseline][dimension], pins[comparison.Candidate][dimension]
		if a.Origin != "observed" || b.Origin != "observed" || a.Value == nil || b.Value == nil || *a.Value != *b.Value {
			return false, nil
		}
	}
	return true, nil
}
