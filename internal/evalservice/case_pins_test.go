package evalservice

import (
	"encoding/json"
	"maps"
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func TestSharedCaseComparisonPinsPreserveAssociationIndependentOfOrder(t *testing.T) {
	draft, data, pre := planInputs(t)
	for i := range data.Cases {
		data.Cases[i].Task.Objective = "Inspect " + data.Cases[i].ID
	}
	cases := slices.Clone(data.Cases)
	configuration := map[string]Pin{"instructions": observedPin("configuration")}
	before, _ := json.Marshal(cases)
	want, err := caseComparisonPins(cases, draft.Variants[0], configuration)
	if err != nil {
		t.Fatal(err)
	}
	reversed := slices.Clone(cases)
	slices.Reverse(reversed)
	got, err := caseComparisonPins(reversed, draft.Variants[0], configuration)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("case order changed provenance: %v", err)
	}
	after, _ := json.Marshal(cases)
	if string(before) != string(after) || len(configuration) != 1 {
		t.Fatal("pin derivation changed its inputs")
	}
	swapped := slices.Clone(cases)
	swapped[0].Task, swapped[1].Task = swapped[1].Task, swapped[0].Task
	got, err = caseComparisonPins(swapped, draft.Variants[0], configuration)
	if err != nil || *got["tasks"].Value == *want["tasks"].Value || *got["source"].Value != *want["source"].Value {
		t.Fatalf("task reassociation was hidden: %v", err)
	}
	// The ordinary native builder consumes the same definition.
	bundle, err := BuildPlan("native-pins", time.Date(2026, 11, 16, 0, 0, 0, 0, time.UTC), draft, data, pre)
	if err != nil {
		t.Fatal(err)
	}
	var plan struct {
		Pins map[string]map[string]Pin `json:"pins"`
	}
	if err := json.Unmarshal(bundle.Plan.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	for _, dim := range []string{"source", "tasks"} {
		if !reflect.DeepEqual(plan.Pins[draft.Variants[0].ID][dim], want[dim]) {
			t.Fatalf("native %s pin differs", dim)
		}
	}
}

func TestSharedCaseComparisonPinsExposeRecipeDifferences(t *testing.T) {
	draft, data, _ := planInputs(t)
	variant := draft.Variants[0]
	want, err := caseComparisonPins(data.Cases, variant, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name   string
		dim    string
		change func([]evaldomain.Case, *evaldomain.Variant)
	}{
		{"revision", "source", func(c []evaldomain.Case, _ *evaldomain.Variant) {
			for role, ref := range c[0].Inputs {
				ref.Revision += "-different"
				c[0].Inputs[role] = ref
			}
		}},
		{"parameters", "tasks", func(_ []evaldomain.Case, v *evaldomain.Variant) {
			v.Parameters = map[string]string{"objective": "different"}
		}},
		{"input mapping", "tasks", func(_ []evaldomain.Case, v *evaldomain.Variant) {
			v.InputMapping = map[string]string{"source": "other"}
		}},
		{"output mapping", "tasks", func(_ []evaldomain.Case, v *evaldomain.Variant) {
			v.OutputMapping = map[string]string{"report": "other"}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			cases := slices.Clone(data.Cases)
			cases[0].Inputs = maps.Clone(cases[0].Inputs)
			v := variant
			test.change(cases, &v)
			got, err := caseComparisonPins(cases, v, nil)
			if err != nil || *got[test.dim].Value == *want[test.dim].Value {
				t.Fatalf("difference hidden: %v", err)
			}
		})
	}
}
