package evaldomain

import (
	"encoding/json"
	"slices"
	"testing"

	evalschema "github.com/grauwolf32/contractor/api/evals/v1"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedLifecycleStatesMatchSharedCases(t *testing.T) {
	t.Parallel()
	data := contractstest.ReadFile(t, "api", "testdata", "evals", "lifecycle-states.json")
	var cases struct {
		States   []State `json:"states"`
		Terminal []State `json:"terminal"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, state := range cases.States {
		if state.Terminal() != slices.Contains(cases.Terminal, state) {
			t.Errorf("state %q: Terminal() = %v", state, state.Terminal())
		}
	}
	for _, state := range cases.Terminal {
		if !slices.Contains(cases.States, state) {
			t.Errorf("terminal state %q is not a lifecycle state", state)
		}
	}
	raw, err := evalschema.Files.ReadFile("managed.schema.json")
	if err != nil {
		t.Fatal(err)
	}
	var schema struct {
		Defs map[string]struct {
			Properties struct {
				State struct {
					Enum []State `json:"enum"`
				} `json:"state"`
			} `json:"properties"`
		} `json:"$defs"`
	}
	if err := json.Unmarshal(raw, &schema); err != nil {
		t.Fatal(err)
	}
	want := slices.Sorted(slices.Values(cases.States))
	for _, name := range []string{"Experiment", "ExperimentSummary", "ExperimentReceipt"} {
		if got := slices.Sorted(slices.Values(schema.Defs[name].Properties.State.Enum)); !slices.Equal(got, want) {
			t.Errorf("%s.state enum = %v, want %v", name, got, want)
		}
	}
}
