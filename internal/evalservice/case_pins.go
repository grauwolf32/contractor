package evalservice

import (
	"cmp"
	"maps"
	"slices"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

type effectiveTask struct {
	CaseID     string            `json:"caseId"`
	Task       evaldomain.Task   `json:"task"`
	Parameters map[string]string `json:"parameters"`
	Inputs     map[string]string `json:"inputs"`
	Outputs    map[string]string `json:"outputs"`
}

// caseComparisonPins follows local Resolver validation of every Artifact ref.
// It hashes the actual recipes, never the producer's attributed case/binding
// digests. Case association is stable regardless of member/execution order.
func caseComparisonPins(cases []evaldomain.Case, variant evaldomain.Variant, configuration map[string]Pin) (map[string]Pin, error) {
	pins := maps.Clone(configuration)
	if pins == nil {
		pins = map[string]Pin{}
	}
	ordered := slices.Clone(cases)
	slices.SortFunc(ordered, func(a, b evaldomain.Case) int { return cmp.Compare(a.ID, b.ID) })
	sources := map[string]evaldomain.Artifact{}
	tasks := make([]effectiveTask, 0, len(ordered))
	for i, c := range ordered {
		if c.ID == "" || i > 0 && ordered[i-1].ID == c.ID {
			return nil, evaldomain.Failure("eval_member_conflict")
		}
		for role, ref := range c.Inputs {
			sources["inputs/"+c.ID+"/"+role] = ref
		}
		tasks = append(tasks, effectiveTask{CaseID: c.ID, Task: c.Task, Parameters: MapParameters(c, variant), Inputs: variant.InputMapping, Outputs: variant.OutputMapping})
	}
	source, err := hashJSON(sources)
	if err != nil {
		return nil, err
	}
	task, err := hashJSON(tasks)
	if err != nil {
		return nil, err
	}
	pins["source"], pins["tasks"] = observedPin(source), observedPin(task)
	return pins, nil
}
