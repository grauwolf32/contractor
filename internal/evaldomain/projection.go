package evaldomain

import "encoding/json"

// ExecutionCase has no source provenance, private checks, rubrics or expectations.
// The explicit type is the only projection allowed into a Run/Audit recipe.
type ExecutionCase struct {
	ID       string              `json:"id"`
	Task     Task                `json:"task"`
	Inputs   map[string]Artifact `json:"inputs"`
	Requires []string            `json:"requires"`
	Outputs  map[string]Output   `json:"outputs"`
}

func ExecutionProjection(c Case) (ExecutionCase, error) {
	data, err := json.Marshal(c)
	if err != nil {
		return ExecutionCase{}, Failure("eval_invalid")
	}
	if err := Validate("Case", data); err != nil {
		return ExecutionCase{}, err
	}
	// Clone mutable maps/slices to prevent later authoring mutations from changing
	// an already reviewed execution projection.
	var out ExecutionCase
	if err := json.Unmarshal(data, &out); err != nil {
		return out, Failure("eval_invalid")
	}
	return out, nil
}

func DatasetProjection(input DatasetInput, projectID, revision string) (Dataset, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return Dataset{}, Failure("eval_invalid")
	}
	if err := Validate("DatasetInput", data); err != nil {
		return Dataset{}, err
	}
	visible, err := json.Marshal(input.Cases)
	if err != nil {
		return Dataset{}, Failure("eval_invalid")
	}
	out := Dataset{DatasetID: input.DatasetID, Name: input.Name, ProjectID: projectID, Revision: revision, VisibleSHA256: Digest(visible), CaseCount: len(input.Cases), Source: input.Source}
	raw, _ := json.Marshal(out)
	if err := Validate("Dataset", raw); err != nil {
		return Dataset{}, err
	}
	return out, nil
}
