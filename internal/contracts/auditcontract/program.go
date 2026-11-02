package auditcontract

import "github.com/grauwolf32/contractor/internal/helpers/strictjson"

// Program is the authored common program. Its selected Workflow closures and
// exact materials are pinned by the Audit owner; none is inferred from outputs.
type Program struct {
	Schema       string                     `json:"schema" yaml:"schema"`
	Inputs       map[string]ProgramInput    `json:"inputs" yaml:"inputs"`
	Workflows    map[string]ProgramWorkflow `json:"workflows" yaml:"workflows"`
	Hypotheses   map[string]HypothesisSeed  `json:"hypotheses" yaml:"hypotheses"`
	InitialWork  []InitialWork              `json:"initial_work" yaml:"initial_work"`
	Continuation *Continuation              `json:"continuation,omitempty" yaml:"continuation,omitempty"`
	Limits       ProgramLimits              `json:"limits" yaml:"limits"`
	Policy       OperationPolicy            `json:"policy" yaml:"policy"`
}

type ProgramInput struct {
	Required   bool     `json:"required" yaml:"required"`
	MediaTypes []string `json:"media_types" yaml:"media_types"`
}

type ProgramWorkflow struct {
	Ref        string                  `json:"ref" yaml:"ref"`
	Inputs     map[string]InputMapping `json:"inputs" yaml:"inputs"`
	Parameters map[string]string       `json:"parameters" yaml:"parameters"`
	Outputs    map[string]string       `json:"outputs" yaml:"outputs"`
}

type InputMapping struct {
	Source string `json:"source" yaml:"source"`
	Name   string `json:"name,omitempty" yaml:"name,omitempty"`
}

type HypothesisSeed struct {
	Statement string            `json:"statement" yaml:"statement"`
	Scope     map[string]string `json:"scope" yaml:"scope"`
	Grounds   string            `json:"grounds" yaml:"grounds"`
}

type InitialWork struct {
	Key        string            `json:"key" yaml:"key"`
	Workflow   string            `json:"workflow" yaml:"workflow"`
	Objective  string            `json:"objective" yaml:"objective"`
	Hypothesis string            `json:"hypothesis,omitempty" yaml:"hypothesis,omitempty"`
	Inputs     map[string]string `json:"inputs,omitempty" yaml:"inputs,omitempty"`
}

type Continuation struct {
	Workflow string `json:"workflow" yaml:"workflow"`
}

type ProgramLimits struct {
	MaxTasks               int   `json:"max_tasks" yaml:"max_tasks"`
	MaxIterations          int   `json:"max_iterations" yaml:"max_iterations"`
	MaxTasksPerIteration   int   `json:"max_tasks_per_iteration" yaml:"max_tasks_per_iteration"`
	BatchSize              int   `json:"batch_size" yaml:"batch_size"`
	MaxSubmittedRuns       int   `json:"max_submitted_runs" yaml:"max_submitted_runs"`
	MaxAttempts            int   `json:"max_attempts" yaml:"max_attempts"`
	DeadlineSeconds        int   `json:"deadline_seconds" yaml:"deadline_seconds"`
	MaxEvidenceBytes       int64 `json:"max_evidence_bytes" yaml:"max_evidence_bytes"`
	MaxHypotheses          int   `json:"max_hypotheses" yaml:"max_hypotheses"`
	MaxHypothesisRevisions int   `json:"max_hypothesis_revisions" yaml:"max_hypothesis_revisions"`
	MaxHypothesisBytes     int64 `json:"max_hypothesis_bytes" yaml:"max_hypothesis_bytes"`
}

type OperationPolicy struct {
	ActiveOperations string `json:"active_operations" yaml:"active_operations"`
}

// DecodeProgram applies the shared strict document contract before exposing a
// typed program to catalog and execution callers. Rejection returns no program.
func DecodeProgram(data []byte) (Program, error) {
	canonical, err := Decode(data, nil)
	if err != nil {
		return Program{}, err
	}
	var program Program
	if strictjson.Decode(canonical, &program) != nil || program.Schema != "contractor.audit.program.v1" {
		return Program{}, errInvalid
	}
	return program, nil
}
