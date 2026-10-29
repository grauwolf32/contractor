// Package operations dispatches bounded Server operations directly for Scheduler.
// Operation modules adapt inputs and outputs; domain modules own their algorithms.
package operations

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

// Call contains only pinned Stage authority. Configuration cannot select a foreign owner,
// Project, artifact scope or implementation path. Domain arguments may include
// an explicit target origin without authorizing target network calls.
type Call struct {
	RunID            string
	StageExecutionID string
	ClaimID          string
	Arguments        map[string]any
	Inputs           map[string]contracts.ArtifactRef
	Outputs          map[string]contracts.ArtifactRef
}

type Operation interface {
	Ref() string
	Run(context.Context, Call) (stageresult.StageContentResult, error)
}

// Failure is a classified operation outcome, independent of Planner/Worker.
// Messages describe the failure class without echoing artifact content.
type Failure struct {
	Code      string
	Message   string
	Retryable bool
}

func (f *Failure) Error() string { return f.Message }

type Registry struct{ operations map[string]Operation }

func NewRegistry(modules ...Operation) (*Registry, error) {
	r := &Registry{operations: make(map[string]Operation, len(modules))}
	for _, module := range modules {
		if module == nil || module.Ref() == "" || r.operations[module.Ref()] != nil {
			return nil, fmt.Errorf("operation modules require distinct registered references")
		}
		r.operations[module.Ref()] = module
	}
	return r, nil
}

// Run joins the operation before returning, including cancellation. Recovering
// an interrupted call uses the same Stage identity and existing domain receipts.
func (r *Registry) Run(ctx context.Context, ref string, call Call) (stageresult.StageContentResult, error) {
	module := r.operations[ref]
	if module == nil {
		return stageresult.StageContentResult{}, contracts.Invalidf("unknown Server operation")
	}
	if !contracts.ValidResourceID(call.RunID) || !contracts.ValidResourceID(call.StageExecutionID) || !contracts.ValidResourceID(call.ClaimID) {
		return stageresult.StageContentResult{}, contracts.Invalidf("operation requires pinned Stage ownership")
	}
	for _, input := range call.Inputs {
		if input.ValidateExact() != nil {
			return stageresult.StageContentResult{}, contracts.ErrValidation
		}
	}
	if err := ctx.Err(); err != nil {
		return stageresult.StageContentResult{}, err
	}
	result, err := module.Run(ctx, call)
	if err != nil {
		return stageresult.StageContentResult{}, err
	}
	if err := ctx.Err(); err != nil {
		return stageresult.StageContentResult{}, err
	}
	if err := result.Validate(); err != nil {
		return stageresult.StageContentResult{}, err
	}
	for name, actual := range result.Artifacts {
		binding, declared := call.Outputs[name]
		if !declared || actual.Namespace != binding.Namespace || actual.Name != binding.Name {
			return stageresult.StageContentResult{}, contracts.Invalidf("operation returned an undeclared output binding")
		}
	}
	return result, nil
}
