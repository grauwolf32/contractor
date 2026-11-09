// Package publishfindings adapts a Stage call to the existing Finding owner.
package publishfindings

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/operations"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

type Module struct{ service *findingintake.Service }

func New(service *findingintake.Service) *Module { return &Module{service: service} }
func (m *Module) Ref() string                    { return contracts.PublishFindingsOperation }
func (m *Module) Run(ctx context.Context, call operations.Call) (stageresult.StageContentResult, error) {
	var summary findingintake.CandidatePublicationSummary
	var err error
	if len(call.Arguments) != 0 || len(call.Outputs) != 0 {
		return stageresult.StageContentResult{}, &operations.Failure{Code: "invalid_request", Message: "Finding publication forbids arguments and Artifact outputs"}
	}
	switch {
	case len(call.Inputs) == 1 && call.Inputs["materials"].ValidateExact() == nil:
		summary, err = m.service.PublishCandidateSet(ctx, call.RunID, call.StageExecutionID, call.ClaimID, call.Inputs["materials"])
	case len(call.Inputs) == 2 && call.Inputs["candidates"].ValidateExact() == nil && call.Inputs["evidence"].ValidateExact() == nil:
		summary, err = m.service.PublishCandidates(ctx, findingintake.CandidatePublication{RunID: call.RunID, StageID: call.StageExecutionID, ClaimID: call.ClaimID, Candidates: call.Inputs["candidates"], Evidence: call.Inputs["evidence"]})
	default:
		return stageresult.StageContentResult{}, &operations.Failure{Code: "invalid_request", Message: "Finding publication requires exact normalized inputs"}
	}

	if err != nil && len(summary.Sources) > 0 && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, runstore.ErrConflict) {
		summary.Complete = false
		encoded, encodeErr := json.Marshal(summary)
		if encodeErr != nil {
			return stageresult.StageContentResult{}, encodeErr
		}
		return stageresult.StageContentResult{APIVersion: contracts.APIVersion, Outcome: stageresult.StageFailed, Summary: string(encoded), Artifacts: map[string]contracts.ArtifactRef{}, Error: &contracts.TerminationError{Code: "server_operation_failed", Message: "Finding publication stopped after committed items; replay the selected inputs", Retryable: persistencepostgres.IsTransientFailure(err)}}, nil
	}
	if err != nil {
		if errors.Is(err, findingintake.ErrInvalid) || errors.Is(err, findingintake.ErrAccessDenied) {
			return stageresult.StageContentResult{}, &operations.Failure{Code: "invalid_request", Message: "Finding publication input or authority is invalid"}
		}
		return stageresult.StageContentResult{}, err
	}
	encoded, err := json.Marshal(summary)
	if err != nil {
		return stageresult.StageContentResult{}, err
	}
	return stageresult.StageContentResult{APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded, Summary: string(encoded), Artifacts: map[string]contracts.ArtifactRef{}}, nil
}
