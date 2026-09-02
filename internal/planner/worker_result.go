package planner

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// ValidateWorkerCompletion enforces the trusted request correlation again at
// the framework-neutral Planner boundary. A2A transports are expected to run
// the same validation, but injected/test WorkerInvokers are not trusted.
func ValidateWorkerCompletion(
	completion contracts.WorkerCompletion,
	expectedSubtaskID string,
) *Error {
	if err := completion.Validate(); err != nil {
		return NewError(
			"invalid_worker_result", "Worker returned an invalid WorkerCompletion", false, err,
		)
	}
	if completion.Result != nil && completion.Result.SubtaskID != expectedSubtaskID {
		return NewError(
			"worker_result_subtask_mismatch",
			"Worker result does not match the requested subtask",
			true,
			nil,
		)
	}
	return nil
}

// StageCandidateFromWorkerCompletion is the deterministic passthrough@1
// adapter. Modeled Planners must never call it: their finish tool alone owns
// the Stage candidate.
func StageCandidateFromWorkerCompletion(
	completion contracts.WorkerCompletion,
) (contracts.StageContentResult, error) {
	if err := completion.Validate(); err != nil {
		return contracts.StageContentResult{}, fmt.Errorf("invalid WorkerCompletion: %w", err)
	}
	if completion.Failure != nil {
		failure := completion.Failure
		return contracts.StageContentResult{
			APIVersion: contracts.APIVersion,
			Outcome:    contracts.StageFailed,
			Summary:    failure.Message,
			Artifacts:  map[string]contracts.ArtifactRef{},
			Error: &contracts.TerminationError{
				Code: failure.Code, Message: failure.Message, Retryable: failure.Retryable,
			},
		}, nil
	}
	result := completion.Result
	return contracts.StageContentResult{
		APIVersion: contracts.APIVersion,
		Outcome:    contracts.StageSucceeded,
		Summary:    result.Result,
		Artifacts:  cloneArtifactMap(result.Artifacts),
	}, nil
}

// CloneWorkerCompletion detaches all mutable Worker result maps.
func CloneWorkerCompletion(input contracts.WorkerCompletion) contracts.WorkerCompletion {
	result := input
	if input.Result != nil {
		workerResult := *input.Result
		workerResult.Artifacts = cloneArtifactMap(input.Result.Artifacts)
		workerResult.Observations = cloneWorkerObservations(input.Result.Observations)
		result.Result = &workerResult
	}
	if input.Failure != nil {
		failure := *input.Failure
		result.Failure = &failure
	}
	return result
}

// CloneWorkerResult detaches all mutable maps and slices in the safe
// model-facing Worker result projection.
func CloneWorkerResult(input contracts.WorkerResult) contracts.WorkerResult {
	result := input
	result.Artifacts = cloneArtifactMap(input.Artifacts)
	result.Observations = cloneWorkerObservations(input.Observations)
	return result
}

func cloneWorkerObservations(input contracts.WorkerObservations) contracts.WorkerObservations {
	result := input
	result.Tools = make(map[string]contracts.ToolObservationCount, len(input.Tools))
	for name, counters := range input.Tools {
		result.Tools[name] = counters
	}
	if input.Workspace != nil {
		workspace := *input.Workspace
		workspace.FilesRead = append([]string(nil), input.Workspace.FilesRead...)
		if input.Workspace.UnreadFiles != nil {
			unread := *input.Workspace.UnreadFiles
			workspace.UnreadFiles = &unread
		}
		result.Workspace = &workspace
	}
	return result
}

func cloneArtifactMap(input map[string]contracts.ArtifactRef) map[string]contracts.ArtifactRef {
	result := make(map[string]contracts.ArtifactRef, len(input))
	for name, ref := range input {
		result[name] = cloneArtifactRef(ref)
	}
	return result
}
