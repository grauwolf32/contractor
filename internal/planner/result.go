package planner

import (
	"context"
	"encoding/json"
	"fmt"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func validateCandidate(
	ctx context.Context,
	runID string,
	contract map[string]workflowconfig.ArtifactSlot,
	result contracts.StageContentResult,
	inspector ArtifactInspector,
) *Error {
	if err := result.Validate(); err != nil {
		return NewError(
			"invalid_worker_result", "Worker returned an invalid StageContentResult", false, err,
		)
	}
	encoded, err := json.Marshal(result)
	if err != nil || len(encoded) > maxStageContentBytes ||
		len(result.Summary) > maxResultSummaryBytes || len(result.Artifacts) > maxResultArtifacts {
		return NewError(
			"invalid_worker_result", "Worker returned an oversized StageContentResult", false, err,
		)
	}
	for name := range result.Artifacts {
		if _, ok := contract[name]; !ok {
			return NewError(
				"result_contract_violation",
				fmt.Sprintf("Worker returned undeclared result artifact %q", name),
				false,
				nil,
			)
		}
	}
	if result.Outcome == contracts.StageSucceeded {
		for name, slot := range contract {
			if _, ok := result.Artifacts[name]; slot.Required && !ok {
				return NewError(
					"result_contract_violation",
					fmt.Sprintf("Worker omitted required result artifact %q", name),
					false,
					nil,
				)
			}
		}
	}
	for name, ref := range result.Artifacts {
		metadata, err := inspector.Inspect(ctx, runID, cloneArtifactRef(ref))
		if err != nil {
			return NewError(
				"result_artifact_unavailable",
				fmt.Sprintf("Result artifact %q could not be verified", name),
				true,
				err,
			)
		}
		if !acceptsMediaType(contract[name].MediaTypes, metadata.MediaType) {
			return NewError(
				"result_contract_violation",
				fmt.Sprintf("Result artifact %q has an incompatible media type", name),
				false,
				nil,
			)
		}
	}
	return nil
}

func acceptsMediaType(accepted []string, actual string) bool {
	for _, mediaType := range accepted {
		if mediaType == "*/*" || mediaType == actual {
			return true
		}
	}
	return false
}
