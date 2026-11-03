package planner

import (
	"context"
	"fmt"
	"maps"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

// validateCandidate checks result names, then verifies each result artifact
// in result-name order, so a candidate with several invalid artifacts is
// always rejected for the same one (PLN-R118).
func validateCandidate(
	ctx context.Context,
	runID string,
	contract map[string]resolved.ArtifactSlot,
	result stageresult.StageContentResult,
	inspector ArtifactInspector,
) *Error {
	if err := result.Validate(); err != nil {
		return NewError(
			"invalid_worker_result", "Worker returned an invalid stageresult.StageContentResult", false, err,
		)
	}
	names := slices.Sorted(maps.Keys(result.Artifacts))
	for _, name := range names {
		if _, ok := contract[name]; !ok {
			return NewError(
				"result_contract_violation",
				fmt.Sprintf("Worker returned undeclared result artifact %q", name),
				false,
				nil,
			)
		}
	}
	if result.Outcome == stageresult.StageSucceeded {
		for _, name := range slices.Sorted(maps.Keys(contract)) {
			if _, ok := result.Artifacts[name]; contract[name].Required && !ok {
				return NewError(
					"result_contract_violation",
					fmt.Sprintf("Worker omitted required result artifact %q", name),
					false,
					nil,
				)
			}
		}
	}
	for _, name := range names {
		ref := result.Artifacts[name]
		if artifactpolicy.IsReservedMemoryBinding(ref.Namespace, ref.Name) {
			return NewError(
				"result_contract_violation",
				fmt.Sprintf("Result artifact %q identifies a reserved Memory binding", name),
				false,
				nil,
			)
		}
		metadata, err := inspector.Inspect(ctx, runID, ref.Clone())
		if err != nil {
			if artifacts.IsExactReferenceRejection(err) {
				return NewError(
					"result_contract_violation",
					fmt.Sprintf("Result artifact %q does not identify an exact revision in the RunScope", name),
					false,
					err,
				)
			}
			return NewError(
				"result_artifact_unavailable",
				fmt.Sprintf("Result artifact %q could not be verified", name),
				true,
				err,
			)
		}
		if !contracts.AcceptsMediaType(contract[name].MediaTypes, metadata.MediaType) {
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

// ValidateCandidate checks a Planner candidate against the immutable Stage
// result contract without accepting or advancing any Artifact binding.
func ValidateCandidate(
	ctx context.Context,
	runID string,
	contract map[string]resolved.ArtifactSlot,
	result stageresult.StageContentResult,
	inspector ArtifactInspector,
) *Error {
	return validateCandidate(ctx, runID, contract, result, inspector)
}
