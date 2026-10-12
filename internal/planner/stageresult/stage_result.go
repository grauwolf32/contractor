// Package stageresult holds the result a Planner returns for one Stage
// (PLN-R1). It is a leaf of the planner packages: RunStore, which persists
// Stage results, and the public API read them without importing the Planner
// implementations.
package stageresult

import (
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// The Stage result bounds (WRK-R37). They are separate from the
// StageContentRequest bound of the private wire even where the numbers
// coincide.
const (
	MaxStageResultBytes        = 256 * 1024
	MaxStageResultSummaryBytes = 64 * 1024
	MaxStageResultArtifacts    = 128
)

type StageOutcome string

const (
	StageSucceeded StageOutcome = "succeeded"
	StageFailed    StageOutcome = "failed"
)

// StageContentResult is what Planner.Run returns to the Scheduler (PLN-R1),
// which RunStore persists with the Stage. It never crosses the Server/Runtime
// boundary; its members keep their JSON names because stored Stage results
// carry them.
type StageContentResult struct {
	APIVersion string                           `json:"apiVersion" strictjson:"required"`
	Outcome    StageOutcome                     `json:"outcome" strictjson:"required"`
	Summary    string                           `json:"summary" strictjson:"required"`
	Artifacts  map[string]contracts.ArtifactRef `json:"artifacts" strictjson:"required"`
	Error      *contracts.TerminationError      `json:"error,omitempty"`
}

// Clone returns a copy that shares no artifact map, revision or error value
// with r.
func (r StageContentResult) Clone() StageContentResult {
	result := r
	result.Artifacts = contracts.CloneArtifactRefs(r.Artifacts)
	if r.Error != nil {
		terminationError := *r.Error
		result.Error = &terminationError
	}
	return result
}

// Validate is the only definition of the StageResult bounds: every Planner,
// the Scheduler and RunStore call it, so they cannot disagree about a
// candidate's size (WRK-R37).
func (r StageContentResult) Validate() error {
	if err := contracts.ValidateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if strings.TrimSpace(r.Summary) == "" || len(r.Summary) > MaxStageResultSummaryBytes {
		return contracts.Invalidf("summary must be non-empty and at most %d bytes", MaxStageResultSummaryBytes)
	}
	if r.Artifacts == nil || len(r.Artifacts) > MaxStageResultArtifacts {
		return contracts.Invalidf("StageContentResult artifacts must be a non-null map of at most %d entries", MaxStageResultArtifacts)
	}
	switch r.Outcome {
	case StageSucceeded:
		if r.Error != nil {
			return contracts.Invalidf("successful StageContentResult must not contain error")
		}
	case StageFailed:
		if r.Error == nil {
			return contracts.Invalidf("failed StageContentResult requires error")
		}
		if err := r.Error.Validate(); err != nil {
			return err
		}
	default:
		return contracts.Invalidf("unknown StageContentResult outcome %q", r.Outcome)
	}
	for key, ref := range r.Artifacts {
		if err := contracts.ValidateOpaqueID("result artifact slot", key); err != nil {
			return err
		}
		if err := ref.ValidateExact(); err != nil {
			return err
		}
	}
	size, err := contracts.ResultJSONSize(r)
	if err != nil || size > MaxStageResultBytes {
		return contracts.Invalidf("StageContentResult exceeds its bounded contract")
	}
	return nil
}
