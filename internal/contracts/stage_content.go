package contracts

import (
	"strings"
	"time"
)

type StageContentRequest struct {
	APIVersion      string                 `json:"apiVersion" strictjson:"required"`
	SubtaskID       string                 `json:"subtaskId" strictjson:"required"`
	Objective       string                 `json:"objective" strictjson:"required"`
	Instructions    string                 `json:"instructions" strictjson:"required"`
	Parameters      map[string]string      `json:"parameters" strictjson:"required"`
	Artifacts       map[string]ArtifactRef `json:"artifacts" strictjson:"required"`
	ResultArtifacts map[string]ArtifactRef `json:"resultArtifacts,omitempty"`
	Deadline        *time.Time             `json:"deadline,omitempty"`
}

func (r StageContentRequest) Validate() error {
	if err := ValidateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if err := ValidateWorkerSubtaskID(r.SubtaskID); err != nil {
		return err
	}
	if strings.TrimSpace(r.Objective) == "" || strings.TrimSpace(r.Instructions) == "" {
		return Invalidf("objective and instructions must not be empty")
	}
	// The schema requires both maps, so Go never sends the null a nil map
	// encodes to, which the Runtime would reject.
	if r.Parameters == nil || r.Artifacts == nil {
		return Invalidf("parameters and artifacts must be non-null maps")
	}
	for key := range r.Parameters {
		if err := ValidateOpaqueID("parameter name", key); err != nil {
			return err
		}
	}
	for key, ref := range r.Artifacts {
		if err := ValidateOpaqueID("artifact context name", key); err != nil {
			return err
		}
		if err := ref.ValidateExact(); err != nil {
			return err
		}
	}
	for key, ref := range r.ResultArtifacts {
		if err := ValidateOpaqueID("result artifact slot", key); err != nil {
			return err
		}
		if err := ref.Validate(); err != nil {
			return err
		}
		if ref.Revision != nil {
			return Invalidf("result artifact binding must be versionless")
		}
	}
	// The Runtime measures the same compact UTF-8 encoding (WRK-R37), so a
	// request the Planner sends is never rejected by the Worker for its size.
	size, err := ResultJSONSize(r)
	if err != nil || size > MaxStageRequestBytes {
		return Invalidf("StageContentRequest exceeds its bounded contract")
	}
	return nil
}

type StageOutcome string

const (
	StageSucceeded StageOutcome = "succeeded"
	StageFailed    StageOutcome = "failed"
)

type StageContentResult struct {
	APIVersion string                 `json:"apiVersion" strictjson:"required"`
	Outcome    StageOutcome           `json:"outcome" strictjson:"required"`
	Summary    string                 `json:"summary" strictjson:"required"`
	Artifacts  map[string]ArtifactRef `json:"artifacts" strictjson:"required"`
	Error      *TerminationError      `json:"error,omitempty"`
}

// Clone returns a copy that shares no artifact map, revision or error value
// with r.
func (r StageContentResult) Clone() StageContentResult {
	result := r
	result.Artifacts = CloneArtifactRefs(r.Artifacts)
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
	if err := ValidateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if strings.TrimSpace(r.Summary) == "" || len(r.Summary) > MaxStageResultSummaryBytes {
		return Invalidf("summary must be non-empty and at most %d bytes", MaxStageResultSummaryBytes)
	}
	if r.Artifacts == nil || len(r.Artifacts) > MaxStageResultArtifacts {
		return Invalidf("StageContentResult artifacts must be a non-null map of at most %d entries", MaxStageResultArtifacts)
	}
	switch r.Outcome {
	case StageSucceeded:
		if r.Error != nil {
			return Invalidf("successful StageContentResult must not contain error")
		}
	case StageFailed:
		if r.Error == nil {
			return Invalidf("failed StageContentResult requires error")
		}
		if err := r.Error.Validate(); err != nil {
			return err
		}
	default:
		return Invalidf("unknown StageContentResult outcome %q", r.Outcome)
	}
	for key, ref := range r.Artifacts {
		if err := ValidateOpaqueID("result artifact slot", key); err != nil {
			return err
		}
		if err := ref.ValidateExact(); err != nil {
			return err
		}
	}
	size, err := ResultJSONSize(r)
	if err != nil || size > MaxStageResultBytes {
		return Invalidf("StageContentResult exceeds its bounded contract")
	}
	return nil
}

type TerminationError struct {
	Code      string `json:"code" strictjson:"required"`
	Message   string `json:"message" strictjson:"required"`
	Retryable bool   `json:"retryable" strictjson:"required"`
}

func (e TerminationError) Validate() error {
	if strings.TrimSpace(e.Code) == "" || strings.TrimSpace(e.Message) == "" {
		return Invalidf("termination error code/message must not be empty")
	}
	return nil
}
