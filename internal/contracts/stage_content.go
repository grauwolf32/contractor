package contracts

import "strings"

type StageContentRequest struct {
	APIVersion      string                 `json:"apiVersion"`
	SubtaskID       string                 `json:"subtaskId"`
	Objective       string                 `json:"objective"`
	Instructions    string                 `json:"instructions"`
	Parameters      map[string]string      `json:"parameters"`
	Artifacts       map[string]ArtifactRef `json:"artifacts"`
	ResultArtifacts map[string]ArtifactRef `json:"resultArtifacts,omitempty"`
}

func (r StageContentRequest) Validate() error {
	if err := validateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if err := validateWorkerSubtaskID(r.SubtaskID); err != nil {
		return err
	}
	if strings.TrimSpace(r.Objective) == "" || strings.TrimSpace(r.Instructions) == "" {
		return invalidf("objective and instructions must not be empty")
	}
	for key := range r.Parameters {
		if err := validateOpaqueID("parameter name", key); err != nil {
			return err
		}
	}
	for key, ref := range r.Artifacts {
		if err := validateOpaqueID("artifact context name", key); err != nil {
			return err
		}
		if err := ref.ValidateExact(); err != nil {
			return err
		}
	}
	for key, ref := range r.ResultArtifacts {
		if err := validateOpaqueID("result artifact slot", key); err != nil {
			return err
		}
		if err := ref.Validate(); err != nil {
			return err
		}
		if ref.Revision != nil {
			return invalidf("result artifact binding must be versionless")
		}
	}
	return nil
}

type StageOutcome string

const (
	StageSucceeded StageOutcome = "succeeded"
	StageFailed    StageOutcome = "failed"
)

type StageContentResult struct {
	APIVersion string                 `json:"apiVersion"`
	Outcome    StageOutcome           `json:"outcome"`
	Summary    string                 `json:"summary"`
	Artifacts  map[string]ArtifactRef `json:"artifacts"`
	Error      *TerminationError      `json:"error,omitempty"`
}

func (r StageContentResult) Validate() error {
	if err := validateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if strings.TrimSpace(r.Summary) == "" {
		return invalidf("summary must not be empty")
	}
	switch r.Outcome {
	case StageSucceeded:
		if r.Error != nil {
			return invalidf("successful StageContentResult must not contain error")
		}
	case StageFailed:
		if r.Error == nil {
			return invalidf("failed StageContentResult requires error")
		}
		if err := validateTerminationError(*r.Error); err != nil {
			return err
		}
	default:
		return invalidf("unknown StageContentResult outcome %q", r.Outcome)
	}
	for key, ref := range r.Artifacts {
		if err := validateOpaqueID("result artifact slot", key); err != nil {
			return err
		}
		if err := ref.ValidateExact(); err != nil {
			return err
		}
	}
	return nil
}
