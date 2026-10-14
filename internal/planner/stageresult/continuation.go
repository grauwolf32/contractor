package stageresult

import (
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// SummaryContinuation is explicit saved progress for a fresh Stage attempt.
// Its deadline is control data and never enters the Planner model input.
type SummaryContinuation struct {
	SourceStageExecutionID string                           `json:"sourceStageExecutionId"`
	Summary                string                           `json:"summary"`
	SummaryArtifact        contracts.ArtifactRef            `json:"summaryArtifact"`
	Artifacts              map[string]contracts.ArtifactRef `json:"artifacts"`
	Deadline               time.Time                        `json:"deadline"`
}

func (s SummaryContinuation) Clone() SummaryContinuation {
	result := s
	result.SummaryArtifact = s.SummaryArtifact.Clone()
	result.Artifacts = contracts.CloneArtifactRefs(s.Artifacts)
	return result
}

func (s SummaryContinuation) Validate() error {
	if !contracts.ValidResourceID(s.SourceStageExecutionID) || s.Deadline.IsZero() {
		return contracts.Invalidf("summary continuation requires its source execution and deadline")
	}
	if s.SummaryArtifact.Name != SummaryArtifactPrefix+s.SourceStageExecutionID {
		return contracts.Invalidf("summary continuation artifact differs from its source execution")
	}
	if err := s.SummaryArtifact.ValidateExact(); err != nil {
		return err
	}
	// The same semantic bounds apply before publication and after recovery.
	return (StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: StageFailed,
		Summary: s.Summary, Artifacts: s.Artifacts, Summarized: true,
		SummaryArtifact: &s.SummaryArtifact,
		Error:           &contracts.TerminationError{Code: SummaryRequiredCode, Message: "Continuation required"},
	}).Validate()
}
