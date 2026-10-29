package stageresult

import "github.com/grauwolf32/contractor/internal/contracts"

// SummaryContinuation is explicit saved progress for a fresh Stage attempt.
// Attempt deadlines come from the pinned policy and admission time.
type SummaryContinuation struct {
	SourceStageExecutionID string                           `json:"sourceStageExecutionId"`
	Summary                string                           `json:"summary"`
	SummaryArtifact        contracts.ArtifactRef            `json:"summaryArtifact"`
	Artifacts              map[string]contracts.ArtifactRef `json:"artifacts"`
}

func (s SummaryContinuation) Clone() SummaryContinuation {
	result := s
	result.SummaryArtifact = s.SummaryArtifact.Clone()
	result.Artifacts = contracts.CloneArtifactRefs(s.Artifacts)
	return result
}

func (s SummaryContinuation) Validate() error {
	if !contracts.ValidResourceID(s.SourceStageExecutionID) {
		return contracts.Invalidf("summary continuation requires its source execution")
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
