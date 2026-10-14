package scheduler

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner/stageresult"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// These private transaction labels are outside the authored alias grammar.
const summaryContextPin = "continuation/summary"
const priorArtifactContextPin = "continuation/result/"

func stageContextPins(snapshot runstore.StageContextSnapshot) []ContextPin {
	result := make([]ContextPin, 0, len(snapshot.Artifacts))
	for name, value := range snapshot.Artifacts {
		if value.Artifact != nil {
			result = append(result, ContextPin{Name: name, Ref: value.Artifact.Clone()})
		}
	}
	if value := snapshot.Continuation; value != nil {
		result = append(result, ContextPin{Name: summaryContextPin, Ref: value.SummaryArtifact.Clone()})
		for name, ref := range value.Artifacts {
			result = append(result, ContextPin{Name: priorArtifactContextPin + name, Ref: ref.Clone()})
		}
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	return result
}

func (s *Scheduler) validateSummarySource(ctx context.Context, run runstore.WorkflowRun, execution runstore.StageExecution) error {
	continuation := execution.StageContext.Continuation
	if continuation == nil {
		return nil
	}
	readContext, cancel := context.WithTimeout(ctx, s.options.OperationTimeout)
	defer cancel()
	source, err := s.store.GetStageExecution(readContext, continuation.SourceStageExecutionID)
	if errors.Is(err, runstore.ErrNotFound) {
		return fmt.Errorf("%w: saved summary source is missing", errInvalidRunState)
	}
	if err != nil {
		return err
	}
	result := source.AcceptedResult
	if source.RunID != run.RunID || source.StageName != execution.StageName || source.State != runstore.StageFailed ||
		result == nil || !result.Summarized || result.SummaryArtifact == nil ||
		!result.SummaryArtifact.SameExact(continuation.SummaryArtifact) || result.Summary != continuation.Summary ||
		!reflect.DeepEqual(result.Artifacts, continuation.Artifacts) ||
		!reflect.DeepEqual(source.StageContext.Parameters, execution.StageContext.Parameters) ||
		!reflect.DeepEqual(source.StageContext.Artifacts, execution.StageContext.Artifacts) {
		return fmt.Errorf("%w: continuation differs from its saved source task or summary", errInvalidRunState)
	}
	return nil
}

func summaryEscalationConfiguration(
	execution runstore.StageExecution,
	action resolved.EscalateTransition,
	deadline time.Time,
) (stageExecutionConfiguration, error) {
	if action.Planner == nil || execution.CandidateResult == nil ||
		!execution.CandidateResult.Summarized || execution.CandidateResult.SummaryArtifact == nil {
		return stageExecutionConfiguration{}, fmt.Errorf("%w: summary escalation has no saved source result", errInvalidRunState)
	}
	context := execution.StageContext.Clone()
	context.Continuation = &stageresult.SummaryContinuation{
		SourceStageExecutionID: execution.StageExecutionID,
		Summary:                execution.CandidateResult.Summary,
		SummaryArtifact:        execution.CandidateResult.SummaryArtifact.Clone(),
		Artifacts:              contracts.CloneArtifactRefs(execution.CandidateResult.Artifacts),
		Deadline:               deadline,
	}
	return stageExecutionConfiguration{
		variant: runstore.StageExecutionConfigSummaryEscalation,
		ordinal: intPointer(1), effective: &action.ExecutionConfig.Effective,
		planner: action.Planner, context: &context,
	}, nil
}
