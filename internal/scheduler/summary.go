package scheduler

import (
	"context"
	"errors"
	"fmt"
	"reflect"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/planner/stageresult"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func (s *Scheduler) recoverCompletedSummary(ctx context.Context, run runstore.WorkflowRun, workflow executableWorkflow, execution runstore.StageExecution) (bool, error) {
	readContext, cancel := context.WithTimeout(ctx, s.options.OperationTimeout)
	defer cancel()
	started, err := s.options.PlannerSessions.Begin(readContext, execution.StageExecutionID)
	if errors.Is(err, planner.ErrInvocationInProgress) {
		return false, nil // Existing process-loss rules own unfinished calls.
	}
	if err != nil {
		if errors.Is(err, context.Canceled) || persistencepostgres.IsTransientFailure(err) {
			return false, err
		}
		return true, s.failInvalidRunState(ctx, run.RunID, err)
	}
	if started.Completion == nil || started.Completion.Result == nil || !started.Completion.Result.Summarized {
		return false, nil
	}
	if !s.now().Before(s.stageDeadline(execution, workflow.stage)) {
		return true, s.beginAbort(ctx, run, workflow, execution, nil, stageDeadlineFailure())
	}
	// Read the trusted completion before checking old live grants. A restart
	// may have lost them, but this Worker already finished: no adoption,
	// allocation preparation or source dispatch is needed to save its summary.
	return true, s.enterFinalizing(ctx, run, workflow, execution, nil, started.Completion.Result.Clone())
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
		source.ExecutionConfigVariant != runstore.StageExecutionConfigBase ||
		result == nil || !result.Summarized || result.SummaryArtifact == nil ||
		!result.SummaryArtifact.SameExact(continuation.SummaryArtifact) || result.Summary != continuation.Summary ||
		!reflect.DeepEqual(result.Artifacts, continuation.Artifacts) ||
		!reflect.DeepEqual(source.StageContext.Parameters, execution.StageContext.Parameters) ||
		!reflect.DeepEqual(source.StageContext.Artifacts, execution.StageContext.Artifacts) {
		return fmt.Errorf("%w: continuation differs from its saved source task or summary", errInvalidRunState)
	}
	if execution.PreviousExecutionID == nil {
		return fmt.Errorf("%w: summary continuation has no preceding attempt", errInvalidRunState)
	}
	parent := source
	if *execution.PreviousExecutionID != source.StageExecutionID {
		parent, err = s.store.GetStageExecution(readContext, *execution.PreviousExecutionID)
		if errors.Is(err, runstore.ErrNotFound) {
			return fmt.Errorf("%w: summary continuation parent is missing", errInvalidRunState)
		}
		if err != nil {
			return err
		}
		if parent.ExecutionConfigVariant != runstore.StageExecutionConfigSummaryEscalation ||
			parent.EscalationOrdinal == nil || *parent.EscalationOrdinal != 1 ||
			!reflect.DeepEqual(parent.StageContext, execution.StageContext) {
			return fmt.Errorf("%w: summary retry or resume changed its parent context", errInvalidRunState)
		}
	} else if execution.ResumeSourceExecutionID != nil {
		return fmt.Errorf("%w: an automatic summary root cannot be a manual resume", errInvalidRunState)
	}
	if parent.RunID != execution.RunID || parent.StageName != execution.StageName || parent.Attempt+1 != execution.Attempt ||
		(parent.State != runstore.StageFailed && parent.State != runstore.StageInterrupted) ||
		(execution.ResumeSourceExecutionID != nil && *execution.ResumeSourceExecutionID != parent.StageExecutionID) {
		return fmt.Errorf("%w: summary continuation has invalid immediate attempt lineage", errInvalidRunState)
	}
	return nil
}

func summaryEscalationConfiguration(
	execution runstore.StageExecution,
	action resolved.EscalateTransition,
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
	}
	return stageExecutionConfiguration{
		variant: runstore.StageExecutionConfigSummaryEscalation,
		ordinal: intPointer(1), effective: &action.ExecutionConfig.Effective,
		planner: action.Planner, context: &context,
	}, nil
}
