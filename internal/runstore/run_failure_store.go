package runstore

import (
	"context"
	"encoding/json"
	"fmt"
)

// FailRunWithActiveStagesParams ends a Run that the Scheduler cannot
// progress together with its active StageExecutions (SCH-R102).
type FailRunWithActiveStagesParams struct {
	RunID         string
	ExpectedState WorkflowRunState
	NextState     WorkflowRunState
	Reason        Reason
	// Termination ends every preparing or running Stage; each records the
	// phase it was terminated from, whatever Termination.Phase says.
	TerminationSchemaVersion string
	Termination              StageTermination
	AbortID                  string
}

// FailRunWithActiveStages moves Run RunID from ExpectedState to NextState,
// failed or, once cancellation owns the Run, cancelled, with Reason in the
// caller's transaction, together with every active StageExecution of the
// Run: preparing and running Stages become terminal with Termination and
// AbortID, skipping aborting; aborting Stages commit their stored
// StageTermination; and a finalizing Stage, which already won its result
// race, accepts its durable candidate without Workflow progression or output
// binding, as during cancellation. Every Stage is then terminal, so terminal
// release recovery releases its allocations. The Run row is locked before
// the Stage rows, the order of every commit point (RUN-R29); a Run no longer
// in ExpectedState is a StateConflictError and changes nothing.
func (s *PostgresTxStore) FailRunWithActiveStages(ctx context.Context, params FailRunWithActiveStagesParams) error {
	for field, value := range map[string]string{
		"runID":                    params.RunID,
		"terminationSchemaVersion": params.TerminationSchemaVersion,
		"abortID":                  params.AbortID,
	} {
		if err := validateOpaque(field, value); err != nil {
			return err
		}
	}
	// The Run ends failed, or cancelled once cancellation owns it.
	if params.NextState != RunFailed && params.NextState != RunCancelled {
		return invalidf("a Run with active Stages ends failed or cancelled, not %s", params.NextState)
	}
	if err := ValidateRunTransition(params.ExpectedState, params.NextState); err != nil {
		return err
	}
	if err := validateReason(params.Reason); err != nil {
		return err
	}
	// The statement replaces the phase with each Stage's own state.
	termination := params.Termination
	termination.Phase = TerminationPreparing
	if err := termination.Validate(); err != nil {
		return err
	}
	encoded, err := json.Marshal(termination)
	if err != nil {
		return fmt.Errorf("fail WorkflowRun %q: encode StageTermination: %w", params.RunID, err)
	}
	if _, err := s.LockRunState(ctx, params.RunID, params.ExpectedState); err != nil {
		return err
	}
	if _, err := s.tx.Exec(ctx, terminateActiveStagesSQL,
		params.RunID, encoded, params.TerminationSchemaVersion, params.AbortID,
	); err != nil {
		return fmt.Errorf("terminate active StageExecutions of WorkflowRun %q: %w", params.RunID, err)
	}
	if _, err := s.tx.Exec(ctx, commitAbortingStagesSQL, params.RunID); err != nil {
		return fmt.Errorf("complete aborting StageExecutions of WorkflowRun %q: %w", params.RunID, err)
	}
	if _, err := s.tx.Exec(ctx, acceptFinalizingCandidatesSQL, params.RunID); err != nil {
		return fmt.Errorf("accept finalizing StageExecutions of WorkflowRun %q: %w", params.RunID, err)
	}
	_, err = s.TransitionRun(ctx, params.RunID, params.ExpectedState, params.NextState, params.Reason)
	return err
}
