package scheduler

import (
	"context"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/operations"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

func (p *PostgresPersistence) StartStageOperation(ctx context.Context, runID, stageID, claimID string) (runstore.StageExecution, error) {
	expected, err := claimIDForRun(ctx, runID)
	if err != nil {
		return runstore.StageExecution{}, err
	}
	if claimID != expected {
		return runstore.StageExecution{}, errors.Join(ErrClaimLost, runstore.ErrConflict)
	}
	var execution runstore.StageExecution
	err = postgres.InTx(ctx, p.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		var err error
		execution, err = runstore.NewPostgresTxStore(tx, p.storage).StartStageOperation(ctx, runID, stageID, claimID)
		return err
	})
	return execution, err
}

func (s *Scheduler) executeServerOperation(ctx context.Context, run runstore.WorkflowRun, workflow executableWorkflow, execution runstore.StageExecution, deadline time.Time) error {
	fail := func(code, message string, retryable bool) error {
		return s.beginTermination(ctx, run, workflow, execution, nil, runstore.StageTermination{
			Outcome: runstore.TerminationInterrupted, Code: code, Message: message, Retryable: retryable,
		})
	}
	persistence, ok := s.persistence.(operationPersistence)
	if !ok || s.options.ServerOperations == nil {
		return fail("server_operation_unavailable", "Server operation execution is unavailable", false)
	}
	if execution.AdmittedAt == nil {
		admissionContext, cancelAdmission := context.WithTimeout(ctx, s.options.OperationTimeout)
		admitted, err := s.persistence.AdmitStage(admissionContext, run.RunID, execution.StageExecutionID)
		cancelAdmission()
		if err != nil {
			return err
		}
		execution = admitted
		deadline = s.stageDeadline(execution, workflow.stage)
	}
	if !s.now().Before(deadline) {
		return fail("stage_deadline_exceeded", "Stage execution deadline expired", true)
	}
	claimID, err := claimIDForRun(ctx, run.RunID)
	if err != nil {
		return err
	}
	startContext, cancelStart := context.WithTimeout(ctx, s.options.OperationTimeout)
	started, err := persistence.StartStageOperation(startContext, run.RunID, execution.StageExecutionID, claimID)
	cancelStart()
	if err != nil {
		return err
	}
	execution = started
	call := operations.Call{RunID: run.RunID, StageExecutionID: execution.StageExecutionID, ClaimID: claimID,
		Arguments: map[string]any{}, Inputs: map[string]contracts.ArtifactRef{}, Outputs: map[string]contracts.ArtifactRef{}}
	for name, binding := range workflow.stage.Operation.Arguments {
		switch binding.Source {
		case "literal":
			call.Arguments[name] = binding.Value
		case "parameter":
			value, err := binding.ResolveParameter(execution.StageContext.Parameters)
			if err != nil {
				return fail("invalid_request", "Server operation arguments are invalid", false)
			}
			call.Arguments[name] = value
		}
	}
	for name, input := range execution.StageContext.Artifacts {
		if input.Artifact != nil {
			call.Inputs[name] = input.Artifact.Clone()
		}
	}
	for name, slot := range workflow.stage.Result.Artifacts {
		call.Outputs[name] = contracts.ArtifactRef{Namespace: slot.From.Namespace, Name: slot.From.Name}
	}
	operationContext, cancel := context.WithDeadline(ctx, deadline)
	result, err := s.options.ServerOperations.Run(operationContext, workflow.stage.Operation.Ref, call)
	deadlineError := operationContext.Err()
	cancel()
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if errors.Is(err, runstore.ErrConflict) {
			return err // Cancellation or claim/Stage progression won the fence.
		}
		if errors.Is(err, context.DeadlineExceeded) {
			if errors.Is(deadlineError, context.DeadlineExceeded) {
				return fail("stage_deadline_exceeded", "Stage execution deadline expired", true)
			}
			return fail("server_operation_timeout", "Server operation deadline expired", true)
		}
		var failure *operations.Failure
		if errors.As(err, &failure) {
			return fail(failure.Code, failure.Message, failure.Retryable)
		}
		return fail("server_operation_failed", "Server operation failed", postgres.IsTransientFailure(err))
	}
	return s.enterFinalizing(ctx, run, workflow, execution, nil, result)
}
