package scheduler

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// PostgresPersistence composes transaction-bound RunStore and ArtifactStore
// repositories. No method performs network I/O or reads artifact payloads.
type PostgresPersistence struct {
	pool *pgxpool.Pool
}

func NewPostgresPersistence(pool *pgxpool.Pool) (*PostgresPersistence, error) {
	if pool == nil {
		return nil, fmt.Errorf("PostgreSQL pool is required")
	}
	return &PostgresPersistence{pool: pool}, nil
}

func (p *PostgresPersistence) CreateStageWithContext(
	ctx context.Context,
	params runstore.CreateStageExecutionParams,
	pins []ContextPin,
) (runstore.StageExecution, error) {
	var created runstore.StageExecution
	err := persistencepostgres.InTx(ctx, p.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		store := runstore.NewPostgresStore(tx)
		if err := lockRunState(ctx, tx, params.RunID, runstore.RunRunning); err != nil {
			return err
		}
		var err error
		created, err = store.CreateStageExecution(ctx, params)
		if err != nil {
			return err
		}
		artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
		scope, err := artifacts.RunScope(params.RunID)
		if err != nil {
			return err
		}
		sorted := append([]ContextPin(nil), pins...)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })
		for _, pin := range sorted {
			if err := artifactService.PinExact(
				ctx,
				scope,
				pin.Ref,
				artifacts.PinStageContext,
				params.StageExecutionID+":"+pin.Name,
			); err != nil {
				return fmt.Errorf("pin StageContext artifact %q: %w", pin.Name, err)
			}
		}
		return nil
	})
	if err != nil {
		return runstore.StageExecution{}, err
	}
	return created, nil
}

func (p *PostgresPersistence) EnterFinalizingWithResult(
	ctx context.Context,
	params runstore.EnterFinalizingParams,
) error {
	return persistencepostgres.InTx(ctx, p.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		store := runstore.NewPostgresStore(tx)
		execution, err := store.GetStageExecution(ctx, params.StageExecutionID)
		if err != nil {
			return err
		}
		if err := lockRunState(ctx, tx, execution.RunID, runstore.RunRunning); err != nil {
			return err
		}
		artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
		scope, err := artifacts.RunScope(execution.RunID)
		if err != nil {
			return err
		}
		names := sortedArtifactNames(params.Candidate.Artifacts)
		for _, name := range names {
			if err := artifactService.PinExact(
				ctx,
				scope,
				params.Candidate.Artifacts[name],
				artifacts.PinStageResult,
				params.StageExecutionID+":"+name,
			); err != nil {
				return fmt.Errorf("pin StageResult artifact %q: %w", name, err)
			}
		}
		return store.EnterFinalizing(ctx, params)
	})
}

func (p *PostgresPersistence) AcceptResultAndFinishRun(
	ctx context.Context,
	acceptance ResultAcceptance,
) error {
	if err := validateAcceptance(acceptance); err != nil {
		return err
	}
	return persistencepostgres.InTx(ctx, p.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		store := runstore.NewPostgresStore(tx)
		if err := lockRunState(ctx, tx, acceptance.RunID, runstore.RunRunning); err != nil {
			return err
		}
		if err := lockStageForRun(ctx, tx, acceptance.StageExecutionID, acceptance.RunID); err != nil {
			return err
		}
		if err := store.CompleteStageResult(
			ctx,
			acceptance.StageExecutionID,
			contracts.APIVersion,
			acceptance.Result,
		); err != nil {
			return err
		}

		if acceptance.ExpectedRunOutcome == runstore.RunSucceeded {
			artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
			outputNames := make([]string, 0, len(acceptance.WorkflowOutputs))
			for outputName := range acceptance.WorkflowOutputs {
				outputNames = append(outputNames, outputName)
			}
			sort.Strings(outputNames)
			for _, outputName := range outputNames {
				resultName := acceptance.WorkflowOutputs[outputName]
				ref, present := acceptance.Result.Artifacts[resultName]
				if !present {
					if acceptance.OutputContracts[outputName].Required {
						return fmt.Errorf("required Workflow output %q has no Stage result", outputName)
					}
					continue
				}
				bound, err := artifactService.BindOutputExact(
					ctx,
					acceptance.RunID,
					outputName,
					ref,
					nil,
				)
				if err != nil {
					return fmt.Errorf("bind Workflow output %q: %w", outputName, err)
				}
				if !acceptsMediaType(acceptance.OutputContracts[outputName].MediaTypes, bound.MediaType) {
					return fmt.Errorf("Workflow output %q has incompatible media type", outputName)
				}
			}
			if err := artifactService.FreezeRunOutputs(ctx, acceptance.RunID); err != nil {
				return fmt.Errorf("freeze Workflow outputs: %w", err)
			}
		}

		reason := runstore.Reason{Code: "stage_failed"}
		if acceptance.ExpectedRunOutcome == runstore.RunSucceeded {
			reason.Code = "workflow_succeeded"
		}
		_, err := store.TransitionRun(
			ctx,
			acceptance.RunID,
			runstore.RunRunning,
			acceptance.ExpectedRunOutcome,
			reason,
		)
		return err
	})
}

func (p *PostgresPersistence) AcceptResultDuringCancellation(
	ctx context.Context,
	runID string,
	stageExecutionID string,
	result contracts.StageContentResult,
) error {
	if runID == "" || stageExecutionID == "" {
		return fmt.Errorf("Run and StageExecution IDs are required")
	}
	if err := result.Validate(); err != nil {
		return fmt.Errorf("invalid StageResult acceptance: %w", err)
	}
	return persistencepostgres.InTx(ctx, p.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := lockRunState(ctx, tx, runID, runstore.RunCancelling); err != nil {
			return err
		}
		if err := lockStageForRun(ctx, tx, stageExecutionID, runID); err != nil {
			return err
		}
		store := runstore.NewPostgresStore(tx)
		if err := store.CompleteStageResult(ctx, stageExecutionID, contracts.APIVersion, result); err != nil {
			return err
		}
		_, err := store.TransitionRun(
			ctx, runID, runstore.RunCancelling, runstore.RunCancelled,
			runstore.Reason{Code: runstore.CancellationUserRequested},
		)
		return err
	})
}

func (p *PostgresPersistence) CommitTerminationAndFinishRun(
	ctx context.Context,
	runID string,
	stageExecutionID string,
	expectedRunState runstore.WorkflowRunState,
	nextRunState runstore.WorkflowRunState,
	reason runstore.Reason,
) error {
	return persistencepostgres.InTx(ctx, p.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := lockRunState(ctx, tx, runID, expectedRunState); err != nil {
			return err
		}
		if err := lockStageForRun(ctx, tx, stageExecutionID, runID); err != nil {
			return err
		}
		store := runstore.NewPostgresStore(tx)
		if err := store.CompleteStageTermination(ctx, stageExecutionID); err != nil {
			return err
		}
		_, err := store.TransitionRun(
			ctx,
			runID,
			expectedRunState,
			nextRunState,
			reason,
		)
		return err
	})
}

func lockRunState(
	ctx context.Context,
	tx pgx.Tx,
	runID string,
	expected runstore.WorkflowRunState,
) error {
	var actual runstore.WorkflowRunState
	err := tx.QueryRow(ctx, `SELECT state FROM workflow_runs WHERE run_id = $1 FOR UPDATE`, runID).Scan(&actual)
	if errors.Is(err, pgx.ErrNoRows) {
		return fmt.Errorf("lock WorkflowRun %q: %w", runID, runstore.ErrNotFound)
	}
	if err != nil {
		return fmt.Errorf("lock WorkflowRun %q: %w", runID, err)
	}
	if actual != expected {
		return &runstore.StateConflictError{
			Resource: "WorkflowRun", ID: runID, Expected: string(expected),
		}
	}
	return nil
}

func lockStageForRun(ctx context.Context, tx pgx.Tx, stageExecutionID, runID string) error {
	var actualRunID string
	err := tx.QueryRow(
		ctx,
		`SELECT run_id FROM stage_executions WHERE stage_execution_id = $1 FOR UPDATE`,
		stageExecutionID,
	).Scan(&actualRunID)
	if errors.Is(err, pgx.ErrNoRows) {
		return fmt.Errorf("lock StageExecution %q: %w", stageExecutionID, runstore.ErrNotFound)
	}
	if err != nil {
		return fmt.Errorf("lock StageExecution %q: %w", stageExecutionID, err)
	}
	if actualRunID != runID {
		return &runstore.StateConflictError{
			Resource: "StageExecution Run", ID: stageExecutionID, Expected: runID,
		}
	}
	return nil
}

func validateAcceptance(value ResultAcceptance) error {
	if value.RunID == "" || value.StageExecutionID == "" {
		return fmt.Errorf("Run and StageExecution IDs are required")
	}
	if err := value.Result.Validate(); err != nil {
		return fmt.Errorf("invalid StageResult acceptance: %w", err)
	}
	switch value.Result.Outcome {
	case contracts.StageSucceeded:
		if value.ExpectedRunOutcome != runstore.RunSucceeded {
			return fmt.Errorf("successful StageResult must finish the MVP Run successfully")
		}
		for outputName, slot := range value.OutputContracts {
			resultName, mapped := value.WorkflowOutputs[outputName]
			if slot.Required && !mapped {
				return fmt.Errorf("required Workflow output %q is not mapped", outputName)
			}
			if mapped {
				if _, exists := value.Result.Artifacts[resultName]; slot.Required && !exists {
					return fmt.Errorf("required Workflow output %q has no Stage result", outputName)
				}
			}
		}
		for outputName := range value.WorkflowOutputs {
			if _, declared := value.OutputContracts[outputName]; !declared {
				return fmt.Errorf("undeclared Workflow output %q is mapped", outputName)
			}
		}
	case contracts.StageFailed:
		if value.ExpectedRunOutcome != runstore.RunFailed {
			return fmt.Errorf("failed StageResult must fail the MVP Run")
		}
	default:
		return fmt.Errorf("unknown StageResult outcome %q", value.Result.Outcome)
	}
	return nil
}

func sortedArtifactNames(values map[string]contracts.ArtifactRef) []string {
	result := make([]string, 0, len(values))
	for name := range values {
		result = append(result, name)
	}
	sort.Strings(result)
	return result
}

func acceptsMediaType(accepted []string, actual string) bool {
	for _, mediaType := range accepted {
		if mediaType == "*/*" || mediaType == actual {
			return true
		}
	}
	return false
}
