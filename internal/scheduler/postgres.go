package scheduler

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/artifacts"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
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

func (p *PostgresPersistence) CommitResultProgression(
	ctx context.Context,
	value ResultProgression,
) error {
	if err := validateResultProgression(value); err != nil {
		return err
	}
	return persistencepostgres.InTx(ctx, p.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		store := runstore.NewPostgresStore(tx)
		if err := lockRunState(ctx, tx, value.RunID, runstore.RunRunning); err != nil {
			return err
		}
		if err := lockStageForRun(ctx, tx, value.StageExecutionID, value.RunID); err != nil {
			return err
		}
		if err := store.CompleteStageResult(
			ctx,
			value.StageExecutionID,
			contracts.APIVersion,
			value.Result,
		); err != nil {
			return err
		}

		if value.Result.Outcome == contracts.StageSucceeded {
			artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
			outputNames := make([]string, 0, len(value.WorkflowOutputs))
			for outputName := range value.WorkflowOutputs {
				outputNames = append(outputNames, outputName)
			}
			sort.Strings(outputNames)
			for _, outputName := range outputNames {
				resultName := value.WorkflowOutputs[outputName]
				ref, present := value.Result.Artifacts[resultName]
				if !present {
					if value.OutputContracts[outputName].Required {
						return fmt.Errorf("required Workflow output %q has no Stage result", outputName)
					}
					continue
				}
				bound, err := artifactService.BindOutputExact(
					ctx,
					value.RunID,
					outputName,
					ref,
					nil,
				)
				if err != nil {
					return fmt.Errorf("bind Workflow output %q: %w", outputName, err)
				}
				if !acceptsMediaType(value.OutputContracts[outputName].MediaTypes, bound.MediaType) {
					return fmt.Errorf("Workflow output %q has incompatible media type", outputName)
				}
			}
		}

		if err := commitNextStage(ctx, tx, store, value.Progression); err != nil {
			return err
		}
		if _, err := store.RecordStageTransitionDecision(ctx, value.Progression.Decision); err != nil {
			return err
		}
		return commitTerminalRun(ctx, tx, store, value.RunID, value.OutputContracts, value.Progression)
	})
}

func (p *PostgresPersistence) CommitTerminationProgression(
	ctx context.Context,
	value TerminationProgression,
) error {
	if err := validateTerminationProgression(value); err != nil {
		return err
	}
	return persistencepostgres.InTx(ctx, p.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := lockRunState(ctx, tx, value.RunID, runstore.RunRunning); err != nil {
			return err
		}
		if err := lockStageForRun(ctx, tx, value.StageExecutionID, value.RunID); err != nil {
			return err
		}
		store := runstore.NewPostgresStore(tx)
		if err := store.CompleteStageTermination(ctx, value.StageExecutionID); err != nil {
			return err
		}
		if err := commitNextStage(ctx, tx, store, value.Progression); err != nil {
			return err
		}
		if _, err := store.RecordStageTransitionDecision(ctx, value.Progression.Decision); err != nil {
			return err
		}
		return commitTerminalRun(ctx, tx, store, value.RunID, nil, value.Progression)
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
		artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
		if err := artifactService.FreezeRunOutputs(ctx, runID); err != nil {
			return fmt.Errorf("freeze Workflow outputs: %w", err)
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
		artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
		if err := artifactService.FreezeRunOutputs(ctx, runID); err != nil {
			return fmt.Errorf("freeze Workflow outputs: %w", err)
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

func validateResultProgression(value ResultProgression) error {
	if value.RunID == "" || value.StageExecutionID == "" {
		return fmt.Errorf("Run and StageExecution IDs are required")
	}
	if err := value.Result.Validate(); err != nil {
		return fmt.Errorf("invalid StageResult acceptance: %w", err)
	}
	if value.Result.Outcome == contracts.StageSucceeded {
		for outputName, slot := range value.OutputContracts {
			resultName, mapped := value.WorkflowOutputs[outputName]
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
	}
	return validateProgression(value.RunID, value.StageExecutionID, value.Progression)
}

func validateTerminationProgression(value TerminationProgression) error {
	if value.RunID == "" || value.StageExecutionID == "" {
		return fmt.Errorf("Run and StageExecution IDs are required")
	}
	return validateProgression(value.RunID, value.StageExecutionID, value.Progression)
}

func validateProgression(runID, sourceExecutionID string, value StageProgression) error {
	decision := value.Decision
	if decision.RunID != runID || decision.SourceExecutionID != sourceExecutionID {
		return fmt.Errorf("Stage progression decision identifies another source")
	}
	if err := decision.Validate(); err != nil {
		return fmt.Errorf("invalid Stage progression decision: %w", err)
	}
	switch decision.Action {
	case runstore.StageTransitionNext, runstore.StageTransitionRetry, runstore.StageTransitionEscalate:
		if value.NextStage == nil || value.TerminalRunState != "" ||
			decision.TargetStageName == nil || decision.TargetExecutionID == nil {
			return fmt.Errorf("%s Stage progression requires exactly one next execution", decision.Action)
		}
		params := value.NextStage.Params
		if params.RunID != runID || params.StageExecutionID != *decision.TargetExecutionID ||
			params.StageName != *decision.TargetStageName {
			return fmt.Errorf("Stage progression target differs from the durable decision")
		}
		if err := validateNextStagePins(*value.NextStage); err != nil {
			return err
		}
		if decision.Action == runstore.StageTransitionEscalate {
			if params.ExecutionConfigVariant != runstore.StageExecutionConfigFailedEscalation &&
				params.ExecutionConfigVariant != runstore.StageExecutionConfigInterruptedEscalation {
				return fmt.Errorf("escalate Stage progression requires an escalation configuration variant")
			}
			if decision.EscalationOrdinal == nil || params.EscalationOrdinal == nil ||
				*decision.EscalationOrdinal != *params.EscalationOrdinal || decision.EscalationExhausted {
				return fmt.Errorf("escalate Stage progression ordinal differs from its target")
			}
		} else if params.ExecutionConfigVariant != runstore.StageExecutionConfigBase ||
			params.EscalationOrdinal != nil {
			return fmt.Errorf("next/retry Stage progression must use the base execution configuration")
		}
	case runstore.StageTransitionSucceed:
		if value.NextStage != nil || value.TerminalRunState != runstore.RunSucceeded ||
			decision.TargetStageName != nil || decision.TargetExecutionID != nil {
			return fmt.Errorf("succeed Stage progression has an invalid shape")
		}
	case runstore.StageTransitionFail:
		if value.NextStage != nil || value.TerminalRunState != runstore.RunFailed ||
			decision.TargetStageName != nil || decision.TargetExecutionID != nil {
			return fmt.Errorf("fail Stage progression has an invalid shape")
		}
	default:
		return fmt.Errorf("unknown Stage progression action %q", decision.Action)
	}
	return nil
}

func validateNextStagePins(value NextStageCreation) error {
	pins := make(map[string]contracts.ArtifactRef, len(value.ContextPins))
	for _, pin := range value.ContextPins {
		if _, duplicate := pins[pin.Name]; duplicate {
			return fmt.Errorf("StageContext artifact %q has duplicate pins", pin.Name)
		}
		pins[pin.Name] = pin.Ref
	}
	for name, contextArtifact := range value.Params.StageContext.Artifacts {
		pin, present := pins[name]
		if contextArtifact.Artifact == nil {
			if present {
				return fmt.Errorf("absent StageContext artifact %q must not be pinned", name)
			}
			continue
		}
		if !present || !sameExactRef(pin, *contextArtifact.Artifact) {
			return fmt.Errorf("StageContext artifact %q pin differs from its exact snapshot", name)
		}
		delete(pins, name)
	}
	if len(pins) != 0 {
		return fmt.Errorf("StageContext contains an undeclared artifact pin")
	}
	return nil
}

func commitNextStage(
	ctx context.Context,
	tx pgx.Tx,
	store *runstore.PostgresStore,
	progression StageProgression,
) error {
	if progression.NextStage == nil {
		return nil
	}
	created, err := store.CreateStageExecution(ctx, progression.NextStage.Params)
	if err != nil {
		return err
	}
	if created.StageExecutionID != progression.NextStage.Params.StageExecutionID {
		return fmt.Errorf("created StageExecution identity differs from progression target")
	}
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
	scope, err := artifacts.RunScope(progression.NextStage.Params.RunID)
	if err != nil {
		return err
	}
	pins := append([]ContextPin(nil), progression.NextStage.ContextPins...)
	sort.Slice(pins, func(i, j int) bool { return pins[i].Name < pins[j].Name })
	for _, pin := range pins {
		if err := artifactService.PinExact(
			ctx,
			scope,
			pin.Ref,
			artifacts.PinStageContext,
			progression.NextStage.Params.StageExecutionID+":"+pin.Name,
		); err != nil {
			return fmt.Errorf("pin StageContext artifact %q: %w", pin.Name, err)
		}
	}
	return nil
}

func commitTerminalRun(
	ctx context.Context,
	tx pgx.Tx,
	store *runstore.PostgresStore,
	runID string,
	outputContracts map[string]workflowconfig.ArtifactSlot,
	progression StageProgression,
) error {
	if progression.TerminalRunState == "" {
		return nil
	}
	if progression.TerminalRunState == runstore.RunSucceeded {
		if err := verifyRequiredWorkflowOutputs(ctx, tx, runID, outputContracts); err != nil {
			return err
		}
	}
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
	if err := artifactService.FreezeRunOutputs(ctx, runID); err != nil {
		return fmt.Errorf("freeze Workflow outputs: %w", err)
	}
	_, err := store.TransitionRun(
		ctx,
		runID,
		runstore.RunRunning,
		progression.TerminalRunState,
		progression.RunReason,
	)
	return err
}

func verifyRequiredWorkflowOutputs(
	ctx context.Context,
	tx pgx.Tx,
	runID string,
	contractsByName map[string]workflowconfig.ArtifactSlot,
) error {
	names := make([]string, 0, len(contractsByName))
	for name := range contractsByName {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		slot := contractsByName[name]
		var mediaType string
		err := tx.QueryRow(ctx, `
SELECT version.media_type
FROM artifact_bindings AS binding
JOIN artifact_binding_revisions AS revision
  ON revision.scope_kind = binding.scope_kind
 AND revision.scope_id = binding.scope_id
 AND revision.namespace = binding.namespace
 AND revision.name = binding.name
 AND revision.revision = binding.current_revision
JOIN artifact_versions AS version ON version.version_id = revision.version_id
WHERE binding.scope_kind = 'run' AND binding.scope_id = $1
  AND binding.namespace = 'outputs' AND binding.name = $2`, runID, name).Scan(&mediaType)
		if errors.Is(err, pgx.ErrNoRows) && !slot.Required {
			continue
		}
		if errors.Is(err, pgx.ErrNoRows) {
			return fmt.Errorf("required Workflow output %q is not bound", name)
		}
		if err != nil {
			return fmt.Errorf("verify Workflow output %q: %w", name, err)
		}
		if !acceptsMediaType(slot.MediaTypes, mediaType) {
			return fmt.Errorf("Workflow output %q has incompatible media type", name)
		}
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
