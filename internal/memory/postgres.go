package memory

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// PostgresStore binds every Planner Memory operation to one still-running Run
// and StageExecution before accessing its ordinary RunScope ArtifactStore.
type PostgresStore struct {
	pool *pgxpool.Pool
}

func NewPostgresStore(pool *pgxpool.Pool) (*PostgresStore, error) {
	if pool == nil {
		return nil, fmt.Errorf("PostgreSQL pool is required for Planner Memory")
	}
	return &PostgresStore{pool: pool}, nil
}

func (s *PostgresStore) List(
	ctx context.Context,
	binding Binding,
) ([]artifacts.ArtifactRef, error) {
	if !validBinding(binding) {
		return nil, ErrAccessForbidden
	}
	return withActiveStage(ctx, s.pool, false, binding, func(tx pgx.Tx) ([]artifacts.ArtifactRef, error) {
		store, err := runArtifactStore(tx, binding.RunID)
		if err != nil {
			return nil, err
		}
		namespace := binding.Namespace
		return store.List(ctx, &namespace)
	})
}

func (s *PostgresStore) Read(
	ctx context.Context,
	binding Binding,
	ref artifacts.ArtifactRef,
) (artifacts.ReadResult, error) {
	if !validBinding(binding) || ref.Namespace != binding.Namespace || ref.Revision != nil {
		return artifacts.ReadResult{}, ErrAccessForbidden
	}
	if _, err := NameFromArtifact(ref.Name); err != nil {
		return artifacts.ReadResult{}, ErrAccessForbidden
	}
	return withActiveStage(ctx, s.pool, false, binding, func(tx pgx.Tx) (artifacts.ReadResult, error) {
		store, err := runArtifactStore(tx, binding.RunID)
		if err != nil {
			return artifacts.ReadResult{}, err
		}
		return store.Read(ctx, ref)
	})
}

func (s *PostgresStore) Write(
	ctx context.Context,
	binding Binding,
	target artifacts.ArtifactRef,
	payload artifacts.Payload,
	expectedRevision *string,
) (artifacts.WriteResult, error) {
	if !validBinding(binding) || target.Namespace != binding.Namespace || target.Revision != nil ||
		payload.MediaType != MediaType {
		return artifacts.WriteResult{}, ErrAccessForbidden
	}
	if _, err := NameFromArtifact(target.Name); err != nil {
		return artifacts.WriteResult{}, ErrAccessForbidden
	}
	return withActiveStage(ctx, s.pool, true, binding, func(tx pgx.Tx) (artifacts.WriteResult, error) {
		store, err := runArtifactStore(tx, binding.RunID)
		if err != nil {
			return artifacts.WriteResult{}, err
		}
		return store.Write(ctx, target, payload, expectedRevision)
	})
}

func runArtifactStore(tx pgx.Tx, runID string) (artifacts.ScopedStore, error) {
	service := artifacts.NewService(artifacts.NewPostgresRepository(tx))
	return service.Run(runID)
}

func withActiveStage[T any](
	ctx context.Context,
	pool *pgxpool.Pool,
	mutation bool,
	binding Binding,
	operation func(pgx.Tx) (T, error),
) (result T, err error) {
	if ctx == nil || ctx.Err() != nil {
		return result, ErrAccessForbidden
	}
	tx, err := pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		if ctx.Err() != nil {
			return result, ErrAccessForbidden
		}
		return result, fmt.Errorf("begin Planner Memory transaction: %w", err)
	}
	defer func() {
		rollbackContext, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		rollbackErr := tx.Rollback(rollbackContext)
		if rollbackErr != nil && !errors.Is(rollbackErr, pgx.ErrTxClosed) && err == nil {
			err = fmt.Errorf("rollback Planner Memory transaction: %w", rollbackErr)
		}
	}()

	if err := lockActiveStage(ctx, tx, binding); err != nil {
		return result, err
	}
	result, err = operation(tx)
	if err != nil {
		if ctx.Err() != nil {
			return result, ErrAccessForbidden
		}
		return result, err
	}
	if err := tx.Commit(ctx); err != nil {
		if mutation {
			// A connection/response failure at commit cannot prove whether the
			// immutable Artifact mutation committed. The logical wrapper owns the
			// sole exact replay and current-payload reconciliation.
			return result, ErrMutationOutcomeUnknown
		}
		if ctx.Err() != nil {
			return result, ErrAccessForbidden
		}
		return result, fmt.Errorf("commit Planner Memory transaction: %w", err)
	}
	return result, nil
}

func lockActiveStage(ctx context.Context, tx pgx.Tx, binding Binding) error {
	// Match Scheduler's lifecycle transaction lock order. Holding both rows
	// through the Artifact call is the durable Planner write fence.
	var runState runstore.WorkflowRunState
	err := tx.QueryRow(ctx, `
SELECT state
FROM workflow_runs
WHERE run_id = $1
FOR UPDATE`, binding.RunID).Scan(&runState)
	if errors.Is(err, pgx.ErrNoRows) || err == nil && runState != runstore.RunRunning {
		return ErrAccessForbidden
	}
	if err != nil {
		if ctx.Err() != nil {
			return ErrAccessForbidden
		}
		return fmt.Errorf("lock Planner Memory WorkflowRun: %w", err)
	}

	var stageRunID string
	var stageState runstore.StageExecutionState
	err = tx.QueryRow(ctx, `
SELECT run_id, state
FROM stage_executions
WHERE stage_execution_id = $1
FOR UPDATE`, binding.StageExecutionID).Scan(&stageRunID, &stageState)
	if errors.Is(err, pgx.ErrNoRows) ||
		err == nil && (stageRunID != binding.RunID || stageState != runstore.StageRunning) {
		return ErrAccessForbidden
	}
	if err != nil {
		if ctx.Err() != nil {
			return ErrAccessForbidden
		}
		return fmt.Errorf("lock Planner Memory StageExecution: %w", err)
	}
	return nil
}

var _ Store = (*PostgresStore)(nil)
