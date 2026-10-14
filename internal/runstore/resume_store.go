package runstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type ResumeRunResult struct {
	RunID                  string `json:"runId"`
	SourceStageExecutionID string `json:"sourceStageExecutionId"`
	StageExecutionID       string `json:"stageExecutionId"`
}

// ResumableStage is a capability hint; ResumeFailedRun rechecks it under locks.
func (s *PostgresStore) ResumableStage(ctx context.Context, ownerID, runID string) (*string, error) {
	var source string
	err := s.db.QueryRow(ctx, resumableStageSQL, runID, ownerID).Scan(&source)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read Run continuation capability: %w", err)
	}
	return &source, nil
}

// The expected source attempt is also the idempotency identity. A response-loss
// replay returns its original target, even after that target has failed again.
func (s *PostgresStore) ResumeFailedRun(ctx context.Context, ownerID, runID, sourceID, targetID string) (ResumeRunResult, error) {
	for name, value := range map[string]string{"ownerID": ownerID, "runID": runID, "sourceID": sourceID, "targetID": targetID} {
		if err := validateOpaque(name, value); err != nil {
			return ResumeRunResult{}, err
		}
	}
	var result ResumeRunResult
	operation := func(tx pgx.Tx) error {
		store := NewPostgresStore(tx)
		// Match Project deletion's project-before-Run lock ordering.
		var projectID *string
		err := tx.QueryRow(ctx, `SELECT project_id FROM workflow_runs WHERE run_id=$1 AND owner_id=$2`, runID, ownerID).Scan(&projectID)
		if errors.Is(err, pgx.ErrNoRows) {
			return ErrNotFound
		}
		if err != nil {
			return err
		}
		if projectID != nil {
			var lifecycle, kind string
			err = tx.QueryRow(ctx, `SELECT lifecycle_state,kind FROM projects WHERE project_id=$1 FOR SHARE`, *projectID).Scan(&lifecycle, &kind)
			if errors.Is(err, pgx.ErrNoRows) {
				return ErrNotFound
			}
			if err != nil {
				return err
			}
			if lifecycle != "active" || kind == "evaluation" {
				return ErrConflict
			}
		}
		var locked string
		err = tx.QueryRow(ctx, `SELECT run_id FROM workflow_runs WHERE run_id=$1 AND owner_id=$2 FOR UPDATE`, runID, ownerID).Scan(&locked)
		if errors.Is(err, pgx.ErrNoRows) {
			return ErrNotFound
		}
		if err != nil {
			return err
		}
		result = ResumeRunResult{RunID: runID, SourceStageExecutionID: sourceID}
		err = tx.QueryRow(ctx, `SELECT target_execution_id FROM run_stage_resumptions WHERE run_id=$1 AND source_execution_id=$2`, runID, sourceID).Scan(&result.StageExecutionID)
		if err == nil {
			return nil
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return err
		}
		source, err := store.ResumableStage(ctx, ownerID, runID)
		if err != nil {
			return err
		}
		if source == nil || *source != sourceID {
			return ErrConflict
		}
		previous, err := store.GetStageExecution(ctx, sourceID)
		if err != nil {
			return err
		}
		// A new immutable attempt uses the exact failed attempt's context and effective
		// configuration. No Planner session, allocation or terminal fact is reused.
		_, err = tx.Exec(ctx, `UPDATE workflow_runs SET state='pending',state_reason_code='user_resumed',
    state_reason_message='',finished_at=NULL,updated_at=clock_timestamp()
    WHERE run_id=$1`, runID)
		if err != nil {
			return err
		}
		next, err := store.CreateStageExecution(ctx, CreateStageExecutionParams{
			StageExecutionID: targetID, RunID: runID, StageName: previous.StageName, Attempt: previous.Attempt + 1,
			PreviousExecutionID: &sourceID, ResumeSourceExecutionID: &sourceID, ExecutionConfigVariant: previous.ExecutionConfigVariant,
			EscalationOrdinal: previous.EscalationOrdinal, StageSpecSchemaVersion: previous.StageSpecSchemaVersion,
			StageSpecSnapshot: previous.StageSpecSnapshot, StageContextSchemaVersion: previous.StageContextSchemaVersion,
			StageContext: previous.StageContext,
		})
		if err != nil {
			return err
		}
		service := artifacts.NewService(artifacts.NewPostgresRepository(tx))
		scope, err := artifacts.RunScope(runID)
		if err != nil {
			return err
		}
		// Use the same exact pin set and lock order as automatic progression.
		for _, pin := range previous.StageContext.Pins() {
			if err := service.RequireExact(ctx, scope, pin.Ref); err != nil {
				return err
			}
		}
		_, err = tx.Exec(ctx, `INSERT INTO run_stage_resumptions(source_execution_id,run_id,target_execution_id,requested_by)
    VALUES($1,$2,$3,$4)`, sourceID, runID, next.StageExecutionID, ownerID)
		if err != nil {
			return err
		}
		// The database permits thawing only in this receipt's transaction.
		err = artifacts.NewPostgresTxRepository(tx).ThawOutputs(ctx, scope)
		result.StageExecutionID = next.StageExecutionID
		return err
	}
	err := persistencepostgres.Within(ctx, s.db, pgx.TxOptions{}, operation)
	return result, err
}
