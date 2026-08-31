package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

var digestPattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

func (s *PostgresStore) RecordStageAllocation(ctx context.Context, allocation StageAllocation) error {
	for field, value := range map[string]string{
		"allocationID":           allocation.AllocationID,
		"stageExecutionID":       allocation.StageExecutionID,
		"logicalAgentName":       allocation.LogicalAgentName,
		"runtimeAgentInstanceID": allocation.RuntimeAgentInstanceID,
	} {
		if err := validateOpaque(field, value); err != nil {
			return err
		}
	}
	if err := validateNamespace(allocation.Namespace); err != nil {
		return err
	}
	if err := validateOpaque("AgentTemplate templateID", allocation.AgentTemplateRef.TemplateID); err != nil {
		return err
	}
	if err := validateOpaque("AgentTemplate version", allocation.AgentTemplateRef.Version); err != nil {
		return err
	}
	if !digestPattern.MatchString(allocation.AgentTemplateRef.Digest) {
		return invalidf("AgentTemplate digest is invalid")
	}
	if err := validateOpaque("WorkerRuntime runtimeID", allocation.WorkerRuntimeRef.RuntimeID); err != nil {
		return err
	}
	if err := validateOpaque("WorkerRuntime version", allocation.WorkerRuntimeRef.Version); err != nil {
		return err
	}
	templateRef, err := json.Marshal(allocation.AgentTemplateRef)
	if err != nil {
		return fmt.Errorf("record Stage allocation: encode AgentTemplate ref: %w", err)
	}
	runtimeRef, err := json.Marshal(allocation.WorkerRuntimeRef)
	if err != nil {
		return fmt.Errorf("record Stage allocation: encode WorkerRuntime ref: %w", err)
	}
	var insertedID string
	err = s.db.QueryRow(ctx, `
INSERT INTO stage_allocations (
    allocation_id, stage_execution_id, logical_agent_name, namespace,
    agent_template_ref, worker_runtime_ref, runtime_agent_instance_id
) VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7)
ON CONFLICT DO NOTHING
RETURNING allocation_id`,
		allocation.AllocationID, allocation.StageExecutionID, allocation.LogicalAgentName,
		allocation.Namespace, templateRef, runtimeRef, allocation.RuntimeAgentInstanceID,
	).Scan(&insertedID)
	if errors.Is(err, pgx.ErrNoRows) {
		existing, loadErr := s.ListStageAllocations(ctx, allocation.StageExecutionID)
		if loadErr != nil {
			return fmt.Errorf("inspect conflicting Stage allocation %q: %w", allocation.AllocationID, loadErr)
		}
		for _, current := range existing {
			if current.AllocationID != allocation.AllocationID &&
				current.LogicalAgentName != allocation.LogicalAgentName {
				continue
			}
			if sameStageAllocation(current, allocation) {
				return nil
			}
			return fmt.Errorf("record Stage allocation %q: %w", allocation.AllocationID, ErrConflict)
		}
		return fmt.Errorf("record Stage allocation %q: %w", allocation.AllocationID, ErrConflict)
	}
	if err != nil {
		sqlState := persistencepostgres.SQLState(err)
		if sqlState == "23503" {
			return fmt.Errorf("record Stage allocation for execution %q: %w", allocation.StageExecutionID, ErrNotFound)
		}
		return fmt.Errorf("record Stage allocation %q: %w", allocation.AllocationID, err)
	}
	if insertedID != allocation.AllocationID {
		return fmt.Errorf("record Stage allocation %q: %w", allocation.AllocationID, ErrConflict)
	}
	return nil
}

func sameStageAllocation(left, right StageAllocation) bool {
	return left.AllocationID == right.AllocationID &&
		left.StageExecutionID == right.StageExecutionID &&
		left.LogicalAgentName == right.LogicalAgentName &&
		left.Namespace == right.Namespace &&
		left.AgentTemplateRef == right.AgentTemplateRef &&
		left.WorkerRuntimeRef == right.WorkerRuntimeRef &&
		left.RuntimeAgentInstanceID == right.RuntimeAgentInstanceID
}

func (s *PostgresStore) ListStageAllocations(
	ctx context.Context,
	stageExecutionID string,
) ([]StageAllocation, error) {
	if err := validateOpaque("stageExecutionID", stageExecutionID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
SELECT allocation_id, stage_execution_id, logical_agent_name, namespace,
       agent_template_ref, worker_runtime_ref, runtime_agent_instance_id, created_at,
       release_attempted_at, release_completed_at
FROM stage_allocations
WHERE stage_execution_id = $1
ORDER BY logical_agent_name`, stageExecutionID)
	if err != nil {
		return nil, fmt.Errorf("list allocations for StageExecution %q: %w", stageExecutionID, err)
	}
	defer rows.Close()
	var result []StageAllocation
	for rows.Next() {
		var allocation StageAllocation
		var templateRef []byte
		var runtimeRef []byte
		if err := rows.Scan(
			&allocation.AllocationID, &allocation.StageExecutionID,
			&allocation.LogicalAgentName, &allocation.Namespace,
			&templateRef, &runtimeRef, &allocation.RuntimeAgentInstanceID, &allocation.CreatedAt,
			&allocation.ReleaseAttemptedAt, &allocation.ReleaseCompletedAt,
		); err != nil {
			return nil, fmt.Errorf("scan allocation for StageExecution %q: %w", stageExecutionID, err)
		}
		if err := json.Unmarshal(templateRef, &allocation.AgentTemplateRef); err != nil {
			return nil, fmt.Errorf("decode persisted AgentTemplate ref: %w", err)
		}
		if err := json.Unmarshal(runtimeRef, &allocation.WorkerRuntimeRef); err != nil {
			return nil, fmt.Errorf("decode persisted WorkerRuntime ref: %w", err)
		}
		result = append(result, allocation)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate allocations for StageExecution %q: %w", stageExecutionID, err)
	}
	if len(result) == 0 {
		var exists int
		err := s.db.QueryRow(ctx,
			`SELECT 1 FROM stage_executions WHERE stage_execution_id = $1`, stageExecutionID,
		).Scan(&exists)
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, fmt.Errorf("list allocations for StageExecution %q: %w", stageExecutionID, ErrNotFound)
		}
		if err != nil {
			return nil, fmt.Errorf("verify StageExecution %q: %w", stageExecutionID, err)
		}
	}
	return result, nil
}

func (s *PostgresStore) MarkStageAllocationReleaseAttempt(
	ctx context.Context,
	allocationID string,
) error {
	if err := validateOpaque("allocationID", allocationID); err != nil {
		return err
	}
	tag, err := s.db.Exec(ctx, `
UPDATE stage_allocations
SET release_attempted_at = clock_timestamp()
WHERE allocation_id = $1 AND release_completed_at IS NULL`, allocationID)
	if err != nil {
		return fmt.Errorf("mark Stage allocation %q release attempt: %w", allocationID, err)
	}
	if tag.RowsAffected() == 1 {
		return nil
	}
	return s.verifyStageAllocationExists(ctx, allocationID)
}

func (s *PostgresStore) MarkStageAllocationReleased(
	ctx context.Context,
	allocationID string,
) error {
	if err := validateOpaque("allocationID", allocationID); err != nil {
		return err
	}
	tag, err := s.db.Exec(ctx, `
UPDATE stage_allocations
SET release_attempted_at = statement_timestamp(),
    release_completed_at = statement_timestamp()
WHERE allocation_id = $1 AND release_completed_at IS NULL`, allocationID)
	if err != nil {
		return fmt.Errorf("mark Stage allocation %q released: %w", allocationID, err)
	}
	if tag.RowsAffected() == 1 {
		return nil
	}
	return s.verifyStageAllocationExists(ctx, allocationID)
}

func (s *PostgresStore) verifyStageAllocationExists(ctx context.Context, allocationID string) error {
	var exists bool
	if err := s.db.QueryRow(ctx, `
SELECT EXISTS(SELECT 1 FROM stage_allocations WHERE allocation_id = $1)`, allocationID,
	).Scan(&exists); err != nil {
		return fmt.Errorf("verify Stage allocation %q: %w", allocationID, err)
	}
	if !exists {
		return fmt.Errorf("Stage allocation %q: %w", allocationID, ErrNotFound)
	}
	return nil
}
