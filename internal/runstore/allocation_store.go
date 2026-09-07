package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"regexp"
	"strconv"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

var (
	digestPattern                = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
	runtimeAgentPrincipalPattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

func (s *PostgresStore) RecordStageAllocation(ctx context.Context, allocation StageAllocation) error {
	for field, value := range map[string]string{
		"allocationID":           allocation.AllocationID,
		"stageExecutionID":       allocation.StageExecutionID,
		"logicalAgentName":       allocation.LogicalAgentName,
		"runtimeAgentID":         allocation.RuntimeAgentID,
		"runtimeAgentInstanceID": allocation.RuntimeAgentInstanceID,
	} {
		if err := validateOpaque(field, value); err != nil {
			return err
		}
	}
	if err := validateNamespace(allocation.Namespace); err != nil {
		return err
	}
	if !runtimeAgentPrincipalPattern.MatchString(allocation.RuntimeAgentID) ||
		allocation.RuntimeAgentLabelRevision == 0 || allocation.RuntimeConfiguration == nil ||
		allocation.RuntimeConfigurationSchemaVersion != AllocationRuntimeConfigurationSchemaVersion {
		return invalidf("Runtime Agent principal and configuration are required")
	}
	if err := allocation.RuntimeConfiguration.Validate(); err != nil {
		return invalidf("allocation Runtime configuration is invalid: %v", err)
	}
	if err := allocation.PerformanceCollectionPolicy.ValidatePinned(); err != nil {
		return invalidf("allocation performance collection policy is invalid")
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
	runtimeConfiguration, err := json.Marshal(allocation.RuntimeConfiguration)
	if err != nil {
		return fmt.Errorf("record Stage allocation: encode Runtime configuration: %w", err)
	}
	var completion []byte
	if allocation.CompletionContract != nil {
		if err := allocation.CompletionContract.Validate(); err != nil || allocation.CompletionContract.ResultArtifact.Namespace != allocation.Namespace {
			return invalidf("invalid allocation completion contract")
		}
		completion, err = json.Marshal(allocation.CompletionContract)
		if err != nil {
			return err
		}
	}
	var insertedID string
	err = s.db.QueryRow(ctx, `
INSERT INTO stage_allocations (
    allocation_id, stage_execution_id, logical_agent_name, namespace,
    agent_template_ref, worker_runtime_ref,
    runtime_agent_id, runtime_agent_instance_id, runtime_agent_label_revision,
    runtime_configuration_schema_version, runtime_configuration,
    performance_collection_policy, completion_contract
) VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, $9, $10, $11::jsonb, $12, $13::jsonb)
ON CONFLICT DO NOTHING
RETURNING allocation_id`,
		allocation.AllocationID, allocation.StageExecutionID, allocation.LogicalAgentName,
		allocation.Namespace, templateRef, runtimeRef,
		allocation.RuntimeAgentID, allocation.RuntimeAgentInstanceID,
		strconv.FormatUint(allocation.RuntimeAgentLabelRevision, 10),
		allocation.RuntimeConfigurationSchemaVersion, runtimeConfiguration,
		allocation.PerformanceCollectionPolicy, completion,
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
	return reflect.DeepEqual(left.CompletionContract, right.CompletionContract) && left.AllocationID == right.AllocationID &&
		left.StageExecutionID == right.StageExecutionID &&
		left.LogicalAgentName == right.LogicalAgentName &&
		left.Namespace == right.Namespace &&
		left.AgentTemplateRef == right.AgentTemplateRef &&
		left.WorkerRuntimeRef == right.WorkerRuntimeRef &&
		left.RuntimeAgentID == right.RuntimeAgentID &&
		left.RuntimeAgentInstanceID == right.RuntimeAgentInstanceID &&
		left.RuntimeAgentLabelRevision == right.RuntimeAgentLabelRevision &&
		left.RuntimeConfigurationSchemaVersion == right.RuntimeConfigurationSchemaVersion &&
		left.PerformanceCollectionPolicy == right.PerformanceCollectionPolicy &&
		sameAllocationRuntimeConfiguration(left.RuntimeConfiguration, right.RuntimeConfiguration)
}

func (s *PostgresStore) ListStageAllocations(
	ctx context.Context,
	stageExecutionID string,
) ([]StageAllocation, error) {
	result, err := s.listStageAllocations(ctx, []string{stageExecutionID})
	if err != nil {
		return nil, err
	}
	if len(result) == 0 {
		var exists int
		err := s.db.QueryRow(ctx, `SELECT 1 FROM stage_executions WHERE stage_execution_id = $1`, stageExecutionID).Scan(&exists)
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, fmt.Errorf("list allocations for StageExecution %q: %w", stageExecutionID, ErrNotFound)
		}
		if err != nil {
			return nil, fmt.Errorf("verify StageExecution %q: %w", stageExecutionID, err)
		}
	}
	return result, nil
}

// ListStageAllocationsBatch returns existing allocations for already-authorized
// executions. Missing/empty executions have no map entry; no existence probes.
func (s *PostgresStore) ListStageAllocationsBatch(ctx context.Context, ids []string) (map[string][]StageAllocation, error) {
	rows, err := s.listStageAllocations(ctx, ids)
	if err != nil {
		return nil, err
	}
	result := make(map[string][]StageAllocation)
	for _, row := range rows {
		result[row.StageExecutionID] = append(result[row.StageExecutionID], row)
	}
	return result, nil
}

func (s *PostgresStore) listStageAllocations(ctx context.Context, stageExecutionIDs []string) ([]StageAllocation, error) {
	for _, id := range stageExecutionIDs {
		if err := validateOpaque("stageExecutionID", id); err != nil {
			return nil, err
		}
	}
	if len(stageExecutionIDs) == 0 {
		return nil, nil
	}
	rows, err := s.db.Query(ctx, `
SELECT allocation_id, stage_execution_id, logical_agent_name, namespace,
       agent_template_ref, worker_runtime_ref,
       runtime_agent_id, runtime_agent_instance_id, runtime_agent_label_revision::text,
       runtime_configuration_schema_version, runtime_configuration,
       performance_collection_policy, completion_contract, created_at,
       release_attempted_at, release_completed_at
FROM stage_allocations
WHERE stage_execution_id = ANY($1::text[])
ORDER BY stage_execution_id, logical_agent_name`, stageExecutionIDs)
	if err != nil {
		return nil, fmt.Errorf("list StageExecution allocations: %w", err)
	}
	defer rows.Close()
	var result []StageAllocation
	for rows.Next() {
		var allocation StageAllocation
		var templateRef []byte
		var runtimeRef []byte
		var runtimeAgentID, runtimeAgentLabelRevision, runtimeConfigurationVersion *string
		var performanceCollectionPolicy *string
		var runtimeConfiguration []byte
		var completion []byte
		if err := rows.Scan(
			&allocation.AllocationID, &allocation.StageExecutionID,
			&allocation.LogicalAgentName, &allocation.Namespace,
			&templateRef, &runtimeRef,
			&runtimeAgentID, &allocation.RuntimeAgentInstanceID, &runtimeAgentLabelRevision,
			&runtimeConfigurationVersion, &runtimeConfiguration, &performanceCollectionPolicy, &completion,
			&allocation.CreatedAt,
			&allocation.ReleaseAttemptedAt, &allocation.ReleaseCompletedAt,
		); err != nil {
			return nil, fmt.Errorf("scan StageExecution allocation: %w", err)
		}
		if completion != nil {
			if err := json.Unmarshal(completion, &allocation.CompletionContract); err != nil {
				return nil, err
			}
			if allocation.CompletionContract == nil || allocation.CompletionContract.Validate() != nil {
				return nil, invalidf("invalid persisted completion contract")
			}
		}
		if err := json.Unmarshal(templateRef, &allocation.AgentTemplateRef); err != nil {
			return nil, fmt.Errorf("decode persisted AgentTemplate ref: %w", err)
		}
		if err := json.Unmarshal(runtimeRef, &allocation.WorkerRuntimeRef); err != nil {
			return nil, fmt.Errorf("decode persisted WorkerRuntime ref: %w", err)
		}
		if runtimeAgentID != nil || runtimeAgentLabelRevision != nil || runtimeConfigurationVersion != nil || runtimeConfiguration != nil {
			if runtimeAgentID == nil || runtimeAgentLabelRevision == nil || runtimeConfigurationVersion == nil || runtimeConfiguration == nil {
				return nil, fmt.Errorf("decode persisted allocation Runtime configuration: incomplete legacy projection")
			}
			revision, parseErr := strconv.ParseUint(*runtimeAgentLabelRevision, 10, 64)
			if parseErr != nil {
				return nil, fmt.Errorf("decode persisted Runtime Agent label revision")
			}
			var configuration AllocationRuntimeConfiguration
			if err := json.Unmarshal(runtimeConfiguration, &configuration); err != nil || configuration.Validate() != nil {
				return nil, fmt.Errorf("decode persisted allocation Runtime configuration")
			}
			allocation.RuntimeAgentID = *runtimeAgentID
			allocation.RuntimeAgentLabelRevision = revision
			allocation.RuntimeConfigurationSchemaVersion = *runtimeConfigurationVersion
			allocation.RuntimeConfiguration = &configuration
		}
		if performanceCollectionPolicy != nil {
			allocation.PerformanceCollectionPolicy = contracts.PerformanceCollectionPolicy(*performanceCollectionPolicy)
			if allocation.PerformanceCollectionPolicy.ValidatePinned() != nil {
				return nil, fmt.Errorf("decode persisted allocation performance collection policy")
			}
		}
		result = append(result, allocation)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate StageExecution allocations: %w", err)
	}
	return result, nil
}

func (c AllocationRuntimeConfiguration) Validate() error {
	if err := c.ModelPolicy.ValidateRef(); err != nil {
		return err
	}
	if err := c.Origins.Validate(); err != nil {
		return err
	}
	return c.Provenance.Validate()
}

func sameAllocationRuntimeConfiguration(left, right *AllocationRuntimeConfiguration) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	leftJSON, leftErr := json.Marshal(left)
	rightJSON, rightErr := json.Marshal(right)
	return leftErr == nil && rightErr == nil && string(leftJSON) == string(rightJSON)
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
