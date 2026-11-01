package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strconv"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func (s *PostgresStore) RecordStageAllocation(ctx context.Context, allocation StageAllocation) error {
	if err := allocation.Validate(); err != nil {
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
		completion, err = json.Marshal(allocation.CompletionContract)
		if err != nil {
			return err
		}
	}
	var insertedID string
	err = s.db.QueryRow(ctx, recordStageAllocationSQL,
		allocation.AllocationID, allocation.StageExecutionID, allocation.LogicalAgentName,
		allocation.Namespace, templateRef, runtimeRef,
		allocation.RuntimeAgentID, allocation.RuntimeAgentInstanceID,
		strconv.FormatUint(allocation.RuntimeAgentLabelRevision, 10),
		allocation.RuntimeConfigurationSchemaVersion, runtimeConfiguration,
		allocation.PerformanceCollectionPolicy, completion,
		append([]string{}, allocation.RuntimePlacementLabels...),
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
		failure := persistencepostgres.Classify(err)
		if failure == persistencepostgres.ClassReferenceViolation {
			return fmt.Errorf("record Stage allocation for execution %q: %w", allocation.StageExecutionID, ErrNotFound)
		}
		return fmt.Errorf("record Stage allocation %q: %w", allocation.AllocationID, err)
	}
	if insertedID != allocation.AllocationID {
		return fmt.Errorf("record Stage allocation %q: %w", allocation.AllocationID, ErrConflict)
	}
	return nil
}

// Validate is the check RecordStageAllocation applies before it writes a row:
// it returns ErrInvalid for an allocation the store refuses. Test doubles of
// RecordStageAllocation apply it too, so they reject what the store rejects.
func (allocation StageAllocation) Validate() error {
	if err := contracts.ValidatePlacementLabels(allocation.RuntimePlacementLabels); err != nil {
		return invalidf("allocation placement labels are invalid: %v", err)
	}
	for field, value := range map[string]string{
		"allocationID":           allocation.AllocationID,
		"stageExecutionID":       allocation.StageExecutionID,
		"runtimeAgentInstanceID": allocation.RuntimeAgentInstanceID,
	} {
		if err := validateResourceID(field, value); err != nil {
			return err
		}
	}
	if err := validateName("logicalAgentName", allocation.LogicalAgentName); err != nil {
		return err
	}
	if err := validateNamespace(allocation.Namespace); err != nil {
		return err
	}
	if !contracts.ValidRuntimeAgentID(allocation.RuntimeAgentID) ||
		allocation.RuntimeAgentLabelRevision == 0 || allocation.RuntimeConfiguration == nil ||
		allocation.RuntimeConfigurationSchemaVersion != AllocationRuntimeConfigurationSchemaVersion {
		return invalidf("Runtime Agent principal and configuration are required")
	}
	if err := allocation.RuntimeConfiguration.Validate(); err != nil {
		return invalidf("allocation Runtime configuration is invalid: %v", err)
	}
	if (allocation.WorkerRuntimeRef == (contracts.WorkerRuntimeRef{RuntimeID: "tool", Version: "1"})) != (allocation.RuntimeConfiguration.ModelPolicy == (contracts.ModelPolicyRef{})) {
		return invalidf("allocation model policy does not match Worker runtime")
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
	if !contentdigest.Valid(allocation.AgentTemplateRef.Digest) {
		return invalidf("AgentTemplate digest is invalid")
	}
	if err := validateOpaque("WorkerRuntime runtimeID", allocation.WorkerRuntimeRef.RuntimeID); err != nil {
		return err
	}
	if err := validateOpaque("WorkerRuntime version", allocation.WorkerRuntimeRef.Version); err != nil {
		return err
	}
	if allocation.CompletionContract != nil {
		if err := allocation.CompletionContract.Validate(); err != nil ||
			allocation.CompletionContract.ResultArtifact.Namespace != allocation.Namespace {
			return invalidf("invalid allocation completion contract")
		}
	}
	return nil
}

func sameStageAllocation(left, right StageAllocation) bool {
	return slices.Equal(left.RuntimePlacementLabels, right.RuntimePlacementLabels) && reflect.DeepEqual(left.CompletionContract, right.CompletionContract) && left.AllocationID == right.AllocationID &&
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
	rows, err := s.db.Query(ctx, listStageAllocationsSQL, stageExecutionIDs)
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
			&allocation.RuntimePlacementLabels,
		); err != nil {
			return nil, fmt.Errorf("scan StageExecution allocation: %w", err)
		}
		if completion != nil {
			if err := strictjson.Decode(completion, &allocation.CompletionContract); err != nil {
				return nil, err
			}
			if allocation.CompletionContract == nil {
				return nil, fmt.Errorf("invalid persisted completion contract")
			}
		}
		if allocation.RuntimePlacementLabels == nil {
			return nil, fmt.Errorf("invalid persisted allocation placement labels")
		}
		if err := strictjson.Decode(templateRef, &allocation.AgentTemplateRef); err != nil {
			return nil, fmt.Errorf("decode persisted AgentTemplate ref: %w", err)
		}
		if err := strictjson.Decode(runtimeRef, &allocation.WorkerRuntimeRef); err != nil {
			return nil, fmt.Errorf("decode persisted WorkerRuntime ref: %w", err)
		}
		if runtimeAgentID == nil || runtimeAgentLabelRevision == nil || runtimeConfigurationVersion == nil || runtimeConfiguration == nil {
			return nil, fmt.Errorf("decode persisted allocation Runtime configuration: missing provenance")
		}
		revision, parseErr := strconv.ParseUint(*runtimeAgentLabelRevision, 10, 64)
		if parseErr != nil {
			return nil, fmt.Errorf("decode persisted Runtime Agent label revision")
		}
		var configuration AllocationRuntimeConfiguration
		if err := strictjson.Decode(runtimeConfiguration, &configuration); err != nil {
			return nil, fmt.Errorf("decode persisted allocation Runtime configuration")
		}
		allocation.RuntimeAgentID = *runtimeAgentID
		allocation.RuntimeAgentLabelRevision = revision
		allocation.RuntimeConfigurationSchemaVersion = *runtimeConfigurationVersion
		allocation.RuntimeConfiguration = &configuration
		if performanceCollectionPolicy == nil {
			return nil, fmt.Errorf("decode persisted allocation performance collection policy: missing policy")
		}
		allocation.PerformanceCollectionPolicy = reporting.PerformanceCollectionPolicy(*performanceCollectionPolicy)
		if err := allocation.Validate(); err != nil {
			return nil, fmt.Errorf("invalid persisted Stage allocation %q: %v", allocation.AllocationID, err)
		}
		result = append(result, allocation)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate StageExecution allocations: %w", err)
	}
	return result, nil
}

func (c AllocationRuntimeConfiguration) Validate() error {
	if c.ModelPolicy == (contracts.ModelPolicyRef{}) {
		if c.Provenance.LLMGatewayConfig != nil || c.Provenance.LLMCredential != nil || c.Origins.LLMGateway != nil || c.Origins.LLMCredential != nil {
			return fmt.Errorf("model-free allocation cannot retain model route provenance")
		}
	} else {
		if err := c.ModelPolicy.ValidateRef(); err != nil {
			return err
		}
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

// stageAllocationMarkBatch is the most allocations one release-marker
// statement names (SCH-R182, LIM-SCH-22): a terminal release marks the
// allocations of its Stage in statements of at most this many, never one per
// allocation.
const stageAllocationMarkBatch = 128

// MarkStageAllocationReleaseAttempt records a release attempt of every named
// allocation whose release has not completed. It marks them together, in
// statements of at most stageAllocationMarkBatch allocations, and reports a
// named allocation that does not exist as ErrNotFound for that allocation
// alone: the others are still marked (SCH-R71).
func (s *PostgresStore) MarkStageAllocationReleaseAttempt(
	ctx context.Context,
	allocationIDs ...string,
) error {
	return s.markStageAllocations(ctx, "release attempts", markStageAllocationReleaseAttemptsSQL, allocationIDs)
}

// MarkStageAllocationReleased records the completed release of every named
// allocation whose release has not completed, with the batching and
// per-allocation outcomes of MarkStageAllocationReleaseAttempt.
func (s *PostgresStore) MarkStageAllocationReleased(
	ctx context.Context,
	allocationIDs ...string,
) error {
	return s.markStageAllocations(ctx, "releases", markStageAllocationsReleasedSQL, allocationIDs)
}

func (s *PostgresStore) markStageAllocations(
	ctx context.Context, marker, statement string, allocationIDs []string,
) error {
	ids := make([]string, 0, len(allocationIDs))
	seen := make(map[string]struct{}, len(allocationIDs))
	for _, allocationID := range allocationIDs {
		if err := validateOpaque("allocationID", allocationID); err != nil {
			return err
		}
		if _, duplicate := seen[allocationID]; !duplicate {
			seen[allocationID] = struct{}{}
			ids = append(ids, allocationID)
		}
	}
	var failures []error
	for start := 0; start < len(ids); start += stageAllocationMarkBatch {
		chunk := ids[start:min(start+stageAllocationMarkBatch, len(ids))]
		rows, err := s.db.Query(ctx, statement, chunk)
		var marked []string
		if err == nil {
			marked, err = pgx.CollectRows(rows, pgx.RowTo[string])
		}
		if err != nil {
			failures = append(failures, fmt.Errorf("mark Stage allocation %s: %w", marker, err))
			continue
		}
		if len(marked) == len(chunk) {
			continue
		}
		// A named allocation that was not marked has completed its release
		// already or does not exist; only the second is an error.
		failures = append(failures, s.verifyStageAllocationsExist(ctx, chunk, marked))
	}
	return errors.Join(failures...)
}

// verifyStageAllocationsExist reports each allocation of requested that is
// not in marked and does not exist as ErrNotFound for that allocation.
func (s *PostgresStore) verifyStageAllocationsExist(ctx context.Context, requested, marked []string) error {
	unmarked := make([]string, 0, len(requested)-len(marked))
	for _, allocationID := range requested {
		if !slices.Contains(marked, allocationID) {
			unmarked = append(unmarked, allocationID)
		}
	}
	rows, err := s.db.Query(ctx, `SELECT allocation_id FROM stage_allocations WHERE allocation_id = ANY($1::text[])`, unmarked)
	var existing []string
	if err == nil {
		existing, err = pgx.CollectRows(rows, pgx.RowTo[string])
	}
	if err != nil {
		return fmt.Errorf("verify Stage allocations: %w", err)
	}
	var failures []error
	for _, allocationID := range unmarked {
		if !slices.Contains(existing, allocationID) {
			failures = append(failures, fmt.Errorf("Stage allocation %q: %w", allocationID, ErrNotFound))
		}
	}
	return errors.Join(failures...)
}
