package scheduler

import (
	"context"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// operationStore bounds each individual store call, including pool acquisition,
// independently of a lane, Planner or Runtime batch. It deliberately implements
// Store without embedding it: adding a store method requires its budget here.
type operationStore struct {
	base    Store
	timeout time.Duration
}

func (s operationStore) ClaimRunnableRun(ctx context.Context, claimID string, duration time.Duration, excludedRunIDs []string, laneClass resolved.LaneClass) (runstore.WorkflowRun, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.ClaimRunnableRun(ctx, claimID, duration, excludedRunIDs, laneClass)
}

func (s operationStore) RenewRunClaim(ctx context.Context, runID string, claimID string, duration time.Duration) (runstore.WorkflowRunState, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.RenewRunClaim(ctx, runID, claimID, duration)
}

func (s operationStore) ReleaseRunClaim(ctx context.Context, runID string, claimID string) error {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.ReleaseRunClaim(ctx, runID, claimID)
}

func (s operationStore) DeferRunClaim(ctx context.Context, runID string, claimID string) error {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.DeferRunClaim(ctx, runID, claimID)
}

func (s operationStore) GetRun(ctx context.Context, runID string) (runstore.WorkflowRun, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.GetRun(ctx, runID)
}

func (s operationStore) GetRunState(ctx context.Context, runID string) (runstore.WorkflowRunState, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.GetRunState(ctx, runID)
}

func (s operationStore) GetOwnerQueueControl(ctx context.Context, ownerID string) (runstore.OwnerQueueControl, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.GetOwnerQueueControl(ctx, ownerID)
}

func (s operationStore) GetActiveStageExecution(ctx context.Context, runID string) (runstore.StageExecution, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.GetActiveStageExecution(ctx, runID)
}

func (s operationStore) HasStageExecutions(ctx context.Context, runID string) (bool, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.HasStageExecutions(ctx, runID)
}

func (s operationStore) ListStageLineage(ctx context.Context, runID string, stageName string) ([]runstore.StageLineage, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.ListStageLineage(ctx, runID, stageName)
}

func (s operationStore) ListStageExecutionsWithUnreleasedAllocations(ctx context.Context, runID string) ([]runstore.StageExecution, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.ListStageExecutionsWithUnreleasedAllocations(ctx, runID)
}

func (s operationStore) HasSummaryEscalation(ctx context.Context, runID string, stageName string) (bool, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.HasSummaryEscalation(ctx, runID, stageName)
}

func (s operationStore) ListTerminalStageExecutionsWithAllocations(ctx context.Context, excludedStageIDs []string) ([]runstore.StageExecution, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.ListTerminalStageExecutionsWithAllocations(ctx, excludedStageIDs)
}

func (s operationStore) GetStageExecution(ctx context.Context, stageID string) (runstore.StageExecution, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.GetStageExecution(ctx, stageID)
}

func (s operationStore) ListStageAllocations(ctx context.Context, stageID string) ([]runstore.StageAllocation, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.ListStageAllocations(ctx, stageID)
}

func (s operationStore) ListStageAllocationsBatch(ctx context.Context, stageIDs []string) (map[string][]runstore.StageAllocation, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.ListStageAllocationsBatch(ctx, stageIDs)
}

func (s operationStore) MarkStageAllocationReleaseAttempt(ctx context.Context, allocationIDs ...string) error {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.MarkStageAllocationReleaseAttempt(ctx, allocationIDs...)
}

func (s operationStore) MarkStageAllocationReleased(ctx context.Context, allocationIDs ...string) error {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.MarkStageAllocationReleased(ctx, allocationIDs...)
}

func (s operationStore) RecordStageOutputPublicationFailure(ctx context.Context, runID string, stageID string) error {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	return s.base.RecordStageOutputPublicationFailure(ctx, runID, stageID)
}

// operationArtifacts applies the same store budget to metadata resolution and
// source-keyed summary publication. Runtime calls use their own budgets.
type operationArtifacts struct {
	base    ArtifactResolver
	timeout time.Duration
}

func (r operationArtifacts) Resolve(ctx context.Context, runID string, ref contracts.ArtifactRef) (ResolvedArtifact, error) {
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	return r.base.Resolve(ctx, runID, ref)
}
func (r operationArtifacts) SaveSummary(ctx context.Context, runID, namespace, source, summary string) (contracts.ArtifactRef, error) {
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	return r.base.SaveSummary(ctx, runID, namespace, source, summary)
}
