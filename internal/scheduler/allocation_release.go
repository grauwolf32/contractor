package scheduler

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func (s *Scheduler) releaseTerminal(
	stageExecutionID string,
	reservations []controlplane.Reservation,
) error {
	if len(reservations) == 0 {
		return nil
	}
	if !s.beginTerminalRelease(stageExecutionID) {
		// Another lane or the recovery monitor owns this exact release. Durable
		// release markers make its failure retryable without duplicate Runtime
		// control calls from this process.
		return nil
	}
	defer s.endTerminalRelease(stageExecutionID)
	return s.releaseTerminalOwned(stageExecutionID, reservations)
}

// releaseTerminalOwned performs the Runtime calls after the caller has
// acquired the per-Stage release gate. Keeping recovery under the same gate
// prevents it from mistaking another lane's in-flight release for completed
// work and immediately rescanning the same durable rows.
func (s *Scheduler) releaseTerminalOwned(
	stageExecutionID string,
	reservations []controlplane.Reservation,
) error {
	var failures []error
	for _, reservation := range reservations {
		markContext, cancelMark := context.WithTimeout(context.Background(), s.options.OperationTimeout)
		err := s.store.MarkStageAllocationReleaseAttempt(
			markContext, reservation.Grant.AllocationID,
		)
		cancelMark()
		if err != nil {
			failures = append(failures, err)
		}
	}
	releaseContext, cancelRelease := context.WithTimeout(context.Background(), s.options.OperationTimeout)
	releaseErr := s.workers.ReleaseAll(releaseContext, reservations)
	cancelRelease()
	if releaseErr != nil {
		s.options.Logger.Warn(
			"terminal allocation release was incomplete",
			"stage_execution_id", stageExecutionID,
			"error", releaseErr,
		)
		failures = append(failures, releaseErr)
	}
	for _, reservation := range reservations {
		allocationID := reservation.Grant.AllocationID
		_, err := s.allocator.GetGrant(allocationID)
		if err == nil {
			continue
		}
		if !errors.Is(err, controlplane.ErrAllocationNotFound) {
			failures = append(failures, err)
			continue
		}
		markContext, cancelMark := context.WithTimeout(context.Background(), s.options.OperationTimeout)
		err = s.store.MarkStageAllocationReleased(markContext, allocationID)
		cancelMark()
		if err != nil {
			failures = append(failures, err)
		}
	}
	return errors.Join(failures...)
}

func (s *Scheduler) beginTerminalRelease(stageExecutionID string) bool {
	s.releaseMu.Lock()
	defer s.releaseMu.Unlock()
	if _, active := s.releasing[stageExecutionID]; active {
		return false
	}
	s.releasing[stageExecutionID] = struct{}{}
	return true
}

func (s *Scheduler) endTerminalRelease(stageExecutionID string) {
	s.releaseMu.Lock()
	delete(s.releasing, stageExecutionID)
	s.releaseMu.Unlock()
}

func (s *Scheduler) recoverTerminalRelease(ctx context.Context) (bool, error) {
	listContext, cancelList := context.WithTimeout(ctx, s.options.OperationTimeout)
	executions, err := s.store.ListTerminalStageExecutionsWithAllocations(listContext)
	cancelList()
	if err != nil {
		return false, err
	}
	for _, execution := range executions {
		if !s.beginTerminalRelease(execution.StageExecutionID) {
			// A Run lane already owns this exact cleanup. Do not report work:
			// the monitor must wait rather than hot-loop over unchanged rows.
			continue
		}
		worked, recoveryErr := func() (bool, error) {
			defer s.endTerminalRelease(execution.StageExecutionID)
			return s.recoverTerminalExecution(ctx, execution)
		}()
		if worked || recoveryErr != nil {
			return worked, recoveryErr
		}
	}
	return false, nil
}

func (s *Scheduler) recoverTerminalExecution(
	ctx context.Context,
	execution runstore.StageExecution,
) (bool, error) {
	allocationContext, cancelAllocations := context.WithTimeout(ctx, s.options.OperationTimeout)
	allocations, err := s.store.ListStageAllocations(
		allocationContext, execution.StageExecutionID,
	)
	cancelAllocations()
	if err != nil {
		return true, err
	}
	worked := false
	reservations := make([]controlplane.Reservation, 0, len(allocations))
	var failures []error
	for _, allocation := range allocations {
		if allocation.ReleaseCompletedAt != nil {
			continue
		}
		worked = true
		markContext, cancelMark := context.WithTimeout(ctx, s.options.OperationTimeout)
		markErr := s.store.MarkStageAllocationReleaseAttempt(markContext, allocation.AllocationID)
		cancelMark()
		if markErr != nil {
			failures = append(failures, markErr)
		}
		reservation, reservationErr := s.allocator.GetReservation(allocation.AllocationID)
		if errors.Is(reservationErr, controlplane.ErrAllocationNotFound) {
			markContext, cancelMark = context.WithTimeout(ctx, s.options.OperationTimeout)
			markErr = s.store.MarkStageAllocationReleased(markContext, allocation.AllocationID)
			cancelMark()
			if markErr != nil {
				failures = append(failures, markErr)
			}
			continue
		}
		if reservationErr != nil {
			failures = append(failures, reservationErr)
			continue
		}
		if err := verifyTerminalReleaseReservation(execution, allocation, reservation); err != nil {
			// Retrying cannot repair diverged provenance. Release the live grant
			// once no active Stage can own it; otherwise that Stage's own
			// release removes it and this row is then marked released.
			if releaseErr := s.claimDivergedTerminalGrant(ctx, execution, reservation); releaseErr != nil {
				failures = append(failures, errors.Join(err, releaseErr))
				continue
			}
		}
		reservations = append(reservations, reservation)
	}
	if len(reservations) != 0 {
		failures = append(failures, s.releaseTerminalOwned(execution.StageExecutionID, reservations))
	}
	return worked, errors.Join(failures...)
}

var errDivergedGrantOwned = errors.New("live allocation is still owned by an active StageExecution")

// claimDivergedTerminalGrant decides whether a live grant whose provenance
// differs from a terminal Stage's durable allocation may be released by that
// Stage's recovery. Allocation IDs are never reused, so the grant is safe to
// release when the Stage it names is this terminal Stage, another terminal
// Stage or unknown. It is write-fenced first, like every terminal release.
func (s *Scheduler) claimDivergedTerminalGrant(
	ctx context.Context,
	execution runstore.StageExecution,
	reservation controlplane.Reservation,
) error {
	grant := reservation.Grant
	if grant.StageExecutionID != execution.StageExecutionID {
		lookupContext, cancelLookup := context.WithTimeout(ctx, s.options.OperationTimeout)
		owner, err := s.store.GetStageExecution(lookupContext, grant.StageExecutionID)
		cancelLookup()
		if err != nil && !errors.Is(err, runstore.ErrNotFound) {
			return err
		}
		if err == nil && !terminalStageExecution(owner.State) {
			return errDivergedGrantOwned
		}
	}
	if err := s.allocator.SetWriteFence(grant.AllocationID); err != nil {
		return err
	}
	s.options.Logger.Error(
		"releasing fenced live allocation whose provenance differs from its terminal StageExecution",
		"stage_execution_id", execution.StageExecutionID,
		"allocation_id", grant.AllocationID,
		"live_stage_execution_id", grant.StageExecutionID,
	)
	return nil
}

func terminalStageExecution(state runstore.StageExecutionState) bool {
	switch state {
	case runstore.StageSucceeded, runstore.StageFailed, runstore.StageInterrupted, runstore.StageCancelled:
		return true
	default:
		return false
	}
}

func verifyTerminalReleaseReservation(
	execution runstore.StageExecution,
	allocation runstore.StageAllocation,
	reservation controlplane.Reservation,
) error {
	grant := reservation.Grant
	if grant.AllocationID != allocation.AllocationID ||
		grant.RunID != execution.RunID || grant.StageExecutionID != execution.StageExecutionID ||
		grant.LogicalAgentName != allocation.LogicalAgentName || grant.Namespace != allocation.Namespace ||
		grant.RuntimeInstanceID != allocation.RuntimeAgentInstanceID ||
		reservation.AgentTemplate.Ref != allocation.AgentTemplateRef ||
		reservation.AgentTemplate.Runtime != allocation.WorkerRuntimeRef {
		return fmt.Errorf("live allocation %q differs from durable terminal provenance", allocation.AllocationID)
	}
	return nil
}
