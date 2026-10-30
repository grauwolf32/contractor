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
	// Every caller has committed the Stage's terminal outcome (or its Queue
	// Pause exception, SCH-R68) after persisting its reports, so the Stage's
	// preparation secrets have no further reader (SCH-R178).
	s.preparedSecrets.forget(stageExecutionID)
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
	allocationIDs := make([]string, len(reservations))
	for index, reservation := range reservations {
		allocationIDs[index] = reservation.Grant.AllocationID
	}
	markContext, cancelMark := context.WithTimeout(context.Background(), s.options.OperationTimeout)
	markErr := s.store.MarkStageAllocationReleaseAttempt(markContext, allocationIDs...)
	cancelMark()
	return errors.Join(markErr, s.releaseAttempted(stageExecutionID, reservations))
}

// releaseAttempted releases reservations, whose release attempts are already
// recorded, in one batch. It then marks released, together, exactly those
// whose live grant is gone: a grant the batch could not release keeps its
// fence and its incomplete row for recovery (SCH-R69, SCH-R71, SCH-R182).
// Each marker batch has the store-operation budget; the release batch has its
// own cleanup deadline (SCH-R65).
func (s *Scheduler) releaseAttempted(
	stageExecutionID string,
	reservations []controlplane.Reservation,
) error {
	var failures []error
	// The batch has its own cleanup deadline, owned by the WorkerController
	// (runtimeLifecycle.cleanupTimeout, SCH-R65). The store-operation budget
	// bounds only the durable markers around it and must not cut the batch.
	releaseErr := s.workers.ReleaseAll(context.Background(), reservations)
	if releaseErr != nil {
		s.options.Logger.Warn(
			"terminal allocation release was incomplete",
			"stage_execution_id", stageExecutionID,
			"error", releaseErr,
		)
		failures = append(failures, releaseErr)
	}
	released := make([]string, 0, len(reservations))
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
		released = append(released, allocationID)
	}
	if len(released) != 0 {
		markContext, cancelMark := context.WithTimeout(context.Background(), s.options.OperationTimeout)
		failures = append(failures, s.store.MarkStageAllocationReleased(markContext, released...))
		cancelMark()
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

// terminalReleaseRecoveryConcurrency bounds the terminal release recovery
// Stage operations one Scheduler runs at once (SCH-R180, LIM-SCH-20). Each
// holds one Stage's release gate, short store operations and one release batch
// under its own cleanup deadline (SCH-R65), so up to three Stages whose
// Runtime Agents stop answering leave a slot for the rest of the batch.
const terminalReleaseRecoveryConcurrency = 4

// startTerminalReleaseRecovery lists one bounded batch of incomplete terminal
// releases, least recently attempted first (SCH-R71), and passes at most limit
// of their Stages to start, in that order. It takes each Stage's release gate
// first and skips a Stage whose release a lane or an earlier operation still
// owns; start owns the gate of the Stage it receives and must run
// recoverTerminalStage, which ends it.
func (s *Scheduler) startTerminalReleaseRecovery(
	ctx context.Context, limit int, start func(runstore.StageExecution),
) (int, error) {
	listContext, cancelList := context.WithTimeout(ctx, s.options.OperationTimeout)
	executions, err := s.store.ListTerminalStageExecutionsWithAllocations(listContext)
	cancelList()
	if err != nil {
		return 0, err
	}
	started := 0
	for _, execution := range executions {
		if started == limit {
			break
		}
		if !s.beginTerminalRelease(execution.StageExecutionID) {
			// A lane or another operation already owns this exact cleanup. Do
			// not report work: the monitor must wait rather than hot-loop over
			// unchanged rows.
			continue
		}
		start(execution)
		started++
	}
	return started, nil
}

// recoverTerminalStage is one terminal release recovery Stage operation. The
// caller took the Stage's release gate; the operation ends it.
func (s *Scheduler) recoverTerminalStage(ctx context.Context, execution runstore.StageExecution) (bool, error) {
	defer s.endTerminalRelease(execution.StageExecutionID)
	return s.recoverTerminalExecution(ctx, execution)
}

// recoverTerminalExecution retries the incomplete releases of one terminal
// Stage. It records the release attempts of all of them with one marker
// batch, marks released with another those whose grant is already gone, and
// releases the remaining live grants in one batch through releaseAttempted,
// so its durable markers grow with marker batches, not with allocations
// (SCH-R182, LIM-SCH-22). Each allocation still keeps its own outcome
// (SCH-R71).
func (s *Scheduler) recoverTerminalExecution(
	ctx context.Context,
	execution runstore.StageExecution,
) (bool, error) {
	// A terminal Stage persists no further report, including one that ended
	// without reaching its own release (SCH-R102, SCH-R178).
	s.preparedSecrets.forget(execution.StageExecutionID)
	allocationContext, cancelAllocations := context.WithTimeout(ctx, s.options.OperationTimeout)
	allocations, err := s.store.ListStageAllocations(
		allocationContext, execution.StageExecutionID,
	)
	cancelAllocations()
	if err != nil {
		return true, err
	}
	unreleased := make([]runstore.StageAllocation, 0, len(allocations))
	allocationIDs := make([]string, 0, len(allocations))
	for _, allocation := range allocations {
		if allocation.ReleaseCompletedAt == nil {
			unreleased = append(unreleased, allocation)
			allocationIDs = append(allocationIDs, allocation.AllocationID)
		}
	}
	if len(unreleased) == 0 {
		return false, nil
	}
	var failures []error
	markContext, cancelMark := context.WithTimeout(ctx, s.options.OperationTimeout)
	failures = append(failures, s.store.MarkStageAllocationReleaseAttempt(markContext, allocationIDs...))
	cancelMark()
	reservations := make([]controlplane.Reservation, 0, len(unreleased))
	gone := make([]string, 0, len(unreleased))
	for _, allocation := range unreleased {
		reservation, reservationErr := s.allocator.GetReservation(allocation.AllocationID)
		if errors.Is(reservationErr, controlplane.ErrAllocationNotFound) {
			gone = append(gone, allocation.AllocationID)
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
	if len(gone) != 0 {
		markContext, cancelMark = context.WithTimeout(ctx, s.options.OperationTimeout)
		failures = append(failures, s.store.MarkStageAllocationReleased(markContext, gone...))
		cancelMark()
	}
	if len(reservations) != 0 {
		failures = append(failures, s.releaseAttempted(execution.StageExecutionID, reservations))
	}
	return true, errors.Join(failures...)
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
		if err == nil && !owner.State.Terminal() {
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
