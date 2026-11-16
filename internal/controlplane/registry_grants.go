package controlplane

import (
	"strings"
)

func (r *InMemoryRegistry) GetGrant(allocationID string) (AllocationGrant, error) {
	if strings.TrimSpace(allocationID) == "" {
		return AllocationGrant{}, ErrAllocationNotFound
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	stored, ok := r.allocations[allocationID]
	if !ok {
		return AllocationGrant{}, ErrAllocationNotFound
	}
	return cloneAllocationGrant(stored.reservation.Grant), nil
}

func (r *InMemoryRegistry) GetReservation(allocationID string) (Reservation, error) {
	if strings.TrimSpace(allocationID) == "" {
		return Reservation{}, ErrAllocationNotFound
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	stored, ok := r.allocations[allocationID]
	if !ok {
		return Reservation{}, ErrAllocationNotFound
	}
	return cloneReservation(stored.reservation), nil
}

// PrepareReservation rechecks the owning Agent's confirmed lease at the last
// possible point before Runtime prepare. A Stage can remain durably pinned
// while admission is deferred, so the original reservation timestamp cannot
// be used as the prepare deadline on replay.
func (r *InMemoryRegistry) PrepareReservation(request Reservation) (Reservation, error) {
	allocationID := request.Grant.AllocationID
	r.mu.Lock()
	defer r.mu.Unlock()
	stored, ok := r.allocations[allocationID]
	if !ok || stored.reservation.Grant.StageExecutionID != request.Grant.StageExecutionID {
		return Reservation{}, ErrAllocationNotFound
	}
	entry, ok := r.agents[stored.reservation.Grant.RuntimeInstanceID]
	if !ok {
		return Reservation{}, ErrAllocationLost
	}
	monotonicNow := r.monotonicNow()
	r.expireEntry(entry, monotonicNow)
	stored = r.allocations[allocationID] // expireEntry may have marked it lost.
	if stored.loss != nil || stored.reservation.Grant.Lost || entry.allocationLost || entry.superseded ||
		entry.authoritativeAllocationID == nil || *entry.authoritativeAllocationID != allocationID ||
		entry.leaseExpired || entry.confirmedLeaseDeadline == 0 ||
		monotonicNow >= entry.confirmedLeaseDeadline {
		return Reservation{}, ErrAllocationLost
	}
	if stored.phase == AllocationPreparing && !entry.allocationActivated &&
		entry.confirmedLeaseExpiresAt.After(stored.reservation.LeaseExpiresAt) {
		stored.reservation.LeaseExpiresAt = entry.confirmedLeaseExpiresAt
		r.allocations[allocationID] = stored
		r.recordOperationsChangeLocked(OperationsAllocation, allocationID)
	}
	return cloneReservation(stored.reservation), nil
}

// WithWriteGrant keeps the allocation's write authorization stable for the
// complete mutation. SetWriteFence and Release take the exclusive side of the
// same gate, so once either returns no earlier private Artifact write can
// commit afterward.
func (r *InMemoryRegistry) WithWriteGrant(
	allocationID string,
	operation func(AllocationGrant) error,
) error {
	if strings.TrimSpace(allocationID) == "" || operation == nil {
		return ErrAllocationNotFound
	}
	r.mu.Lock()
	stored, ok := r.allocations[allocationID]
	if !ok || stored.writeGate == nil {
		r.mu.Unlock()
		return ErrAllocationNotFound
	}
	gate := stored.writeGate
	r.mu.Unlock()

	gate.RLock()
	defer gate.RUnlock()
	r.mu.Lock()
	stored, ok = r.allocations[allocationID]
	if !ok || stored.writeGate != gate {
		r.mu.Unlock()
		return ErrAllocationNotFound
	}
	grant := cloneAllocationGrant(stored.reservation.Grant)
	r.mu.Unlock()
	return operation(grant)
}

func (r *InMemoryRegistry) SetWriteFence(allocationID string) error {
	if strings.TrimSpace(allocationID) == "" {
		return ErrAllocationNotFound
	}
	r.mu.Lock()
	stored, ok := r.allocations[allocationID]
	if !ok || stored.writeGate == nil {
		r.mu.Unlock()
		return ErrAllocationNotFound
	}
	gate := stored.writeGate
	r.mu.Unlock()

	gate.Lock()
	defer gate.Unlock()
	r.mu.Lock()
	defer r.mu.Unlock()
	stored, ok = r.allocations[allocationID]
	if !ok || stored.writeGate != gate {
		return ErrAllocationNotFound
	}
	fenceChanged := !stored.reservation.Grant.WriteFenced
	if fenceChanged {
		stored.reservation.Grant.WriteFenced = true
		r.allocations[allocationID] = stored
		r.recordOperationsChangeLocked(OperationsAllocation, allocationID)
	}
	entry, ownsAllocation := r.agents[stored.reservation.Grant.RuntimeInstanceID]
	if ownsAllocation && entry.authoritativeAllocationID != nil &&
		*entry.authoritativeAllocationID == allocationID && !entry.reconciliationRequired {
		entry.reconciliationRequired = true
		r.recordOperationsChangeLocked(OperationsRuntimeAgent, entry.registration.InstanceID)
	}
	return nil
}

func (r *InMemoryRegistry) Release(allocationID string) error {
	_, err := r.releaseAllocation(allocationID, false)
	return err
}

// ReleaseLost removes a grant only after its owning Runtime can no longer
// hold a live Control Plane lease. It takes the same exclusive write gate as
// Release, so no earlier authorized private write can commit afterward.
func (r *InMemoryRegistry) ReleaseLost(allocationID string) (bool, error) {
	return r.releaseAllocation(allocationID, true)
}

func (r *InMemoryRegistry) releaseAllocation(allocationID string, lostOnly bool) (bool, error) {
	if strings.TrimSpace(allocationID) == "" {
		return false, ErrAllocationNotFound
	}
	r.mu.Lock()
	stored, ok := r.allocations[allocationID]
	if !ok || stored.writeGate == nil {
		r.mu.Unlock()
		return false, ErrAllocationNotFound
	}
	gate := stored.writeGate
	r.mu.Unlock()

	gate.Lock()
	defer gate.Unlock()
	r.mu.Lock()
	defer r.mu.Unlock()
	stored, ok = r.allocations[allocationID]
	if !ok || stored.writeGate != gate {
		return false, ErrAllocationNotFound
	}
	entry, ok := r.agents[stored.reservation.Grant.RuntimeInstanceID]
	if !ok || entry.authoritativeAllocationID == nil || *entry.authoritativeAllocationID != allocationID {
		return false, ErrAllocationNotFound
	}
	if lostOnly {
		r.expireEntry(entry, r.monotonicNow())
		stored = r.allocations[allocationID]
		if (!entry.leaseExpired && !entry.superseded) || stored.loss == nil ||
			!stored.reservation.Grant.WriteFenced {
			return false, nil
		}
	}
	// The last heartbeat may still report idle when a short allocation was
	// prepared and finalized entirely between heartbeat intervals. The private
	// release call only prepares Runtime cleanup; it does not prove that the
	// Runtime has applied the authoritative release response. Always require a
	// fresh post-release heartbeat before offering this slot again.
	entry.reconciliationRequired = true
	r.removeAllocationAuthorityLocked(allocationID, entry)
	r.compactStageReservationLocked(stored.reservation.Grant.StageExecutionID)
	return true, nil
}

// removeAllocationAuthorityLocked removes a validated grant and reconciles any
// replacement waiting on its owner. The caller owns stage replay semantics and
// whether the old owner needs a fresh heartbeat before becoming eligible.
func (r *InMemoryRegistry) removeAllocationAuthorityLocked(allocationID string, entry *agentEntry) {
	entry.authoritativeAllocationID = nil
	entry.allocationLost = false
	entry.allocationActivated = false
	delete(r.allocations, allocationID)
	for _, candidate := range r.agents {
		if candidate.blockedByInstanceID != nil && *candidate.blockedByInstanceID == entry.registration.InstanceID {
			candidate.blockedByInstanceID = nil
			candidate.reconciliationRequired = r.entryNeedsReconciliationLocked(candidate)
			r.recordOperationsChangeLocked(OperationsRuntimeAgent, candidate.registration.InstanceID)
		}
	}
	r.recordOperationsChangeLocked(OperationsAllocation, allocationID)
	if !r.retireInactiveAgentLocked(entry.registration.InstanceID) {
		r.recordOperationsChangeLocked(OperationsRuntimeAgent, entry.registration.InstanceID)
	}
}

func (r *InMemoryRegistry) compactStageReservationLocked(stageExecutionID string) {
	existing, ok := r.stageReservations[stageExecutionID]
	if !ok || existing.released {
		return
	}
	for _, allocationID := range existing.allocationIDs {
		if _, live := r.allocations[allocationID]; live {
			return
		}
	}
	existing.allocationIDs = nil
	existing.released = true
	r.stageReservations[stageExecutionID] = existing
	r.stageReservationTombstones = append(r.stageReservationTombstones, stageExecutionID)
	for len(r.stageReservationTombstones) > stageReservationTombstoneLimit {
		oldest := r.stageReservationTombstones[0]
		r.stageReservationTombstones = r.stageReservationTombstones[1:]
		if tombstone, present := r.stageReservations[oldest]; present && tombstone.released {
			delete(r.stageReservations, oldest)
		}
	}
}

// PollAllocationLosses expires monotonic deadlines and drains the one-shot
// loss edge queue. A late heartbeat can update diagnostics but cannot remove a
// loss already attached to an allocation.
func (r *InMemoryRegistry) PollAllocationLosses() []AllocationLoss {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := r.monotonicNow()
	r.expireAndRetireAgentsLocked(now)
	result := append([]AllocationLoss(nil), r.pendingLosses...)
	r.pendingLosses = nil
	return result
}
