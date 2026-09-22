package controlplane

// Reservations and the allocations they become: candidate selection,
// the pinned commit, write grants and fences, release, and the losses
// reported when an allocation disappears without one.

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func (r *InMemoryRegistry) ReserveAll(request ReservationRequest) ([]Reservation, error) {
	return r.reserveAll(request, nil, false)
}

// PlacementCandidates returns detached snapshots of the slots that are
// eligible at this instant. The caller may perform database work after this
// method returns; reserveAll rechecks every selected edge under the Registry
// mutex, so the snapshot itself grants no capacity.
func (r *InMemoryRegistry) PlacementCandidates() []AgentSnapshot {
	monotonicNow := r.monotonicNow()
	r.mu.Lock()
	defer r.mu.Unlock()
	result := make([]AgentSnapshot, 0, len(r.agents))
	for _, entry := range r.agents {
		r.expireEntry(entry, monotonicNow)
		if isPlacementEligible(entry, monotonicNow) {
			result = append(result, snapshotAgent(entry))
		}
	}
	sort.Slice(result, func(i, j int) bool {
		left, right := r.agents[result[i].Registration.InstanceID], r.agents[result[j].Registration.InstanceID]
		if left.orderKey == right.orderKey {
			return result[i].Registration.InstanceID < result[j].Registration.InstanceID
		}
		return left.orderKey < right.orderKey
	})
	return result
}

// ReserveCandidateEdges installs a complete provisional batch using only the
// candidate-specific edges produced outside the Registry lock. The batch must
// be durably pinned and committed before it may be prepared.
func (r *InMemoryRegistry) ReserveCandidateEdges(
	request ReservationRequest,
	edges []CandidateEdge,
) ([]Reservation, error) {
	return r.reserveAll(request, edges, true)
}

func (r *InMemoryRegistry) reserveAll(
	request ReservationRequest,
	edges []CandidateEdge,
	provisional bool,
) ([]Reservation, error) {
	fingerprint, normalizedBindings, err := normalizeReservationRequest(request)
	if err != nil {
		return nil, err
	}
	runMetadataLabels, err := contracts.NormalizeRunMetadataLabels(request.RunMetadataLabels)
	if err != nil {
		return nil, fmt.Errorf("%w: Run metadata labels are invalid", ErrInvalidRequest)
	}
	if provisional {
		if err := validateCandidateEdges(normalizedBindings, edges); err != nil {
			return nil, err
		}
	} else if edges != nil {
		return nil, fmt.Errorf("%w: candidate edges require provisional reservation", ErrInvalidRequest)
	}
	allocationIDs := make([]string, len(normalizedBindings))
	seenIDs := make(map[string]struct{}, len(normalizedBindings))
	for index := range normalizedBindings {
		allocationID, err := r.nextID("allocation_")
		if err != nil {
			return nil, fmt.Errorf("generate allocation ID: %w", err)
		}
		if strings.TrimSpace(allocationID) == "" {
			return nil, fmt.Errorf("%w: allocation ID generator returned an empty value", ErrInvalidRequest)
		}
		if _, duplicate := seenIDs[allocationID]; duplicate {
			return nil, fmt.Errorf("%w: allocation ID generator returned a duplicate", ErrInvalidRequest)
		}
		seenIDs[allocationID] = struct{}{}
		allocationIDs[index] = allocationID
	}

	monotonicNow := r.monotonicNow()
	r.mu.Lock()
	defer r.mu.Unlock()
	if existing, ok := r.stageReservations[request.StageExecutionID]; ok {
		if existing.fingerprint != fingerprint {
			return nil, ErrReservationConflict
		}
		if existing.released {
			return nil, ErrReservationReleased
		}
		if !existing.committed {
			return nil, ErrReservationConflict
		}
		return r.existingReservations(existing)
	}
	for _, allocationID := range allocationIDs {
		if _, collision := r.allocations[allocationID]; collision {
			return nil, fmt.Errorf("%w: generated allocation ID already exists", ErrReservationConflict)
		}
	}

	available := make([]*agentEntry, 0, len(r.agents))
	for _, entry := range r.agents {
		if isPlacementEligible(entry, monotonicNow) {
			available = append(available, entry)
		}
	}
	sort.Slice(available, func(i, j int) bool {
		if available[i].orderKey == available[j].orderKey {
			return available[i].registration.InstanceID < available[j].registration.InstanceID
		}
		return available[i].orderKey < available[j].orderKey
	})
	selected, complete := completeCapabilityAssignmentWithEdges(available, normalizedBindings, edges)
	if !complete {
		return nil, ErrInsufficientCapacity
	}

	reservations := make([]Reservation, len(normalizedBindings))
	for index, binding := range normalizedBindings {
		entry := selected[index]
		allocationID := allocationIDs[index]
		grant := AllocationGrant{
			AllocationID: allocationID, RuntimeAgentID: entry.principal.RuntimeAgentID,
			RuntimeInstanceID: entry.registration.InstanceID,
			RunID:             request.RunID, StageExecutionID: request.StageExecutionID,
			LogicalAgentName: binding.LogicalAgentName, Namespace: binding.Namespace,
			ReadPolicy: ReadCurrentRun, WritePolicy: WriteInputsAndIntermediates,
		}
		reservation := Reservation{
			CompletionContract:     contracts.CloneWorkerCompletionContract(binding.CompletionContract),
			CompletionCapabilities: contracts.NormalizeAgentRegistration(entry.registration).Capabilities,
			Grant:                  grant, ControlURL: entry.registration.ControlURL, A2AURL: entry.registration.A2AURL,
			AgentTemplate:             cloneAgentTemplate(binding.AgentTemplate),
			WorkerSessionMode:         binding.WorkerSessionMode,
			ResolvedSkills:            contracts.CloneResolvedSkills(binding.ResolvedSkills),
			ExecutionConfig:           cloneAllocationExecutionConfig(binding.ExecutionConfig),
			Workspace:                 contracts.CloneAllocationWorkspaceSpec(binding.Workspace),
			RunMetadataLabels:         runMetadataLabels.Clone(),
			RuntimeAgentLabelRevision: entry.principal.LabelRevision,
			LeaseExpiresAt:            entry.confirmedLeaseExpiresAt,
		}
		entry.authoritativeAllocationID = cloneString(&allocationID)
		entry.allocationActivated = false
		r.allocations[allocationID] = storedReservation{
			reservation: reservation, phase: AllocationPreparing, writeGate: &sync.RWMutex{},
		}
		reservations[index] = cloneReservation(reservation)
	}
	r.stageReservations[request.StageExecutionID] = stageReservation{
		fingerprint: fingerprint, allocationIDs: append([]string(nil), allocationIDs...),
		committed: !provisional,
	}
	r.recordOperationsChangeLocked(OperationsAllocation, "")
	return reservations, nil
}

// GetStageReservations returns only a fully pinned batch. Provisional
// reservations are deliberately invisible to Scheduler replay.
func (r *InMemoryRegistry) GetStageReservations(stageExecutionID string) ([]Reservation, error) {
	if strings.TrimSpace(stageExecutionID) == "" {
		return nil, ErrAllocationNotFound
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	existing, ok := r.stageReservations[stageExecutionID]
	if !ok {
		return nil, ErrAllocationNotFound
	}
	if existing.released {
		return nil, ErrReservationReleased
	}
	if !existing.committed {
		return nil, ErrReservationConflict
	}
	return r.existingReservations(existing)
}

// CommitCandidateReservations attaches the detached safe resolution after its
// exact provenance is durable. All checks happen before mutating any stored
// reservation, making a caller retry/discard deterministic.
func (r *InMemoryRegistry) CommitCandidateReservations(
	stageExecutionID string,
	configurations map[string]PinnedReservationConfig,
) ([]Reservation, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	existing, ok := r.stageReservations[stageExecutionID]
	if !ok || existing.released || existing.committed || len(configurations) != len(existing.allocationIDs) {
		return nil, ErrReservationConflict
	}
	for _, allocationID := range existing.allocationIDs {
		stored, present := r.allocations[allocationID]
		configuration, configured := configurations[allocationID]
		if !present || !configured || configuration.RuntimeAgentLabelRevision == 0 ||
			configuration.RuntimeAgentLabelRevision != stored.reservation.RuntimeAgentLabelRevision ||
			configuration.Resolved.Validate() != nil ||
			validatePinnedPerformanceCollection(configuration) != nil {
			return nil, ErrReservationConflict
		}
	}
	for _, allocationID := range existing.allocationIDs {
		stored := r.allocations[allocationID]
		resolved := configurations[allocationID].Resolved.Clone()
		stored.reservation.ExecutionConfig = AllocationExecutionConfig{
			ModelPolicy: resolved.ModelPolicy.Ref,
			LLMGateway:  resolved.LLMGateway.Ref,
			Credential:  cloneCredentialRef(resolved.LLMCredential),
		}
		stored.reservation.ResolvedRuntimeConfig = &resolved
		stored.reservation.PerformanceCollectionPolicy = configurations[allocationID].PerformanceCollectionPolicy
		if configurations[allocationID].PerformanceMetrics != nil {
			request := *configurations[allocationID].PerformanceMetrics
			stored.reservation.PerformanceMetrics = &request
		}
		r.allocations[allocationID] = stored
	}
	existing.committed = true
	r.stageReservations[stageExecutionID] = existing
	return r.existingReservations(existing)
}

func validatePinnedPerformanceCollection(configuration PinnedReservationConfig) error {
	if err := configuration.PerformanceCollectionPolicy.ValidatePinned(); err != nil {
		return err
	}
	if configuration.PerformanceCollectionPolicy == contracts.PerformanceCollectionRequested {
		if configuration.PerformanceMetrics == nil {
			return ErrInvalidRequest
		}
		return configuration.PerformanceMetrics.Validate()
	}
	if configuration.PerformanceMetrics != nil {
		return ErrInvalidRequest
	}
	return nil
}

// DiscardCandidateReservations releases a batch that was never exposed to a
// Runtime Agent. Unlike normal Release it does not require a reconciliation
// heartbeat because the Runtime never observed these allocation IDs.
func (r *InMemoryRegistry) DiscardCandidateReservations(stageExecutionID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	existing, ok := r.stageReservations[stageExecutionID]
	if !ok {
		return nil
	}
	if existing.committed || existing.released {
		return ErrReservationConflict
	}
	for _, allocationID := range existing.allocationIDs {
		stored, present := r.allocations[allocationID]
		if !present {
			return ErrReservationConflict
		}
		entry, present := r.agents[stored.reservation.Grant.RuntimeInstanceID]
		if !present || entry.authoritativeAllocationID == nil ||
			*entry.authoritativeAllocationID != allocationID || entry.allocationActivated {
			return ErrReservationConflict
		}
	}
	for _, allocationID := range existing.allocationIDs {
		stored := r.allocations[allocationID]
		entry := r.agents[stored.reservation.Grant.RuntimeInstanceID]
		entry.authoritativeAllocationID = nil
		entry.allocationLost = false
		entry.allocationActivated = false
		delete(r.allocations, allocationID)
		r.recordOperationsChangeLocked(OperationsAllocation, allocationID)
		r.recordOperationsChangeLocked(OperationsRuntimeAgent, entry.registration.InstanceID)
	}
	delete(r.stageReservations, stageExecutionID)
	return nil
}

func validateCandidateEdges(bindings []BindingRequirement, edges []CandidateEdge) error {
	if edges == nil || len(edges) > len(bindings)*maximumOperationsItems {
		return fmt.Errorf("%w: candidate edge set is invalid", ErrInvalidRequest)
	}
	bindingsByName := make(map[string]struct{}, len(bindings))
	for _, binding := range bindings {
		bindingsByName[binding.LogicalAgentName] = struct{}{}
	}
	seen := make(map[candidateEdgeKey]struct{}, len(edges))
	for _, edge := range edges {
		if _, ok := bindingsByName[edge.LogicalAgentName]; !ok ||
			strings.TrimSpace(edge.RuntimeAgentInstanceID) == "" ||
			validateAuthenticatedPrincipal(AuthenticatedPrincipal{
				RuntimeAgentID: edge.RuntimeAgentID, Labels: []string{},
				LabelRevision: edge.RuntimeAgentLabelRevision,
			}) != nil {
			return fmt.Errorf("%w: candidate edge identity is invalid", ErrInvalidRequest)
		}
		key := candidateEdgeKey{edge.LogicalAgentName, edge.RuntimeAgentInstanceID}
		if _, duplicate := seen[key]; duplicate {
			return fmt.Errorf("%w: candidate edge is duplicated", ErrInvalidRequest)
		}
		seen[key] = struct{}{}
		previous := contracts.RuntimeAdapterRef("")
		for _, adapter := range edge.RequiredRuntimeAdapters {
			if adapter.Validate() != nil || adapter <= previous {
				return fmt.Errorf("%w: candidate adapter requirements are invalid", ErrInvalidRequest)
			}
			previous = adapter
		}
	}
	return nil
}

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
	return stored.reservation.Grant, nil
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
	grant := stored.reservation.Grant
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
	entry, ok := r.agents[stored.reservation.Grant.RuntimeInstanceID]
	if !ok || entry.authoritativeAllocationID == nil || *entry.authoritativeAllocationID != allocationID {
		return ErrAllocationNotFound
	}
	entry.authoritativeAllocationID = nil
	entry.allocationLost = false
	entry.allocationActivated = false
	// The last heartbeat may still report idle when a short allocation was
	// prepared and finalized entirely between heartbeat intervals. The private
	// release call only prepares Runtime cleanup; it does not prove that the
	// Runtime has applied the authoritative release response. Always require a
	// fresh post-release heartbeat before offering this slot again.
	entry.reconciliationRequired = true
	delete(r.allocations, allocationID)
	r.compactStageReservationLocked(stored.reservation.Grant.StageExecutionID)
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
	return nil
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

func (r *InMemoryRegistry) existingReservations(existing stageReservation) ([]Reservation, error) {
	result := make([]Reservation, 0, len(existing.allocationIDs))
	for _, allocationID := range existing.allocationIDs {
		stored, ok := r.allocations[allocationID]
		if !ok {
			return nil, ErrReservationReleased
		}
		result = append(result, cloneReservation(stored.reservation))
	}
	return result, nil
}

func (r *InMemoryRegistry) markAllocationLost(entry *agentEntry, reason AllocationLossReason) {
	if entry.authoritativeAllocationID == nil {
		return
	}
	allocationID := *entry.authoritativeAllocationID
	stored, ok := r.allocations[allocationID]
	if !ok || stored.loss != nil {
		return
	}
	loss := AllocationLoss{
		AllocationID: allocationID, RuntimeAgentID: entry.principal.RuntimeAgentID,
		RuntimeInstanceID: entry.registration.InstanceID,
		RunID:             stored.reservation.Grant.RunID,
		StageExecutionID:  stored.reservation.Grant.StageExecutionID,
		Reason:            reason,
	}
	stored.reservation.Grant.WriteFenced = true
	stored.reservation.Grant.Lost = true
	stored.loss = &loss
	r.allocations[allocationID] = stored
	entry.allocationLost = true
	entry.reconciliationRequired = true
	r.pendingLosses = append(r.pendingLosses, loss)
	r.recordOperationsChangeLocked(OperationsAllocation, allocationID)
}

func (r *InMemoryRegistry) detectObservedLoss(entry *agentEntry, reason AllocationLossReason) {
	if entry.authoritativeAllocationID == nil {
		return
	}
	allocationID := *entry.authoritativeAllocationID
	stored, ok := r.allocations[allocationID]
	if !ok || stored.loss != nil {
		return
	}
	matching := entry.registration.AllocationID != nil &&
		*entry.registration.AllocationID == allocationID
	if matching && entry.registration.ObservedState == contracts.AgentAllocated {
		entry.allocationActivated = true
		return
	}
	if !entry.allocationActivated && entry.registration.ObservedState == contracts.AgentIdle &&
		entry.registration.AllocationID == nil {
		return
	}
	valid := matching && entry.registration.ObservedState == contracts.AgentAllocated
	if matching && stored.reservation.Grant.WriteFenced &&
		(entry.registration.ObservedState == contracts.AgentDraining ||
			entry.registration.ObservedState == contracts.AgentFenced) {
		valid = true
	}
	if !valid {
		r.markAllocationLost(entry, reason)
	}
}

func (r *InMemoryRegistry) resetExpiredLease(entry *agentEntry) {
	entry.leaseExpired = false
	entry.confirmedLeaseDeadline = 0
	entry.confirmedLeaseExpiresAt = time.Time{}
	entry.lastConfirmedAckSeq = 0
	entry.issuedAcks = make(map[uint64]struct{})
	entry.heartbeatResponses = make(map[uint64]contracts.HeartbeatResponse)
	entry.heartbeatOrder = nil
}
