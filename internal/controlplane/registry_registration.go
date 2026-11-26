package controlplane

import (
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/helpers/clone"
)

func (r *InMemoryRegistry) RegistrationResponse() control.AgentRegistrationResponse {
	return control.AgentRegistrationResponse{
		APIVersion:               contracts.APIVersion,
		HeartbeatIntervalSeconds: int(r.heartbeatInterval / time.Second),
		ConfirmedLeaseSeconds:    int(r.confirmedLease / time.Second),
	}
}

func (r *InMemoryRegistry) RegisterAuthenticated(
	principal AuthenticatedPrincipal,
	registration control.AgentRegistration,
) (AgentSnapshot, error) {
	if err := registration.Validate(); err != nil {
		return AgentSnapshot{}, fmt.Errorf("%w: %v", ErrInvalidRequest, err)
	}
	if err := validateAuthenticatedPrincipal(principal); err != nil {
		return AgentSnapshot{}, err
	}
	normalized := normalizeRegistration(registration)
	identity, err := control.AgentRegistrationFingerprint(normalized)
	if err != nil {
		return AgentSnapshot{}, fmt.Errorf("encode registration identity: %w", err)
	}
	toolCapabilities := indexToolCapabilities(normalized.SupportedToolsets)
	orderKey := r.agentOrderKey(normalized.Clone())
	now := r.now()

	r.mu.Lock()
	defer r.mu.Unlock()
	if _, deleting := r.principalDeletions[principal.RuntimeAgentID]; deleting {
		return AgentSnapshot{}, ErrRegistrationConflict
	}
	monotonicNow := r.monotonicNow()
	r.expireAndRetireAgentsLocked(monotonicNow)
	if existing, ok := r.agents[normalized.InstanceID]; ok {
		if existing.identity != identity || existing.principal.RuntimeAgentID != principal.RuntimeAgentID {
			return AgentSnapshot{}, ErrRegistrationConflict
		}
		if principal.LabelRevision >= existing.principal.LabelRevision {
			existing.principal = clonePrincipal(principal)
		}
		existing.principalMissing = false
		existing.registration.ObservedState = normalized.ObservedState
		existing.registration.Allocations = append([]control.ObservedAllocation{}, normalized.Allocations...)
		existing.lastSeenAt = now
		if len(existing.allocationIDs) == 0 {
			// A Runtime can self-fence before our confirmed lease expires.
			// Re-registration must start a fresh claim so its re-armed local
			// watchdog cannot outlive our own deadline.
			r.resetExpiredLease(existing)
			existing.principalClaimDeadline = monotonicNow + r.confirmedLease
		}
		r.detectObservedLoss(existing, LossRuntimeMismatch)
		existing.reconciliationRequired = r.entryNeedsReconciliationLocked(existing)
		r.recordOperationsChangeLocked(OperationsRuntimeAgent, normalized.InstanceID)
		return snapshotAgent(existing), nil
	}
	for _, existing := range r.agents {
		if existing.principal.RuntimeAgentID != principal.RuntimeAgentID {
			continue
		}
		r.expireEntry(existing, monotonicNow)
		if principalInstanceIsLive(existing, monotonicNow) {
			return AgentSnapshot{}, ErrRegistrationConflict
		}
	}
	if len(r.agents) >= maximumOperationsItems {
		return AgentSnapshot{}, fmt.Errorf("%w: Runtime Agent observation capacity is exhausted", ErrInvalidRequest)
	}
	entry := &agentEntry{
		toolCapabilities: toolCapabilities,
		principal:        clonePrincipal(principal), registration: normalized,
		identity: identity, orderKey: orderKey, lastSeenAt: now,
		allocationIDs: make(map[string]struct{}), releaseAcks: make(map[string]uint64),
		principalClaimDeadline: monotonicNow + r.confirmedLease,
		issuedAcks:             make(map[uint64]struct{}), heartbeatResponses: make(map[uint64]control.HeartbeatResponse),
	}
	for instanceID, existing := range r.agents {
		// Only a restart under the same certificate principal replaces an
		// earlier process on its endpoint. Control Plane binds each endpoint to
		// its registrant's SPKI, so another principal's overlapping endpoint
		// cannot reach that process and must not revoke its allocation.
		if existing.superseded || existing.principal.RuntimeAgentID != principal.RuntimeAgentID ||
			!sameRuntimeEndpoint(existing.registration, normalized) {
			continue
		}
		existing.superseded = true
		existing.reconciliationRequired = true
		if len(existing.allocationIDs) != 0 {
			entry.blockedByInstanceID = clone.Pointer(&instanceID)
			r.recordOperationsChangeLocked(OperationsRuntimeAgent, instanceID)
		}
	}
	entry.reconciliationRequired = r.entryNeedsReconciliationLocked(entry)
	r.agents[normalized.InstanceID] = entry
	r.recordOperationsChangeLocked(OperationsRuntimeAgent, normalized.InstanceID)
	r.retireInactiveAgentsLocked()
	return snapshotAgent(entry), nil
}

func (r *InMemoryRegistry) HeartbeatAuthenticated(
	runtimeAgentID string,
	heartbeat control.AgentHeartbeat,
) (control.HeartbeatResponse, error) {
	if err := heartbeat.Validate(); err != nil {
		return control.HeartbeatResponse{AllocationActions: []control.AllocationAction{}}, fmt.Errorf("%w: %v", ErrInvalidRequest, err)
	}
	now := r.now()
	r.mu.Lock()
	defer r.mu.Unlock()
	entry, ok := r.agents[heartbeat.InstanceID]
	if !ok {
		return control.HeartbeatResponse{
			APIVersion: contracts.APIVersion, AckSeq: heartbeat.HeartbeatSeq, Action: control.ActionReregister,
			AllocationActions: []control.AllocationAction{},
		}, nil
	}
	if entry.principal.RuntimeAgentID != runtimeAgentID {
		return control.HeartbeatResponse{AllocationActions: []control.AllocationAction{}}, ErrRegistrationConflict
	}
	if len(heartbeat.Allocations) > entry.registration.AllocationSlots {
		return control.HeartbeatResponse{AllocationActions: []control.AllocationAction{}}, fmt.Errorf("%w: observations exceed Runtime capacity", ErrInvalidRequest)
	}
	r.expireEntry(entry, r.monotonicNow())
	if heartbeat.HeartbeatSeq <= entry.lastHeartbeatSeq {
		if response, exists := entry.heartbeatResponses[heartbeat.HeartbeatSeq]; exists {
			entry.lastSeenAt = now
			entry.lastAcceptedHeartbeatAt = now
			r.recordOperationsChangeLocked(OperationsRuntimeAgent, heartbeat.InstanceID)
			return cloneHeartbeatResponse(response), nil
		}
		return control.HeartbeatResponse{AllocationActions: []control.AllocationAction{}}, ErrHeartbeatOutOfOrder
	}
	if !entry.leaseExpired && heartbeat.EchoedAckSeq > entry.lastConfirmedAckSeq {
		if _, issued := entry.issuedAcks[heartbeat.EchoedAckSeq]; issued {
			entry.lastConfirmedAckSeq = heartbeat.EchoedAckSeq
			entry.confirmedLeaseExpiresAt = now.Add(r.confirmedLease)
			entry.confirmedLeaseDeadline = r.monotonicNow() + r.confirmedLease
			entry.principalClaimDeadline = 0
		}
	}
	entry.lastSeenAt = now
	entry.lastAcceptedHeartbeatAt = now
	entry.lastHeartbeatSeq = heartbeat.HeartbeatSeq
	entry.lastIssuedAckSeq = heartbeat.HeartbeatSeq
	entry.registration.ObservedState = heartbeat.ObservedState
	entry.registration.Allocations = append([]control.ObservedAllocation{}, heartbeat.Allocations...)
	_, echoIssued := entry.issuedAcks[heartbeat.EchoedAckSeq]
	for id, sequence := range entry.releaseAcks {
		if _, present := entry.registration.ObservedAllocation(id); !present && echoIssued && sequence != 0 && heartbeat.EchoedAckSeq >= sequence {
			delete(entry.releaseAcks, id)
		}
	}
	r.detectObservedLoss(entry, LossRuntimeMismatch)
	response := r.heartbeatAction(entry, heartbeat.HeartbeatSeq)
	entry.reconciliationRequired = r.entryNeedsReconciliationLocked(entry)
	r.recordHeartbeat(entry, heartbeat.HeartbeatSeq, response)
	r.recordOperationsChangeLocked(OperationsRuntimeAgent, heartbeat.InstanceID)
	return cloneHeartbeatResponse(response), nil
}

func (r *InMemoryRegistry) recordHeartbeat(entry *agentEntry, sequence uint64, response control.HeartbeatResponse) {
	entry.issuedAcks[sequence] = struct{}{}
	entry.heartbeatResponses[sequence] = cloneHeartbeatResponse(response)
	entry.heartbeatOrder = append(entry.heartbeatOrder, sequence)
	if len(entry.heartbeatOrder) <= heartbeatHistoryLimit {
		return
	}
	oldest := entry.heartbeatOrder[0]
	entry.heartbeatOrder = entry.heartbeatOrder[1:]
	delete(entry.issuedAcks, oldest)
	delete(entry.heartbeatResponses, oldest)
}

func (r *InMemoryRegistry) heartbeatAction(entry *agentEntry, sequence uint64) control.HeartbeatResponse {
	response := control.HeartbeatResponse{APIVersion: contracts.APIVersion, AckSeq: sequence,
		Action: control.ActionContinue, AllocationActions: []control.AllocationAction{}}
	if entry.leaseExpired && len(entry.allocationIDs) == 0 && len(entry.releaseAcks) == 0 && len(entry.registration.Allocations) == 0 {
		response.Action = control.ActionReregister
		return response
	}
	actions := make(map[string]control.ReconciliationAction, 2*control.MaxAllocationSlots)
	for id := range entry.allocationIDs {
		stored := r.allocations[id]
		state, present := entry.registration.ObservedAllocation(id)
		if entry.leaseExpired || entry.superseded || stored.loss != nil || stored.reservation.Grant.WriteFenced ||
			entry.registration.ObservedState == control.AgentFenced ||
			(stored.activated && (!present || state != control.AgentAllocated)) {
			actions[id] = control.ActionDrain
		}
	}
	for _, item := range entry.registration.Allocations {
		if entry.owns(item.AllocationID) {
			continue
		}
		action := control.ActionDrain
		if item.ObservedState == control.AgentFenced {
			action = control.ActionRelease
		}
		actions[item.AllocationID] = action
	}
	for id, ack := range entry.releaseAcks {
		actions[id] = control.ActionRelease
		if ack == 0 {
			entry.releaseAcks[id] = sequence
		}
	}
	for id, action := range actions {
		response.AllocationActions = append(response.AllocationActions, control.AllocationAction{AllocationID: id, Action: action})
	}
	slices.SortFunc(response.AllocationActions, func(left, right control.AllocationAction) int {
		return strings.Compare(left.AllocationID, right.AllocationID)
	})
	return response
}

func (r *InMemoryRegistry) entryNeedsReconciliationLocked(entry *agentEntry) bool {
	if entry.leaseExpired || entry.superseded || entry.blockedByInstanceID != nil || len(entry.releaseAcks) != 0 ||
		entry.registration.ObservedState == control.AgentFenced || entry.registration.ObservedState == control.AgentDraining {
		return true
	}
	for _, item := range entry.registration.Allocations {
		if !entry.owns(item.AllocationID) {
			return true
		}
	}
	for id := range entry.allocationIDs {
		stored := r.allocations[id]
		state, present := entry.registration.ObservedAllocation(id)
		if stored.loss != nil || stored.reservation.Grant.WriteFenced ||
			(stored.activated && (!present || state != control.AgentAllocated)) {
			return true
		}
	}
	return false
}

func (r *InMemoryRegistry) expireAndRetireAgentsLocked(monotonicNow time.Duration) {
	for _, entry := range r.agents {
		wasExpired := entry.leaseExpired
		r.expireEntry(entry, monotonicNow)
		if !wasExpired && entry.leaseExpired && len(entry.allocationIDs) != 0 {
			r.recordOperationsChangeLocked(OperationsRuntimeAgent, entry.registration.InstanceID)
		}
	}
	r.retireInactiveAgentsLocked()
}

func (r *InMemoryRegistry) retireInactiveAgentsLocked() {
	blocked := make(map[string]struct{})
	for _, entry := range r.agents {
		if entry.blockedByInstanceID != nil {
			blocked[*entry.blockedByInstanceID] = struct{}{}
		}
	}
	for instanceID, entry := range r.agents {
		if len(entry.allocationIDs) != 0 || !entry.leaseExpired && !entry.superseded {
			continue
		}
		if _, requiredAsBlocker := blocked[instanceID]; requiredAsBlocker {
			continue
		}
		delete(r.agents, instanceID)
		r.recordOperationsChangeLocked(OperationsRuntimeAgent, instanceID)
	}
}

func (r *InMemoryRegistry) retireInactiveAgentLocked(instanceID string) bool {
	entry, ok := r.agents[instanceID]
	if !ok || len(entry.allocationIDs) != 0 || !entry.leaseExpired && !entry.superseded {
		return false
	}
	for _, candidate := range r.agents {
		if candidate.blockedByInstanceID != nil && *candidate.blockedByInstanceID == instanceID {
			return false
		}
	}
	delete(r.agents, instanceID)
	r.recordOperationsChangeLocked(OperationsRuntimeAgent, instanceID)
	return true
}

func normalizeRegistration(source control.AgentRegistration) control.AgentRegistration {
	return control.NormalizeAgentRegistration(source)
}

func snapshotAgent(entry *agentEntry) AgentSnapshot {
	return AgentSnapshot{
		Principal:    clonePrincipal(entry.principal),
		Registration: entry.registration.Clone(), LastSeenAt: entry.lastSeenAt,
		LastHeartbeatSeq: entry.lastHeartbeatSeq, LastIssuedAckSeq: entry.lastIssuedAckSeq,
		LastConfirmedAckSeq: entry.lastConfirmedAckSeq, ConfirmedLeaseExpiresAt: entry.confirmedLeaseExpiresAt,
		AuthoritativeAllocationIDs: entry.authoritativeIDs(),
		ReconciliationRequired:     entry.reconciliationRequired,
		LeaseExpired:               entry.leaseExpired,
	}
}

func (r *InMemoryRegistry) expireEntry(entry *agentEntry, monotonicNow time.Duration) {
	deadline := entry.confirmedLeaseDeadline
	if deadline == 0 {
		deadline = entry.principalClaimDeadline
	}
	if entry.leaseExpired || deadline == 0 || monotonicNow < deadline {
		return
	}
	entry.leaseExpired = true
	entry.reconciliationRequired = true
	r.markAllocationLost(entry, LossControlLeaseExpired)
}

func (r *InMemoryRegistry) markAllocationLost(entry *agentEntry, reason AllocationLossReason) {
	for id := range entry.allocationIDs {
		r.markOneAllocationLost(entry, id, reason)
	}
}

func (r *InMemoryRegistry) markOneAllocationLost(entry *agentEntry, allocationID string, reason AllocationLossReason) {
	stored, ok := r.allocations[allocationID]
	if !ok || stored.loss != nil {
		return
	}
	loss := AllocationLoss{
		AllocationID: allocationID, RuntimeAgentID: entry.principal.RuntimeAgentID,
		RuntimeInstanceID: entry.registration.InstanceID,
		RunID:             stored.reservation.Grant.RunID, StageExecutionID: stored.reservation.Grant.StageExecutionID, Reason: reason,
	}
	stored.reservation.Grant.WriteFenced = true
	stored.reservation.Grant.Lost = true
	stored.reservation.Grant.LossReason = reason
	stored.loss = &loss
	r.allocations[allocationID] = stored
	entry.reconciliationRequired = true
	r.pendingLosses = append(r.pendingLosses, loss)
	r.recordOperationsChangeLocked(OperationsAllocation, allocationID)
}

func (r *InMemoryRegistry) detectObservedLoss(entry *agentEntry, reason AllocationLossReason) {
	for id := range entry.allocationIDs {
		stored := r.allocations[id]
		if stored.loss != nil {
			continue
		}
		state, present := entry.registration.ObservedAllocation(id)
		if entry.registration.ObservedState == control.AgentFenced && !stored.reservation.Grant.WriteFenced {
			r.markOneAllocationLost(entry, id, reason)
			continue
		}
		if present && state == control.AgentAllocated {
			stored.activated = true
			r.allocations[id] = stored
			continue
		}
		if !stored.activated && !present {
			continue
		}
		if present && stored.reservation.Grant.WriteFenced && (state == control.AgentDraining || state == control.AgentFenced) {
			continue
		}
		r.markOneAllocationLost(entry, id, reason)
	}
}

func (r *InMemoryRegistry) resetExpiredLease(entry *agentEntry) {
	entry.leaseExpired = false
	entry.confirmedLeaseDeadline = 0
	entry.confirmedLeaseExpiresAt = time.Time{}
	entry.lastConfirmedAckSeq = 0
	entry.issuedAcks = make(map[uint64]struct{})
	entry.heartbeatResponses = make(map[uint64]control.HeartbeatResponse)
	entry.heartbeatOrder = nil
	for id := range entry.releaseAcks {
		entry.releaseAcks[id] = 0
	}
}

func principalInstanceIsLive(entry *agentEntry, monotonicNow time.Duration) bool {
	if entry.leaseExpired {
		return false
	}
	deadline := entry.confirmedLeaseDeadline
	if deadline == 0 {
		deadline = entry.principalClaimDeadline
	}
	return deadline > monotonicNow
}

func sameRuntimeEndpoint(left, right control.AgentRegistration) bool {
	return left.ControlURL == right.ControlURL || left.A2AURL == right.A2AURL
}
