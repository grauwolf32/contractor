package controlplane

// Runtime Agent registration and heartbeat, and the lifecycle that
// follows from them: reconciliation, lease expiry and retirement of
// agents that stopped reporting.

import (
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func (r *InMemoryRegistry) RegistrationResponse(principal AuthenticatedPrincipal) contracts.AgentRegistrationResponse {
	return contracts.AgentRegistrationResponse{
		APIVersion:               contracts.APIVersion,
		RuntimeAgentID:           principal.RuntimeAgentID,
		Labels:                   append([]string{}, principal.Labels...),
		LabelRevision:            principal.LabelRevision,
		HeartbeatIntervalSeconds: int(r.heartbeatInterval / time.Second),
		ConfirmedLeaseSeconds:    int(r.confirmedLease / time.Second),
	}
}

func (r *InMemoryRegistry) RegisterAuthenticated(
	principal AuthenticatedPrincipal,
	registration contracts.AgentRegistration,
) (AgentSnapshot, error) {
	if err := registration.Validate(); err != nil {
		return AgentSnapshot{}, fmt.Errorf("%w: %v", ErrInvalidRequest, err)
	}
	if err := validateAuthenticatedPrincipal(principal); err != nil {
		return AgentSnapshot{}, err
	}
	normalized := normalizeRegistration(registration)
	identity, err := contracts.AgentRegistrationFingerprint(normalized)
	if err != nil {
		return AgentSnapshot{}, fmt.Errorf("encode registration identity: %w", err)
	}
	orderKey := r.agentOrderKey(cloneRegistration(normalized))
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
		existing.principal = clonePrincipal(principal)
		existing.registration.ObservedState = normalized.ObservedState
		existing.registration.AllocationID = cloneString(normalized.AllocationID)
		existing.lastSeenAt = now
		if existing.leaseExpired && existing.authoritativeAllocationID == nil {
			r.resetExpiredLease(existing)
		}
		if existing.confirmedLeaseDeadline == 0 {
			existing.principalClaimDeadline = r.monotonicNow() + r.confirmedLease
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
		principal: clonePrincipal(principal), registration: normalized,
		identity: identity, orderKey: orderKey, lastSeenAt: now,
		principalClaimDeadline: monotonicNow + r.confirmedLease,
		issuedAcks:             make(map[uint64]struct{}), heartbeatResponses: make(map[uint64]contracts.HeartbeatResponse),
	}
	for instanceID, existing := range r.agents {
		if existing.superseded || !sameRuntimeEndpoint(existing.registration, normalized) {
			continue
		}
		existing.superseded = true
		existing.reconciliationRequired = true
		r.markAllocationLost(existing, LossRuntimeRestarted)
		if existing.authoritativeAllocationID != nil {
			entry.blockedByInstanceID = cloneString(&instanceID)
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
	heartbeat contracts.AgentHeartbeat,
) (contracts.HeartbeatResponse, error) {
	if err := heartbeat.Validate(); err != nil {
		return contracts.HeartbeatResponse{}, fmt.Errorf("%w: %v", ErrInvalidRequest, err)
	}
	now := r.now()
	r.mu.Lock()
	defer r.mu.Unlock()
	entry, ok := r.agents[heartbeat.InstanceID]
	if !ok {
		return contracts.HeartbeatResponse{
			APIVersion: contracts.APIVersion, AckSeq: heartbeat.HeartbeatSeq, Action: contracts.ActionReregister,
		}, nil
	}
	if entry.principal.RuntimeAgentID != runtimeAgentID {
		return contracts.HeartbeatResponse{}, ErrRegistrationConflict
	}
	r.expireEntry(entry, r.monotonicNow())
	if heartbeat.HeartbeatSeq <= entry.lastHeartbeatSeq {
		if response, exists := entry.heartbeatResponses[heartbeat.HeartbeatSeq]; exists {
			entry.lastSeenAt = now
			entry.lastAcceptedHeartbeatAt = now
			r.recordOperationsChangeLocked(OperationsRuntimeAgent, heartbeat.InstanceID)
			return cloneHeartbeatResponse(response), nil
		}
		return contracts.HeartbeatResponse{}, ErrHeartbeatOutOfOrder
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
	entry.registration.AllocationID = cloneString(heartbeat.AllocationID)
	r.detectObservedLoss(entry, LossRuntimeMismatch)
	response, reconciliation := heartbeatAction(entry, heartbeat.HeartbeatSeq)
	entry.reconciliationRequired = reconciliation || r.entryNeedsReconciliationLocked(entry)
	r.recordHeartbeat(entry, heartbeat.HeartbeatSeq, response)
	r.recordOperationsChangeLocked(OperationsRuntimeAgent, heartbeat.InstanceID)
	return cloneHeartbeatResponse(response), nil
}

// Register and Heartbeat retain a small in-process test/embedding surface for
// callers that do not terminate TLS. The private HTTP API never uses these
// methods: production always supplies its certificate-derived principal to
// RegisterAuthenticated/HeartbeatAuthenticated and validates the current private contract.
func (r *InMemoryRegistry) Register(registration contracts.AgentRegistration) (AgentSnapshot, error) {
	return r.RegisterAuthenticated(inProcessPrincipal(registration.InstanceID), registration)
}

func (r *InMemoryRegistry) Heartbeat(heartbeat contracts.AgentHeartbeat) (contracts.HeartbeatResponse, error) {
	return r.HeartbeatAuthenticated(inProcessPrincipal(heartbeat.InstanceID).RuntimeAgentID, heartbeat)
}

func (r *InMemoryRegistry) GetAgent(instanceID string) (AgentSnapshot, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	entry, ok := r.agents[instanceID]
	if !ok {
		return AgentSnapshot{}, ErrAgentNotFound
	}
	return snapshotAgent(entry), nil
}

func (r *InMemoryRegistry) recordHeartbeat(entry *agentEntry, sequence uint64, response contracts.HeartbeatResponse) {
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

func heartbeatAction(entry *agentEntry, sequence uint64) (contracts.HeartbeatResponse, bool) {
	response := contracts.HeartbeatResponse{APIVersion: contracts.APIVersion, AckSeq: sequence}
	if entry.leaseExpired && entry.authoritativeAllocationID == nil {
		response.Action = contracts.ActionReregister
		return response, true
	}
	if entry.authoritativeAllocationID == nil {
		if entry.registration.ObservedState == contracts.AgentIdle {
			response.Action = contracts.ActionContinue
			return response, entry.blockedByInstanceID != nil || entry.superseded
		}
		response.AllocationID = cloneString(entry.registration.AllocationID)
		if entry.registration.ObservedState == contracts.AgentFenced {
			response.Action = contracts.ActionRelease
		} else {
			response.Action = contracts.ActionDrain
		}
		return response, true
	}
	response.AllocationID = cloneString(entry.authoritativeAllocationID)
	if entry.leaseExpired || entry.superseded {
		response.Action = contracts.ActionDrain
		return response, true
	}
	if !entry.allocationActivated && entry.registration.ObservedState == contracts.AgentIdle &&
		entry.registration.AllocationID == nil {
		// Reservation authority precedes the private prepare call. The Runtime
		// may legitimately report idle during that bounded transition.
		response.Action = contracts.ActionContinue
		return response, false
	}
	if entry.registration.ObservedState == contracts.AgentAllocated && entry.registration.AllocationID != nil &&
		*entry.registration.AllocationID == *entry.authoritativeAllocationID {
		// A grant marked lost never becomes live again, even if a delayed
		// heartbeat happens to match its old identity.
		// heartbeatAction has no Registry pointer, so WriteFenced/lost paths are
		// represented by reconciliationRequired set by detectObservedLoss.
		if entry.reconciliationRequired {
			response.Action = contracts.ActionDrain
			return response, true
		}
		response.Action = contracts.ActionContinue
		return response, false
	}
	response.Action = contracts.ActionDrain
	return response, true
}

func registrationNeedsReconciliation(entry *agentEntry) bool {
	if entry.leaseExpired || entry.allocationLost || entry.superseded || entry.blockedByInstanceID != nil {
		return true
	}
	if entry.authoritativeAllocationID == nil {
		return entry.registration.ObservedState != contracts.AgentIdle
	}
	if !entry.allocationActivated && entry.registration.ObservedState == contracts.AgentIdle &&
		entry.registration.AllocationID == nil {
		return false
	}
	return entry.registration.ObservedState != contracts.AgentAllocated || entry.registration.AllocationID == nil ||
		*entry.registration.AllocationID != *entry.authoritativeAllocationID
}

func (r *InMemoryRegistry) entryNeedsReconciliationLocked(entry *agentEntry) bool {
	if registrationNeedsReconciliation(entry) {
		return true
	}
	if entry.authoritativeAllocationID == nil {
		return false
	}
	stored, ok := r.allocations[*entry.authoritativeAllocationID]
	return ok && stored.reservation.Grant.WriteFenced
}

func (r *InMemoryRegistry) expireAndRetireAgentsLocked(monotonicNow time.Duration) {
	for _, entry := range r.agents {
		wasExpired := entry.leaseExpired
		r.expireEntry(entry, monotonicNow)
		if !wasExpired && entry.leaseExpired && entry.authoritativeAllocationID != nil {
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
		if entry.authoritativeAllocationID != nil || !entry.leaseExpired && !entry.superseded {
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
	if !ok || entry.authoritativeAllocationID != nil || !entry.leaseExpired && !entry.superseded {
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

func normalizeRegistration(source contracts.AgentRegistration) contracts.AgentRegistration {
	return contracts.NormalizeAgentRegistration(source)
}

func snapshotAgent(entry *agentEntry) AgentSnapshot {
	return AgentSnapshot{
		Principal:    clonePrincipal(entry.principal),
		Registration: cloneRegistration(entry.registration), LastSeenAt: entry.lastSeenAt,
		LastHeartbeatSeq: entry.lastHeartbeatSeq, LastIssuedAckSeq: entry.lastIssuedAckSeq,
		LastConfirmedAckSeq: entry.lastConfirmedAckSeq, ConfirmedLeaseExpiresAt: entry.confirmedLeaseExpiresAt,
		AuthoritativeAllocationID: cloneString(entry.authoritativeAllocationID),
		ReconciliationRequired:    entry.reconciliationRequired,
		LeaseExpired:              entry.leaseExpired,
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

func sameRuntimeEndpoint(left, right contracts.AgentRegistration) bool {
	return left.ControlURL == right.ControlURL || left.A2AURL == right.A2AURL
}
