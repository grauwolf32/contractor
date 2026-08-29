package controlplane

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const heartbeatHistoryLimit = 128

type Registry interface {
	RegistrationResponse() contracts.AgentRegistrationResponse
	Register(contracts.AgentRegistration) (AgentSnapshot, error)
	Heartbeat(contracts.AgentHeartbeat) (contracts.HeartbeatResponse, error)
	ReserveAll(ReservationRequest) ([]Reservation, error)
	GetGrant(string) (AllocationGrant, error)
	SetWriteFence(string) error
	Release(string) error
	GetAgent(string) (AgentSnapshot, error)
	PollAllocationLosses() []AllocationLoss
}

type RegistryOptions struct {
	HeartbeatInterval time.Duration
	ConfirmedLease    time.Duration
	Now               func() time.Time
	MonotonicNow      func() time.Duration
	NewID             func(string) (string, error)
	AgentOrderKey     func(contracts.AgentRegistration) string
}

type AgentSnapshot struct {
	Registration              contracts.AgentRegistration
	LastSeenAt                time.Time
	LastHeartbeatSeq          uint64
	LastIssuedAckSeq          uint64
	LastConfirmedAckSeq       uint64
	ConfirmedLeaseExpiresAt   time.Time
	AuthoritativeAllocationID *string
	ReconciliationRequired    bool
	LeaseExpired              bool
}

type InMemoryRegistry struct {
	mu                sync.Mutex
	idMu              sync.Mutex
	agents            map[string]*agentEntry
	allocations       map[string]storedReservation
	stageReservations map[string]stageReservation
	heartbeatInterval time.Duration
	confirmedLease    time.Duration
	now               func() time.Time
	monotonicNow      func() time.Duration
	newID             func(string) (string, error)
	agentOrderKey     func(contracts.AgentRegistration) string
	pendingLosses     []AllocationLoss
}

type agentEntry struct {
	registration              contracts.AgentRegistration
	identity                  string
	orderKey                  string
	lastSeenAt                time.Time
	lastHeartbeatSeq          uint64
	lastIssuedAckSeq          uint64
	lastConfirmedAckSeq       uint64
	confirmedLeaseExpiresAt   time.Time
	authoritativeAllocationID *string
	reconciliationRequired    bool
	confirmedLeaseDeadline    time.Duration
	leaseExpired              bool
	allocationLost            bool
	allocationActivated       bool
	superseded                bool
	blockedByInstanceID       *string
	issuedAcks                map[uint64]struct{}
	heartbeatResponses        map[uint64]contracts.HeartbeatResponse
	heartbeatOrder            []uint64
}

type storedReservation struct {
	reservation Reservation
	loss        *AllocationLoss
}

type stageReservation struct {
	fingerprint   string
	allocationIDs []string
}

func NewRegistry(options RegistryOptions) (*InMemoryRegistry, error) {
	if options.HeartbeatInterval == 0 {
		options.HeartbeatInterval = 10 * time.Second
	}
	if options.ConfirmedLease == 0 {
		options.ConfirmedLease = 60 * time.Second
	}
	if options.HeartbeatInterval < time.Second || options.ConfirmedLease <= options.HeartbeatInterval ||
		options.HeartbeatInterval%time.Second != 0 || options.ConfirmedLease%time.Second != 0 {
		return nil, fmt.Errorf("%w: heartbeat and lease durations must be whole seconds and lease must exceed interval", ErrInvalidRequest)
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	if options.MonotonicNow == nil {
		started := time.Now()
		options.MonotonicNow = func() time.Duration { return time.Since(started) }
	}
	if options.NewID == nil {
		options.NewID = randomID
	}
	if options.AgentOrderKey == nil {
		options.AgentOrderKey = func(registration contracts.AgentRegistration) string {
			return registration.InstanceID
		}
	}
	return &InMemoryRegistry{
		agents: make(map[string]*agentEntry), allocations: make(map[string]storedReservation),
		stageReservations: make(map[string]stageReservation),
		heartbeatInterval: options.HeartbeatInterval, confirmedLease: options.ConfirmedLease,
		now: options.Now, monotonicNow: options.MonotonicNow,
		newID: options.NewID, agentOrderKey: options.AgentOrderKey,
	}, nil
}

func (r *InMemoryRegistry) RegistrationResponse() contracts.AgentRegistrationResponse {
	return contracts.AgentRegistrationResponse{
		APIVersion:               contracts.APIVersion,
		HeartbeatIntervalSeconds: int(r.heartbeatInterval / time.Second),
		ConfirmedLeaseSeconds:    int(r.confirmedLease / time.Second),
	}
}

func (r *InMemoryRegistry) Register(registration contracts.AgentRegistration) (AgentSnapshot, error) {
	if err := registration.Validate(); err != nil {
		return AgentSnapshot{}, fmt.Errorf("%w: %v", ErrInvalidRequest, err)
	}
	normalized := normalizeRegistration(registration)
	identity, err := registrationIdentity(normalized)
	if err != nil {
		return AgentSnapshot{}, fmt.Errorf("encode registration identity: %w", err)
	}
	orderKey := r.agentOrderKey(cloneRegistration(normalized))
	now := r.now()

	r.mu.Lock()
	defer r.mu.Unlock()
	if existing, ok := r.agents[normalized.InstanceID]; ok {
		if existing.identity != identity {
			return AgentSnapshot{}, ErrRegistrationConflict
		}
		existing.registration.ObservedState = normalized.ObservedState
		existing.registration.AllocationID = cloneString(normalized.AllocationID)
		existing.lastSeenAt = now
		if existing.leaseExpired && existing.authoritativeAllocationID == nil {
			r.resetExpiredLease(existing)
		}
		r.detectObservedLoss(existing, LossRuntimeMismatch)
		existing.reconciliationRequired = registrationNeedsReconciliation(existing)
		return snapshotAgent(existing), nil
	}
	entry := &agentEntry{
		registration: normalized, identity: identity, orderKey: orderKey, lastSeenAt: now,
		issuedAcks: make(map[uint64]struct{}), heartbeatResponses: make(map[uint64]contracts.HeartbeatResponse),
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
		}
	}
	entry.reconciliationRequired = registrationNeedsReconciliation(entry)
	r.agents[normalized.InstanceID] = entry
	return snapshotAgent(entry), nil
}

func (r *InMemoryRegistry) Heartbeat(heartbeat contracts.AgentHeartbeat) (contracts.HeartbeatResponse, error) {
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
	r.expireEntry(entry, r.monotonicNow())
	if heartbeat.HeartbeatSeq <= entry.lastHeartbeatSeq {
		if response, exists := entry.heartbeatResponses[heartbeat.HeartbeatSeq]; exists {
			entry.lastSeenAt = now
			return cloneHeartbeatResponse(response), nil
		}
		return contracts.HeartbeatResponse{}, ErrHeartbeatOutOfOrder
	}
	if !entry.leaseExpired && heartbeat.EchoedAckSeq > entry.lastConfirmedAckSeq {
		if _, issued := entry.issuedAcks[heartbeat.EchoedAckSeq]; issued {
			entry.lastConfirmedAckSeq = heartbeat.EchoedAckSeq
			entry.confirmedLeaseExpiresAt = now.Add(r.confirmedLease)
			entry.confirmedLeaseDeadline = r.monotonicNow() + r.confirmedLease
		}
	}
	entry.lastSeenAt = now
	entry.lastHeartbeatSeq = heartbeat.HeartbeatSeq
	entry.lastIssuedAckSeq = heartbeat.HeartbeatSeq
	entry.registration.ObservedState = heartbeat.ObservedState
	entry.registration.AllocationID = cloneString(heartbeat.AllocationID)
	r.detectObservedLoss(entry, LossRuntimeMismatch)
	response, reconciliation := heartbeatAction(entry, heartbeat.HeartbeatSeq)
	entry.reconciliationRequired = reconciliation || registrationNeedsReconciliation(entry)
	r.recordHeartbeat(entry, heartbeat.HeartbeatSeq, response)
	return cloneHeartbeatResponse(response), nil
}

func (r *InMemoryRegistry) ReserveAll(request ReservationRequest) ([]Reservation, error) {
	fingerprint, normalizedBindings, err := normalizeReservationRequest(request)
	if err != nil {
		return nil, err
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
	selected := make([]*agentEntry, len(normalizedBindings))
	used := make(map[string]struct{}, len(normalizedBindings))
	for bindingIndex, binding := range normalizedBindings {
		for _, entry := range available {
			if _, exists := used[entry.registration.InstanceID]; exists || !isCompatible(entry.registration, binding.AgentTemplate) {
				continue
			}
			selected[bindingIndex] = entry
			used[entry.registration.InstanceID] = struct{}{}
			break
		}
		if selected[bindingIndex] == nil {
			return nil, ErrInsufficientCapacity
		}
	}

	reservations := make([]Reservation, len(normalizedBindings))
	for index, binding := range normalizedBindings {
		entry := selected[index]
		allocationID := allocationIDs[index]
		grant := AllocationGrant{
			AllocationID: allocationID, RuntimeInstanceID: entry.registration.InstanceID,
			RunID: request.RunID, StageExecutionID: request.StageExecutionID,
			LogicalAgentName: binding.LogicalAgentName, Namespace: binding.Namespace,
			ReadPolicy: ReadCurrentRun, WritePolicy: WriteInputsAndIntermediates,
		}
		reservation := Reservation{
			Grant: grant, ControlURL: entry.registration.ControlURL, A2AURL: entry.registration.A2AURL,
			AgentTemplate: cloneAgentTemplate(binding.AgentTemplate), LeaseExpiresAt: entry.confirmedLeaseExpiresAt,
		}
		entry.authoritativeAllocationID = cloneString(&allocationID)
		entry.allocationActivated = false
		r.allocations[allocationID] = storedReservation{reservation: reservation}
		reservations[index] = cloneReservation(reservation)
	}
	r.stageReservations[request.StageExecutionID] = stageReservation{
		fingerprint: fingerprint, allocationIDs: append([]string(nil), allocationIDs...),
	}
	return reservations, nil
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

func (r *InMemoryRegistry) SetWriteFence(allocationID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	stored, ok := r.allocations[allocationID]
	if !ok {
		return ErrAllocationNotFound
	}
	stored.reservation.Grant.WriteFenced = true
	r.allocations[allocationID] = stored
	return nil
}

func (r *InMemoryRegistry) Release(allocationID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	stored, ok := r.allocations[allocationID]
	if !ok {
		return ErrAllocationNotFound
	}
	entry, ok := r.agents[stored.reservation.Grant.RuntimeInstanceID]
	if !ok || entry.authoritativeAllocationID == nil || *entry.authoritativeAllocationID != allocationID {
		return ErrAllocationNotFound
	}
	entry.authoritativeAllocationID = nil
	entry.allocationLost = false
	entry.allocationActivated = false
	entry.reconciliationRequired = registrationNeedsReconciliation(entry)
	delete(r.allocations, allocationID)
	for _, candidate := range r.agents {
		if candidate.blockedByInstanceID != nil && *candidate.blockedByInstanceID == entry.registration.InstanceID {
			candidate.blockedByInstanceID = nil
			candidate.reconciliationRequired = registrationNeedsReconciliation(candidate)
		}
	}
	return nil
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

// PollAllocationLosses expires monotonic deadlines and drains the one-shot
// loss edge queue. A late heartbeat can update diagnostics but cannot remove a
// loss already attached to an allocation.
func (r *InMemoryRegistry) PollAllocationLosses() []AllocationLoss {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := r.monotonicNow()
	for _, entry := range r.agents {
		r.expireEntry(entry, now)
	}
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

func isPlacementEligible(entry *agentEntry, monotonicNow time.Duration) bool {
	return entry.authoritativeAllocationID == nil && !entry.reconciliationRequired &&
		!entry.leaseExpired && !entry.superseded && entry.blockedByInstanceID == nil &&
		entry.registration.ObservedState == contracts.AgentIdle &&
		entry.confirmedLeaseDeadline > monotonicNow
}

func isCompatible(registration contracts.AgentRegistration, template contracts.ResolvedAgentTemplate) bool {
	runtime := template.Runtime.RuntimeID + "@" + template.Runtime.Version
	if !contains(registration.SupportedRuntimes, runtime) {
		return false
	}
	sandbox := template.SandboxProfile.SandboxProfileID + "@" + template.SandboxProfile.Version
	if !contains(registration.SupportedSandboxProfiles, sandbox) {
		return false
	}
	capabilities := make(map[string]map[string]struct{}, len(registration.SupportedToolsets))
	for _, capability := range registration.SupportedToolsets {
		tools := make(map[string]struct{}, len(capability.Tools))
		for _, tool := range capability.Tools {
			tools[tool] = struct{}{}
		}
		capabilities[capability.Ref] = tools
	}
	for _, selection := range template.Toolsets {
		ref := selection.Ref.ToolsetID + "@" + selection.Ref.Version
		tools, ok := capabilities[ref]
		if !ok {
			return false
		}
		for _, selected := range selection.Tools {
			if _, ok := tools[selected]; !ok {
				return false
			}
		}
	}
	return true
}

func normalizeReservationRequest(request ReservationRequest) (string, []BindingRequirement, error) {
	if strings.TrimSpace(request.RunID) == "" || strings.TrimSpace(request.StageExecutionID) == "" || len(request.Bindings) == 0 {
		return "", nil, fmt.Errorf("%w: run, StageExecution, and bindings are required", ErrInvalidRequest)
	}
	bindings := make([]BindingRequirement, len(request.Bindings))
	seen := make(map[string]struct{}, len(request.Bindings))
	for index, binding := range request.Bindings {
		if strings.TrimSpace(binding.LogicalAgentName) == "" || strings.TrimSpace(binding.Namespace) == "" || strings.Contains(binding.Namespace, "/") {
			return "", nil, fmt.Errorf("%w: binding name and slash-free namespace are required", ErrInvalidRequest)
		}
		if binding.Namespace == "inputs" || binding.Namespace == "outputs" {
			return "", nil, fmt.Errorf("%w: Agent binding cannot use a Run-reserved namespace", ErrInvalidRequest)
		}
		if err := binding.AgentTemplate.Validate(); err != nil {
			return "", nil, fmt.Errorf("%w: invalid AgentTemplate for %q: %v", ErrInvalidRequest, binding.LogicalAgentName, err)
		}
		if _, duplicate := seen[binding.LogicalAgentName]; duplicate {
			return "", nil, fmt.Errorf("%w: duplicate logical Agent name", ErrInvalidRequest)
		}
		seen[binding.LogicalAgentName] = struct{}{}
		bindings[index] = BindingRequirement{
			LogicalAgentName: binding.LogicalAgentName, Namespace: binding.Namespace,
			AgentTemplate: cloneAgentTemplate(binding.AgentTemplate),
		}
	}
	sort.Slice(bindings, func(i, j int) bool { return bindings[i].LogicalAgentName < bindings[j].LogicalAgentName })
	encoded, err := json.Marshal(struct {
		RunID            string
		StageExecutionID string
		Bindings         []BindingRequirement
	}{request.RunID, request.StageExecutionID, bindings})
	if err != nil {
		return "", nil, fmt.Errorf("encode reservation request: %w", err)
	}
	return string(encoded), bindings, nil
}

func normalizeRegistration(source contracts.AgentRegistration) contracts.AgentRegistration {
	result := cloneRegistration(source)
	sort.Strings(result.SupportedRuntimes)
	sort.Strings(result.SupportedSandboxProfiles)
	for index := range result.SupportedToolsets {
		sort.Strings(result.SupportedToolsets[index].Tools)
	}
	sort.Slice(result.SupportedToolsets, func(i, j int) bool { return result.SupportedToolsets[i].Ref < result.SupportedToolsets[j].Ref })
	return result
}

func registrationIdentity(source contracts.AgentRegistration) (string, error) {
	source.ObservedState = ""
	source.AllocationID = nil
	encoded, err := json.Marshal(source)
	return string(encoded), err
}

func snapshotAgent(entry *agentEntry) AgentSnapshot {
	return AgentSnapshot{
		Registration: cloneRegistration(entry.registration), LastSeenAt: entry.lastSeenAt,
		LastHeartbeatSeq: entry.lastHeartbeatSeq, LastIssuedAckSeq: entry.lastIssuedAckSeq,
		LastConfirmedAckSeq: entry.lastConfirmedAckSeq, ConfirmedLeaseExpiresAt: entry.confirmedLeaseExpiresAt,
		AuthoritativeAllocationID: cloneString(entry.authoritativeAllocationID),
		ReconciliationRequired:    entry.reconciliationRequired,
		LeaseExpired:              entry.leaseExpired,
	}
}

func (r *InMemoryRegistry) expireEntry(entry *agentEntry, monotonicNow time.Duration) {
	if entry.leaseExpired || entry.confirmedLeaseDeadline == 0 ||
		monotonicNow < entry.confirmedLeaseDeadline {
		return
	}
	entry.leaseExpired = true
	entry.reconciliationRequired = true
	r.markAllocationLost(entry, LossControlLeaseExpired)
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
		AllocationID: allocationID, RuntimeInstanceID: entry.registration.InstanceID,
		RunID:            stored.reservation.Grant.RunID,
		StageExecutionID: stored.reservation.Grant.StageExecutionID,
		Reason:           reason,
	}
	stored.reservation.Grant.WriteFenced = true
	stored.reservation.Grant.Lost = true
	stored.loss = &loss
	r.allocations[allocationID] = stored
	entry.allocationLost = true
	entry.reconciliationRequired = true
	r.pendingLosses = append(r.pendingLosses, loss)
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

func sameRuntimeEndpoint(left, right contracts.AgentRegistration) bool {
	return left.ControlURL == right.ControlURL || left.A2AURL == right.A2AURL
}

func cloneRegistration(source contracts.AgentRegistration) contracts.AgentRegistration {
	result := source
	result.AllocationID = cloneString(source.AllocationID)
	result.SupportedRuntimes = append([]string(nil), source.SupportedRuntimes...)
	result.SupportedSandboxProfiles = append([]string(nil), source.SupportedSandboxProfiles...)
	result.SupportedToolsets = make([]contracts.ToolsetCapability, len(source.SupportedToolsets))
	for index, capability := range source.SupportedToolsets {
		result.SupportedToolsets[index] = capability
		result.SupportedToolsets[index].Tools = append([]string(nil), capability.Tools...)
	}
	return result
}

func cloneHeartbeatResponse(source contracts.HeartbeatResponse) contracts.HeartbeatResponse {
	result := source
	result.AllocationID = cloneString(source.AllocationID)
	return result
}

func cloneReservation(source Reservation) Reservation {
	result := source
	result.AgentTemplate = cloneAgentTemplate(source.AgentTemplate)
	return result
}

func cloneAgentTemplate(source contracts.ResolvedAgentTemplate) contracts.ResolvedAgentTemplate {
	result := source
	if source.ModelPolicy.Temperature != nil {
		temperature := *source.ModelPolicy.Temperature
		result.ModelPolicy.Temperature = &temperature
	}
	result.Toolsets = make([]contracts.ToolsetSelection, len(source.Toolsets))
	for index, selection := range source.Toolsets {
		result.Toolsets[index] = selection
		result.Toolsets[index].Tools = append([]string(nil), selection.Tools...)
	}
	return result
}

func cloneString(source *string) *string {
	if source == nil {
		return nil
	}
	result := *source
	return &result
}

func contains(values []string, expected string) bool {
	index := sort.SearchStrings(values, expected)
	return index < len(values) && values[index] == expected
}

func randomID(prefix string) (string, error) {
	buffer := make([]byte, 16)
	if _, err := rand.Read(buffer); err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(buffer), nil
}

func (r *InMemoryRegistry) nextID(prefix string) (string, error) {
	r.idMu.Lock()
	defer r.idMu.Unlock()
	return r.newID(prefix)
}

var _ Registry = (*InMemoryRegistry)(nil)
