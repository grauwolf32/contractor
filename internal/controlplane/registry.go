package controlplane

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

const (
	heartbeatHistoryLimit          = 128
	stageReservationTombstoneLimit = 1024
)

type Registry interface {
	RegistrationResponse(AuthenticatedPrincipal) contracts.AgentRegistrationResponseV2
	RegisterAuthenticated(AuthenticatedPrincipal, contracts.AgentRegistrationV2) (AgentSnapshot, error)
	HeartbeatAuthenticated(string, contracts.AgentHeartbeat) (contracts.HeartbeatResponse, error)
	ReserveAll(ReservationRequest) ([]Reservation, error)
	GetGrant(string) (AllocationGrant, error)
	GetReservation(string) (Reservation, error)
	WithWriteGrant(string, func(AllocationGrant) error) error
	SetWriteFence(string) error
	Release(string) error
	GetAgent(string) (AgentSnapshot, error)
	PollAllocationLosses() []AllocationLoss
	SnapshotOperations() OperationsSnapshot
	SetAllocationPhase(string, AllocationAuthoritativePhase, *SafeReason) error
	RecordAllocationReport(string, contracts.AllocationFinalReport) error
}

type AuthenticatedPrincipal struct {
	RuntimeAgentID string
	Labels         []string
	LabelRevision  uint64
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
	Principal                 AuthenticatedPrincipal
	Registration              contracts.AgentRegistrationV2
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
	mu                         sync.Mutex
	idMu                       sync.Mutex
	agents                     map[string]*agentEntry
	allocations                map[string]storedReservation
	stageReservations          map[string]stageReservation
	stageReservationTombstones []string
	heartbeatInterval          time.Duration
	confirmedLease             time.Duration
	now                        func() time.Time
	monotonicNow               func() time.Duration
	newID                      func(string) (string, error)
	agentOrderKey              func(contracts.AgentRegistration) string
	principalDeletions         map[string]struct{}
	pendingLosses              []AllocationLoss
	operationsGeneration       string
	operationsRevision         uint64
	operationsHistory          []OperationsChange
	operationsWatchers         map[uint64]chan struct{}
	nextOperationsWatcher      uint64
}

type agentEntry struct {
	principal                 AuthenticatedPrincipal
	registration              contracts.AgentRegistrationV2
	identity                  string
	orderKey                  string
	lastSeenAt                time.Time
	lastAcceptedHeartbeatAt   time.Time
	lastHeartbeatSeq          uint64
	lastIssuedAckSeq          uint64
	lastConfirmedAckSeq       uint64
	confirmedLeaseExpiresAt   time.Time
	authoritativeAllocationID *string
	reconciliationRequired    bool
	confirmedLeaseDeadline    time.Duration
	principalClaimDeadline    time.Duration
	leaseExpired              bool
	allocationLost            bool
	allocationActivated       bool
	superseded                bool
	blockedByInstanceID       *string
	issuedAcks                map[uint64]struct{}
	heartbeatResponses        map[uint64]contracts.HeartbeatResponse
	heartbeatOrder            []uint64
}

type stageReservation struct {
	fingerprint   string
	allocationIDs []string
	committed     bool
	released      bool
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
	operationsGeneration, err := randomID("operations-generation-")
	if err != nil {
		return nil, fmt.Errorf("generate Operations snapshot identity: %w", err)
	}
	return &InMemoryRegistry{
		agents: make(map[string]*agentEntry), allocations: make(map[string]storedReservation),
		stageReservations: make(map[string]stageReservation),
		heartbeatInterval: options.HeartbeatInterval, confirmedLease: options.ConfirmedLease,
		now: options.Now, monotonicNow: options.MonotonicNow,
		newID: options.NewID, agentOrderKey: options.AgentOrderKey,
		operationsGeneration: operationsGeneration,
		operationsWatchers:   make(map[uint64]chan struct{}),
		principalDeletions:   make(map[string]struct{}),
	}, nil
}

func (r *InMemoryRegistry) RegistrationResponse(principal AuthenticatedPrincipal) contracts.AgentRegistrationResponseV2 {
	return contracts.AgentRegistrationResponseV2{
		APIVersion:               contracts.APIVersion,
		PrivateProtocolVersion:   contracts.PrivateProtocolVersionV2,
		RuntimeAgentID:           principal.RuntimeAgentID,
		Labels:                   append([]string{}, principal.Labels...),
		LabelRevision:            principal.LabelRevision,
		HeartbeatIntervalSeconds: int(r.heartbeatInterval / time.Second),
		ConfirmedLeaseSeconds:    int(r.confirmedLease / time.Second),
	}
}

func (r *InMemoryRegistry) RegisterAuthenticated(
	principal AuthenticatedPrincipal,
	registration contracts.AgentRegistrationV2,
) (AgentSnapshot, error) {
	if err := registration.Validate(); err != nil {
		return AgentSnapshot{}, fmt.Errorf("%w: %v", ErrInvalidRequest, err)
	}
	if err := validateAuthenticatedPrincipal(principal); err != nil {
		return AgentSnapshot{}, err
	}
	normalized := normalizeRegistration(registration)
	identity, err := contracts.AgentRegistrationFingerprintV2(normalized)
	if err != nil {
		return AgentSnapshot{}, fmt.Errorf("encode registration identity: %w", err)
	}
	orderKey := r.agentOrderKey(legacyRegistrationProjection(normalized))
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
// RegisterAuthenticated/HeartbeatAuthenticated and serves protocol v2 only.
func (r *InMemoryRegistry) Register(registration contracts.AgentRegistration) (AgentSnapshot, error) {
	v2 := contracts.AgentRegistrationV2{
		APIVersion:             registration.APIVersion,
		PrivateProtocolVersion: contracts.PrivateProtocolVersionV2,
		InstanceID:             registration.InstanceID, SoftwareVersion: registration.SoftwareVersion,
		StartedAt: registration.StartedAt, ControlURL: registration.ControlURL, A2AURL: registration.A2AURL,
		InitialLabels: []string{}, SupportedRuntimes: registration.SupportedRuntimes,
		SupportedToolsets:        registration.SupportedToolsets,
		SupportedSandboxProfiles: registration.SupportedSandboxProfiles,
		SupportedRuntimeAdapters: []contracts.RuntimeAdapterRef{},
		ObservedState:            registration.ObservedState, AllocationID: registration.AllocationID,
	}
	return r.RegisterAuthenticated(legacyPrincipal(registration.InstanceID), v2)
}

func (r *InMemoryRegistry) Heartbeat(heartbeat contracts.AgentHeartbeat) (contracts.HeartbeatResponse, error) {
	return r.HeartbeatAuthenticated(legacyPrincipal(heartbeat.InstanceID).RuntimeAgentID, heartbeat)
}

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
			Grant: grant, ControlURL: entry.registration.ControlURL, A2AURL: entry.registration.A2AURL,
			AgentTemplate:             cloneAgentTemplate(binding.AgentTemplate),
			WorkerSessionMode:         binding.WorkerSessionMode,
			ResolvedSkills:            contracts.CloneResolvedSkills(binding.ResolvedSkills),
			ExecutionConfig:           cloneAllocationExecutionConfig(binding.ExecutionConfig),
			Workspace:                 contracts.CloneAllocationWorkspaceSpecV2(binding.Workspace),
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
			configuration.Resolved.Validate() != nil {
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
		r.allocations[allocationID] = stored
	}
	existing.committed = true
	r.stageReservations[stageExecutionID] = existing
	return r.existingReservations(existing)
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

// completeCapabilityAssignment finds a complete injective binding-to-slot
// assignment over two already stable-ordered collections. Each augmentation
// uses an explicit queue and parent edges so request-controlled binding depth
// never becomes call-stack depth. It does not mutate Runtime Agent entries.
func completeCapabilityAssignment(
	available []*agentEntry,
	bindings []BindingRequirement,
) ([]*agentEntry, bool) {
	return completeCapabilityAssignmentWithEdges(available, bindings, nil)
}

func completeCapabilityAssignmentWithEdges(
	available []*agentEntry,
	bindings []BindingRequirement,
	edges []CandidateEdge,
) ([]*agentEntry, bool) {
	if len(bindings) > len(available) {
		return nil, false
	}
	edgeSet := make(map[string]CandidateEdge, len(edges))
	for _, edge := range edges {
		edgeSet[edge.LogicalAgentName+"\x00"+edge.RuntimeAgentInstanceID] = edge
	}
	candidates := make([][]int, len(bindings))
	for bindingIndex, binding := range bindings {
		for agentIndex, entry := range available {
			if !isCompatible(entry.registration, binding.AgentTemplate, binding.Workspace) {
				continue
			}
			if edges != nil {
				edge, ok := edgeSet[binding.LogicalAgentName+"\x00"+entry.registration.InstanceID]
				if !ok || edge.RuntimeAgentID != entry.principal.RuntimeAgentID ||
					edge.RuntimeAgentLabelRevision != entry.principal.LabelRevision ||
					!containsRuntimeAdapters(entry.registration.SupportedRuntimeAdapters, edge.RequiredRuntimeAdapters) {
					continue
				}
			}
			candidates[bindingIndex] = append(candidates[bindingIndex], agentIndex)
		}
		if len(candidates[bindingIndex]) == 0 {
			return nil, false
		}
	}

	bindingToAgent := integersFilled(len(bindings), -1)
	agentToBinding := integersFilled(len(available), -1)
	for startBinding := range bindings {
		seenBindings := make([]bool, len(bindings))
		seenAgents := make([]bool, len(available))
		parentBindingForAgent := integersFilled(len(available), -1)
		queue := make([]int, 1, len(bindings))
		queue[0] = startBinding
		seenBindings[startBinding] = true
		augmented := false

		for len(queue) > 0 && !augmented {
			bindingIndex := queue[0]
			queue = queue[1:]
			for _, agentIndex := range candidates[bindingIndex] {
				if seenAgents[agentIndex] {
					continue
				}
				seenAgents[agentIndex] = true
				parentBindingForAgent[agentIndex] = bindingIndex
				occupiedBy := agentToBinding[agentIndex]
				if occupiedBy == -1 {
					for currentAgent := agentIndex; currentAgent != -1; {
						currentBinding := parentBindingForAgent[currentAgent]
						previousAgent := bindingToAgent[currentBinding]
						bindingToAgent[currentBinding] = currentAgent
						agentToBinding[currentAgent] = currentBinding
						currentAgent = previousAgent
					}
					augmented = true
					break
				}
				if !seenBindings[occupiedBy] {
					seenBindings[occupiedBy] = true
					queue = append(queue, occupiedBy)
				}
			}
		}
		if !augmented {
			return nil, false
		}
	}

	selected := make([]*agentEntry, len(bindings))
	for bindingIndex, agentIndex := range bindingToAgent {
		if agentIndex < 0 {
			return nil, false
		}
		selected[bindingIndex] = available[agentIndex]
	}
	return selected, true
}

func validateCandidateEdges(bindings []BindingRequirement, edges []CandidateEdge) error {
	if edges == nil || len(edges) > len(bindings)*maximumOperationsItems {
		return fmt.Errorf("%w: candidate edge set is invalid", ErrInvalidRequest)
	}
	bindingsByName := make(map[string]struct{}, len(bindings))
	for _, binding := range bindings {
		bindingsByName[binding.LogicalAgentName] = struct{}{}
	}
	seen := make(map[string]struct{}, len(edges))
	for _, edge := range edges {
		if _, ok := bindingsByName[edge.LogicalAgentName]; !ok ||
			strings.TrimSpace(edge.RuntimeAgentInstanceID) == "" ||
			validateAuthenticatedPrincipal(AuthenticatedPrincipal{
				RuntimeAgentID: edge.RuntimeAgentID, Labels: []string{},
				LabelRevision: edge.RuntimeAgentLabelRevision,
			}) != nil {
			return fmt.Errorf("%w: candidate edge identity is invalid", ErrInvalidRequest)
		}
		key := edge.LogicalAgentName + "\x00" + edge.RuntimeAgentInstanceID
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

func containsRuntimeAdapters(
	available []contracts.RuntimeAdapterRef,
	required []contracts.RuntimeAdapterRef,
) bool {
	availableIndex := 0
	for _, expected := range required {
		for availableIndex < len(available) && available[availableIndex] < expected {
			availableIndex++
		}
		if availableIndex == len(available) || available[availableIndex] != expected {
			return false
		}
	}
	return true
}

func integersFilled(length int, value int) []int {
	result := make([]int, length)
	for index := range result {
		result[index] = value
	}
	return result
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

func isPlacementEligible(entry *agentEntry, monotonicNow time.Duration) bool {
	return entry.authoritativeAllocationID == nil && !entry.reconciliationRequired &&
		!entry.leaseExpired && !entry.superseded && entry.blockedByInstanceID == nil &&
		entry.registration.ObservedState == contracts.AgentIdle &&
		entry.confirmedLeaseDeadline > monotonicNow
}

func isCompatible(
	registration contracts.AgentRegistrationV2,
	template contracts.ResolvedAgentTemplate,
	workspace *contracts.AllocationWorkspaceSpecV2,
) bool {
	if !workflowconfig.SandboxWorkspaceCompatible(template, workspace, registration.WorkspaceCapabilities) {
		return false
	}
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
	if workspace != nil && !supportsWorkspaceMode(registration.WorkspaceCapabilities, workspace.Mode) {
		return false
	}
	return true
}

func supportsWorkspaceMode(
	capabilities *contracts.WorkspaceCapabilitiesV2,
	mode contracts.WorkspaceModeV2,
) bool {
	if capabilities == nil {
		return false
	}
	for _, supported := range capabilities.Modes {
		if supported == mode {
			return true
		}
	}
	return false
}

func normalizeReservationRequest(request ReservationRequest) (string, []BindingRequirement, error) {
	if strings.TrimSpace(request.RunID) == "" || strings.TrimSpace(request.StageExecutionID) == "" || len(request.Bindings) == 0 {
		return "", nil, fmt.Errorf("%w: run, StageExecution, and bindings are required", ErrInvalidRequest)
	}
	if request.RuntimeConfig != nil {
		if err := request.RuntimeConfig.Validate(); err != nil {
			return "", nil, fmt.Errorf("%w: Run RuntimeConfig snapshot is invalid", ErrInvalidRequest)
		}
	}
	metadataLabels, err := contracts.NormalizeRunMetadataLabels(request.RunMetadataLabels)
	if err != nil {
		return "", nil, fmt.Errorf("%w: Run metadata labels are invalid", ErrInvalidRequest)
	}
	bindings := make([]BindingRequirement, len(request.Bindings))
	seen := make(map[string]struct{}, len(request.Bindings))
	for index, binding := range request.Bindings {
		if strings.TrimSpace(binding.LogicalAgentName) == "" || contracts.ValidateArtifactName(binding.Namespace) != nil {
			return "", nil, fmt.Errorf("%w: binding name and valid Artifact namespace are required", ErrInvalidRequest)
		}
		if binding.Namespace == "inputs" || binding.Namespace == "outputs" {
			return "", nil, fmt.Errorf("%w: Agent binding cannot use a Run-reserved namespace", ErrInvalidRequest)
		}
		if err := binding.AgentTemplate.Validate(); err != nil {
			return "", nil, fmt.Errorf("%w: invalid AgentTemplate for %q: %v", ErrInvalidRequest, binding.LogicalAgentName, err)
		}
		if err := binding.WorkerSessionMode.Validate(); err != nil {
			return "", nil, fmt.Errorf("%w: invalid Worker session mode for %q", ErrInvalidRequest, binding.LogicalAgentName)
		}
		resolvedSkills := binding.ResolvedSkills
		if resolvedSkills == nil && len(binding.AgentTemplate.Skills) == 0 {
			resolvedSkills = []contracts.ResolvedSkill{}
		}
		if err := contracts.ValidateResolvedSkills(binding.AgentTemplate, resolvedSkills); err != nil {
			return "", nil, fmt.Errorf("%w: invalid resolved Skills for %q: %v", ErrInvalidRequest, binding.LogicalAgentName, err)
		}
		if err := binding.ExecutionConfig.Validate(); err != nil {
			return "", nil, fmt.Errorf("%w: invalid execution config for %q: %v", ErrInvalidRequest, binding.LogicalAgentName, err)
		}
		if binding.Workspace != nil {
			if err := binding.Workspace.Validate(); err != nil {
				return "", nil, fmt.Errorf("%w: invalid workspace for %q: %v", ErrInvalidRequest, binding.LogicalAgentName, err)
			}
		}
		if (request.RuntimeConfig == nil) != (binding.RuntimeSelection == nil) {
			return "", nil, fmt.Errorf("%w: candidate Runtime selection is incomplete", ErrInvalidRequest)
		}
		if binding.RuntimeSelection != nil {
			if err := validateRuntimeSelection(*binding.RuntimeSelection); err != nil {
				return "", nil, fmt.Errorf("%w: invalid Runtime selection for %q", ErrInvalidRequest, binding.LogicalAgentName)
			}
		}
		if _, duplicate := seen[binding.LogicalAgentName]; duplicate {
			return "", nil, fmt.Errorf("%w: duplicate logical Agent name", ErrInvalidRequest)
		}
		seen[binding.LogicalAgentName] = struct{}{}
		bindings[index] = BindingRequirement{
			LogicalAgentName: binding.LogicalAgentName, Namespace: binding.Namespace,
			WorkerSessionMode: binding.WorkerSessionMode,
			AgentTemplate:     cloneAgentTemplate(binding.AgentTemplate),
			ResolvedSkills:    contracts.CloneResolvedSkills(resolvedSkills),
			ExecutionConfig:   cloneAllocationExecutionConfig(binding.ExecutionConfig),
			RuntimeSelection:  cloneRuntimeSelection(binding.RuntimeSelection),
			Workspace:         contracts.CloneAllocationWorkspaceSpecV2(binding.Workspace),
		}
	}
	sort.Slice(bindings, func(i, j int) bool { return bindings[i].LogicalAgentName < bindings[j].LogicalAgentName })
	encoded, err := json.Marshal(struct {
		RunID             string
		StageExecutionID  string
		RunMetadataLabels contracts.RunMetadataLabels
		Bindings          []BindingRequirement
		RuntimeConfig     *runtimeconfig.RunSnapshot
	}{
		RunID:             request.RunID,
		StageExecutionID:  request.StageExecutionID,
		RunMetadataLabels: metadataLabels,
		Bindings:          bindings,
		RuntimeConfig:     cloneRunSnapshot(request.RuntimeConfig),
	})
	if err != nil {
		return "", nil, fmt.Errorf("encode reservation request: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), bindings, nil
}

func normalizeRegistration(source contracts.AgentRegistrationV2) contracts.AgentRegistrationV2 {
	return contracts.NormalizeAgentRegistrationV2(source)
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

// BeginPrincipalDeletion establishes an in-memory exclusion point without
// running SQL under the Registry mutex. The returned release function must be
// called after the durable transaction finishes, successfully or otherwise.
func (r *InMemoryRegistry) BeginPrincipalDeletion(runtimeAgentID string) (func(), error) {
	if err := validateAuthenticatedPrincipal(AuthenticatedPrincipal{
		RuntimeAgentID: runtimeAgentID, Labels: []string{}, LabelRevision: 1,
	}); err != nil {
		return nil, err
	}
	r.mu.Lock()
	if _, exists := r.principalDeletions[runtimeAgentID]; exists {
		r.mu.Unlock()
		return nil, ErrRegistrationConflict
	}
	monotonicNow := r.monotonicNow()
	for _, entry := range r.agents {
		if entry.principal.RuntimeAgentID != runtimeAgentID {
			continue
		}
		r.expireEntry(entry, monotonicNow)
		if entry.authoritativeAllocationID != nil || principalInstanceIsLive(entry, monotonicNow) {
			r.mu.Unlock()
			return nil, ErrRegistrationConflict
		}
	}
	r.principalDeletions[runtimeAgentID] = struct{}{}
	r.mu.Unlock()

	var once sync.Once
	return func() {
		once.Do(func() {
			r.mu.Lock()
			delete(r.principalDeletions, runtimeAgentID)
			r.mu.Unlock()
		})
	}, nil
}

// ApplyPrincipalLabels advances the process-local placement snapshot after a
// durable principal CAS. Active reservations retain their separately pinned
// Runtime configuration and label revision.
func (r *InMemoryRegistry) ApplyPrincipalLabels(principal AuthenticatedPrincipal) error {
	if err := validateAuthenticatedPrincipal(principal); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, entry := range r.agents {
		if entry.principal.RuntimeAgentID != principal.RuntimeAgentID {
			continue
		}
		if entry.principal.LabelRevision > principal.LabelRevision {
			continue
		}
		if entry.principal.LabelRevision == principal.LabelRevision &&
			equalAuthenticatedPrincipal(entry.principal, principal) {
			continue
		}
		entry.principal = clonePrincipal(principal)
		r.recordOperationsChangeLocked(OperationsRuntimeAgent, entry.registration.InstanceID)
	}
	return nil
}

func equalAuthenticatedPrincipal(left, right AuthenticatedPrincipal) bool {
	if left.RuntimeAgentID != right.RuntimeAgentID || left.LabelRevision != right.LabelRevision ||
		len(left.Labels) != len(right.Labels) {
		return false
	}
	for index := range left.Labels {
		if left.Labels[index] != right.Labels[index] {
			return false
		}
	}
	return true
}

// PrincipalRuntimeObservation returns only a current live or allocation-bound
// process incarnation. Expired unallocated registry history is not projected
// as durable liveness.
func (r *InMemoryRegistry) PrincipalRuntimeObservation(
	runtimeAgentID string,
) (*RuntimeAgentObservation, bool) {
	if err := validateAuthenticatedPrincipal(AuthenticatedPrincipal{
		RuntimeAgentID: runtimeAgentID, Labels: []string{}, LabelRevision: 1,
	}); err != nil {
		return nil, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	monotonicNow := r.monotonicNow()
	var selected *agentEntry
	for _, entry := range r.agents {
		if entry.principal.RuntimeAgentID != runtimeAgentID {
			continue
		}
		r.expireEntry(entry, monotonicNow)
		if entry.authoritativeAllocationID == nil && !principalInstanceIsLive(entry, monotonicNow) {
			continue
		}
		if selected == nil || selected.lastSeenAt.Before(entry.lastSeenAt) {
			selected = entry
		}
	}
	if selected == nil {
		return nil, false
	}
	result := runtimeObservation(selected)
	return &result, true
}

func sameRuntimeEndpoint(left, right contracts.AgentRegistrationV2) bool {
	return left.ControlURL == right.ControlURL || left.A2AURL == right.A2AURL
}

func cloneRegistration(source contracts.AgentRegistrationV2) contracts.AgentRegistrationV2 {
	result := source
	result.AllocationID = cloneString(source.AllocationID)
	result.InitialLabels = append([]string{}, source.InitialLabels...)
	result.SupportedRuntimes = append([]string{}, source.SupportedRuntimes...)
	result.SupportedSandboxProfiles = append([]string{}, source.SupportedSandboxProfiles...)
	result.SupportedRuntimeAdapters = append([]contracts.RuntimeAdapterRef{}, source.SupportedRuntimeAdapters...)
	if source.WorkspaceCapabilities != nil {
		capabilities := *source.WorkspaceCapabilities
		capabilities.Modes = append([]contracts.WorkspaceModeV2{}, source.WorkspaceCapabilities.Modes...)
		result.WorkspaceCapabilities = &capabilities
	}
	result.SupportedToolsets = make([]contracts.ToolsetCapability, len(source.SupportedToolsets))
	for index, capability := range source.SupportedToolsets {
		result.SupportedToolsets[index] = capability
		result.SupportedToolsets[index].Tools = append([]string(nil), capability.Tools...)
	}
	return result
}

func clonePrincipal(source AuthenticatedPrincipal) AuthenticatedPrincipal {
	result := source
	result.Labels = append([]string{}, source.Labels...)
	return result
}

func legacyPrincipal(instanceID string) AuthenticatedPrincipal {
	sum := sha256.Sum256([]byte("in-process-runtime-principal\x00" + instanceID))
	return AuthenticatedPrincipal{
		RuntimeAgentID: hex.EncodeToString(sum[:]), Labels: []string{}, LabelRevision: 1,
	}
}

func legacyRegistrationProjection(source contracts.AgentRegistrationV2) contracts.AgentRegistration {
	return contracts.AgentRegistration{
		APIVersion: source.APIVersion, InstanceID: source.InstanceID,
		SoftwareVersion: source.SoftwareVersion, StartedAt: source.StartedAt,
		ControlURL: source.ControlURL, A2AURL: source.A2AURL,
		SupportedRuntimes:        append([]string(nil), source.SupportedRuntimes...),
		SupportedToolsets:        cloneRegistration(source).SupportedToolsets,
		SupportedSandboxProfiles: append([]string(nil), source.SupportedSandboxProfiles...),
		ObservedState:            source.ObservedState, AllocationID: cloneString(source.AllocationID),
	}
}

func validateAuthenticatedPrincipal(principal AuthenticatedPrincipal) error {
	probe := contracts.AgentRegistrationResponseV2{
		APIVersion: contracts.APIVersion, PrivateProtocolVersion: contracts.PrivateProtocolVersionV2,
		RuntimeAgentID: principal.RuntimeAgentID, Labels: principal.Labels,
		LabelRevision: principal.LabelRevision, HeartbeatIntervalSeconds: 1, ConfirmedLeaseSeconds: 2,
	}
	if err := probe.Validate(); err != nil {
		return fmt.Errorf("%w: authenticated Runtime Agent principal is invalid", ErrInvalidRequest)
	}
	return nil
}

func cloneHeartbeatResponse(source contracts.HeartbeatResponse) contracts.HeartbeatResponse {
	result := source
	result.AllocationID = cloneString(source.AllocationID)
	return result
}

func cloneReservation(source Reservation) Reservation {
	result := source
	result.AgentTemplate = cloneAgentTemplate(source.AgentTemplate)
	result.ResolvedSkills = contracts.CloneResolvedSkills(source.ResolvedSkills)
	result.ExecutionConfig = cloneAllocationExecutionConfig(source.ExecutionConfig)
	result.Workspace = contracts.CloneAllocationWorkspaceSpecV2(source.Workspace)
	result.RunMetadataLabels = source.RunMetadataLabels.Clone()
	if source.ResolvedRuntimeConfig != nil {
		resolved := source.ResolvedRuntimeConfig.Clone()
		result.ResolvedRuntimeConfig = &resolved
	}
	return result
}

func validateRuntimeSelection(value workflowconfig.ResolvedConsumerExecutionConfig) error {
	if err := value.ModelPolicy.Validate(); err != nil || strings.TrimSpace(value.Origins.ModelPolicy) == "" {
		return ErrInvalidRequest
	}
	if value.LLMGateway != nil {
		if err := value.LLMGateway.Validate(); err != nil || strings.TrimSpace(value.Origins.LLMGateway) == "" {
			return ErrInvalidRequest
		}
	}
	if value.Credential != nil {
		if err := value.Credential.Validate(); err != nil || strings.TrimSpace(value.Origins.Credential) == "" {
			return ErrInvalidRequest
		}
	}
	return nil
}

func cloneRuntimeSelection(
	source *workflowconfig.ResolvedConsumerExecutionConfig,
) *workflowconfig.ResolvedConsumerExecutionConfig {
	if source == nil {
		return nil
	}
	result := *source
	result.ModelPolicy = cloneModelPolicy(source.ModelPolicy)
	if source.LLMGateway != nil {
		gateway := *source.LLMGateway
		if source.LLMGateway.CredentialManager != nil {
			manager := *source.LLMGateway.CredentialManager
			gateway.CredentialManager = &manager
		}
		result.LLMGateway = &gateway
	}
	if source.Credential != nil {
		credential := *source.Credential
		result.Credential = &credential
	}
	return &result
}

func cloneRunSnapshot(source *runtimeconfig.RunSnapshot) *runtimeconfig.RunSnapshot {
	if source == nil {
		return nil
	}
	result := source.Clone()
	return &result
}

func cloneAgentTemplate(source contracts.ResolvedAgentTemplate) contracts.ResolvedAgentTemplate {
	result := source
	result.ModelPolicy = cloneModelPolicy(source.ModelPolicy)
	if source.Summarizer != nil {
		summarizer := *source.Summarizer
		summarizer.ModelPolicy = cloneModelPolicy(source.Summarizer.ModelPolicy)
		summarizer.CumulativeBudget = cloneIntPointer(source.Summarizer.CumulativeBudget)
		result.Summarizer = &summarizer
	}
	result.Toolsets = make([]contracts.ToolsetSelection, len(source.Toolsets))
	for index, selection := range source.Toolsets {
		result.Toolsets[index] = selection
		result.Toolsets[index].Tools = append([]string(nil), selection.Tools...)
	}
	result.Skills = make([]contracts.ArtifactRef, len(source.Skills))
	for index, skill := range source.Skills {
		result.Skills[index] = skill
		if skill.Revision != nil {
			revision := *skill.Revision
			result.Skills[index].Revision = &revision
		}
	}
	return result
}

func cloneIntPointer(source *int) *int {
	if source == nil {
		return nil
	}
	value := *source
	return &value
}

func cloneModelPolicy(source contracts.ResolvedModelPolicy) contracts.ResolvedModelPolicy {
	result := source
	if source.Temperature != nil {
		temperature := *source.Temperature
		result.Temperature = &temperature
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

func cloneCredentialRef(source *contracts.LLMCredentialRef) *contracts.LLMCredentialRef {
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
