package controlplane

import (
	"fmt"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	"github.com/grauwolf32/contractor/internal/helpers/clone"
)

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

// liveCandidates returns detached snapshots of every slot that placement
// could use once its current allocation, if any, is released. Only a model
// Stage hold reads them (CTL-R155); they grant no capacity.
func (r *InMemoryRegistry) liveCandidates() []AgentSnapshot {
	monotonicNow := r.monotonicNow()
	r.mu.Lock()
	defer r.mu.Unlock()
	result := make([]AgentSnapshot, 0, len(r.agents))
	for _, entry := range r.agents {
		r.expireEntry(entry, monotonicNow)
		if isCapacityLive(entry, monotonicNow) {
			result = append(result, snapshotAgent(entry))
		}
	}
	return result
}

// holdForModelStage keeps every live slot that the model Stage of request
// could use away from model-free placement for the hold window (CTL-R155).
// edges are resolved for live slots, idle or allocated, exactly as placement
// resolves them. A Stage with no complete matching among live slots, which
// could not be placed even with every allocation released, holds nothing.
func (r *InMemoryRegistry) holdForModelStage(request ReservationRequest, edges []CandidateEdge) error {
	if request.ModelFree {
		return fmt.Errorf("%w: a model-free Stage holds no Agent", ErrInvalidRequest)
	}
	_, bindings, err := normalizeReservationRequest(request)
	if err != nil {
		return err
	}
	if err := validateCandidateEdges(bindings, edges); err != nil {
		return err
	}
	monotonicNow := r.monotonicNow()
	r.mu.Lock()
	defer r.mu.Unlock()
	live := make([]*agentEntry, 0, len(r.agents))
	for _, entry := range r.agents {
		r.expireEntry(entry, monotonicNow)
		if isCapacityLive(entry, monotonicNow) && contracts.PlacementLabelsMatch(request.RequiredPlacementLabels, entry.registration.PlacementLabels) {
			live = append(live, entry)
		}
	}
	if len(bindings) > len(live) {
		return nil
	}
	candidates, complete := buildCapabilityCandidates(live, bindings, edges)
	if !complete {
		return nil
	}
	if _, complete := matchCapabilityCandidates(candidates, len(live)); !complete {
		return nil
	}
	heldUntil := monotonicNow + r.modelStageHold
	for _, agentIndexes := range candidates {
		for _, agentIndex := range agentIndexes {
			live[agentIndex].modelStageHeldUntil = max(live[agentIndex].modelStageHeldUntil, heldUntil)
		}
	}
	return nil
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
	runMetadataLabels, err := runlabels.NormalizeRunMetadataLabels(request.RunMetadataLabels)
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
		if isPlacementEligible(entry, monotonicNow) &&
			contracts.PlacementLabelsMatch(request.RequiredPlacementLabels, entry.registration.PlacementLabels) &&
			(!request.ModelFree || entry.modelStageHeldUntil <= monotonicNow) {
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
			SitemapPolicy: binding.AgentTemplate.Sitemap.Clone(),
		}
		for _, selection := range binding.AgentTemplate.Toolsets {
			if selection.Ref == (contracts.ToolsetRef{ToolsetID: "audit-results", Version: "2"}) && slices.Contains(selection.Tools, "submit_check_result") {
				grant.HTTPResultEvidence = true
			}
			if selection.Ref == (contracts.ToolsetRef{ToolsetID: "security-findings", Version: "1"}) {
				grant.FindingTools = append([]string(nil), selection.Tools...)
			}
			if selection.Ref == (contracts.ToolsetRef{ToolsetID: "sitemap", Version: "1"}) {
				grant.SitemapTools = append([]string(nil), selection.Tools...)
			}
		}
		grant.ProjectArtifactReads = projectArtifactReadTargets(grant)
		// normalizeReservationRequest detached these binding fields; the Registry
		// takes ownership, then cloneReservation detaches the returned value.
		reservation := Reservation{
			PlacementLabels:        append([]string{}, entry.registration.PlacementLabels...),
			CompletionContract:     binding.CompletionContract,
			CompletionCapabilities: control.NormalizeAgentRegistration(entry.registration).Capabilities,
			Grant:                  grant, ControlURL: entry.registration.ControlURL, A2AURL: entry.registration.A2AURL,
			AgentTemplate:             binding.AgentTemplate,
			WorkerSessionMode:         binding.WorkerSessionMode,
			ContextManagement:         binding.ContextManagement,
			ResolvedSkills:            binding.ResolvedSkills,
			ExecutionConfig:           binding.ExecutionConfig,
			Workspace:                 binding.Workspace,
			RunMetadataLabels:         runMetadataLabels.Clone(),
			RuntimeAgentLabelRevision: entry.principal.LabelRevision,
			LeaseExpiresAt:            entry.confirmedLeaseExpiresAt,
			initialLeaseExpiresAt:     entry.confirmedLeaseExpiresAt,
		}
		entry.authoritativeAllocationID = clone.Pointer(&allocationID)
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

// reservationsForRequest reconciles an ambiguous durable placement without
// resolving mutable dependencies again, while checking its immutable request.
func (r *InMemoryRegistry) reservationsForRequest(request ReservationRequest) ([]Reservation, error) {
	fingerprint, _, err := normalizeReservationRequest(request)
	if err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	existing, ok := r.stageReservations[request.StageExecutionID]
	if !ok {
		return nil, ErrAllocationNotFound
	}
	if existing.fingerprint != fingerprint || !existing.committed {
		return nil, ErrReservationConflict
	}
	if existing.released {
		return nil, ErrReservationReleased
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
		runtimeConfig := configurations[allocationID].Resolved.Clone()
		stored.reservation.ExecutionConfig = AllocationExecutionConfig{
			ModelPolicy: runtimeConfig.ModelPolicy.Ref,
			LLMGateway:  runtimeConfig.LLMGateway.Ref,
			Credential:  clone.Pointer(runtimeConfig.LLMCredential),
		}
		stored.reservation.ResolvedRuntimeConfig = &runtimeConfig
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
		r.removeAllocationAuthorityLocked(allocationID, entry)
	}
	delete(r.stageReservations, stageExecutionID)
	return nil
}

// DiscardUnpreparedReservations removes a committed Stage reservation that no
// Runtime Agent was asked to prepare, as when Queue Pause defers admission of
// a never admitted Stage. Like a discarded candidate the Runtime never
// observed a prepared allocation, so its slot needs no reconciliation
// heartbeat, and no released tombstone is kept: placement for the same
// StageExecution can run again. The caller guarantees that no prepare request
// was sent. A batch that is activated, fenced, lost or past preparing is not
// discarded. Each allocation's write gate is held exclusively, as Release
// does, so no earlier private write can commit afterward.
func (r *InMemoryRegistry) DiscardUnpreparedReservations(stageExecutionID string) error {
	r.mu.Lock()
	existing, ok := r.stageReservations[stageExecutionID]
	if !ok {
		r.mu.Unlock()
		return nil
	}
	if !existing.committed || existing.released {
		r.mu.Unlock()
		return ErrReservationConflict
	}
	allocationIDs := append([]string(nil), existing.allocationIDs...)
	sort.Strings(allocationIDs)
	gates := make([]*sync.RWMutex, 0, len(allocationIDs))
	for _, allocationID := range allocationIDs {
		stored, present := r.allocations[allocationID]
		if !present || stored.writeGate == nil {
			r.mu.Unlock()
			return ErrReservationConflict
		}
		gates = append(gates, stored.writeGate)
	}
	r.mu.Unlock()

	for _, gate := range gates {
		gate.Lock()
		defer gate.Unlock()
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	current, ok := r.stageReservations[stageExecutionID]
	if !ok || !current.committed || current.released || len(current.allocationIDs) != len(allocationIDs) {
		return ErrReservationConflict
	}
	for index, allocationID := range allocationIDs {
		stored, present := r.allocations[allocationID]
		if !present || stored.writeGate != gates[index] || stored.loss != nil ||
			stored.reservation.Grant.WriteFenced || stored.reservation.Grant.Lost ||
			stored.phase != AllocationPreparing {
			return ErrReservationConflict
		}
		entry, present := r.agents[stored.reservation.Grant.RuntimeInstanceID]
		if !present || entry.authoritativeAllocationID == nil ||
			*entry.authoritativeAllocationID != allocationID || entry.allocationActivated {
			return ErrReservationConflict
		}
	}
	for _, allocationID := range allocationIDs {
		stored := r.allocations[allocationID]
		entry := r.agents[stored.reservation.Grant.RuntimeInstanceID]
		r.removeAllocationAuthorityLocked(allocationID, entry)
	}
	delete(r.stageReservations, stageExecutionID)
	return nil
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

func isPlacementEligible(entry *agentEntry, monotonicNow time.Duration) bool {
	return entry.authoritativeAllocationID == nil && !entry.reconciliationRequired &&
		!entry.leaseExpired && !entry.superseded && !entry.principalMissing && entry.blockedByInstanceID == nil &&
		entry.registration.ObservedState == control.AgentIdle &&
		entry.confirmedLeaseDeadline > monotonicNow
}

// isCapacityLive reports whether entry is a slot that placement could use
// once its current allocation, if any, is released.
func isCapacityLive(entry *agentEntry, monotonicNow time.Duration) bool {
	return !entry.leaseExpired && !entry.superseded && !entry.principalMissing &&
		entry.blockedByInstanceID == nil && entry.confirmedLeaseDeadline > monotonicNow
}
