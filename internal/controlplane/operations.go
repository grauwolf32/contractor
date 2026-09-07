package controlplane

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	maximumOperationsItems        = 10_000
	maximumRuntimeCapabilityRefs  = 128
	maximumRuntimeToolsetRefs     = 128
	maximumRuntimeToolsPerToolset = 256
	operationsHistoryLimit        = 255
)

var (
	operationsResourceIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$`)
	operationsConfigIDPattern   = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`)
	softwareVersionPattern      = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$`)
	runtimeCapabilityRefPattern = regexp.MustCompile(`^[a-z][a-z0-9_-]*@[A-Za-z0-9][A-Za-z0-9._+-]*$`)
)

type SlotState string

const (
	SlotIdle     SlotState = "idle"
	SlotReserved SlotState = "reserved"
	SlotBusy     SlotState = "busy"
	SlotDraining SlotState = "draining"
	SlotFenced   SlotState = "fenced"
)

type AllocationAuthoritativePhase string

const (
	AllocationPreparing  AllocationAuthoritativePhase = "preparing"
	AllocationActive     AllocationAuthoritativePhase = "active"
	AllocationFinalizing AllocationAuthoritativePhase = "finalizing"
	AllocationAborting   AllocationAuthoritativePhase = "aborting"
	AllocationReleasing  AllocationAuthoritativePhase = "releasing"
)

type AllocationObservedPhase string

const (
	AllocationObservedAbsent   AllocationObservedPhase = "absent"
	AllocationObservedPrepared AllocationObservedPhase = "prepared"
	AllocationObservedDraining AllocationObservedPhase = "draining"
	AllocationObservedFenced   AllocationObservedPhase = "fenced"
)

type SafeReason struct {
	Code      string `json:"code"`
	Retryable bool   `json:"retryable"`
}

// MetricsSummary is the only Runtime report projection retained by the live
// Control Plane registry. It deliberately has no participant IDs, tool names,
// arguments, messages, provider fields, or URLs.
type MetricsSummary struct {
	ReportsComplete bool  `json:"reportsComplete"`
	ModelCalls      int64 `json:"modelCalls"`
	InputTokens     int64 `json:"inputTokens"`
	OutputTokens    int64 `json:"outputTokens"`
	TotalTokens     int64 `json:"totalTokens"`
	ToolCalls       int64 `json:"toolCalls"`
	ToolFailures    int64 `json:"toolFailures"`
	ErrorCount      int64 `json:"errorCount"`
	Truncated       bool  `json:"truncated"`
}

// AllocationExecutionConfig contains only exact, non-secret refs. It is
// captured with the reservation so Operations never reconstructs authority
// from a mutable configuration catalog.
type AllocationExecutionConfig struct {
	ModelPolicy contracts.ModelPolicyRef      `json:"modelPolicy"`
	LLMGateway  contracts.LLMGatewayConfigRef `json:"llmGateway"`
	Credential  *contracts.LLMCredentialRef   `json:"credential,omitempty"`
}

func (c AllocationExecutionConfig) Validate() error {
	if err := c.ModelPolicy.ValidateRef(); err != nil {
		return fmt.Errorf("invalid allocation ModelPolicy ref: %w", err)
	}
	if err := c.LLMGateway.ValidateRef(); err != nil {
		return fmt.Errorf("invalid allocation LLMGatewayConfig ref: %w", err)
	}
	if c.Credential != nil {
		if err := c.Credential.Validate(); err != nil {
			return fmt.Errorf("invalid allocation credential ref: %w", err)
		}
	}
	return nil
}

type OperationsCursor struct {
	Generation string `json:"generation"`
	Revision   uint64 `json:"-"`
}

type OperationsResource string

const (
	OperationsRuntimeAgent      OperationsResource = "runtimeAgent"
	OperationsAllocation        OperationsResource = "allocation"
	OperationsConfiguration     OperationsResource = "configuration"
	OperationsCredential        OperationsResource = "credential"
	OperationsSchedulerSettings OperationsResource = "schedulerSettings"
)

// OperationsChange is a reduced process-local invalidation. SnapshotOperations
// remains authoritative; the change only identifies a query family to refresh
// at one exact cursor revision.
type OperationsChange struct {
	Cursor     OperationsCursor
	Resource   OperationsResource
	ResourceID string
	OccurredAt time.Time
}

func (c OperationsCursor) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Generation string `json:"generation"`
		Revision   string `json:"revision"`
	}{Generation: c.Generation, Revision: strconv.FormatUint(c.Revision, 10)})
}

type RuntimeAgentObservation struct {
	InstanceID                string                           `json:"instanceId"`
	SoftwareVersion           string                           `json:"softwareVersion"`
	SupportedRuntimes         []string                         `json:"supportedRuntimes"`
	SupportedToolsets         []RuntimeToolsetCapability       `json:"supportedToolsets"`
	SupportedSandboxProfiles  []string                         `json:"supportedSandboxProfiles"`
	SupportedRuntimeAdapters  []string                         `json:"supportedRuntimeAdapters"`
	WorkspaceCapabilities     *contracts.WorkspaceCapabilities `json:"workspaceCapabilities,omitempty"`
	ObservedState             contracts.AgentObservedState     `json:"observedState"`
	SlotState                 SlotState                        `json:"slotState"`
	LastAcceptedHeartbeat     *time.Time                       `json:"lastAcceptedHeartbeat,omitempty"`
	ConfirmedLeaseUntil       *time.Time                       `json:"confirmedLeaseUntil,omitempty"`
	CurrentAllocationID       *string                          `json:"currentAllocationId,omitempty"`
	AuthoritativeAllocationID *string                          `json:"authoritativeAllocationId,omitempty"`
	ReconciliationReason      *SafeReason                      `json:"reconciliationReason,omitempty"`
}

type RuntimeToolsetCapability struct {
	Ref   string   `json:"ref"`
	Tools []string `json:"tools"`
}

type AllocationObservation struct {
	AllocationID           string                       `json:"allocationId"`
	RunID                  string                       `json:"runId"`
	StageExecutionID       string                       `json:"stageExecutionId"`
	RuntimeAgentInstanceID string                       `json:"runtimeAgentInstanceId"`
	LogicalWorker          string                       `json:"logicalWorker"`
	AgentTemplate          contracts.AgentTemplateRef   `json:"agentTemplate"`
	ExecutionConfig        AllocationExecutionConfig    `json:"executionConfig"`
	AuthoritativePhase     AllocationAuthoritativePhase `json:"authoritativePhase"`
	ObservedPhase          AllocationObservedPhase      `json:"observedPhase"`
	Reason                 *SafeReason                  `json:"reason,omitempty"`
	Metrics                MetricsSummary               `json:"metrics"`
	ExhaustedDimension     *string                      `json:"exhaustedDimension,omitempty"`
}

type OperationsSnapshot struct {
	Cursor        OperationsCursor          `json:"cursor"`
	RuntimeAgents []RuntimeAgentObservation `json:"runtimeAgents"`
	Allocations   []AllocationObservation   `json:"allocations"`
}

func (s OperationsSnapshot) Validate() error {
	if !operationsResourceIDPattern.MatchString(s.Cursor.Generation) ||
		len(s.RuntimeAgents) > maximumOperationsItems || len(s.Allocations) > maximumOperationsItems {
		return fmt.Errorf("invalid Operations cursor or collection bound")
	}
	seenAgents := make(map[string]struct{}, len(s.RuntimeAgents))
	for _, agent := range s.RuntimeAgents {
		if err := agent.Validate(); err != nil {
			return err
		}
		if _, duplicate := seenAgents[agent.InstanceID]; duplicate {
			return fmt.Errorf("duplicate Runtime Agent observation")
		}
		seenAgents[agent.InstanceID] = struct{}{}
	}
	seenAllocations := make(map[string]struct{}, len(s.Allocations))
	for _, allocation := range s.Allocations {
		if err := allocation.Validate(); err != nil {
			return err
		}
		if _, duplicate := seenAllocations[allocation.AllocationID]; duplicate {
			return fmt.Errorf("duplicate allocation observation")
		}
		seenAllocations[allocation.AllocationID] = struct{}{}
	}
	return nil
}

func (o RuntimeAgentObservation) Validate() error {
	if !operationsResourceIDPattern.MatchString(o.InstanceID) ||
		!softwareVersionPattern.MatchString(o.SoftwareVersion) || o.LastAcceptedHeartbeat != nil &&
		o.LastAcceptedHeartbeat.IsZero() || o.ConfirmedLeaseUntil != nil && o.ConfirmedLeaseUntil.IsZero() {
		return fmt.Errorf("invalid Runtime Agent observation identity or time")
	}
	if err := validateRuntimeCapabilities(o); err != nil {
		return err
	}
	switch o.ObservedState {
	case contracts.AgentIdle, contracts.AgentAllocated, contracts.AgentDraining, contracts.AgentFenced:
	default:
		return fmt.Errorf("invalid Runtime Agent observed state")
	}
	switch o.SlotState {
	case SlotIdle, SlotReserved, SlotBusy, SlotDraining, SlotFenced:
	default:
		return fmt.Errorf("invalid Runtime Agent slot state")
	}
	for _, value := range []*string{o.CurrentAllocationID, o.AuthoritativeAllocationID} {
		if value != nil && !operationsResourceIDPattern.MatchString(*value) {
			return fmt.Errorf("invalid Runtime Agent allocation identity")
		}
	}
	if o.ReconciliationReason != nil && !validSafeReason(*o.ReconciliationReason) {
		return fmt.Errorf("invalid Runtime Agent reconciliation reason")
	}
	return nil
}

func validateRuntimeCapabilities(observation RuntimeAgentObservation) error {
	if len(observation.SupportedRuntimes) == 0 ||
		len(observation.SupportedRuntimes) > maximumRuntimeCapabilityRefs ||
		len(observation.SupportedSandboxProfiles) == 0 ||
		len(observation.SupportedSandboxProfiles) > maximumRuntimeCapabilityRefs ||
		len(observation.SupportedToolsets) > maximumRuntimeToolsetRefs ||
		len(observation.SupportedRuntimeAdapters) > maximumRuntimeCapabilityRefs {
		return fmt.Errorf("invalid Runtime Agent capability collection bound")
	}
	if !validUniqueCapabilityRefs(observation.SupportedRuntimes) ||
		!validUniqueCapabilityRefs(observation.SupportedSandboxProfiles) ||
		!validUniqueCapabilityRefs(observation.SupportedRuntimeAdapters) {
		return fmt.Errorf("invalid or duplicate Runtime Agent capability ref")
	}
	if observation.WorkspaceCapabilities != nil && observation.WorkspaceCapabilities.Validate() != nil {
		return fmt.Errorf("invalid Runtime Agent workspace capability")
	}
	seenToolsets := make(map[string]struct{}, len(observation.SupportedToolsets))
	for _, capability := range observation.SupportedToolsets {
		if !validRuntimeCapabilityRef(capability.Ref) {
			return fmt.Errorf("invalid Runtime Agent Toolset capability ref")
		}
		if _, duplicate := seenToolsets[capability.Ref]; duplicate {
			return fmt.Errorf("duplicate Runtime Agent Toolset capability ref")
		}
		seenToolsets[capability.Ref] = struct{}{}
		if len(capability.Tools) == 0 || len(capability.Tools) > maximumRuntimeToolsPerToolset {
			return fmt.Errorf("invalid Runtime Agent Toolset tool collection bound")
		}
		seenTools := make(map[string]struct{}, len(capability.Tools))
		for _, tool := range capability.Tools {
			if !operationsConfigIDPattern.MatchString(tool) {
				return fmt.Errorf("invalid Runtime Agent tool capability name")
			}
			if _, duplicate := seenTools[tool]; duplicate {
				return fmt.Errorf("duplicate Runtime Agent tool capability name")
			}
			seenTools[tool] = struct{}{}
		}
	}
	return nil
}

func validUniqueCapabilityRefs(values []string) bool {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if !validRuntimeCapabilityRef(value) {
			return false
		}
		if _, duplicate := seen[value]; duplicate {
			return false
		}
		seen[value] = struct{}{}
	}
	return true
}

func validRuntimeCapabilityRef(value string) bool {
	return len(value) <= 256 && runtimeCapabilityRefPattern.MatchString(value)
}

func (o AllocationObservation) Validate() error {
	for _, value := range []string{
		o.AllocationID, o.RunID, o.StageExecutionID, o.RuntimeAgentInstanceID,
	} {
		if !operationsResourceIDPattern.MatchString(value) {
			return fmt.Errorf("invalid allocation observation identity")
		}
	}
	if !operationsConfigIDPattern.MatchString(o.LogicalWorker) ||
		o.AgentTemplate.ValidateRef() != nil || o.ExecutionConfig.Validate() != nil ||
		!validAllocationPhase(o.AuthoritativePhase) || !validObservedPhase(o.ObservedPhase) ||
		o.Reason != nil && !validSafeReason(*o.Reason) || !validMetricsSummary(o.Metrics) ||
		o.ExhaustedDimension != nil && !validExhaustedDimension(*o.ExhaustedDimension) {
		return fmt.Errorf("invalid allocation observation body")
	}
	return nil
}

type storedReservation struct {
	reservation        Reservation
	writeGate          *sync.RWMutex
	loss               *AllocationLoss
	phase              AllocationAuthoritativePhase
	reason             *SafeReason
	metrics            MetricsSummary
	exhaustedDimension *string
}

func (r *InMemoryRegistry) SnapshotOperations() OperationsSnapshot {
	r.mu.Lock()
	defer r.mu.Unlock()

	result := OperationsSnapshot{
		Cursor:        OperationsCursor{Generation: r.operationsGeneration, Revision: r.operationsRevision},
		RuntimeAgents: make([]RuntimeAgentObservation, 0, len(r.agents)),
		Allocations:   make([]AllocationObservation, 0, len(r.allocations)),
	}
	for _, entry := range r.agents {
		if entry.superseded && entry.authoritativeAllocationID == nil {
			continue
		}
		result.RuntimeAgents = append(result.RuntimeAgents, runtimeObservation(entry))
	}
	sort.Slice(result.RuntimeAgents, func(left, right int) bool {
		return result.RuntimeAgents[left].InstanceID < result.RuntimeAgents[right].InstanceID
	})
	for _, stored := range r.allocations {
		result.Allocations = append(result.Allocations, allocationObservation(stored, r.agents))
	}
	sort.Slice(result.Allocations, func(left, right int) bool {
		return result.Allocations[left].AllocationID < result.Allocations[right].AllocationID
	})
	return result
}

// ReplayOperations returns every retained change strictly after the supplied
// cursor. This ring is deliberately not an audit log; callers must fetch a new
// REST snapshot for every cursor error.
func (r *InMemoryRegistry) ReplayOperations(
	after OperationsCursor,
) ([]OperationsChange, OperationsCursor, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	current := OperationsCursor{Generation: r.operationsGeneration, Revision: r.operationsRevision}
	if after.Generation != r.operationsGeneration {
		return nil, current, ErrOperationsGeneration
	}
	if after.Revision > r.operationsRevision {
		return nil, current, ErrOperationsCursor
	}
	if after.Revision == r.operationsRevision {
		return []OperationsChange{}, current, nil
	}
	want := after.Revision + 1
	if len(r.operationsHistory) == 0 || r.operationsHistory[0].Cursor.Revision > want {
		return nil, current, ErrOperationsCursor
	}
	result := make([]OperationsChange, 0, r.operationsRevision-after.Revision)
	for _, change := range r.operationsHistory {
		if change.Cursor.Revision < want {
			continue
		}
		if change.Cursor.Revision != want {
			return nil, current, ErrOperationsGap
		}
		result = append(result, change)
		want++
	}
	if want != r.operationsRevision+1 {
		return nil, current, ErrOperationsGap
	}
	return result, current, nil
}

// SubscribeOperations emits coalescing wake-up hints. ReplayOperations closes
// every race and detects dropped hints, so producers never wait for readers.
func (r *InMemoryRegistry) SubscribeOperations() (<-chan struct{}, func()) {
	r.mu.Lock()
	r.nextOperationsWatcher++
	id := r.nextOperationsWatcher
	updates := make(chan struct{}, 1)
	r.operationsWatchers[id] = updates
	r.mu.Unlock()
	var once sync.Once
	cancel := func() {
		once.Do(func() {
			r.mu.Lock()
			delete(r.operationsWatchers, id)
			close(updates)
			r.mu.Unlock()
		})
	}
	return updates, cancel
}

func (r *InMemoryRegistry) InvalidateOperations(
	resource OperationsResource,
	resourceID string,
) error {
	if !validOperationsResource(resource) ||
		resourceID != "" && !operationsResourceIDPattern.MatchString(resourceID) ||
		resource == OperationsSchedulerSettings && resourceID != "" {
		return ErrInvalidRequest
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.recordOperationsChangeLocked(resource, resourceID)
	return nil
}

func (r *InMemoryRegistry) recordOperationsChangeLocked(
	resource OperationsResource,
	resourceID string,
) {
	r.operationsRevision++
	change := OperationsChange{
		Cursor: OperationsCursor{
			Generation: r.operationsGeneration,
			Revision:   r.operationsRevision,
		},
		Resource: resource, ResourceID: resourceID,
		OccurredAt: r.now().UTC().Round(0),
	}
	r.operationsHistory = append(r.operationsHistory, change)
	if len(r.operationsHistory) > operationsHistoryLimit {
		r.operationsHistory = append(
			[]OperationsChange(nil),
			r.operationsHistory[len(r.operationsHistory)-operationsHistoryLimit:]...,
		)
	}
	for _, watcher := range r.operationsWatchers {
		select {
		case watcher <- struct{}{}:
		default:
		}
	}
}

func validOperationsResource(resource OperationsResource) bool {
	switch resource {
	case OperationsRuntimeAgent, OperationsAllocation, OperationsConfiguration, OperationsCredential,
		OperationsSchedulerSettings:
		return true
	default:
		return false
	}
}

func (r *InMemoryRegistry) SetAllocationPhase(
	allocationID string,
	phase AllocationAuthoritativePhase,
	reason *SafeReason,
) error {
	if !validAllocationPhase(phase) || reason != nil && !validSafeReason(*reason) {
		return ErrInvalidRequest
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	stored, ok := r.allocations[allocationID]
	if !ok {
		return ErrAllocationNotFound
	}
	if !validPhaseTransition(stored.phase, phase) {
		return ErrInvalidRequest
	}
	if stored.phase == phase && (reason == nil || stored.reason != nil && *stored.reason == *reason) {
		return nil
	}
	stored.phase = phase
	if reason != nil {
		stored.reason = cloneSafeReason(reason)
	}
	r.allocations[allocationID] = stored
	r.recordOperationsChangeLocked(OperationsAllocation, allocationID)
	return nil
}

func (r *InMemoryRegistry) RecordAllocationReport(
	allocationID string,
	report contracts.AllocationFinalReport,
) error {
	if report.AllocationID != allocationID || report.Validate() != nil {
		return ErrInvalidRequest
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	stored, ok := r.allocations[allocationID]
	if !ok {
		return ErrAllocationNotFound
	}
	summary, valid := summarizeAllocationReport(report)
	if !valid {
		return ErrInvalidRequest
	}
	stored.metrics = summary
	stored.exhaustedDimension = nil
	if report.Worker.Metrics.WorkerBudget != nil && report.Worker.Metrics.WorkerBudget.Exhausted != nil {
		value := *report.Worker.Metrics.WorkerBudget.Exhausted
		stored.exhaustedDimension = &value
	}
	if existing := r.allocations[allocationID]; existing.metrics == stored.metrics &&
		equalOptionalString(existing.exhaustedDimension, stored.exhaustedDimension) {
		return nil
	}
	r.allocations[allocationID] = stored
	r.recordOperationsChangeLocked(OperationsAllocation, allocationID)
	return nil
}

func runtimeObservation(entry *agentEntry) RuntimeAgentObservation {
	result := RuntimeAgentObservation{
		InstanceID:                entry.registration.InstanceID,
		SoftwareVersion:           entry.registration.SoftwareVersion,
		SupportedRuntimes:         append([]string{}, entry.registration.SupportedRuntimes...),
		SupportedToolsets:         make([]RuntimeToolsetCapability, len(entry.registration.SupportedToolsets)),
		SupportedSandboxProfiles:  append([]string{}, entry.registration.SupportedSandboxProfiles...),
		SupportedRuntimeAdapters:  contracts.RuntimeAdapterCapabilityProjection(entry.registration),
		WorkspaceCapabilities:     cloneWorkspaceCapabilities(entry.registration.WorkspaceCapabilities),
		ObservedState:             entry.registration.ObservedState,
		SlotState:                 slotState(entry),
		CurrentAllocationID:       cloneString(entry.registration.AllocationID),
		AuthoritativeAllocationID: cloneString(entry.authoritativeAllocationID),
	}
	for index, capability := range entry.registration.SupportedToolsets {
		result.SupportedToolsets[index] = RuntimeToolsetCapability{
			Ref: capability.Ref, Tools: append([]string{}, capability.Tools...),
		}
	}
	sort.Strings(result.SupportedRuntimes)
	sort.Strings(result.SupportedSandboxProfiles)
	sort.Strings(result.SupportedRuntimeAdapters)
	for index := range result.SupportedToolsets {
		sort.Strings(result.SupportedToolsets[index].Tools)
	}
	sort.Slice(result.SupportedToolsets, func(left, right int) bool {
		return result.SupportedToolsets[left].Ref < result.SupportedToolsets[right].Ref
	})
	if !entry.lastAcceptedHeartbeatAt.IsZero() {
		accepted := entry.lastAcceptedHeartbeatAt
		result.LastAcceptedHeartbeat = &accepted
	}
	if !entry.confirmedLeaseExpiresAt.IsZero() {
		confirmed := entry.confirmedLeaseExpiresAt
		result.ConfirmedLeaseUntil = &confirmed
	}
	if entry.reconciliationRequired {
		code := "reconciliation_required"
		if entry.leaseExpired {
			code = string(LossControlLeaseExpired)
		}
		result.ReconciliationReason = &SafeReason{Code: code, Retryable: true}
	}
	return result
}

func cloneWorkspaceCapabilities(
	source *contracts.WorkspaceCapabilities,
) *contracts.WorkspaceCapabilities {
	if source == nil {
		return nil
	}
	result := *source
	result.Modes = append([]contracts.WorkspaceMode{}, source.Modes...)
	return &result
}

func allocationObservation(
	stored storedReservation,
	agents map[string]*agentEntry,
) AllocationObservation {
	reservation := stored.reservation
	result := AllocationObservation{
		AllocationID:           reservation.Grant.AllocationID,
		RunID:                  reservation.Grant.RunID,
		StageExecutionID:       reservation.Grant.StageExecutionID,
		RuntimeAgentInstanceID: reservation.Grant.RuntimeInstanceID,
		LogicalWorker:          reservation.Grant.LogicalAgentName,
		AgentTemplate:          reservation.AgentTemplate.Ref,
		ExecutionConfig:        cloneAllocationExecutionConfig(reservation.ExecutionConfig),
		AuthoritativePhase:     stored.phase,
		ObservedPhase:          AllocationObservedAbsent,
		Reason:                 cloneSafeReason(stored.reason),
		Metrics:                stored.metrics,
		ExhaustedDimension:     cloneString(stored.exhaustedDimension),
	}
	if entry, ok := agents[reservation.Grant.RuntimeInstanceID]; ok {
		result.ObservedPhase = observedAllocationPhase(entry, reservation.Grant.AllocationID)
	}
	if stored.loss != nil {
		result.Reason = &SafeReason{Code: string(stored.loss.Reason), Retryable: true}
	}
	return result
}

func slotState(entry *agentEntry) SlotState {
	if entry.leaseExpired || entry.allocationLost || entry.registration.ObservedState == contracts.AgentFenced {
		return SlotFenced
	}
	if entry.registration.ObservedState == contracts.AgentDraining {
		return SlotDraining
	}
	if entry.reconciliationRequired {
		if entry.registration.ObservedState == contracts.AgentAllocated {
			return SlotDraining
		}
		return SlotFenced
	}
	if entry.authoritativeAllocationID == nil {
		return SlotIdle
	}
	if !entry.allocationActivated && entry.registration.ObservedState == contracts.AgentIdle {
		return SlotReserved
	}
	if entry.registration.ObservedState == contracts.AgentAllocated && entry.registration.AllocationID != nil &&
		*entry.registration.AllocationID == *entry.authoritativeAllocationID {
		return SlotBusy
	}
	return SlotFenced
}

func observedAllocationPhase(entry *agentEntry, allocationID string) AllocationObservedPhase {
	if entry.registration.ObservedState == contracts.AgentFenced {
		return AllocationObservedFenced
	}
	if entry.registration.AllocationID == nil || *entry.registration.AllocationID != allocationID {
		return AllocationObservedAbsent
	}
	switch entry.registration.ObservedState {
	case contracts.AgentAllocated:
		return AllocationObservedPrepared
	case contracts.AgentDraining:
		return AllocationObservedDraining
	default:
		return AllocationObservedAbsent
	}
}

func validAllocationPhase(phase AllocationAuthoritativePhase) bool {
	switch phase {
	case AllocationPreparing, AllocationActive, AllocationFinalizing, AllocationAborting, AllocationReleasing:
		return true
	default:
		return false
	}
}

func validObservedPhase(phase AllocationObservedPhase) bool {
	switch phase {
	case AllocationObservedAbsent, AllocationObservedPrepared, AllocationObservedDraining, AllocationObservedFenced:
		return true
	default:
		return false
	}
}

func validPhaseTransition(from, to AllocationAuthoritativePhase) bool {
	if from == to {
		return true
	}
	switch to {
	case AllocationActive:
		return from == AllocationPreparing
	case AllocationFinalizing:
		return from == AllocationPreparing || from == AllocationActive
	case AllocationAborting:
		return from == AllocationPreparing || from == AllocationActive
	case AllocationReleasing:
		return from != AllocationReleasing
	default:
		return false
	}
}

func validSafeReason(reason SafeReason) bool {
	if reason.Code == "" || len(reason.Code) > 128 {
		return false
	}
	for index := range len(reason.Code) {
		value := reason.Code[index]
		letterOrDigit := value >= 'a' && value <= 'z' || value >= '0' && value <= '9'
		if index == 0 && !letterOrDigit || index != 0 && !letterOrDigit && value != '_' && value != '-' && value != '.' {
			return false
		}
	}
	return true
}

func summarizeAllocationReport(report contracts.AllocationFinalReport) (MetricsSummary, bool) {
	result := MetricsSummary{
		ReportsComplete: report.Worker.Complete && report.Runtime.Complete,
		ErrorCount:      int64(len(report.Worker.Errors)),
		Truncated:       report.Worker.Truncated,
	}
	if !addMetric(&result.ModelCalls, report.Worker.Metrics.ModelCalls) ||
		!addMetric(&result.InputTokens, report.Worker.Metrics.InputTokens) ||
		!addMetric(&result.OutputTokens, report.Worker.Metrics.OutputTokens) ||
		!addMetric(&result.TotalTokens, report.Worker.Metrics.TotalTokens) {
		return MetricsSummary{}, false
	}
	for _, tool := range report.Worker.Metrics.Tools {
		if !addMetric(&result.ToolCalls, tool.Calls) || !addMetric(&result.ToolFailures, tool.Failed) {
			return MetricsSummary{}, false
		}
	}
	return result, true
}

func validMetricsSummary(summary MetricsSummary) bool {
	return summary.ModelCalls >= 0 && summary.InputTokens >= 0 && summary.OutputTokens >= 0 &&
		summary.TotalTokens >= 0 && summary.ToolCalls >= 0 && summary.ToolFailures >= 0 &&
		summary.ErrorCount >= 0
}

func validExhaustedDimension(value string) bool {
	switch value {
	case "model_calls", "tool_calls", "total_tokens", "worker_calls", "wall_time":
		return true
	default:
		return false
	}
}

func addMetric(target *int64, value *int64) bool {
	if value == nil {
		return true
	}
	if *value < 0 || *target > math.MaxInt64-*value {
		return false
	}
	*target += *value
	return true
}

func cloneSafeReason(source *SafeReason) *SafeReason {
	if source == nil {
		return nil
	}
	result := *source
	return &result
}

func cloneAllocationExecutionConfig(source AllocationExecutionConfig) AllocationExecutionConfig {
	result := source
	if source.Credential != nil {
		credential := *source.Credential
		result.Credential = &credential
	}
	return result
}

func equalOptionalString(left, right *string) bool {
	return left == nil && right == nil || left != nil && right != nil && *left == *right
}
