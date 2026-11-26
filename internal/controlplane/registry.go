package controlplane

import (
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/helpers/randomid"
)

const (
	heartbeatHistoryLimit          = 128
	stageReservationTombstoneLimit = 1024
)

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
	AgentOrderKey     func(control.AgentRegistration) string
	// ModelStageHold is how long an Agent stays held after the last failed
	// placement of a model Stage that could use it (CTL-R155). Zero selects
	// five seconds, five default Scheduler poll intervals.
	ModelStageHold time.Duration
}

type AgentSnapshot struct {
	Principal                  AuthenticatedPrincipal
	Registration               control.AgentRegistration
	LastSeenAt                 time.Time
	LastHeartbeatSeq           uint64
	LastIssuedAckSeq           uint64
	LastConfirmedAckSeq        uint64
	ConfirmedLeaseExpiresAt    time.Time
	AuthoritativeAllocationIDs []string
	ReconciliationRequired     bool
	LeaseExpired               bool
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
	agentOrderKey              func(control.AgentRegistration) string
	modelStageHold             time.Duration
	principalDeletions         map[string]struct{}
	pendingLosses              []AllocationLoss
	operationsGeneration       string
	operationsRevision         uint64
	operationsHistory          operationsChanges
	operationsWatchers         map[uint64]chan struct{}
	nextOperationsWatcher      uint64
}

// agentEntry owns process liveness and bounded allocation membership. Per-allocation
// activation, loss and write fencing live on storedReservation under Registry.mu.
type agentEntry struct {
	toolCapabilities        toolCapabilityIndex
	principal               AuthenticatedPrincipal
	principalMissing        bool
	registration            control.AgentRegistration
	identity                string
	orderKey                string
	lastSeenAt              time.Time
	lastAcceptedHeartbeatAt time.Time
	lastHeartbeatSeq        uint64
	lastIssuedAckSeq        uint64
	lastConfirmedAckSeq     uint64
	confirmedLeaseExpiresAt time.Time
	allocationIDs           map[string]struct{}
	releaseAcks             map[string]uint64
	reconciliationRequired  bool
	confirmedLeaseDeadline  time.Duration
	principalClaimDeadline  time.Duration
	leaseExpired            bool
	superseded              bool
	blockedByInstanceID     *string
	issuedAcks              map[uint64]struct{}
	heartbeatResponses      map[uint64]control.HeartbeatResponse
	heartbeatOrder          []uint64
	// modelStageHeldUntil is the monotonic instant until which model-free
	// placement must not select this Agent (CTL-R155).
	modelStageHeldUntil time.Duration
}

type stageReservation struct {
	fingerprint   string
	allocationIDs []string
	committed     bool
	released      bool
}

const (
	defaultHeartbeatInterval = 10 * time.Second
	defaultConfirmedLease    = 60 * time.Second
	defaultModelStageHold    = 5 * time.Second
)

func NewRegistry(options RegistryOptions) (*InMemoryRegistry, error) {
	if options.HeartbeatInterval == 0 {
		options.HeartbeatInterval = defaultHeartbeatInterval
	}
	if options.ConfirmedLease == 0 {
		options.ConfirmedLease = defaultConfirmedLease
	}
	if options.HeartbeatInterval < time.Second || options.ConfirmedLease <= options.HeartbeatInterval ||
		options.HeartbeatInterval%time.Second != 0 || options.ConfirmedLease%time.Second != 0 {
		return nil, fmt.Errorf("%w: heartbeat and lease durations must be whole seconds and lease must exceed interval", ErrInvalidRequest)
	}
	if options.ModelStageHold == 0 {
		options.ModelStageHold = defaultModelStageHold
	}
	if options.ModelStageHold < 0 {
		return nil, fmt.Errorf("%w: model Stage hold must be positive", ErrInvalidRequest)
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	if options.MonotonicNow == nil {
		started := time.Now()
		options.MonotonicNow = func() time.Duration { return time.Since(started) }
	}
	if options.NewID == nil {
		options.NewID = randomid.New
	}
	if options.AgentOrderKey == nil {
		options.AgentOrderKey = func(registration control.AgentRegistration) string {
			return registration.InstanceID
		}
	}
	operationsGeneration, err := randomid.New("operations-generation-")
	if err != nil {
		return nil, fmt.Errorf("generate Operations snapshot identity: %w", err)
	}
	return &InMemoryRegistry{
		agents: make(map[string]*agentEntry), allocations: make(map[string]storedReservation),
		stageReservations: make(map[string]stageReservation),
		heartbeatInterval: options.HeartbeatInterval, confirmedLease: options.ConfirmedLease,
		now: options.Now, monotonicNow: options.MonotonicNow,
		newID: options.NewID, agentOrderKey: options.AgentOrderKey,
		modelStageHold:       options.ModelStageHold,
		operationsGeneration: operationsGeneration,
		operationsWatchers:   make(map[uint64]chan struct{}),
		principalDeletions:   make(map[string]struct{}),
	}, nil
}

func (r *InMemoryRegistry) nextID(prefix string) (string, error) {
	r.idMu.Lock()
	defer r.idMu.Unlock()
	return r.newID(prefix)
}

var _ Registry = (*InMemoryRegistry)(nil)

func (entry *agentEntry) owns(allocationID string) bool {
	_, present := entry.allocationIDs[allocationID]
	return present
}

func (entry *agentEntry) authoritativeIDs() []string {
	result := make([]string, 0, len(entry.allocationIDs))
	for identity := range entry.allocationIDs {
		result = append(result, identity)
	}
	slices.Sort(result)
	return result
}

func (entry *agentEntry) availableSlots() int {
	if entry.leaseExpired || entry.superseded || entry.principalMissing || entry.blockedByInstanceID != nil ||
		entry.registration.ObservedState == control.AgentFenced || entry.registration.ObservedState == control.AgentDraining {
		return 0
	}
	occupied := make(map[string]struct{}, 2*control.MaxAllocationSlots)
	for id := range entry.allocationIDs {
		occupied[id] = struct{}{}
	}
	for id := range entry.releaseAcks {
		occupied[id] = struct{}{}
	}
	for _, item := range entry.registration.Allocations {
		occupied[item.AllocationID] = struct{}{}
	}
	return max(0, entry.registration.AllocationSlots-len(occupied))
}
