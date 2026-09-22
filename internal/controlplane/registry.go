package controlplane

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	heartbeatHistoryLimit          = 128
	stageReservationTombstoneLimit = 1024
)

type Registry interface {
	RegistrationResponse(AuthenticatedPrincipal) contracts.AgentRegistrationResponse
	RegisterAuthenticated(AuthenticatedPrincipal, contracts.AgentRegistration) (AgentSnapshot, error)
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
	registration              contracts.AgentRegistration
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

const (
	defaultHeartbeatInterval = 10 * time.Second
	defaultConfirmedLease    = 60 * time.Second
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
