package controlplane

// The certificate-derived principal behind a Runtime Agent: its labels,
// its deletion, and the observation projected to Operations.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

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

func inProcessPrincipal(instanceID string) AuthenticatedPrincipal {
	sum := sha256.Sum256([]byte("in-process-runtime-principal\x00" + instanceID))
	return AuthenticatedPrincipal{
		RuntimeAgentID: hex.EncodeToString(sum[:]), Labels: []string{}, LabelRevision: 1,
	}
}

func validateAuthenticatedPrincipal(principal AuthenticatedPrincipal) error {
	probe := contracts.AgentRegistrationResponse{
		APIVersion: contracts.APIVersion, RuntimeAgentID: principal.RuntimeAgentID, Labels: principal.Labels,
		LabelRevision: principal.LabelRevision, HeartbeatIntervalSeconds: 1, ConfirmedLeaseSeconds: 2,
	}
	if err := probe.Validate(); err != nil {
		return fmt.Errorf("%w: authenticated Runtime Agent principal is invalid", ErrInvalidRequest)
	}
	return nil
}
