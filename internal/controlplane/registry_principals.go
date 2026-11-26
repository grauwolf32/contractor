package controlplane

import (
	"sync"
)

// MarkPrincipalMissing excludes an instance whose durable principal was
// deleted on another Server. A later authenticated registration may restore
// it only after reading a principal from the database.
func (r *InMemoryRegistry) MarkPrincipalMissing(runtimeAgentID string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	changed := false
	for _, entry := range r.agents {
		if entry.principal.RuntimeAgentID != runtimeAgentID || entry.principalMissing {
			continue
		}
		entry.principalMissing = true
		r.recordOperationsChangeLocked(OperationsRuntimeAgent, entry.registration.InstanceID)
		changed = true
	}
	return changed
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
		if len(entry.allocationIDs) != 0 || principalInstanceIsLive(entry, monotonicNow) {
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
			entry.principalMissing = false
			continue
		}
		entry.principal = clonePrincipal(principal)
		entry.principalMissing = false
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
		if len(entry.allocationIDs) == 0 && !principalInstanceIsLive(entry, monotonicNow) {
			continue
		}
		if selected == nil || selected.lastSeenAt.Before(entry.lastSeenAt) {
			selected = entry
		}
	}
	if selected == nil {
		return nil, false
	}
	result := r.runtimeObservation(selected)
	return &result, true
}
