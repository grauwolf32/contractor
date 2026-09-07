package controlplane

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestPrincipalOperationsSeparatesOfflineBusyAndAdapterMismatch(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	principal := runtimeconfig.RuntimeAgentPrincipal{
		RuntimeAgentID: strings.Repeat("e", 64), Labels: []string{"debug"}, LabelRevision: 1,
		CreatedBy: "runtime-registration", CreatedAt: clock.Now(),
		UpdatedBy: "runtime-registration", UpdatedAt: clock.Now(),
	}
	registration := testRegistration("agent-principal-operations")
	registration.SupportedRuntimeAdapters = []contracts.RuntimeAdapterRef{}
	if _, err := registry.RegisterAuthenticated(AuthenticatedPrincipal{
		RuntimeAgentID: principal.RuntimeAgentID,
		Labels:         principal.Labels, LabelRevision: principal.LabelRevision,
	}, registration); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.HeartbeatAuthenticated(
		principal.RuntimeAgentID, heartbeat(registration.InstanceID, 1, 0),
	); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.HeartbeatAuthenticated(
		principal.RuntimeAgentID, heartbeat(registration.InstanceID, 2, 1),
	); err != nil {
		t.Fatal(err)
	}
	catalog := &principalOperationsCatalogFake{
		principal: principal, required: []string{"otlp-http@1"},
	}
	operations, err := NewPrincipalOperations(catalog, registry)
	if err != nil {
		t.Fatal(err)
	}
	projection, err := operations.Get(t.Context(), principal.RuntimeAgentID)
	if err != nil || projection.Availability != PrincipalAdapterCapabilityMismatch ||
		len(projection.MissingRuntimeAdapters) != 1 || projection.Live == nil ||
		projection.Live.InstanceID != registration.InstanceID {
		t.Fatalf("capability mismatch projection = (%+v, %v)", projection, err)
	}

	catalog.required = []string{}
	updated, replayed, err := operations.ReplaceLabels(
		t.Context(), principal.RuntimeAgentID, 1, []string{}, "replace-key", "operator", clock.Now(),
	)
	if err != nil || replayed || updated.Availability != PrincipalAvailable || updated.Principal.LabelRevision != 2 {
		t.Fatalf("updated projection = (%+v, %v, %v)", updated, replayed, err)
	}
	candidates := registry.PlacementCandidates()
	if len(candidates) != 1 || candidates[0].Principal.LabelRevision != 2 || len(candidates[0].Principal.Labels) != 0 {
		t.Fatalf("placement principal was not synchronized: %+v", candidates)
	}

	clock.Advance(61 * time.Second)
	registry.PollAllocationLosses()
	offline, err := operations.Get(t.Context(), principal.RuntimeAgentID)
	if err != nil || offline.Availability != PrincipalOffline || offline.Live != nil {
		t.Fatalf("offline projection = (%+v, %v)", offline, err)
	}
}

type principalOperationsCatalogFake struct {
	principal runtimeconfig.RuntimeAgentPrincipal
	required  []string
}

func (f *principalOperationsCatalogFake) Get(
	context.Context, string,
) (runtimeconfig.RuntimeAgentPrincipal, error) {
	return f.principal, nil
}

func (f *principalOperationsCatalogFake) List(
	context.Context, string, int,
) ([]runtimeconfig.RuntimeAgentPrincipal, error) {
	return []runtimeconfig.RuntimeAgentPrincipal{f.principal}, nil
}

func (f *principalOperationsCatalogFake) RequiredRuntimeAdapters(
	context.Context, []string,
) ([]string, error) {
	return append([]string{}, f.required...), nil
}

func (f *principalOperationsCatalogFake) ReplaceLabelsIdempotent(
	_ context.Context, _ string, expected uint64, labels []string, _ string, actor string, at time.Time,
) (runtimeconfig.PrincipalMutationResult, error) {
	if f.principal.LabelRevision != expected {
		return runtimeconfig.PrincipalMutationResult{}, runtimeconfig.ErrPrecondition
	}
	f.principal.Labels = append([]string{}, labels...)
	f.principal.LabelRevision++
	f.principal.UpdatedBy = actor
	f.principal.UpdatedAt = at
	result := f.principal
	return runtimeconfig.PrincipalMutationResult{Principal: &result}, nil
}

func (f *principalOperationsCatalogFake) DeleteIdempotent(
	context.Context, string, uint64, string, string, time.Time,
) (runtimeconfig.PrincipalMutationResult, error) {
	return runtimeconfig.PrincipalMutationResult{Deleted: true}, nil
}
