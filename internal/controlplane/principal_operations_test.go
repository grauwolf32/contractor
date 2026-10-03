package controlplane

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestPrincipalOperationsAppliesCommittedLabelsBeforeFollowupRead(t *testing.T) {
	for _, scenario := range []string{"read-failure", "cancelled-context"} {
		t.Run(scenario, func(t *testing.T) {
			clock := newTestClock()
			registry := newTestRegistry(t, clock)
			principal := runtimeconfig.RuntimeAgentPrincipal{
				RuntimeAgentID: strings.Repeat("f", 64), Labels: []string{"old"}, LabelRevision: 1,
				CreatedBy: "runtime-registration", CreatedAt: clock.Now(),
				UpdatedBy: "runtime-registration", UpdatedAt: clock.Now(),
			}
			registration := testRegistration("agent-followup-read")
			if _, err := registry.RegisterAuthenticated(AuthenticatedPrincipal{
				RuntimeAgentID: principal.RuntimeAgentID, Labels: principal.Labels, LabelRevision: 1,
			}, registration); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			catalog := &principalOperationsCatalogFake{principal: principal}
			catalog.afterReplace = func() {
				if scenario == "cancelled-context" {
					cancel()
				} else {
					catalog.getErr = errors.New("follow-up read failed")
				}
			}
			operations, err := NewPrincipalOperations(catalog, registry)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := operations.ReplaceLabels(ctx, principal.RuntimeAgentID, 1,
				[]string{"new"}, "replace-key", "operator", clock.Now()); err == nil {
				t.Fatal("fixture follow-up read unexpectedly succeeded")
			}
			snapshot, err := registry.GetAgent(registration.InstanceID)
			if err != nil || snapshot.Principal.LabelRevision != 2 ||
				!equalTestStrings(snapshot.Principal.Labels, []string{"new"}) {
				t.Fatalf("committed labels did not reach Registry: %+v, %v", snapshot, err)
			}
		})
	}
}

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
	principal    runtimeconfig.RuntimeAgentPrincipal
	required     []string
	getErr       error
	afterReplace func()
}

func (f *principalOperationsCatalogFake) Get(
	ctx context.Context, _ string,
) (runtimeconfig.RuntimeAgentPrincipal, error) {
	if err := ctx.Err(); err != nil {
		return runtimeconfig.RuntimeAgentPrincipal{}, err
	}
	if f.getErr != nil {
		return runtimeconfig.RuntimeAgentPrincipal{}, f.getErr
	}
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
	if f.afterReplace != nil {
		f.afterReplace()
	}
	return runtimeconfig.PrincipalMutationResult{Principal: &result}, nil
}

func (f *principalOperationsCatalogFake) DeleteIdempotent(
	context.Context, string, uint64, string, string, time.Time,
) (runtimeconfig.PrincipalMutationResult, error) {
	return runtimeconfig.PrincipalMutationResult{Deleted: true}, nil
}
