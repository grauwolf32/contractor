package controlplane

import (
	"errors"
	"slices"
	"testing"
	"time"
)

func TestRegistrationPlacementLabelsAreOwnedByIncarnation(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	principal := testPrincipal("a")
	registration := testRegistration("placement-first")
	registration.PlacementLabels = []string{"cloud/aws"}
	first, err := registry.RegisterAuthenticated(principal, registration)
	if err != nil {
		t.Fatal(err)
	}
	first.Registration.PlacementLabels[0] = "local/lab"
	registration.PlacementLabels[0] = "cloud/hetzner"
	if _, err := registry.RegisterAuthenticated(principal, registration); !errors.Is(err, ErrRegistrationConflict) {
		t.Fatalf("same-instance declaration change = %v", err)
	}
	registration.PlacementLabels = []string{"cloud/aws"}
	replayed, err := registry.RegisterAuthenticated(principal, registration)
	if err != nil || !slices.Equal(replayed.Registration.PlacementLabels, []string{"cloud/aws"}) {
		t.Fatalf("registration replay = %+v, %v", replayed, err)
	}
	if len(replayed.Principal.Labels) != 0 {
		t.Fatal("placement labels changed durable RuntimeConfig principal labels")
	}
	operations := registry.SnapshotOperations()
	if len(operations.RuntimeAgents) != 1 || !slices.Equal(operations.RuntimeAgents[0].PlacementLabels, []string{"cloud/aws"}) {
		t.Fatalf("actual placement declaration missing from Operations: %+v", operations)
	}
	operations.RuntimeAgents[0].PlacementLabels[0] = "local/lab"
	if err := registry.SnapshotOperations().Validate(); err != nil {
		t.Fatal(err)
	}
	if got := registry.SnapshotOperations().RuntimeAgents[0].PlacementLabels; !slices.Equal(got, []string{"cloud/aws"}) {
		t.Fatalf("Operations snapshot aliased registration labels: %v", got)
	}
	clock.Advance(2 * time.Minute)
	replacement := testRegistration("placement-second")
	replacement.PlacementLabels = []string{"local/lab"}
	second, err := registry.RegisterAuthenticated(principal, replacement)
	if err != nil || !slices.Equal(second.Registration.PlacementLabels, []string{"local/lab"}) {
		t.Fatalf("replacement incarnation declaration = %+v, %v", second, err)
	}
}
