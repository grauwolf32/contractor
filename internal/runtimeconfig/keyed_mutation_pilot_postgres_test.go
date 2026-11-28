package runtimeconfig

import (
	"context"
	"errors"
	"io"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func TestPostgresKeyedMutationPilotBindingLostReply(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedRuntimeConfigPool(t, ctx, url)
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 11, 28, 0, 0, 0, 0, time.UTC)
	publisher, err := NewPublisher(PublisherOptions{Pool: pool, RuntimeCredentials: allowRuntimeCredentialCatalog{}, Now: func() time.Time { return now }, PlannerTelemetryAdapters: plannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" })})
	if err != nil {
		t.Fatal(err)
	}
	bindings, err := NewBindingService(allowRuntimeCredentialCatalog{})
	if err != nil {
		t.Fatal(err)
	}
	management, err := NewManagementService(pool, publisher, bindings, noLabelHolders{})
	if err != nil {
		t.Fatal(err)
	}
	var refs []Ref
	for _, name := range []string{"pilot-a", "pilot-b"} {
		document := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"` + name + `","version":"1"},"spec":{"worker":{"telemetry":null}}}`)
		result, err := management.Publish(ctx, document, "publish-"+name, "operator")
		if err != nil {
			t.Fatal(err)
		}
		refs = append(refs, result.Version.Ref)
	}
	// The service committed successfully, but its caller receives no result.
	lostCreate := func() error {
		if _, err := management.CreateBinding(ctx, "pilot", refs[0], "pilot-create", "operator", now); err != nil {
			return err
		}
		return io.ErrUnexpectedEOF
	}
	if err := lostCreate(); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("lost create: %v", err)
	}
	original, err := management.GetBinding(ctx, "pilot")
	if err != nil || original.Revision != 1 {
		t.Fatalf("committed original: %+v %v", original, err)
	}
	// Another accepted operation advances the live resource before reconciliation.
	changed, err := management.Rebind(ctx, "pilot", original.Revision, refs[1], "pilot-rebind", "operator", now.Add(time.Minute))
	if err != nil || changed.Binding == nil || changed.Binding.Revision != 2 {
		t.Fatalf("later mutation: %+v %v", changed, err)
	}
	fresh, err := NewManagementService(pool, publisher, bindings, noLabelHolders{})
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		replay, err := fresh.CreateBinding(ctx, "pilot", refs[0], "pilot-create", "retry-operator", now.Add(2*time.Minute))
		if err != nil || !replay.Replayed || replay.Binding == nil || !reflect.DeepEqual(*replay.Binding, original) {
			t.Fatalf("original result replay: %+v %v", replay, err)
		}
	}
	if _, err := fresh.CreateBinding(ctx, "pilot", refs[1], "pilot-create", "operator", now); !errors.Is(err, contracts.ErrIdempotencyConflict) {
		t.Fatalf("changed request: %v", err)
	}
	live, err := fresh.GetBinding(ctx, "pilot")
	if err != nil || !reflect.DeepEqual(live, *changed.Binding) {
		t.Fatalf("old replay rewrote current binding: %+v %v", live, err)
	}
}
