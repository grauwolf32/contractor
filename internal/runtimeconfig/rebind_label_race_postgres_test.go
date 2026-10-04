package runtimeconfig

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// TestPostgresRebindUnrelatedLabelRemovalIsPrecondition proves that when a
// rebind optimistically lists a principal holding another label and that other
// label's binding is removed before the lock acquires it, the rebind reports a
// precondition failure (412) rather than a raw not-found that the public
// handler would answer as 400 runtime_label_unknown for a label the request
// never selected.
func TestPostgresRebindUnrelatedLabelRemovalIsPrecondition(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRuntimeConfigPool(t, ctx, databaseURL)
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	publisher, err := NewPublisher(PublisherOptions{
		Pool: pool, RuntimeCredentials: allowRuntimeCredentialCatalog{}, Now: func() time.Time { return now },
		PlannerTelemetryAdapters: PlannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
	})
	if err != nil {
		t.Fatal(err)
	}
	refs := map[string]Ref{}
	repository := NewRepository(pool)
	for _, name := range []string{"selected", "unrelated"} {
		document := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"` + name + `","version":"1"},"spec":{"worker":{"telemetry":null}}}`)
		published, publishErr := publisher.Publish(ctx, document, "publish-"+name, "operator")
		if publishErr != nil {
			t.Fatal(publishErr)
		}
		refs[name] = published.Version.Ref
		if _, err := repository.CreateBinding(ctx, name, published.Version.Ref, "operator", now); err != nil {
			t.Fatal(err)
		}
	}

	principalService, err := NewPrincipalService(PrincipalServiceOptions{
		Pool: pool, DeletionGuard: &recordingDeletionGuard{}, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	// One principal holds both labels, so a rebind of "selected" also locks
	// "unrelated".
	if _, err := principalService.Register(
		ctx, strings.Repeat("e", 64), []string{"selected", "unrelated"},
	); err != nil {
		t.Fatal(err)
	}

	bindings, err := NewBindingService(allowRuntimeCredentialCatalog{})
	if err != nil {
		t.Fatal(err)
	}
	management, err := NewManagementService(pool, publisher, bindings)
	if err != nil {
		t.Fatal(err)
	}

	// Remove the unrelated label's binding behind the optimistic snapshot the
	// rebind already read: the principal row still references "unrelated", so
	// the lock set still includes it, but its binding row is gone.
	if _, err := pool.Exec(ctx, `DELETE FROM runtime_label_bindings WHERE label = 'unrelated'`); err != nil {
		t.Fatal(err)
	}

	_, err = management.Rebind(ctx, "selected", 1, refs["selected"], "rebind-race", "operator", now.Add(time.Minute))
	if !errors.Is(err, ErrPrecondition) {
		t.Fatalf("concurrent unrelated-label removal rebind error = %v, want ErrPrecondition", err)
	}
	if errors.Is(err, ErrNotFound) || errors.Is(err, ErrLabelNotFound) {
		t.Fatalf("rebind reported the selected label as unknown: %v", err)
	}
	// The selected label's binding is untouched.
	if current, getErr := repository.GetBinding(ctx, "selected"); getErr != nil || current.Revision != 1 {
		t.Fatalf("selected binding after failed rebind = (%+v, %v)", current, getErr)
	}
}
