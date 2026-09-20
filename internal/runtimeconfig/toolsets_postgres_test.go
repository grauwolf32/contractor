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

func TestPostgresToolsetPublicationAndPrincipalAssignmentRebindRace(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRuntimeConfigPool(t, ctx, dsn)
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	publisher, err := NewPublisher(PublisherOptions{Pool: pool, RuntimeCredentials: allowRuntimeCredentialCatalog{}, PlannerTelemetryAdapters: PlannerTelemetryAdapterCatalogFunc(func(string) bool { return true })})
	if err != nil {
		t.Fatal(err)
	}
	publish := func(name string, worker any) Ref {
		t.Helper()
		raw := toolsetDocument(name, worker)
		first, err := publisher.Publish(ctx, raw, "publish-"+name, "operator")
		if err != nil {
			t.Fatal(err)
		}
		replay, err := publisher.Publish(ctx, raw, "publish-"+name, "operator")
		if err != nil || !replay.Replayed || first.Version.Ref != replay.Version.Ref {
			t.Fatalf("publication replay failed: %v", err)
		}
		stored, err := NewRepository(pool).GetVersionByRef(ctx, first.Version.Ref)
		if err != nil || !specEqual(stored.Spec, first.Version.Spec) {
			t.Fatalf("stored typed config differs: %v", err)
		}
		return first.Version.Ref
	}
	safe := publish("safe", map[string]any{"telemetry": nil})
	bindings, err := NewBindingService(pool, allowRuntimeCredentialCatalog{})
	if err != nil {
		t.Fatal(err)
	}
	principals, err := NewPrincipalService(PrincipalServiceOptions{Pool: pool, DeletionGuard: &recordingDeletionGuard{}})
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	if _, err := bindings.Create(ctx, "safe", safe, "operator", now); err != nil {
		t.Fatal(err)
	}
	principalID := strings.Repeat("a", 64)
	if _, err := principals.Register(ctx, principalID, []string{"safe"}); err != nil {
		t.Fatal(err)
	}
	var last Ref
	for i, value := range []any{map[string]any{"endpoint": "https://x.example/mcp"}, nil} {
		name := []string{"configured", "cleared"}[i]
		last = publish(name, map[string]any{"telemetry": nil, "toolsets": map[string]any{"mcp@1": value}})
		if _, err := bindings.Create(ctx, name, last, "operator", now); err != nil {
			t.Fatal(err)
		}
		if _, err := principals.Register(ctx, strings.Repeat([]string{"b", "c"}[i], 64), []string{name}); !errors.Is(err, ErrAgentLabelNotApplicable) {
			t.Fatalf("toolset principal assignment: %v", err)
		}
		if _, err := principals.ReplaceLabelsIdempotent(ctx, principalID, 1, []string{name}, "replace-"+name, "operator", now); !errors.Is(err, ErrAgentLabelNotApplicable) {
			t.Fatalf("mixed principal replacement: %v", err)
		}
		if _, err := bindings.Rebind(ctx, "safe", 1, last, "operator", now); !errors.Is(err, ErrAgentLabelNotApplicable) {
			t.Fatalf("assigned label rebind: %v", err)
		}
		binding, err := NewRepository(pool).GetBinding(ctx, "safe")
		if err != nil || binding.Ref != safe || binding.Revision != 1 {
			t.Fatal("invalid rebind was not rolled back")
		}
	}
	if _, err := bindings.Create(ctx, "race", safe, "operator", now); err != nil {
		t.Fatal(err)
	}
	start := make(chan struct{})
	assigned := make(chan error, 1)
	rebound := make(chan error, 1)
	go func() {
		<-start
		_, err := principals.Register(ctx, strings.Repeat("d", 64), []string{"race"})
		assigned <- err
	}()
	go func() { <-start; _, err := bindings.Rebind(ctx, "race", 1, last, "operator", now); rebound <- err }()
	close(start)
	assignmentErr, rebindErr := <-assigned, <-rebound
	if assignmentErr == nil && rebindErr == nil {
		t.Fatal("assignment and forbidden rebind both committed")
	}
	if assignmentErr != nil && !errors.Is(assignmentErr, ErrAgentLabelNotApplicable) && !errors.Is(assignmentErr, ErrPrecondition) {
		t.Fatalf("unexpected assignment race error: %v", assignmentErr)
	}
	if rebindErr != nil && !errors.Is(rebindErr, ErrAgentLabelNotApplicable) && !errors.Is(rebindErr, ErrPrecondition) {
		t.Fatalf("unexpected rebind race error: %v", rebindErr)
	}
	if assignmentErr == nil {
		binding, err := NewRepository(pool).GetBinding(ctx, "race")
		if err != nil || binding.Ref != safe {
			t.Fatal("principal retained toolset-bearing label")
		}
	}
}
