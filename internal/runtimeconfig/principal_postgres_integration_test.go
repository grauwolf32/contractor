package runtimeconfig

import (
	"context"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func TestPostgresRuntimeAgentPrincipalSeedCASAndDelete(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRuntimeConfigPool(t, ctx, databaseURL)
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatalf("apply migrations: %v", err)
	}

	now := time.Date(2026, 9, 1, 1, 0, 0, 0, time.UTC)
	publisher, err := NewPublisher(PublisherOptions{
		Pool: pool,
		GatewayResolver: GatewayResolverFunc(func(_ context.Context, selector string) (contracts.ResolvedLLMGatewayConfig, error) {
			gatewayID, version, _ := strings.Cut(selector, "@")
			return contracts.ResolvedLLMGatewayConfig{
				Ref: contracts.LLMGatewayConfigRef{
					GatewayID: gatewayID, Version: version,
					Digest: "sha256:" + strings.Repeat("a", 64),
				},
				Protocol: contracts.OpenAICompatibleProtocol,
				URL:      "http://127.0.0.1:4000/v1",
			}, nil
		}),
		RuntimeCredentials: allowRuntimeCredentialCatalog{},
		Now:                func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	published, err := publisher.Publish(ctx, []byte(`{
  "apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
  "metadata":{"name":"agent-route","version":"1"},
  "spec":{"worker":{"llmGateway":{"gateway":"local-litellm@1"}}}
}`), "principal-config", "operator")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewRepository(pool).CreateBinding(
		ctx, "agent-route", published.Version.Ref, "operator", now,
	); err != nil {
		t.Fatal(err)
	}

	guard := &recordingDeletionGuard{}
	service, err := NewPrincipalService(PrincipalServiceOptions{
		Pool: pool, DeletionGuard: guard, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	principalID := strings.Repeat("a", 64)
	created, err := service.Register(ctx, principalID, []string{"agent-route"})
	if err != nil || created.LabelRevision != 1 || !equalStrings(created.Labels, []string{"agent-route"}) {
		t.Fatalf("created principal = (%+v, %v)", created, err)
	}
	// Startup arguments are seed-only after the first successful registration.
	replayed, err := service.Register(ctx, principalID, []string{})
	if err != nil || replayed.LabelRevision != 1 || !equalStrings(replayed.Labels, created.Labels) {
		t.Fatalf("replayed principal = (%+v, %v)", replayed, err)
	}
	if err := service.Delete(ctx, principalID, 1); !errors.Is(err, ErrConflict) {
		t.Fatalf("delete labeled principal error = %v", err)
	}

	updated, err := service.ReplaceLabels(ctx, principalID, 1, []string{}, "operator")
	if err != nil || updated.LabelRevision != 2 || len(updated.Labels) != 0 {
		t.Fatalf("replace labels = (%+v, %v)", updated, err)
	}
	if _, err := service.ReplaceLabels(ctx, principalID, 1, []string{}, "operator"); !errors.Is(err, ErrPrecondition) {
		t.Fatalf("stale principal CAS error = %v", err)
	}
	if err := service.Delete(ctx, principalID, 2); err != nil {
		t.Fatalf("delete empty offline principal: %v", err)
	}
	if _, err := NewPrincipalRepository(pool).Get(ctx, principalID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("deleted principal lookup error = %v", err)
	}
	if guard.begins != 2 || guard.releases != 2 {
		t.Fatalf("deletion guard calls = begin %d release %d", guard.begins, guard.releases)
	}
}

func TestPostgresLabelRebindCannotInvalidateAssignedPrincipalLayer(t *testing.T) {
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
	now := time.Date(2026, 9, 1, 2, 0, 0, 0, time.UTC)
	resolver := GatewayResolverFunc(func(_ context.Context, selector string) (contracts.ResolvedLLMGatewayConfig, error) {
		gatewayID, version, _ := strings.Cut(selector, "@")
		digit := "b"
		if gatewayID == "other-litellm" {
			digit = "c"
		}
		return contracts.ResolvedLLMGatewayConfig{
			Ref: contracts.LLMGatewayConfigRef{
				GatewayID: gatewayID, Version: version,
				Digest: "sha256:" + strings.Repeat(digit, 64),
			},
			Protocol: contracts.OpenAICompatibleProtocol,
			URL:      "http://127.0.0.1:4000/v1",
		}, nil
	})
	publisher, err := NewPublisher(PublisherOptions{
		Pool: pool, GatewayResolver: resolver,
		RuntimeCredentials: allowRuntimeCredentialCatalog{}, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	publish := func(name, gateway string) Ref {
		t.Helper()
		document := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"` +
			name + `","version":"1"},"spec":{"worker":{"llmGateway":{"gateway":"` + gateway + `@1"}}}}`)
		result, publishErr := publisher.Publish(ctx, document, "publish-"+name, "operator")
		if publishErr != nil {
			t.Fatal(publishErr)
		}
		return result.Version.Ref
	}
	firstRef := publish("route-one", "local-litellm")
	identicalRef := publish("route-two", "local-litellm")
	conflictingRef := publish("route-other", "other-litellm")
	repository := NewRepository(pool)
	for label, ref := range map[string]Ref{"route-one": firstRef, "route-two": identicalRef} {
		if _, err := repository.CreateBinding(ctx, label, ref, "operator", now); err != nil {
			t.Fatal(err)
		}
	}
	principalService, err := NewPrincipalService(PrincipalServiceOptions{
		Pool: pool, DeletionGuard: &recordingDeletionGuard{}, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	principalID := strings.Repeat("d", 64)
	if _, err := principalService.Register(
		ctx, principalID, []string{"route-one", "route-two"},
	); err != nil {
		t.Fatal(err)
	}
	bindingService, err := NewBindingService(pool, allowRuntimeCredentialCatalog{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := bindingService.Rebind(
		ctx, "route-two", 1, conflictingRef, "operator", now.Add(time.Minute),
	); err == nil {
		t.Fatal("conflicting label rebind was accepted")
	} else {
		var conflict *MergeConflictError
		if !errors.As(err, &conflict) || conflict.Path != "worker.llmGateway.gateway" {
			t.Fatalf("conflicting rebind error = %v", err)
		}
	}
	current, err := repository.GetBinding(ctx, "route-two")
	if err != nil || current.Revision != 1 || current.Ref != identicalRef {
		t.Fatalf("binding after rolled-back conflict = (%+v, %v)", current, err)
	}
	if err := bindingService.Delete(ctx, "route-one", 1); !errors.Is(err, ErrConflict) {
		t.Fatalf("delete assigned label error = %v", err)
	}
}

type recordingDeletionGuard struct {
	mu       sync.Mutex
	begins   int
	releases int
}

func (g *recordingDeletionGuard) BeginPrincipalDeletion(string) (func(), error) {
	g.mu.Lock()
	g.begins++
	g.mu.Unlock()
	return func() {
		g.mu.Lock()
		g.releases++
		g.mu.Unlock()
	}, nil
}
