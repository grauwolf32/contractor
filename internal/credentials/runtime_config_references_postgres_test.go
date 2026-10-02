package credentials

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestRuntimeConfigLLMCredentialReferencesValidateAndFenceDeletion(t *testing.T) {
	pool, ctx := lifecycleTestPool(t)
	manager := newFakeGatewayManager()
	fixture := newLifecycleFixture(t, pool, manager, ServiceOptions{})
	if err := fixture.service.Recover(ctx); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"bound-worker", "deleted-worker"} {
		if _, err := fixture.service.Create(ctx, fixture.createRequest(id, "create-"+id)); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := fixture.service.Delete(ctx, DeleteRequest{
		CredentialID: "deleted-worker", IdempotencyKey: "delete-deleted-worker", ActorID: "user-1",
	}); err != nil {
		t.Fatal(err)
	}

	mismatch := fixture.gateway.Ref
	mismatch.Digest = "sha256:" + strings.Repeat("e", 64)
	static, err := NewStaticProvider([]StaticEntry{{
		Metadata: config.CredentialMetadata{
			Ref: contracts.LLMCredentialRef{CredentialID: "wrong-gateway"}, LLMGateway: mismatch,
		},
		Token: contracts.NewSecretString("static-test-token"),
	}})
	if err != nil {
		t.Fatal(err)
	}
	factory, err := NewTransactionLookupFactory(static)
	if err != nil {
		t.Fatal(err)
	}
	publisher, bindings, resolver := newReferenceTestServices(t, pool, fixture.gateway, factory)
	repository := runtimeconfig.NewRepository(pool)
	management, err := runtimeconfig.NewManagementService(pool, publisher, bindings)
	if err != nil {
		t.Fatal(err)
	}

	for _, id := range []string{"missing-worker", "deleted-worker", "wrong-gateway"} {
		name := "invalid-" + id
		if _, err := publisher.Publish(ctx, referenceTestDocument(name, id), "publish-"+name, "operator"); !errors.Is(err, runtimeconfig.ErrInvalid) {
			t.Fatalf("publish %s = %v", id, err)
		}
		if _, err := repository.GetVersion(ctx, name, "1"); !errors.Is(err, runtimeconfig.ErrNotFound) {
			t.Fatalf("invalid publication %s stored a version: %v", id, err)
		}
	}
	valid, err := publisher.Publish(ctx, referenceTestDocument("valid-worker", "bound-worker"), "publish-valid", "operator")
	if err != nil {
		t.Fatal(err)
	}
	created, err := management.CreateBinding(ctx, "probe", valid.Version.Ref, "create-probe", "operator", time.Now().UTC())
	if err != nil || created.Binding == nil || created.Binding.Revision != 1 {
		t.Fatalf("create valid binding = (%+v, %v)", created, err)
	}

	for _, id := range []string{"missing-worker", "deleted-worker", "wrong-gateway"} {
		name := "legacy-" + id
		prepared, err := runtimeconfig.PreparePublication(referenceTestDocument(name, id))
		if err != nil {
			t.Fatal(err)
		}
		version, err := prepared.Resolve(ctx, resolver)
		if err != nil {
			t.Fatal(err)
		}
		version.ActorID, version.CreatedAt = "operator", time.Now().UTC()
		if _, err := repository.InsertVersion(ctx, version); err != nil {
			t.Fatal(err)
		}
		if _, err := management.CreateBinding(ctx, "label-"+id, version.Ref, "create-label-"+id, "operator", time.Now().UTC()); !errors.Is(err, runtimeconfig.ErrInvalid) {
			t.Fatalf("create binding to %s = %v", id, err)
		}
		if _, err := repository.GetBinding(ctx, "label-"+id); !errors.Is(err, runtimeconfig.ErrNotFound) {
			t.Fatalf("invalid binding to %s was stored: %v", id, err)
		}
		if _, err := management.Rebind(ctx, "probe", 1, version.Ref, "rebind-probe-"+id, "operator", time.Now().UTC()); !errors.Is(err, runtimeconfig.ErrInvalid) {
			t.Fatalf("rebind to %s = %v", id, err)
		}
		current, err := repository.GetBinding(ctx, "probe")
		if err != nil || current.Revision != 1 || current.Ref != valid.Version.Ref {
			t.Fatalf("failed rebind changed probe binding: (%+v, %v)", current, err)
		}
	}
	if _, err := management.Rebind(ctx, runtimeconfig.DefaultLabel, 1, valid.Version.Ref, "rebind-default-valid", "operator", time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
	deletion := DeleteRequest{CredentialID: "bound-worker", IdempotencyKey: "delete-bound-worker", ActorID: "user-1"}
	_, err = fixture.service.Delete(ctx, deletion)
	var inUse *CredentialInUseError
	if !errors.As(err, &inUse) || !reflect.DeepEqual(inUse.BindingLabels, []string{"default", "probe"}) {
		t.Fatalf("bound credential deletion = %#v", err)
	}
	if manager.deleteCalls() != 1 || manager.remoteCount() != 1 || countOperations(t, ctx, pool, OperationDelete) != 1 {
		t.Fatalf("bound delete had side effects: delete calls=%d remote=%d operations=%d",
			manager.deleteCalls(), manager.remoteCount(), countOperations(t, ctx, pool, OperationDelete))
	}
	if _, err := fixture.service.GetCredential(ctx, "bound-worker"); err != nil {
		t.Fatalf("bound credential disappeared: %v", err)
	}
	builtIn, err := repository.GetVersion(ctx, runtimeconfig.BuiltInName, runtimeconfig.BuiltInVersion)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := management.Rebind(ctx, "probe", 1, builtIn.Ref, "rebind-probe-empty", "operator", time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
	if _, err := management.Rebind(ctx, runtimeconfig.DefaultLabel, 2, builtIn.Ref, "rebind-default-empty", "operator", time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.service.Delete(ctx, deletion); err != nil {
		t.Fatalf("delete after rebind = %v", err)
	}
	if manager.deleteCalls() != 2 || manager.remoteCount() != 0 {
		t.Fatalf("delete after rebind: calls=%d remote=%d", manager.deleteCalls(), manager.remoteCount())
	}
}

func TestRuntimeConfigBindingAndCredentialDeleteSerializeAcrossBarriers(t *testing.T) {
	pool, ctx := lifecycleTestPool(t)
	manager := newFakeGatewayManager()
	fixture := newLifecycleFixture(t, pool, manager, ServiceOptions{})
	if err := fixture.service.Recover(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.service.Create(ctx, fixture.createRequest("racing-worker", "create-racing-worker")); err != nil {
		t.Fatal(err)
	}
	static, err := NewStaticProvider(nil)
	if err != nil {
		t.Fatal(err)
	}
	factory, err := NewTransactionLookupFactory(static)
	if err != nil {
		t.Fatal(err)
	}
	publisher, bindings, _ := newReferenceTestServices(t, pool, fixture.gateway, factory)
	published, err := publisher.Publish(ctx, referenceTestDocument("racing-config", "racing-worker"), "publish-racing", "operator")
	if err != nil {
		t.Fatal(err)
	}
	start := make(chan struct{})
	var wait sync.WaitGroup
	var bindErr, deleteErr error
	wait.Add(2)
	go func() {
		defer wait.Done()
		<-start
		_, bindErr = bindings.Create(ctx, "racing", published.Version.Ref, "operator", time.Now().UTC())
	}()
	go func() {
		defer wait.Done()
		<-start
		_, deleteErr = fixture.service.Delete(ctx, DeleteRequest{
			CredentialID: "racing-worker", IdempotencyKey: "delete-racing-worker", ActorID: "user-1",
		})
	}()
	close(start)
	wait.Wait()
	var inUse *CredentialInUseError
	switch {
	case bindErr == nil && errors.As(deleteErr, &inUse):
		if !reflect.DeepEqual(inUse.BindingLabels, []string{"racing"}) {
			t.Fatalf("concurrent delete labels = %v", inUse.BindingLabels)
		}
		if _, err := fixture.service.GetCredential(ctx, "racing-worker"); err != nil {
			t.Fatalf("bound credential was deleted: %v", err)
		}
	case deleteErr == nil && errors.Is(bindErr, runtimeconfig.ErrInvalid):
		if _, err := runtimeconfig.NewRepository(pool).GetBinding(ctx, "racing"); !errors.Is(err, runtimeconfig.ErrNotFound) {
			t.Fatalf("binding survived concurrent delete: %v", err)
		}
	default:
		t.Fatalf("concurrent binding/delete = (%v, %v)", bindErr, deleteErr)
	}
}

func newReferenceTestServices(
	t *testing.T, pool *pgxpool.Pool, gateway contracts.ResolvedLLMGatewayConfig,
	factory *TransactionLookupFactory,
) (*runtimeconfig.Publisher, *runtimeconfig.BindingService, runtimeconfig.GatewayResolver) {
	t.Helper()
	resolver := runtimeconfig.GatewayResolverFunc(func(_ context.Context, selector string) (contracts.ResolvedLLMGatewayConfig, error) {
		if selector != gateway.Ref.GatewayID+"@"+gateway.Ref.Version {
			return contracts.ResolvedLLMGatewayConfig{}, fmt.Errorf("unexpected Gateway %q", selector)
		}
		return gateway, nil
	})
	catalog := transactionTestRuntimeCredentialCatalog{}
	publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
		Pool: pool, GatewayResolver: resolver, RuntimeCredentials: catalog,
		TransactionLLMCredentials: factory,
		PlannerTelemetryAdapters:  runtimeconfig.PlannerTelemetryAdapterCatalogFunc(func(string) bool { return true }),
	})
	if err != nil {
		t.Fatal(err)
	}
	bindings, err := runtimeconfig.NewBindingService(pool, catalog, factory)
	if err != nil {
		t.Fatal(err)
	}
	return publisher, bindings, resolver
}

func referenceTestDocument(name, credentialID string) []byte {
	return []byte(fmt.Sprintf(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
"metadata":{"name":%q,"version":"1"},
"spec":{"worker":{"llmGateway":{"gateway":"local-litellm@1","credential":%q}}}}`, name, credentialID))
}
