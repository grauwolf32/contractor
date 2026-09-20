package credentials

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestRuntimeCredentialRebindDoesNotBorrowAnotherConnection(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	for _, maxConns := range []int32{1, 2} {
		t.Run(fmt.Sprint(maxConns), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			pool := isolatedCredentialPool(t, ctx, databaseURL)
			configuration := pool.Config().Copy()
			configuration.MaxConns, configuration.MinConns, configuration.MinIdleConns = maxConns, 0, 0
			limited, err := pgxpool.NewWithConfig(ctx, configuration)
			if err != nil {
				t.Fatal(err)
			}
			defer limited.Close()
			if maxConns == 2 {
				listener, err := limited.Acquire(ctx)
				if err != nil {
					t.Fatal(err)
				}
				defer listener.Release()
				if _, err := listener.Exec(ctx, `LISTEN runtime_binding_test`); err != nil {
					t.Fatal(err)
				}
			}
			cipher, err := NewTokenCipher(bytes.Repeat([]byte{0x73}, 32))
			if err != nil {
				t.Fatal(err)
			}
			service, err := NewRuntimeCredentialService(RuntimeCredentialServiceOptions{
				Pool: limited, Cipher: cipher, Usage: NewRuntimeCredentialRepository(limited), Barrier: NewLifecycleBarrier(),
			})
			if err != nil {
				t.Fatal(err)
			}
			material, err := NewOTLPHeadersCredential(map[string]string{"Authorization": "Bearer test-credential"})
			if err != nil {
				t.Fatal(err)
			}
			defer material.Destroy()
			if _, err := service.Create(ctx, RuntimeCredentialCreateRequest{
				CredentialID: "binding-auth", Material: material, IdempotencyKey: "create-binding-auth", ActorID: "operator",
			}); err != nil {
				t.Fatal(err)
			}
			publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
				Pool: limited, RuntimeCredentials: service,
				PlannerTelemetryAdapters: runtimeconfig.PlannerTelemetryAdapterCatalogFunc(func(string) bool { return true }),
			})
			if err != nil {
				t.Fatal(err)
			}
			refs := make([]runtimeconfig.Ref, 2)
			for index := range refs {
				document := []byte(fmt.Sprintf(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
"metadata":{"name":"binding-config","version":"%d"},"spec":{"worker":{"telemetry":{
"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces","credential":"binding-auth"}}}}`, index+1))
				published, err := publisher.Publish(ctx, document, fmt.Sprintf("publish-binding-%d", index), "operator")
				if err != nil {
					t.Fatal(err)
				}
				refs[index] = published.Version.Ref
			}
			bindings, err := runtimeconfig.NewBindingService(limited, service)
			if err != nil {
				t.Fatal(err)
			}
			binding, err := bindings.Create(ctx, "authenticated", refs[0], "operator", time.Now().UTC())
			if err != nil {
				t.Fatal(err)
			}
			attempt, cancelAttempt := context.WithTimeout(ctx, 2*time.Second)
			defer cancelAttempt()
			updated, err := bindings.Rebind(attempt, binding.Label, binding.Revision, refs[1], "operator", time.Now().UTC())
			if err != nil {
				t.Fatalf("credential-backed rebind with one available connection: %v", err)
			}
			if updated.Ref != refs[1] || updated.Revision != binding.Revision+1 {
				t.Fatalf("rebind did not preserve revision/target contract: %+v", updated)
			}
			management, err := runtimeconfig.NewManagementService(limited, publisher, bindings)
			if err != nil {
				t.Fatal(err)
			}
			created, err := management.CreateBinding(attempt, "managed", refs[0], "managed-create", "operator", time.Now().UTC())
			if err != nil {
				t.Fatalf("management create with one available connection: %v", err)
			}
			rebound, err := management.Rebind(attempt, "managed", created.Binding.Revision, refs[1], "managed-rebind", "operator", time.Now().UTC())
			if err != nil {
				t.Fatalf("management rebind with one available connection: %v", err)
			}
			replayed, err := management.Rebind(attempt, "managed", created.Binding.Revision, refs[1], "managed-rebind", "operator", time.Now().UTC())
			if err != nil || !replayed.Replayed || replayed.Binding.Revision != rebound.Binding.Revision {
				t.Fatalf("management replay = %+v, %v", replayed, err)
			}
		})
	}
}
