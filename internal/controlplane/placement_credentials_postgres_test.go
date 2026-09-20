package controlplane

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPlacementManagedCredentialsDoNotBorrowAnotherConnection(t *testing.T) {
	for _, maxConns := range []int32{1, 2} {
		t.Run(fmt.Sprint(maxConns), func(t *testing.T) {
			setupCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			pool := isolatedPlacementPool(t, setupCtx)
			fixture := newPlacementFixture(t, setupCtx, pool, nil)
			fixture.registerCandidate(t, setupCtx, "runtime-review", "1", nil)
			cipher, err := credentials.NewTokenCipher(bytes.Repeat([]byte{0x71}, 32))
			if err != nil {
				t.Fatal(err)
			}
			token, err := credentials.NewToken("review-test-token")
			if err != nil {
				t.Fatal(err)
			}
			selection := fixture.selection
			envelope, err := cipher.Seal(selection.Credential.CredentialID, selection.LLMGateway.Ref, token)
			if err != nil {
				t.Fatal(err)
			}
			record := credentials.Record{
				CredentialID: selection.Credential.CredentialID,
				LLMGateway:   selection.LLMGateway.Ref,
				RemoteKeyID:  strings.Repeat("a", 64),
				EffectivePolicy: credentials.EffectiveGatewayPolicy{
					ModelPolicies: []contracts.ModelPolicyRef{selection.ModelPolicy.Ref},
					Models:        []string{selection.ModelPolicy.Model},
				},
				Envelope:  envelope,
				CreatedAt: time.Now().UTC(),
			}
			if _, err := pool.Exec(setupCtx, `INSERT INTO llm_credential_identities (credential_id, reserved_at) VALUES ($1, now())`, selection.Credential.CredentialID); err != nil {
				t.Fatal(err)
			}
			if err := credentials.NewRepository(pool).InsertCredential(setupCtx, record); err != nil {
				t.Fatal(err)
			}
			cfg := pool.Config().Copy()
			cfg.MaxConns, cfg.MinConns, cfg.MinIdleConns = maxConns, 0, 0
			limited, err := pgxpool.NewWithConfig(setupCtx, cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer limited.Close()
			if maxConns == 2 {
				listener, err := limited.Acquire(setupCtx)
				if err != nil {
					t.Fatal(err)
				}
				defer listener.Release()
				if _, err := listener.Exec(setupCtx, `LISTEN placement_credential_test`); err != nil {
					t.Fatal(err)
				}
			}
			provider, err := credentials.NewEncryptedProvider(credentials.NewRepository(limited), cipher)
			if err != nil {
				t.Fatal(err)
			}
			fixture.allocator.pool = limited
			fixture.allocator.llmCredentials = provider
			development, err := credentials.NewStaticProvider(nil)
			if err != nil {
				t.Fatal(err)
			}
			fixture.allocator.transactionLLMCredentials, err = credentials.NewTransactionLookupFactory(development)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := provider.LookupLLMCredential(setupCtx, selection.Credential.CredentialID); err != nil {
				t.Fatal(err)
			}
			attemptCtx, attemptCancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer attemptCancel()
			started := time.Now()
			result, err := fixture.allocator.ReserveAllContext(attemptCtx, fixture.request())
			t.Logf("MaxConns=%d allocations=%d elapsed=%v err=%v acquiredAfter=%d", maxConns, len(result), time.Since(started), err, limited.Stat().AcquiredConns())
			if err != nil {
				t.Fatalf("valid placement could not complete: %v", err)
			}
			if len(result) != 1 || result[0].ResolvedRuntimeConfig == nil ||
				result[0].ResolvedRuntimeConfig.LLMCredential == nil ||
				*result[0].ResolvedRuntimeConfig.LLMCredential != *selection.Credential {
				t.Fatalf("placement lost the managed credential selection: %+v", result)
			}
			allocations, err := runstore.NewPostgresStore(limited).ListStageAllocations(attemptCtx, fixture.stageExecutionID)
			if err != nil || len(allocations) != 1 || allocations[0].AllocationID != result[0].Grant.AllocationID {
				t.Fatalf("placement did not retain its durable allocation: %+v, %v", allocations, err)
			}
		})
	}
}

func TestPlacementRuntimeCredentialsDoNotBorrowAnotherConnection(t *testing.T) {
	for _, maxConns := range []int32{1, 2} {
		t.Run(fmt.Sprint(maxConns), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			pool := isolatedPlacementPool(t, ctx)
			fixture := newPlacementFixture(t, ctx, pool, nil)
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
				if _, err := listener.Exec(ctx, `LISTEN runtime_placement_test`); err != nil {
					t.Fatal(err)
				}
			}
			cipher, err := credentials.NewTokenCipher(bytes.Repeat([]byte{0x72}, 32))
			if err != nil {
				t.Fatal(err)
			}
			barrier := credentials.NewLifecycleBarrier()
			service, err := credentials.NewRuntimeCredentialService(credentials.RuntimeCredentialServiceOptions{
				Pool: limited, Cipher: cipher, Usage: credentials.NewRuntimeCredentialRepository(limited), Barrier: barrier,
			})
			if err != nil {
				t.Fatal(err)
			}
			material, err := credentials.NewOTLPHeadersCredential(map[string]string{"Authorization": "Bearer test-auth"})
			if err != nil {
				t.Fatal(err)
			}
			defer material.Destroy()
			if _, err := service.Create(ctx, credentials.RuntimeCredentialCreateRequest{
				CredentialID: "placement-auth", Material: material, IdempotencyKey: "create-placement-auth", ActorID: "operator",
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
			published, err := publisher.Publish(ctx, []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
"metadata":{"name":"authenticated-placement","version":"1"},"spec":{"worker":{"telemetry":{
"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces","credential":"placement-auth"}}}}`),
				"publish-placement-auth", "operator")
			if err != nil {
				t.Fatal(err)
			}
			if _, err := runtimeconfig.NewRepository(pool).CreateBinding(ctx, "authenticated", published.Version.Ref, "operator", time.Now().UTC()); err != nil {
				t.Fatal(err)
			}
			fixture.registerCandidateWithLabels(t, ctx, "runtime-authenticated", "1",
				[]contracts.RuntimeAdapterRef{contracts.RuntimeAdapterOTLPHTTP}, []string{"authenticated"})
			fixture.allocator.pool = limited
			fixture.allocator.runtimeCredentials = service
			fixture.allocator.credentialGuard = placementGuardFunc(barrier.WithCredentialReferences)
			attempt, cancelAttempt := context.WithTimeout(ctx, 2*time.Second)
			defer cancelAttempt()
			reservations, err := fixture.allocator.ReserveAllContext(attempt, fixture.request())
			if err != nil {
				t.Fatalf("credential-backed placement with one available connection: %v", err)
			}
			if len(reservations) != 1 || reservations[0].ResolvedRuntimeConfig == nil ||
				reservations[0].ResolvedRuntimeConfig.WorkerTelemetry == nil ||
				reservations[0].ResolvedRuntimeConfig.WorkerTelemetry.Credential != "placement-auth" {
				t.Fatalf("placement lost pinned Runtime credential: %+v", reservations)
			}
		})
	}
}
