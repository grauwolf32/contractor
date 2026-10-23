package public

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// A Server started without a credential master key answers every LLM and
// Runtime credential creation and deletion with the permanent 503
// credential_key_unavailable, and stores nothing and calls no Gateway
// (RCF-R44, FAIL-RCF-1, FAIL-RCF-2, FAIL-RCF-8). The services are the real
// ones on PostgreSQL, built without a cipher as the Server composes them.
func TestPostgresCredentialMutationsWithoutMasterKeyArePermanent(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	manager := &countingGatewayCredentialManager{}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t),
		mustTestOrigins(t), false, nil, func(dependencies *testServices) {
			gateways, ok := dependencies.Config.(credentials.GatewayLookup)
			if !ok {
				t.Fatal("the test catalog does not resolve LLM Gateways")
			}
			managers, err := credentials.NewManagerRegistry(credentials.ManagerRegistration{
				Implementation: llmgateway.LiteLLMVirtualKeysManager, Manager: manager,
			})
			if err != nil {
				t.Fatal(err)
			}
			barrier := credentials.NewLifecycleBarrier()
			managed, err := credentials.NewService(credentials.ServiceOptions{
				Pool: pool, Gateways: gateways, Managers: managers,
				Runs: runstore.NewPostgresStore(pool), Audits: hookstore.NewCredentialHolds(pool),
				Barrier: barrier,
			})
			if err != nil {
				t.Fatal(err)
			}
			if err := managed.Recover(ctx); err != nil {
				t.Fatal(err)
			}
			runtimeCredentials, err := credentials.NewRuntimeCredentialService(
				credentials.RuntimeCredentialServiceOptions{
					Pool: pool, Usage: credentials.NewRuntimeCredentialRepository(pool),
					Audits: hookstore.NewCredentialHolds(pool), Barrier: barrier,
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			dependencies.ManagedCredentials = managed
			dependencies.RuntimeCredentials = runtimeCredentials
		})

	gateway, err := fixture.configs.Snapshot().LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	policy, err := fixture.configs.Snapshot().ModelPolicy("worker@1")
	if err != nil {
		t.Fatal(err)
	}
	createBody, err := json.Marshal(createCredentialRequest{
		CredentialID: "managed-worker", LLMGateway: gateway.Ref,
		GatewayPolicy: credentials.GatewayPolicy{ModelPolicies: []contracts.ModelPolicyRef{policy.Ref}},
	})
	if err != nil {
		t.Fatal(err)
	}
	requests := map[string]func() *http.Request{
		"create LLM credential": func() *http.Request {
			return runtimeMutationRequest(t, http.MethodPost, "/v1/operations/credentials",
				string(createBody), "create-without-key", "", "")
		},
		"delete LLM credential": func() *http.Request {
			return runtimeMutationRequest(t, http.MethodDelete,
				"/v1/operations/credentials/managed-worker", "", "delete-without-key", "", "")
		},
		"create Runtime credential": func() *http.Request {
			return runtimeMutationRequest(t, http.MethodPost, "/v1/operations/runtime-credentials",
				`{"credentialId":"otel-debug","kind":"otlp-headers@1",`+
					`"material":{"headers":{"authorization":"Bearer canary"}}}`,
				"runtime-create-without-key", "", "")
		},
		"delete Runtime credential": func() *http.Request {
			return runtimeMutationRequest(t, http.MethodDelete,
				"/v1/operations/runtime-credentials/otel-debug", "", "", "", "")
		},
	}
	for name, request := range requests {
		// A retry sees the same answer: the condition lasts until an operator
		// configures the key and restarts the Server.
		for attempt := range 2 {
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, request())
			assertErrorEnvelope(t, response, http.StatusServiceUnavailable, "credential_key_unavailable", false)
			var envelope errorResponse
			if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
				t.Fatal(err)
			}
			if envelope.Message != "credential master key is not configured on this Server" ||
				envelope.RequestID == "" || bytes.Contains(response.Body.Bytes(), []byte("canary")) {
				t.Fatalf("%s attempt %d = %s", name, attempt, response.Body.String())
			}
		}
	}
	if calls := manager.calls.Load(); calls != 0 {
		t.Fatalf("Gateway credential manager calls = %d", calls)
	}
	assertNoCredentialRows(t, ctx, pool)
	if revision := fixture.operations.SnapshotOperations().Cursor.Revision; revision != 0 {
		t.Fatalf("refused credential mutations advanced the Operations revision to %d", revision)
	}
}

func assertNoCredentialRows(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	for _, table := range []string{"llm_credentials", "credential_operations", "runtime_credentials"} {
		var count int
		if err := pool.QueryRow(ctx, `SELECT count(*) FROM `+table).Scan(&count); err != nil {
			t.Fatal(err)
		}
		if count != 0 {
			t.Fatalf("%s holds %d rows after refused mutations", table, count)
		}
	}
}

// countingGatewayCredentialManager fails every call and counts it, so a test
// can prove that a refused credential mutation made no Gateway call.
type countingGatewayCredentialManager struct{ calls atomic.Int32 }

func (m *countingGatewayCredentialManager) ValidateCreate(context.Context, credentials.ManagerCreateRequest) error {
	m.calls.Add(1)
	return credentials.ErrGatewayUnavailable
}

func (m *countingGatewayCredentialManager) Create(
	context.Context, credentials.ManagerCreateRequest,
) (credentials.GeneratedCredential, error) {
	m.calls.Add(1)
	return credentials.GeneratedCredential{}, credentials.ErrGatewayUnavailable
}

func (m *countingGatewayCredentialManager) Delete(context.Context, credentials.ManagerDeleteRequest) error {
	m.calls.Add(1)
	return credentials.ErrGatewayUnavailable
}

func (m *countingGatewayCredentialManager) RecoverCreate(context.Context, credentials.ManagerCreateRequest) error {
	m.calls.Add(1)
	return credentials.ErrGatewayUnavailable
}
