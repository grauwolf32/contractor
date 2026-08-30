package litellm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
)

func TestManagerMapsCanonicalPolicyAndUsesSeparateDeleteIdentities(t *testing.T) {
	t.Parallel()
	const (
		adminSecret  = "sk-recognizable-manager-admin"
		generatedKey = "sk-generated-virtual-key"
		tokenID      = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	)
	var mu sync.Mutex
	var generatedRequest generateKeyRequest
	var deleteRequests []deleteKeyRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+adminSecret ||
			r.Header.Get("Accept") != "application/json" || r.Header.Get("Content-Type") != "application/json" {
			http.Error(w, "bad headers", http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/key/generate":
			decoder := json.NewDecoder(io.LimitReader(r.Body, maximumRequestBytes+1))
			decoder.DisallowUnknownFields()
			if err := decoder.Decode(&generatedRequest); err != nil {
				http.Error(w, "bad request", http.StatusBadRequest)
				return
			}
			response := validGenerateResponse(generatedRequest, generatedKey, tokenID)
			effectiveBudget := 8.5
			response.MaxBudget = &effectiveBudget
			if err := json.NewEncoder(w).Encode(response); err != nil {
				panic(err)
			}
		case "/key/delete":
			var request deleteKeyRequest
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				http.Error(w, "bad request", http.StatusBadRequest)
				return
			}
			mu.Lock()
			deleteRequests = append(deleteRequests, request)
			mu.Unlock()
			if len(request.KeyAliases) == 1 {
				w.WriteHeader(http.StatusNotFound)
				_, _ = io.WriteString(w, `{"error":{"message":"{'error': 'No keys found'}","type":"internal_server_error","param":"None","code":"404"}}`)
				return
			}
			_ = json.NewEncoder(w).Encode(deleteKeyResponse{DeletedKeys: request.Keys})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	gateway := testGateway(server.URL, "d")
	policies, refs := testPolicies()
	manager := newTestManager(t, gateway, adminSecret, policies, Options{})
	maxBudget := 10.0
	tpm, rpm, parallel := 1000, 20, 3
	request := credentials.ManagerCreateRequest{
		OperationID: "credop-create-1", CredentialID: "managed-worker", LLMGateway: gateway,
		Label: "Worker key",
		Policy: credentials.GatewayPolicy{
			ModelPolicies: []contracts.ModelPolicyRef{refs[1], refs[0]},
			MaxBudget:     &maxBudget, BudgetDuration: "1d",
			TPMLimit: &tpm, RPMLimit: &rpm, MaxParallelRequests: &parallel,
		},
	}
	if err := manager.ValidateCreate(t.Context(), request); err != nil {
		t.Fatal(err)
	}
	generated, err := manager.Create(t.Context(), request)
	if err != nil || generated.RemoteKeyID != tokenID || generated.Validate() != nil {
		t.Fatalf("generated credential = (%+v, %v)", generated, err)
	}
	if generated.EffectivePolicy.MaxBudget == nil || *generated.EffectivePolicy.MaxBudget != 8.5 ||
		!equalStrings(generated.EffectivePolicy.Models, []string{"alpha-model", "zeta-model"}) ||
		len(generated.EffectivePolicy.ModelPolicies) != 2 {
		t.Fatalf("effective policy = %+v", generated.EffectivePolicy)
	}
	alias, _ := KeyAlias(gateway.Ref, request.CredentialID)
	const expectedAlias = "contractor-a1e314c68f4609179c485eafbe62acbefbf5c243057792e24b869a543107a49d"
	if alias != expectedAlias {
		t.Fatalf("deterministic alias = %q, want %q", alias, expectedAlias)
	}
	if generatedRequest.KeyAlias != alias || generatedRequest.KeyType != "llm_api" ||
		!equalStrings(generatedRequest.Models, []string{"alpha-model", "zeta-model"}) ||
		generatedRequest.MaxBudget == nil || *generatedRequest.MaxBudget != maxBudget ||
		generatedRequest.BudgetDuration != "1d" || generatedRequest.Metadata.CredentialID != request.CredentialID ||
		generatedRequest.Metadata.GatewayDigest != gateway.Ref.Digest || generatedRequest.Metadata.OperationID != request.OperationID {
		t.Fatalf("LiteLLM generate request = %+v", generatedRequest)
	}
	if err := manager.Delete(t.Context(), credentials.ManagerDeleteRequest{
		OperationID: "credop-delete-1", CredentialID: request.CredentialID,
		LLMGateway: gateway, RemoteKeyID: tokenID,
	}); err != nil {
		t.Fatal(err)
	}
	if err := manager.RecoverCreate(t.Context(), request); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	deletes := append([]deleteKeyRequest(nil), deleteRequests...)
	mu.Unlock()
	if len(deletes) != 2 || len(deletes[0].Keys) != 1 || deletes[0].Keys[0] != tokenID ||
		len(deletes[0].KeyAliases) != 0 || len(deletes[1].KeyAliases) != 1 ||
		deletes[1].KeyAliases[0] != alias || len(deletes[1].Keys) != 0 {
		t.Fatalf("LiteLLM delete requests = %+v", deletes)
	}
	if strings.Contains(fmt.Sprint(generated), generatedKey) || strings.Contains(fmt.Sprint(manager), adminSecret) {
		t.Fatal("secret-bearing value has unsafe default formatting")
	}
}

func TestManagerRejectsMalformedOrUnboundedGatewayResponsesWithoutSecrets(t *testing.T) {
	t.Parallel()
	const adminSecret = "sk-provider-secret-admin"
	policies, refs := testPolicies()
	baseRequest := func(gateway contracts.ResolvedLLMGatewayConfig) credentials.ManagerCreateRequest {
		return credentials.ManagerCreateRequest{
			OperationID: "credop-malformed", CredentialID: "managed-malformed", LLMGateway: gateway,
			Policy: credentials.GatewayPolicy{ModelPolicies: refs[:1]},
		}
	}
	tests := map[string]func(generateKeyRequest) (int, string, string){
		"provider error": func(generateKeyRequest) (int, string, string) {
			return http.StatusBadGateway, "application/json", `{"error":"sk-provider-body-secret"}`
		},
		"unknown field": func(request generateKeyRequest) (int, string, string) {
			response := validGenerateResponse(request, "sk-generated", strings.Repeat("a", 64))
			encoded, _ := json.Marshal(response)
			return http.StatusOK, "application/json", strings.TrimSuffix(string(encoded), "}") + `,"new_provider_field":true}`
		},
		"invalid key": func(request generateKeyRequest) (int, string, string) {
			response := validGenerateResponse(request, "not-a-key", strings.Repeat("a", 64))
			encoded, _ := json.Marshal(response)
			return http.StatusOK, "application/json", string(encoded)
		},
		"invalid token id": func(request generateKeyRequest) (int, string, string) {
			response := validGenerateResponse(request, "sk-generated", "ABC")
			encoded, _ := json.Marshal(response)
			return http.StatusOK, "application/json", string(encoded)
		},
		"management route": func(request generateKeyRequest) (int, string, string) {
			response := validGenerateResponse(request, "sk-generated", strings.Repeat("a", 64))
			response.AllowedRoutes = []string{"management_routes"}
			encoded, _ := json.Marshal(response)
			return http.StatusOK, "application/json", string(encoded)
		},
		"mismatched models": func(request generateKeyRequest) (int, string, string) {
			response := validGenerateResponse(request, "sk-generated", strings.Repeat("a", 64))
			response.Models = []string{"another-model"}
			encoded, _ := json.Marshal(response)
			return http.StatusOK, "application/json", string(encoded)
		},
		"invalid effective policy": func(request generateKeyRequest) (int, string, string) {
			response := validGenerateResponse(request, "sk-generated", strings.Repeat("a", 64))
			duration := "1d"
			response.MaxBudget = nil
			response.BudgetDuration = &duration
			encoded, _ := json.Marshal(response)
			return http.StatusOK, "application/json", string(encoded)
		},
		"wrong content type": func(request generateKeyRequest) (int, string, string) {
			response := validGenerateResponse(request, "sk-generated", strings.Repeat("a", 64))
			encoded, _ := json.Marshal(response)
			return http.StatusOK, "text/plain", string(encoded)
		},
		"oversized": func(generateKeyRequest) (int, string, string) {
			return http.StatusOK, "application/json", strings.Repeat("x", 2048)
		},
	}
	for name, responseFor := range tests {
		name, responseFor := name, responseFor
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") != "Bearer "+adminSecret {
					t.Error("admin Authorization header missing")
				}
				var request generateKeyRequest
				_ = json.NewDecoder(r.Body).Decode(&request)
				status, contentType, body := responseFor(request)
				w.Header().Set("Content-Type", contentType)
				w.WriteHeader(status)
				_, _ = io.WriteString(w, body)
			}))
			defer server.Close()
			gateway := testGateway(server.URL, "e")
			manager := newTestManager(t, gateway, adminSecret, policies, Options{MaxResponseBytes: 1024})
			_, err := manager.Create(t.Context(), baseRequest(gateway))
			if !errors.Is(err, credentials.ErrGatewayUnavailable) {
				t.Fatalf("Create error = %v", err)
			}
			if strings.Contains(fmt.Sprint(err), adminSecret) || strings.Contains(fmt.Sprint(err), "provider-body-secret") {
				t.Fatalf("unsafe Gateway error = %v", err)
			}
		})
	}
}

func TestManagerNeverFollowsManagementRedirect(t *testing.T) {
	t.Parallel()
	var redirected atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		redirected.Add(1)
	}))
	defer target.Close()
	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Location", target.URL+"/key/generate")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTemporaryRedirect)
		_, _ = io.WriteString(w, `{}`)
	}))
	defer source.Close()
	gateway := testGateway(source.URL, "f")
	policies, refs := testPolicies()
	manager := newTestManager(t, gateway, "sk-redirect-admin", policies, Options{})
	_, err := manager.Create(t.Context(), credentials.ManagerCreateRequest{
		OperationID: "credop-redirect", CredentialID: "managed-redirect", LLMGateway: gateway,
		Policy: credentials.GatewayPolicy{ModelPolicies: refs[:1]},
	})
	if !errors.Is(err, credentials.ErrGatewayUnavailable) || redirected.Load() != 0 {
		t.Fatalf("redirect result = (err=%v, target calls=%d)", err, redirected.Load())
	}
}

func TestManagerRejectsUnconfirmedOrMalformedDeleteResponses(t *testing.T) {
	t.Parallel()
	const tokenID = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
	policies, _ := testPolicies()
	tests := map[string]struct {
		status      int
		contentType string
		body        string
	}{
		"different deleted key": {
			status: http.StatusOK, contentType: "application/json",
			body: `{"deleted_keys":["0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"]}`,
		},
		"unknown success field": {
			status: http.StatusOK, contentType: "application/json",
			body: `{"deleted_keys":["` + tokenID + `"],"provider":true}`,
		},
		"unconfirmed not found": {
			status: http.StatusNotFound, contentType: "application/json",
			body: `{"error":{"message":"another resource was absent","type":"internal_server_error","param":"None","code":"404"}}`,
		},
		"non JSON": {
			status: http.StatusOK, contentType: "text/plain",
			body: `{"deleted_keys":["` + tokenID + `"]}`,
		},
	}
	for name, test := range tests {
		name, test := name, test
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", test.contentType)
				w.WriteHeader(test.status)
				_, _ = io.WriteString(w, test.body)
			}))
			defer server.Close()
			gateway := testGateway(server.URL, "5")
			manager := newTestManager(t, gateway, "sk-delete-admin", policies, Options{})
			err := manager.Delete(t.Context(), credentials.ManagerDeleteRequest{
				OperationID: "credop-delete-malformed", CredentialID: "managed-delete",
				LLMGateway: gateway, RemoteKeyID: tokenID,
			})
			if !errors.Is(err, credentials.ErrGatewayUnavailable) {
				t.Fatalf("Delete error = %v", err)
			}
		})
	}
}

func TestManagerValidatesExactBindingAndPolicyBeforeNetwork(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { calls.Add(1) }))
	defer server.Close()
	gateway := testGateway(server.URL, "1")
	policies, refs := testPolicies()
	manager := newTestManager(t, gateway, "sk-validation-admin", policies, Options{})
	wrongGateway := gateway
	wrongGateway.Ref.Digest = "sha256:" + strings.Repeat("2", 64)
	request := credentials.ManagerCreateRequest{
		OperationID: "credop-validation", CredentialID: "managed-validation", LLMGateway: wrongGateway,
		Policy: credentials.GatewayPolicy{ModelPolicies: refs[:1]},
	}
	if err := manager.ValidateCreate(t.Context(), request); !errors.Is(err, credentials.ErrManagerUnavailable) {
		t.Fatalf("wrong binding error = %v", err)
	}
	request.LLMGateway = gateway
	request.Policy.ModelPolicies[0].Digest = "sha256:" + strings.Repeat("3", 64)
	if err := manager.ValidateCreate(t.Context(), request); !errors.Is(err, credentials.ErrInvalid) {
		t.Fatalf("wrong ModelPolicy error = %v", err)
	}
	oversizedModel := policies["zeta@1"]
	request.Policy.ModelPolicies = []contracts.ModelPolicyRef{oversizedModel.Ref}
	oversizedModel.Model = strings.Repeat("m", 257)
	policies["zeta@1"] = oversizedModel
	if err := manager.ValidateCreate(t.Context(), request); !errors.Is(err, credentials.ErrInvalid) {
		t.Fatalf("oversized derived model error = %v", err)
	}
	if err := manager.Delete(t.Context(), credentials.ManagerDeleteRequest{
		OperationID: "credop-invalid-delete", CredentialID: "managed-validation",
		LLMGateway: gateway, RemoteKeyID: "not-a-token-id",
	}); !errors.Is(err, credentials.ErrInvalid) {
		t.Fatalf("invalid remote key ID error = %v", err)
	}
	if calls.Load() != 0 {
		t.Fatalf("preflight made %d network calls", calls.Load())
	}
}

func TestManagerRequestTimeoutIsBounded(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(time.Second):
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{}`)
		}
	}))
	defer server.Close()
	gateway := testGateway(server.URL, "4")
	policies, refs := testPolicies()
	manager := newTestManager(t, gateway, "sk-timeout-admin", policies, Options{RequestTimeout: 25 * time.Millisecond})
	started := time.Now()
	_, err := manager.Create(context.Background(), credentials.ManagerCreateRequest{
		OperationID: "credop-timeout", CredentialID: "managed-timeout", LLMGateway: gateway,
		Policy: credentials.GatewayPolicy{ModelPolicies: refs[:1]},
	})
	if !errors.Is(err, credentials.ErrGatewayUnavailable) || time.Since(started) > 500*time.Millisecond {
		t.Fatalf("bounded timeout = (%v, %s)", err, time.Since(started))
	}
}

type staticPolicies map[string]contracts.ResolvedModelPolicy

func (s staticPolicies) ModelPolicy(selector string) (contracts.ResolvedModelPolicy, error) {
	policy, exists := s[selector]
	if !exists {
		return contracts.ResolvedModelPolicy{}, errors.New("not found")
	}
	return policy, nil
}

func testPolicies() (staticPolicies, []contracts.ModelPolicyRef) {
	first := contracts.ModelPolicyRef{
		PolicyID: "zeta", Version: "1", Digest: "sha256:" + strings.Repeat("6", 64),
	}
	second := contracts.ModelPolicyRef{
		PolicyID: "alpha", Version: "1", Digest: "sha256:" + strings.Repeat("7", 64),
	}
	return staticPolicies{
		"zeta@1":  {Ref: first, Model: "zeta-model"},
		"alpha@1": {Ref: second, Model: "alpha-model"},
	}, []contracts.ModelPolicyRef{first, second}
}

func newTestManager(
	t *testing.T,
	gateway contracts.ResolvedLLMGatewayConfig,
	adminSecret string,
	policies staticPolicies,
	options Options,
) *Manager {
	t.Helper()
	bindings := &AdminBindings{bindings: map[contracts.LLMGatewayConfigRef]exactBinding{
		gateway.Ref: {managementURL: gateway.CredentialManager.ManagementURL, key: adminKey{value: adminSecret}},
	}}
	manager, err := NewManager(bindings, policies, options)
	if err != nil {
		t.Fatal(err)
	}
	return manager
}

func validGenerateResponse(request generateKeyRequest, key, tokenID string) generateKeyResponse {
	return generateKeyResponse{
		Key: key, Token: tokenID, TokenID: tokenID, KeyAlias: request.KeyAlias,
		Models: request.Models, Metadata: request.Metadata,
		MaxBudget: cloneFloat(request.MaxBudget), TPMLimit: cloneInt(request.TPMLimit),
		RPMLimit: cloneInt(request.RPMLimit), MaxParallelRequests: cloneInt(request.MaxParallelRequests),
		AllowedRoutes: []string{liteLLMAllowedRoute},
		BudgetDuration: func() *string {
			if request.BudgetDuration == "" {
				return nil
			}
			value := request.BudgetDuration
			return &value
		}(),
	}
}
