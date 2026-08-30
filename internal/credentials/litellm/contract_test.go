package litellm

import (
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
)

func TestPinnedLiteLLMContract(t *testing.T) {
	origin := os.Getenv("CONTRACTOR_LITELLM_CONTRACT_URL")
	adminSecret := os.Getenv("CONTRACTOR_LITELLM_CONTRACT_ADMIN_KEY")
	if origin == "" || adminSecret == "" {
		t.Skip("pinned LiteLLM contract environment is not configured")
	}
	gateway := testGateway(origin, "8")
	plannerRef := contracts.ModelPolicyRef{
		PolicyID: "planner", Version: "1", Digest: "sha256:" + strings.Repeat("9", 64),
	}
	workerRef := contracts.ModelPolicyRef{
		PolicyID: "worker", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
	}
	policies := staticPolicies{
		"planner@1": {Ref: plannerRef, Model: "planner-model"},
		"worker@1":  {Ref: workerRef, Model: "worker-model"},
	}
	manager := newTestManager(t, gateway, adminSecret, policies, Options{RequestTimeout: 30 * time.Second})
	maxBudget := 7.5
	tpm, rpm, parallel := 12345, 23, 2
	request := credentials.ManagerCreateRequest{
		OperationID: "credop-pinned-contract", CredentialID: "pinned-contract", LLMGateway: gateway,
		Label: "Pinned compatibility contract",
		Policy: credentials.GatewayPolicy{
			ModelPolicies: []contracts.ModelPolicyRef{workerRef, plannerRef},
			MaxBudget:     &maxBudget, BudgetDuration: "1d",
			TPMLimit: &tpm, RPMLimit: &rpm, MaxParallelRequests: &parallel,
		},
	}
	if err := manager.ValidateCreate(t.Context(), request); err != nil {
		t.Fatal(err)
	}
	if err := manager.RecoverCreate(t.Context(), request); err != nil {
		t.Fatalf("initial deterministic cleanup: %v", err)
	}
	defer func() { _ = manager.RecoverCreate(t.Context(), request) }()
	first, err := manager.Create(t.Context(), request)
	if err != nil || first.Validate() != nil ||
		!equalStrings(first.EffectivePolicy.Models, []string{"planner-model", "worker-model"}) ||
		first.EffectivePolicy.MaxBudget == nil || *first.EffectivePolicy.MaxBudget != maxBudget ||
		first.EffectivePolicy.TPMLimit == nil || *first.EffectivePolicy.TPMLimit != tpm ||
		first.EffectivePolicy.RPMLimit == nil || *first.EffectivePolicy.RPMLimit != rpm ||
		first.EffectivePolicy.MaxParallelRequests == nil || *first.EffectivePolicy.MaxParallelRequests != parallel {
		t.Fatalf("pinned create result = (%+v, %v)", first, err)
	}
	if _, err := manager.Create(t.Context(), request); !errors.Is(err, credentials.ErrGatewayUnavailable) {
		t.Fatalf("pinned image did not enforce unique deterministic alias: %v", err)
	}
	if err := manager.RecoverCreate(t.Context(), request); err != nil {
		t.Fatalf("crash-window alias cleanup: %v", err)
	}
	if err := manager.RecoverCreate(t.Context(), request); err != nil {
		t.Fatalf("already-absent alias cleanup: %v", err)
	}
	second, err := manager.Create(t.Context(), request)
	if err != nil || second.Validate() != nil || second.RemoteKeyID == first.RemoteKeyID {
		t.Fatalf("create after recovery = (%+v, %v)", second, err)
	}
	deleteRequest := credentials.ManagerDeleteRequest{
		OperationID: "credop-pinned-delete", CredentialID: request.CredentialID,
		LLMGateway: gateway, RemoteKeyID: second.RemoteKeyID,
	}
	if err := manager.Delete(t.Context(), deleteRequest); err != nil {
		t.Fatalf("delete by token_id: %v", err)
	}
	if err := manager.Delete(t.Context(), deleteRequest); err != nil {
		t.Fatalf("already-absent token_id deletion: %v", err)
	}
}
