package credentials

import (
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestManagerRegistryRequiresExactDeclaredImplementation(t *testing.T) {
	t.Parallel()
	manager := newFakeGatewayManager()
	registry, err := NewManagerRegistry(ManagerRegistration{
		Implementation: contracts.LiteLLMVirtualKeysManager, Manager: manager,
	})
	if err != nil {
		t.Fatal(err)
	}
	gateway := contracts.ResolvedLLMGatewayConfig{
		Ref:      testPinnedGatewayRef("1", strings.Repeat("1", 64)),
		Protocol: contracts.OpenAICompatibleProtocol, URL: "http://127.0.0.1:4000/v1",
		CredentialManager: &contracts.LLMGatewayCredentialManager{
			Implementation: contracts.LiteLLMVirtualKeysManager,
			ManagementURL:  "http://127.0.0.1:4000",
		},
	}
	if got, err := registry.ForGateway(gateway); err != nil || got != manager {
		t.Fatalf("registered manager = (%v, %v)", got, err)
	}
	gateway.CredentialManager = nil
	if _, err := registry.ForGateway(gateway); !errors.Is(err, ErrManagerUnavailable) {
		t.Fatalf("unmanaged Gateway error = %v", err)
	}
	if _, err := NewManagerRegistry(
		ManagerRegistration{Implementation: contracts.LiteLLMVirtualKeysManager, Manager: manager},
		ManagerRegistration{Implementation: contracts.LiteLLMVirtualKeysManager, Manager: manager},
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("duplicate manager registration error = %v", err)
	}
}
