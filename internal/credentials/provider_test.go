package credentials

import (
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestStaticProviderResolvesOnlyExactCredentialGatewayBinding(t *testing.T) {
	const secret = "recognizable-development-secret"
	gateway := testGatewayRef("local")
	provider, err := NewStaticProvider([]StaticEntry{{
		Metadata: config.CredentialMetadata{
			Ref:        contracts.LLMCredentialRef{CredentialID: "development-worker"},
			LLMGateway: gateway,
			ModelPolicies: []contracts.ModelPolicyRef{{
				PolicyID: "worker", Version: "1", Digest: "sha256:" + strings.Repeat("b", 64),
			}},
			Models: []string{"qwen/test"},
		},
		Token: contracts.NewSecretString(secret),
	}})
	if err != nil {
		t.Fatal(err)
	}

	metadata, err := provider.LookupLLMCredential(t.Context(), "development-worker")
	if err != nil || metadata.Ref.CredentialID != "development-worker" || metadata.LLMGateway != gateway {
		t.Fatalf("LookupLLMCredential = (%+v, %v)", metadata, err)
	}
	metadata.ModelPolicies[0].PolicyID = "mutated"
	metadata.Models[0] = "mutated"
	again, err := provider.LookupLLMCredential(t.Context(), "development-worker")
	if err != nil || again.ModelPolicies[0].PolicyID != "worker" || again.Models[0] != "qwen/test" {
		t.Fatalf("LookupLLMCredential returned aliased policy metadata: (%+v, %v)", again, err)
	}
	token, err := provider.ResolveLLMCredential(t.Context(), metadata.Ref, gateway)
	if err != nil || token.Reveal() != secret {
		t.Fatalf("ResolveLLMCredential = (%s, %v)", token, err)
	}

	wrongGateway := testGatewayRef("other")
	for _, call := range []func() error{
		func() error {
			_, currentErr := provider.LookupLLMCredential(t.Context(), "absent")
			return currentErr
		},
		func() error {
			_, currentErr := provider.ResolveLLMCredential(t.Context(), metadata.Ref, wrongGateway)
			return currentErr
		},
		func() error {
			_, currentErr := provider.ResolveLLMCredential(
				t.Context(), contracts.LLMCredentialRef{CredentialID: "absent"}, gateway,
			)
			return currentErr
		},
	} {
		currentErr := call()
		if !errors.Is(currentErr, ErrNotFound) || strings.Contains(currentErr.Error(), secret) {
			t.Fatalf("safe exact-binding error = %v", currentErr)
		}
	}
}

func TestStaticProviderRejectsInvalidDevelopmentEntries(t *testing.T) {
	gateway := testGatewayRef("local")
	valid := StaticEntry{
		Metadata: config.CredentialMetadata{
			Ref: contracts.LLMCredentialRef{CredentialID: "development-worker"}, LLMGateway: gateway,
		},
		Token: contracts.NewSecretString("secret"),
	}
	tests := []struct {
		name    string
		entries []StaticEntry
	}{
		{"invalid ID", []StaticEntry{{
			Metadata: config.CredentialMetadata{
				Ref: contracts.LLMCredentialRef{CredentialID: "two words"}, LLMGateway: gateway,
			},
			Token: contracts.NewSecretString("secret"),
		}}},
		{"invalid Gateway", []StaticEntry{{
			Metadata: config.CredentialMetadata{
				Ref: contracts.LLMCredentialRef{CredentialID: "development-worker"},
			},
			Token: contracts.NewSecretString("secret"),
		}}},
		{"empty token", []StaticEntry{{
			Metadata: valid.Metadata,
		}}},
		{"duplicate ID", []StaticEntry{valid, valid}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			provider, err := NewStaticProvider(test.entries)
			if err == nil || provider != nil {
				t.Fatalf("NewStaticProvider = (%v, %v), want nil/error", provider, err)
			}
		})
	}
}

func testGatewayRef(id string) contracts.LLMGatewayConfigRef {
	return contracts.LLMGatewayConfigRef{
		GatewayID: id, Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
	}
}
