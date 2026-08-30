package litellm

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
)

func TestLoadAdminBindingsResolvesExactGatewayAndRedactsKey(t *testing.T) {
	t.Parallel()
	gateway := testGateway("http://127.0.0.1:4011", "a")
	secret := "sk-recognizable-admin-key"
	keyPath := writeTestFile(t, "admin-key", secret+"\n", 0o600)
	bindingsPath := writeBindingsDocument(t, gateway.Ref, keyPath, "")
	bindings, err := LoadAdminBindings(bindingsPath, staticGateways{gateway: gateway})
	if err != nil {
		t.Fatal(err)
	}
	binding, err := bindings.bindingFor(gateway)
	if err != nil || binding.managementURL != gateway.CredentialManager.ManagementURL ||
		binding.key.reveal() != secret {
		t.Fatalf("exact binding = (%+v, %v)", binding, err)
	}
	formatted := fmt.Sprintf("%v %#v", binding.key, binding.key)
	if strings.Contains(formatted, secret) || !strings.Contains(formatted, "REDACTED") {
		t.Fatalf("admin key formatting is unsafe: %q", formatted)
	}
	forged := gateway
	forged.CredentialManager = &contracts.LLMGatewayCredentialManager{
		Implementation: contracts.LiteLLMVirtualKeysManager,
		ManagementURL:  "http://127.0.0.1:4012",
	}
	if _, err := bindings.bindingFor(forged); !errors.Is(err, credentials.ErrManagerUnavailable) {
		t.Fatalf("forged management origin error = %v", err)
	}
}

func TestLoadAdminBindingsRejectsUnsafeOrInexactBootstrap(t *testing.T) {
	t.Parallel()
	gateway := testGateway("http://127.0.0.1:4013", "b")
	secret := "sk-secret-must-not-appear"
	validKey := writeTestFile(t, "valid-admin-key", secret, 0o600)

	tests := map[string]func(*testing.T) string{
		"unknown field": func(t *testing.T) string {
			return writeBindingsDocument(t, gateway.Ref, validKey, "unknown: true\n")
		},
		"duplicate field": func(t *testing.T) string {
			body := bindingYAML(gateway.Ref, validKey) + "bindings: []\n"
			return writeTestFile(t, "duplicate-bindings.yaml", body, 0o600)
		},
		"duplicate ref": func(t *testing.T) string {
			entry := strings.TrimPrefix(bindingYAML(gateway.Ref, validKey), "bindings:\n")
			return writeTestFile(t, "duplicate-ref.yaml", "bindings:\n"+entry+entry, 0o600)
		},
		"wrong digest": func(t *testing.T) string {
			wrong := gateway.Ref
			wrong.Digest = "sha256:" + strings.Repeat("c", 64)
			return writeBindingsDocument(t, wrong, validKey, "")
		},
		"relative key path": func(t *testing.T) string {
			return writeBindingsDocument(t, gateway.Ref, "relative-admin-key", "")
		},
		"insecure key permissions": func(t *testing.T) string {
			path := writeTestFile(t, "insecure-admin-key", secret, 0o640)
			return writeBindingsDocument(t, gateway.Ref, path, "")
		},
		"malformed key": func(t *testing.T) string {
			path := writeTestFile(t, "malformed-admin-key", "not-an-sk-key", 0o600)
			return writeBindingsDocument(t, gateway.Ref, path, "")
		},
		"key symlink": func(t *testing.T) string {
			link := filepath.Join(t.TempDir(), "admin-link")
			if err := os.Symlink(validKey, link); err != nil {
				t.Fatal(err)
			}
			return writeBindingsDocument(t, gateway.Ref, link, "")
		},
		"writable document": func(t *testing.T) string {
			path := writeBindingsDocument(t, gateway.Ref, validKey, "")
			if err := os.Chmod(path, 0o660); err != nil {
				t.Fatal(err)
			}
			return path
		},
		"document symlink": func(t *testing.T) string {
			target := writeBindingsDocument(t, gateway.Ref, validKey, "")
			link := filepath.Join(t.TempDir(), "bindings-link")
			if err := os.Symlink(target, link); err != nil {
				t.Fatal(err)
			}
			return link
		},
	}
	for name, makePath := range tests {
		name, makePath := name, makePath
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			path := makePath(t)
			_, err := LoadAdminBindings(path, staticGateways{gateway: gateway})
			if !errors.Is(err, credentials.ErrManagerUnavailable) {
				t.Fatalf("LoadAdminBindings error = %v", err)
			}
			if strings.Contains(fmt.Sprint(err), secret) {
				t.Fatalf("binding error contains admin key: %v", err)
			}
		})
	}
}

func TestLoadAdminBindingsAllowsExplicitlyEmptyConfigurationOnly(t *testing.T) {
	t.Parallel()
	bindings, err := LoadAdminBindings("", nil)
	if err != nil || bindings == nil || len(bindings.bindings) != 0 {
		t.Fatalf("empty binding configuration = (%+v, %v)", bindings, err)
	}
	path := writeTestFile(t, "empty-bindings.yaml", "bindings: []\n", 0o600)
	if _, err := LoadAdminBindings(path, staticGateways{}); !errors.Is(err, credentials.ErrManagerUnavailable) {
		t.Fatalf("explicit empty binding document error = %v", err)
	}
}

type staticGateways struct {
	gateway contracts.ResolvedLLMGatewayConfig
}

func (s staticGateways) LLMGateway(selector string) (contracts.ResolvedLLMGatewayConfig, error) {
	if selector != s.gateway.Ref.GatewayID+"@"+s.gateway.Ref.Version {
		return contracts.ResolvedLLMGatewayConfig{}, errors.New("not found")
	}
	return s.gateway, nil
}

func testGateway(origin, digestCharacter string) contracts.ResolvedLLMGatewayConfig {
	return contracts.ResolvedLLMGatewayConfig{
		Ref: contracts.LLMGatewayConfigRef{
			GatewayID: "local-litellm", Version: "1",
			Digest: "sha256:" + strings.Repeat(digestCharacter, 64),
		},
		Protocol: contracts.OpenAICompatibleProtocol,
		URL:      origin + "/v1",
		CredentialManager: &contracts.LLMGatewayCredentialManager{
			Implementation: contracts.LiteLLMVirtualKeysManager,
			ManagementURL:  origin,
		},
	}
}

func writeBindingsDocument(
	t *testing.T,
	ref contracts.LLMGatewayConfigRef,
	keyPath, suffix string,
) string {
	t.Helper()
	return writeTestFile(t, "admin-bindings.yaml", bindingYAML(ref, keyPath)+suffix, 0o600)
}

func bindingYAML(ref contracts.LLMGatewayConfigRef, keyPath string) string {
	return fmt.Sprintf(`bindings:
  - llmGateway:
      gatewayId: %s
      version: %q
      digest: %s
    adminKeyFile: %s
`, ref.GatewayID, ref.Version, ref.Digest, keyPath)
}

func writeTestFile(t *testing.T, name, contents string, mode os.FileMode) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(contents), mode); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(path, mode); err != nil {
		t.Fatal(err)
	}
	return path
}
