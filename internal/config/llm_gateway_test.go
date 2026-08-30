package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestLoadLLMGatewayConfig(t *testing.T) {
	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	gateway, err := snapshot.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	if gateway.Protocol != contracts.OpenAICompatibleProtocol ||
		gateway.URL != "http://127.0.0.1:4000/v1" || gateway.CredentialManager == nil ||
		gateway.CredentialManager.Implementation != contracts.LiteLLMVirtualKeysManager ||
		gateway.CredentialManager.ManagementURL != "http://127.0.0.1:4000" {
		t.Fatalf("resolved LLMGatewayConfig = %+v", gateway)
	}
	const expectedDigest = "sha256:6e1bcf93a5d1fc64307dcd256a5c120f1bac54fe85e9d23d0aaafb84d4f14376"
	if gateway.Ref.Digest != expectedDigest {
		t.Fatalf("LLMGatewayConfig digest = %q, want %q", gateway.Ref.Digest, expectedDigest)
	}
	if listed := snapshot.LLMGateways(); len(listed) != 1 || listed[0].Ref != gateway.Ref {
		t.Fatalf("LLMGateways() = %+v", listed)
	}
}

func TestLLMGatewayDigestUsesNormalizedManifest(t *testing.T) {
	baseline := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	baselineGateway, _ := baseline.LLMGateway("local-litellm@1")
	root := copyConfigTree(t)
	path := filepath.Join(root, "llm-gateways/local_litellm.yaml")
	writeFile(t, path, []byte(`# presentation order and a root slash are normalized
kind: LLMGatewayConfig
spec:
  credentialManager:
    managementUrl: http://127.0.0.1:4000/
    implementation: litellm-virtual-keys@1
  url: http://127.0.0.1:4000/v1
  protocol: openai-compatible@1
metadata: {version: "1", name: local-litellm}
apiVersion: contractor/v1alpha1
`))
	variant := mustLoad(t, root, MVPDescriptors())
	variantGateway, _ := variant.LLMGateway("local-litellm@1")
	if variantGateway.Ref.Digest != baselineGateway.Ref.Digest ||
		variantGateway.CredentialManager.ManagementURL != "http://127.0.0.1:4000" {
		t.Fatalf("normalized variant = %+v, baseline = %+v", variantGateway, baselineGateway)
	}

	changedRoot := copyConfigTree(t)
	replaceFile(t, filepath.Join(changedRoot, "llm-gateways/local_litellm.yaml"),
		"url: http://127.0.0.1:4000/v1", "url: http://127.0.0.1:4001/v1")
	changed := mustLoad(t, changedRoot, MVPDescriptors())
	changedGateway, _ := changed.LLMGateway("local-litellm@1")
	if changedGateway.Ref.Digest == baselineGateway.Ref.Digest {
		t.Fatal("semantic URL change did not change the LLMGatewayConfig digest")
	}
}

func TestLLMGatewayIdentityComesFromManifest(t *testing.T) {
	root := copyConfigTree(t)
	oldPath := filepath.Join(root, "llm-gateways/local_litellm.yaml")
	newPath := filepath.Join(root, "llm-gateways/nested/arbitrary_filename.yaml")
	if err := os.MkdirAll(filepath.Dir(newPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(oldPath, newPath); err != nil {
		t.Fatal(err)
	}
	snapshot := mustLoad(t, root, MVPDescriptors())
	if _, err := snapshot.LLMGateway("local-litellm@1"); err != nil {
		t.Fatalf("metadata identity was not preserved: %v", err)
	}
}

func TestLLMGatewayManagerOriginRules(t *testing.T) {
	t.Run("loopback HTTP", func(t *testing.T) {
		mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	})
	t.Run("HTTPS", func(t *testing.T) {
		root := copyConfigTree(t)
		replaceFile(t, filepath.Join(root, "llm-gateways/local_litellm.yaml"),
			"http://127.0.0.1:4000\n", "https://gateway.example\n")
		mustLoad(t, root, MVPDescriptors())
	})
	for _, raw := range []string{
		"http://gateway.example", "http://localhost:4000", "https://gateway.example/admin",
	} {
		t.Run(raw, func(t *testing.T) {
			root := copyConfigTree(t)
			replaceFile(t, filepath.Join(root, "llm-gateways/local_litellm.yaml"),
				"http://127.0.0.1:4000\n", raw+"\n")
			if snapshot, err := Load(root, MVPDescriptors()); err == nil || snapshot != nil {
				t.Fatalf("Load(%q) = (%v, %v), want origin error", raw, snapshot, err)
			}
		})
	}
}

func TestInvalidLLMGatewayPreventsSnapshotPublication(t *testing.T) {
	const secret = "recognizable-litellm-admin-secret"
	tests := []struct {
		name   string
		mutate func(*testing.T, string)
		want   string
	}{
		{"URL userinfo", replaceGateway("url: http://127.0.0.1:4000/v1", "url: http://user:pass@127.0.0.1:4000/v1"), "spec.url"},
		{"URL query", replaceGateway("url: http://127.0.0.1:4000/v1", "url: http://127.0.0.1:4000/v1?route=x"), "spec.url"},
		{"URL fragment", replaceGateway("url: http://127.0.0.1:4000/v1", `url: "http://127.0.0.1:4000/v1#fragment"`), "spec.url"},
		{"URL without path", replaceGateway("url: http://127.0.0.1:4000/v1", "url: http://127.0.0.1:4000"), "inference path"},
		{"unknown protocol", replaceGateway("protocol: openai-compatible@1", "protocol: proprietary@9"), "spec.protocol"},
		{"secret field", appendGateway("  adminKey: " + secret + "\n"), "adminKey"},
		{"duplicate key", appendGateway("  url: http://127.0.0.1:4001/v1\n"), "decode strict YAML"},
		{"second document", appendGateway("---\napiVersion: contractor/v1alpha1\nkind: LLMGatewayConfig\n"), "exactly one"},
		{"wrong kind", replaceGateway("kind: LLMGatewayConfig", "kind: ModelPolicy"), "does not match LLMGatewayConfig subtree"},
		{"unknown manager", replaceGateway("implementation: litellm-virtual-keys@1", "implementation: custom@1"), "credentialManager.implementation"},
		{"duplicate identity", duplicateGateway, "duplicate LLMGatewayConfig identity local-litellm@1"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := copyConfigTree(t)
			test.mutate(t, root)
			snapshot, err := Load(root, MVPDescriptors())
			if err == nil || snapshot != nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Load() = (%v, %v), want error containing %q", snapshot, err, test.want)
			}
			if strings.Contains(err.Error(), secret) {
				t.Fatalf("configuration error leaked a secret value: %v", err)
			}
		})
	}
}

func replaceGateway(old, replacement string) func(*testing.T, string) {
	return func(t *testing.T, root string) {
		replaceFile(t, filepath.Join(root, "llm-gateways/local_litellm.yaml"), old, replacement)
	}
}

func appendGateway(value string) func(*testing.T, string) {
	return func(t *testing.T, root string) {
		appendFile(t, filepath.Join(root, "llm-gateways/local_litellm.yaml"), value)
	}
}

func duplicateGateway(t *testing.T, root string) {
	content := readFile(t, filepath.Join(root, "llm-gateways/local_litellm.yaml"))
	writeFile(t, filepath.Join(root, "llm-gateways/nested/duplicate.yaml"), content)
}
