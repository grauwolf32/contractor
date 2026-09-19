package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestDevelopmentGatewayPrecedenceAndExactSelector(t *testing.T) {
	path := filepath.Join(t.TempDir(), "server.yaml")
	writeServerConfigTestFile(t, path, "apiVersion: contractor/v1alpha1\nkind: ServerConfig\nspec:\n  developmentLlmGateway: file-gateway@2\n")
	for _, test := range []struct {
		name        string
		args        []string
		environment string
		want        string
	}{
		{"default", nil, "", "local-litellm@1"},
		{"file", []string{"--config", path}, "", "file-gateway@2"},
		{"environment", []string{"--config", path}, "env-gateway@3", "env-gateway@3"},
		{"flag", []string{"--config", path, "--development-llm-gateway=flag-gateway@4"}, "env-gateway@3", "flag-gateway@4"},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg, err := ParseConfig(test.args, func(key string) string {
				if key == "CONTRACTOR_DEVELOPMENT_LLM_GATEWAY" {
					return test.environment
				}
				return ""
			})
			if err != nil || cfg.DevelopmentLLMGateway != test.want {
				t.Fatalf("selector = %q, %v", cfg.DevelopmentLLMGateway, err)
			}
		})
	}
	for _, invalid := range []string{"", "local-litellm", "local-litellm@", "gateway@one@two"} {
		if _, err := ParseConfig([]string{"--development-llm-gateway=" + invalid}, func(string) string { return "" }); err == nil {
			t.Fatalf("accepted non-exact selector %q", invalid)
		}
	}
}

func TestDevelopmentCredentialsUseSelectedGatewayWithoutChangingManagedCredentials(t *testing.T) {
	root := t.TempDir()
	if err := os.CopyFS(root, os.DirFS("../../configs")); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(root, "llm-gateways/local_litellm.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	raw = []byte(strings.Replace(string(raw), "name: local-litellm", "name: selected-gateway", 1))
	if err := os.WriteFile(filepath.Join(root, "llm-gateways/selected.yaml"), raw, 0600); err != nil {
		t.Fatal(err)
	}
	snapshot, err := workflowconfig.Load(root, workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	provider, err := developmentCredentials(snapshot, Config{
		DevelopmentLLMGateway:  "selected-gateway@1",
		DevelopmentWorkerToken: contracts.NewSecretString("test-token"),
	})
	if err != nil {
		t.Fatal(err)
	}
	selected, _ := snapshot.LLMGateway("selected-gateway@1")
	original, _ := snapshot.LLMGateway("local-litellm@1")
	metadata, err := provider.LookupLLMCredential(t.Context(), developmentWorkerCredential)
	if err != nil || metadata.LLMGateway != selected.Ref {
		t.Fatalf("wrong credential gateway: %+v, %v", metadata, err)
	}
	if _, err := provider.ResolveLLMCredential(t.Context(), metadata.Ref, original.Ref); err == nil {
		t.Fatal("development token resolved for another gateway")
	}
	// Ordinary managed credentials do not require the development gateway to exist.
	if _, err := developmentCredentials(snapshot, Config{DevelopmentLLMGateway: "missing@1"}); err != nil {
		t.Fatal(err)
	}
	if _, err := developmentCredentials(snapshot, Config{DevelopmentLLMGateway: "missing@1", DevelopmentWorkerToken: contracts.NewSecretString("test-token")}); err == nil {
		t.Fatal("missing development gateway accepted with bootstrap token")
	}
}
