package config

import (
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"testing"
)

func TestSharedGatewayEndpointAdmission(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Name  string
			URL   string
			Valid bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "runtime-endpoint-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			_, err := resolveLLMGatewayConfig(resolved.Selector{ID: "gateway", Version: "1"}, &llmGatewayConfigSpecSource{Protocol: "openai-compatible@1", URL: c.URL})
			if (err == nil) != c.Valid {
				t.Fatalf("admission valid=%v, want %v: %v", err == nil, c.Valid, err)
			}
		})
	}
}
