package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
)

// The Runtime recomputes these digests from the same cases in
// runtime/tests/test_digest_cases.py.
func TestSharedDigestCases(t *testing.T) {
	t.Parallel()
	data, err := os.ReadFile(filepath.Join("..", "..", "api", "testdata", "v1alpha1", "digest-cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases struct {
		ModelPolicies     []contracts.ResolvedModelPolicy       `json:"modelPolicies"`
		LLMGatewayConfigs []llmgateway.ResolvedLLMGatewayConfig `json:"llmGatewayConfigs"`
		AgentTemplates    []contracts.ResolvedAgentTemplate     `json:"agentTemplates"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	checkPolicy := func(policy contracts.ResolvedModelPolicy) {
		t.Helper()
		got, err := modelPolicyDigest(Selector{ID: policy.Ref.PolicyID, Version: policy.Ref.Version}, policy)
		if err != nil || got != policy.Ref.Digest {
			t.Errorf("ModelPolicy %s digest = %s, %v; want %s", policy.Ref.PolicyID, got, err, policy.Ref.Digest)
		}
	}
	checkInstructions := func(owner string, instructions contracts.ResolvedInstructions) {
		t.Helper()
		if got := contentdigest.Bytes([]byte(instructions.Text)); got != instructions.Digest {
			t.Errorf("%s instructions digest = %s, want %s", owner, got, instructions.Digest)
		}
	}
	for _, policy := range cases.ModelPolicies {
		checkPolicy(policy)
	}
	for _, gateway := range cases.LLMGatewayConfigs {
		got, err := llmGatewayConfigDigest(Selector{ID: gateway.Ref.GatewayID, Version: gateway.Ref.Version}, gateway)
		if err != nil || got != gateway.Ref.Digest {
			t.Errorf("LLMGatewayConfig %s digest = %s, %v; want %s", gateway.Ref.GatewayID, got, err, gateway.Ref.Digest)
		}
	}
	for _, template := range cases.AgentTemplates {
		if !template.IsToolWorker() {
			checkInstructions(template.Ref.TemplateID, template.Instructions)
			checkPolicy(template.ModelPolicy)
		}
		if summarizer := template.Summarizer; summarizer != nil {
			checkPolicy(summarizer.ModelPolicy)
			if summarizer.Instructions != nil {
				checkInstructions(template.Ref.TemplateID+" summarizer", *summarizer.Instructions)
			}
		}
		got, err := agentTemplateDigest(Selector{ID: template.Ref.TemplateID, Version: template.Ref.Version}, template)
		if err != nil || got != template.Ref.Digest {
			t.Errorf("AgentTemplate %s digest = %s, %v; want %s", template.Ref.TemplateID, got, err, template.Ref.Digest)
		}
	}
}
