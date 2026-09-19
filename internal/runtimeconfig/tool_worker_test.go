package runtimeconfig

import (
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestToolWorkerProjectsOnlyModelProxyTargets(t *testing.T) {
	input := resolverInput()
	input.ModelFree, input.ModelPolicy = true, contracts.ResolvedModelPolicy{}
	input.Gateways, input.LLMCredentials = nil, nil
	input.Default.Spec.Worker.HTTPProxy = proxyPatch("")
	modelOnly := proxyPatch("unavailable-model-proxy")
	modelOnly.Value.Targets = []string{"llm-gateway"}
	input.RunLabels = []PinnedRuntimeConfig{testPin("model-proxy", "1", Spec{Worker: WorkerPatch{HTTPProxy: modelOnly}})}
	result, err := ResolveRuntimeConfig(input)
	if err != nil {
		t.Fatal(err)
	}
	if result.HTTPProxy == nil || !reflect.DeepEqual(result.HTTPProxy.Targets, []string{"tool-http"}) || result.Origins.HTTPProxy.Layer != LayerDefault {
		t.Fatalf("model-only label changed tool proxy: %+v", result)
	}
	if !reflect.DeepEqual(input.Default.Spec.Worker.HTTPProxy.Value.Targets, []string{"llm-gateway", "tool-http"}) {
		t.Fatal("projection mutated input targets")
	}
}

func TestToolWorkerIgnoresUnavailableConflictingModelRoutes(t *testing.T) {
	input := resolverInput()
	input.ModelFree = true
	input.ModelPolicy = contracts.ResolvedModelPolicy{}
	input.Gateways, input.LLMCredentials = nil, nil
	input.RunLabels = []PinnedRuntimeConfig{
		testPin("one", "1", Spec{Worker: WorkerPatch{LLMGateway: LLMGatewayPatch{Present: true, Gateway: gatewayField(testGatewayRef("missing-one", "1")), Credential: credentialField("missing-one")}}}),
		testPin("two", "1", Spec{Worker: WorkerPatch{LLMGateway: LLMGatewayPatch{Present: true, Gateway: gatewayField(testGatewayRef("missing-two", "1")), Credential: credentialField("missing-two")}, Telemetry: telemetryPatch("https://otel.example/worker", "")}}),
	}
	input.AgentLabels = []PinnedRuntimeConfig{testPin("model-only", "1", Spec{Worker: input.Default.Spec.Worker})}
	result, err := ResolveRuntimeConfig(input)
	if err != nil {
		t.Fatal(err)
	}
	if err := result.Validate(); err != nil {
		t.Fatal(err)
	}
	if !result.ModelFree || !result.ModelPolicy.IsZero() || result.Provenance.LLMGatewayConfig != nil || result.LLMCredential != nil || result.WorkerTelemetry == nil {
		t.Fatalf("bad projection: %+v", result)
	}
	if !input.RunLabels[0].Spec.Worker.LLMGateway.Present {
		t.Fatal("caller mutated")
	}
	input.RunOverride.Credential = Field[string]{Present: true, Clear: true}
	if _, err := ResolveRuntimeConfig(input); err == nil {
		t.Fatal("explicit model override accepted")
	}
}
