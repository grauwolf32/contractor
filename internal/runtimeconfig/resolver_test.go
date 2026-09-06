package runtimeconfig

import (
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestResolverComplexPrecedenceFixture(t *testing.T) {
	input := resolverInput()
	input.Default.Spec.Worker.LLMGateway.Credential = credentialField("default-key")
	input.Default.Spec.Planner.Telemetry = telemetryPatch(
		"https://otel.example/default", "default-otel",
	)
	input.Workflow = WorkerRoutePatch{
		Gateway:    gatewayField(testGatewayRef("secondary", "2")),
		Credential: credentialField("workflow-key"),
	}
	input.RunLabels = []PinnedRuntimeConfig{
		testPin("debug", "4", Spec{
			Worker:  WorkerPatch{Telemetry: telemetryPatch("https://otel.example/run", "run-otel")},
			Planner: PlannerPatch{Telemetry: telemetryPatch("https://otel.example/planner", "planner-otel")},
		}),
		testPin("caido", "3", Spec{Worker: WorkerPatch{HTTPProxy: proxyPatch("proxy-auth")}}),
	}
	input.RunOverride.Credential = Field[string]{Present: true, Clear: true}
	input.Escalation = WorkerRoutePatch{
		Gateway:    gatewayField(testGatewayRef("primary", "1")),
		Credential: credentialField("escalation-key"),
	}
	input.AgentLabels = []PinnedRuntimeConfig{testPin("site", "5", Spec{
		Worker: WorkerPatch{
			LLMGateway: LLMGatewayPatch{Present: true, Credential: credentialField("agent-key")},
			HTTPProxy:  AtomicPatch[HTTPProxyConfig]{Present: true, Clear: true},
		},
		// This value is deliberately hostile: Agent labels never enter Planner
		// resolution even when the immutable config also contains this block.
		Planner: PlannerPatch{Telemetry: telemetryPatch("https://evil.example/traces", "")},
	})}
	for _, credentialID := range []string{"default-key", "workflow-key", "escalation-key", "agent-key"} {
		input.LLMCredentials[credentialID] = authorization(input, credentialID, testGatewayRef("primary", "1"))
	}
	input.LLMCredentials["workflow-key"] = authorization(
		input, "workflow-key", testGatewayRef("secondary", "2"),
	)
	input.RuntimeCredentials = map[string]contracts.RuntimeCredentialKind{
		"default-otel": contracts.RuntimeCredentialOTLPHeaders,
		"run-otel":     contracts.RuntimeCredentialOTLPHeaders,
		"planner-otel": contracts.RuntimeCredentialOTLPHeaders,
		"proxy-auth":   contracts.RuntimeCredentialProxyBearer,
	}

	result, err := ResolveRuntimeConfig(input)
	if err != nil {
		t.Fatal(err)
	}
	if result.LLMGateway.Ref.GatewayID != "primary" || result.LLMCredential == nil ||
		result.LLMCredential.CredentialID != "agent-key" || result.HTTPProxy != nil ||
		result.WorkerTelemetry == nil || result.PlannerTelemetry == nil ||
		result.PlannerTelemetry.Endpoint != "https://otel.example/planner" {
		t.Fatalf("complex resolution = %+v", result)
	}
	if !reflect.DeepEqual(result.RequiredRuntimeAdapters, []contracts.RuntimeAdapterRef{contracts.RuntimeAdapterOTLPHTTP}) {
		t.Fatalf("required adapters = %v", result.RequiredRuntimeAdapters)
	}
	if result.Origins.LLMGateway.Layer != LayerEscalation ||
		result.Origins.LLMCredential.Layer != LayerAgentLabels ||
		result.Origins.WorkerTelemetry.Layer != LayerRunLabels ||
		result.Origins.HTTPProxy.Layer != LayerAgentLabels ||
		result.Origins.PlannerTelemetry.Layer != LayerRunLabels {
		t.Fatalf("complex origins = %+v", result.Origins)
	}
	assertResolverFixture(t, result)
}

func TestResolverRunLabelAdaptersDoNotRequireMatchingAgentLabels(t *testing.T) {
	input := resolverInput()
	input.RunLabels = []PinnedRuntimeConfig{testPin("debug", "3", Spec{Worker: WorkerPatch{
		Telemetry: telemetryPatch("https://otel.example/run", ""),
	}})}
	result, err := ResolveRuntimeConfig(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Provenance.AgentLabels) != 0 || len(result.Provenance.RunLabels) != 1 ||
		result.Provenance.RunLabels[0].Label != "debug" ||
		!reflect.DeepEqual(result.RequiredRuntimeAdapters, []contracts.RuntimeAdapterRef{contracts.RuntimeAdapterOTLPHTTP}) {
		t.Fatalf("unlabeled candidate result = %+v", result)
	}
}

func TestResolverPreservesTrustedContentOptIn(t *testing.T) {
	input := resolverInput()
	worker := telemetryPatch("https://otel.example/worker", "")
	planner := telemetryPatch("https://otel.example/planner", "")
	worker.Value.CaptureContent = true
	planner.Value.CaptureContent = true
	input.RunLabels = []PinnedRuntimeConfig{testPin("debug", "2", Spec{Worker: WorkerPatch{Telemetry: worker}, Planner: PlannerPatch{Telemetry: planner}})}
	result, err := ResolveRuntimeConfig(input)
	if err != nil {
		t.Fatal(err)
	}
	if result.WorkerTelemetry == nil || result.PlannerTelemetry == nil || !result.WorkerTelemetry.CaptureContent || !result.PlannerTelemetry.CaptureContent {
		t.Fatal("trusted capture lost at resolution")
	}
}

func TestResolverCaidoAtomicDefaultRunAgentPrecedenceAndClear(t *testing.T) {
	input := resolverInput()
	input.Default.Spec.Worker.Caido = caidoPatch("https://default.example", "")
	input.RunLabels = []PinnedRuntimeConfig{testPin("debug", "3", Spec{Worker: WorkerPatch{
		Caido: caidoPatch("https://run.example/prefix", "run-caido"),
	}})}
	input.AgentLabels = []PinnedRuntimeConfig{testPin("lab", "4", Spec{Worker: WorkerPatch{
		Caido: caidoPatch("https://agent.example", ""),
	}})}
	input.RuntimeCredentials["run-caido"] = contracts.RuntimeCredentialCaidoBearer

	result, err := ResolveRuntimeConfig(input)
	if err != nil {
		t.Fatal(err)
	}
	if result.Caido == nil || result.Caido.Endpoint != "https://agent.example" ||
		result.Caido.Credential != "" || result.Origins.Caido.Layer != LayerAgentLabels ||
		!reflect.DeepEqual(result.RequiredRuntimeAdapters, []contracts.RuntimeAdapterRef{contracts.RuntimeAdapterCaidoGraphQL}) ||
		len(result.Provenance.RuntimeCredentialRefs) != 0 {
		t.Fatalf("Agent Caido precedence = %+v", result)
	}

	input.AgentLabels = nil
	result, err = ResolveRuntimeConfig(input)
	if err != nil {
		t.Fatal(err)
	}
	if result.Caido == nil || result.Caido.Endpoint != "https://run.example/prefix" ||
		result.Origins.Caido.Layer != LayerRunLabels ||
		!reflect.DeepEqual(result.Provenance.RuntimeCredentialRefs, []contracts.RuntimeCredentialRefV2{{
			CredentialID: "run-caido", Kind: contracts.RuntimeCredentialCaidoBearer,
		}}) {
		t.Fatalf("Run Caido precedence = %+v", result)
	}

	input.AgentLabels = []PinnedRuntimeConfig{testPin("no-caido", "5", Spec{Worker: WorkerPatch{
		Caido: AtomicPatch[CaidoConfig]{Present: true, Clear: true},
	}})}
	result, err = ResolveRuntimeConfig(input)
	if err != nil || result.Caido != nil || len(result.RequiredRuntimeAdapters) != 0 ||
		result.Origins.Caido.Layer != LayerAgentLabels {
		t.Fatalf("Agent Caido clear = (%+v, %v)", result, err)
	}

	input.AgentLabels = nil
	input.RuntimeCredentials["run-caido"] = contracts.RuntimeCredentialProxyBearer
	_, err = ResolveRuntimeConfig(input)
	assertResolutionError(t, err, ResolutionCredentialKindMismatch, "worker.caido.credential")
}

func TestResolverSameLayerPermutationConflictAndDeduplication(t *testing.T) {
	one := testPin("one", "4", Spec{Worker: WorkerPatch{
		Telemetry: telemetryPatch("https://one.example/traces", ""),
	}})
	two := testPin("two", "5", Spec{Worker: WorkerPatch{
		Telemetry: telemetryPatch("https://two.example/traces", ""),
	}})
	for _, labels := range [][]PinnedRuntimeConfig{{one, two}, {two, one}} {
		input := resolverInput()
		input.RunLabels = labels
		_, err := ResolveRuntimeConfig(input)
		assertResolutionError(t, err, ResolutionConflict, "worker.telemetry")
		if strings.Contains(err.Error(), "https://") {
			t.Fatalf("conflict exposed endpoint: %v", err)
		}
	}

	two.Spec.Worker.Telemetry = one.Spec.Worker.Telemetry
	var canonical []byte
	for _, labels := range [][]PinnedRuntimeConfig{{one, two}, {two, one}} {
		input := resolverInput()
		input.RunLabels = labels
		result, err := ResolveRuntimeConfig(input)
		if err != nil {
			t.Fatal(err)
		}
		if len(result.Origins.WorkerTelemetry.Configs) != 2 ||
			result.Provenance.RunLabels[0].Label != "one" || result.Provenance.RunLabels[1].Label != "two" {
			t.Fatalf("deduplicated provenance = %+v", result)
		}
		encoded, err := json.Marshal(result)
		if err != nil {
			t.Fatal(err)
		}
		if canonical == nil {
			canonical = encoded
		} else if !reflect.DeepEqual(canonical, encoded) {
			t.Fatalf("permutation changed result:\n%s\n%s", canonical, encoded)
		}
	}
}

func TestResolverOmissionClearAndEscalationPrecedenceMatrix(t *testing.T) {
	tests := []struct {
		name        string
		run         WorkerRoutePatch
		escalation  WorkerRoutePatch
		agent       *PinnedRuntimeConfig
		wantID      string
		wantOrigin  RuntimeLayer
		wantPresent bool
	}{
		{name: "omission inherits", wantID: "default-key", wantOrigin: LayerDefault, wantPresent: true},
		{name: "run clear", run: WorkerRoutePatch{Credential: Field[string]{Present: true, Clear: true}}, wantOrigin: LayerRunOverride},
		{name: "escalation replaces clear", run: WorkerRoutePatch{Credential: Field[string]{Present: true, Clear: true}}, escalation: WorkerRoutePatch{Credential: credentialField("escalation-key")}, wantID: "escalation-key", wantOrigin: LayerEscalation, wantPresent: true},
		{name: "agent clear wins", escalation: WorkerRoutePatch{Credential: credentialField("escalation-key")}, agent: pointerPin(testPin("direct", "6", Spec{Worker: WorkerPatch{LLMGateway: LLMGatewayPatch{Present: true, Credential: Field[string]{Present: true, Clear: true}}}})), wantOrigin: LayerAgentLabels},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			input := resolverInput()
			input.Default.Spec.Worker.LLMGateway.Credential = credentialField("default-key")
			input.RunOverride = test.run
			input.Escalation = test.escalation
			if test.agent != nil {
				input.AgentLabels = []PinnedRuntimeConfig{*test.agent}
			}
			for _, id := range []string{"default-key", "escalation-key"} {
				input.LLMCredentials[id] = authorization(input, id, testGatewayRef("primary", "1"))
			}
			result, err := ResolveRuntimeConfig(input)
			if err != nil {
				t.Fatal(err)
			}
			if (result.LLMCredential != nil) != test.wantPresent ||
				test.wantPresent && result.LLMCredential.CredentialID != test.wantID ||
				result.Origins.LLMCredential.Layer != test.wantOrigin {
				t.Fatalf("credential precedence = ref:%+v origin:%+v", result.LLMCredential, result.Origins.LLMCredential)
			}
		})
	}
}

func TestResolverAgentLayerCannotAffectPlanner(t *testing.T) {
	input := resolverInput()
	input.RunLabels = []PinnedRuntimeConfig{testPin("debug", "2", Spec{
		Planner: PlannerPatch{Telemetry: telemetryPatch("https://planner.example/traces", "")},
	})}
	input.AgentLabels = []PinnedRuntimeConfig{testPin("agent-debug", "3", Spec{
		Worker:  WorkerPatch{Telemetry: telemetryPatch("https://worker.example/traces", "")},
		Planner: PlannerPatch{Telemetry: telemetryPatch("https://wrong.example/traces", "")},
	})}
	result, err := ResolveRuntimeConfig(input)
	if err != nil {
		t.Fatal(err)
	}
	if result.PlannerTelemetry == nil || result.PlannerTelemetry.Endpoint != "https://planner.example/traces" ||
		result.Origins.PlannerTelemetry.Layer != LayerRunLabels {
		t.Fatalf("Planner telemetry = %+v origin=%+v", result.PlannerTelemetry, result.Origins.PlannerTelemetry)
	}

	input.AgentLabels[0].Spec.Worker = WorkerPatch{}
	_, err = ResolveRuntimeConfig(input)
	assertResolutionError(t, err, ResolutionInvalid, "agent_labels.worker")
}

func TestResolverRejectsUnauthorizedRoutesAndCredentialKinds(t *testing.T) {
	t.Run("unauthorized policy", func(t *testing.T) {
		input := resolverInput()
		input.Default.Spec.Worker.LLMGateway.Credential = credentialField("agent-key")
		input.LLMCredentials["agent-key"] = LLMCredentialAuthorization{
			Ref:        contracts.LLMCredentialRef{CredentialID: "agent-key"},
			LLMGateway: testGatewayRef("primary", "1"),
			ModelPolicies: []contracts.ModelPolicyRef{{
				PolicyID: "other", Version: "1", Digest: testDigest("9"),
			}},
			Models: []string{"other-model"},
		}
		_, err := ResolveRuntimeConfig(input)
		assertResolutionError(t, err, ResolutionModelUnauthorized, "worker.llmGateway.credential")
	})

	t.Run("Gateway mismatch", func(t *testing.T) {
		input := resolverInput()
		input.Default.Spec.Worker.LLMGateway.Credential = credentialField("agent-key")
		input.LLMCredentials["agent-key"] = authorization(
			input, "agent-key", testGatewayRef("secondary", "2"),
		)
		_, err := ResolveRuntimeConfig(input)
		assertResolutionError(t, err, ResolutionGatewayMismatch, "worker.llmGateway.credential")
	})

	t.Run("unauthorized summarizer policy", func(t *testing.T) {
		input := resolverInput()
		summarizer := contracts.ResolvedModelPolicy{
			Ref: contracts.ModelPolicyRef{
				PolicyID: "terminal-summarizer", Version: "1", Digest: testDigest("8"),
			},
			Model: "summary-model", ContextWindowTokens: 8192,
			MaxOutputTokens: 1024, MaxModelCalls: 1,
		}
		input.SummarizerModelPolicy = &summarizer
		input.Default.Spec.Worker.LLMGateway.Credential = credentialField("agent-key")
		input.LLMCredentials["agent-key"] = authorization(
			input, "agent-key", testGatewayRef("primary", "1"),
		)
		_, err := ResolveRuntimeConfig(input)
		assertResolutionError(t, err, ResolutionModelUnauthorized, "worker.llmGateway.credential")

		authorized := input.LLMCredentials["agent-key"]
		authorized.ModelPolicies = append(authorized.ModelPolicies, summarizer.Ref)
		authorized.Models = append(authorized.Models, summarizer.Model)
		input.LLMCredentials["agent-key"] = authorized
		resolved, err := ResolveRuntimeConfig(input)
		if err != nil || resolved.LLMGateway.Ref != testGatewayRef("primary", "1") ||
			resolved.LLMCredential == nil || resolved.LLMCredential.CredentialID != "agent-key" {
			t.Fatalf("summarizer-authorized shared route = (%+v, %v)", resolved, err)
		}
	})

	t.Run("Runtime credential kind", func(t *testing.T) {
		input := resolverInput()
		input.RunLabels = []PinnedRuntimeConfig{testPin("debug", "4", Spec{Worker: WorkerPatch{
			Telemetry: telemetryPatch("https://otel.example/traces", "wrong-kind"),
		}})}
		input.RuntimeCredentials["wrong-kind"] = contracts.RuntimeCredentialProxyBearer
		_, err := ResolveRuntimeConfig(input)
		assertResolutionError(t, err, ResolutionCredentialKindMismatch, "worker.telemetry.credential")
	})

	t.Run("incomplete Gateway body", func(t *testing.T) {
		input := resolverInput()
		input.Gateways = nil
		_, err := ResolveRuntimeConfig(input)
		assertResolutionError(t, err, ResolutionIncomplete, "worker.llmGateway.gateway")
	})
}

func TestResolverAllowsExplicitUnrestrictedDevelopmentCredentialOnlyOnExactGateway(t *testing.T) {
	input := resolverInput()
	input.Default.Spec.Worker.LLMGateway.Credential = credentialField("development-key")
	input.LLMCredentials["development-key"] = LLMCredentialAuthorization{
		Ref:        contracts.LLMCredentialRef{CredentialID: "development-key"},
		LLMGateway: testGatewayRef("primary", "1"), Unrestricted: true,
	}
	result, err := ResolveRuntimeConfig(input)
	if err != nil || result.LLMCredential == nil ||
		result.LLMCredential.CredentialID != "development-key" {
		t.Fatalf("unrestricted development resolution = (%+v, %v)", result, err)
	}

	input.LLMCredentials["development-key"] = LLMCredentialAuthorization{
		Ref:        contracts.LLMCredentialRef{CredentialID: "development-key"},
		LLMGateway: testGatewayRef("secondary", "2"), Unrestricted: true,
	}
	_, err = ResolveRuntimeConfig(input)
	assertResolutionError(t, err, ResolutionGatewayMismatch, "worker.llmGateway.credential")
}

func TestResolverReturnsCallerOwnedValuesAndPreservesModelPolicy(t *testing.T) {
	input := resolverInput()
	input.RunLabels = []PinnedRuntimeConfig{testPin("caido", "3", Spec{Worker: WorkerPatch{
		HTTPProxy: proxyPatch(""),
	}})}
	result, err := ResolveRuntimeConfig(input)
	if err != nil {
		t.Fatal(err)
	}
	if result.ModelPolicy.Ref != input.ModelPolicy.Ref || result.ModelPolicy.Model != input.ModelPolicy.Model {
		t.Fatalf("resolver changed ModelPolicy: %+v", result.ModelPolicy)
	}
	result.HTTPProxy.Targets[0] = "changed"
	result.Provenance.RunLabels[0].Label = "changed"
	result.Origins.HTTPProxy.Configs[0].Name = "changed"
	if input.RunLabels[0].Spec.Worker.HTTPProxy.Value.Targets[0] != "llm-gateway" ||
		input.RunLabels[0].Label != "caido" || input.RunLabels[0].Config.Name == "changed" {
		t.Fatal("resolved value aliases caller input")
	}
}

type resolverFixtureProjection struct {
	GatewayID       string                                      `json:"gatewayId"`
	CredentialID    string                                      `json:"credentialId"`
	WorkerTelemetry *TelemetryConfig                            `json:"workerTelemetry"`
	ProxyEnabled    bool                                        `json:"proxyEnabled"`
	PlannerEndpoint string                                      `json:"plannerEndpoint"`
	Adapters        []contracts.RuntimeAdapterRef               `json:"adapters"`
	Origins         ResolvedRuntimeConfigOrigins                `json:"origins"`
	Provenance      contracts.ResolvedRuntimeConfigProvenanceV2 `json:"provenance"`
}

func assertResolverFixture(t *testing.T, result ResolvedRuntimeConfig) {
	t.Helper()
	projection := resolverFixtureProjection{
		GatewayID:       result.LLMGateway.Ref.GatewayID,
		WorkerTelemetry: result.WorkerTelemetry,
		ProxyEnabled:    result.HTTPProxy != nil,
		Adapters:        result.RequiredRuntimeAdapters,
		Origins:         result.Origins,
		Provenance:      result.Provenance,
	}
	if result.LLMCredential != nil {
		projection.CredentialID = result.LLMCredential.CredentialID
	}
	if result.PlannerTelemetry != nil {
		projection.PlannerEndpoint = result.PlannerTelemetry.Endpoint
	}
	actual, err := json.Marshal(projection)
	if err != nil {
		t.Fatal(err)
	}
	expected, err := os.ReadFile("testdata/complex_resolution.json")
	if err != nil {
		t.Fatal(err)
	}
	var actualValue, expectedValue any
	if json.Unmarshal(actual, &actualValue) != nil || json.Unmarshal(expected, &expectedValue) != nil {
		t.Fatal("decode complex resolution fixture")
	}
	if !reflect.DeepEqual(actualValue, expectedValue) {
		t.Fatalf("complex fixture mismatch\nactual: %s\nexpected: %s", actual, expected)
	}
}

func resolverInput() ResolveRuntimeConfigInput {
	primaryRef := testGatewayRef("primary", "1")
	secondaryRef := testGatewayRef("secondary", "2")
	policy := contracts.ResolvedModelPolicy{
		Ref:   contracts.ModelPolicyRef{PolicyID: "worker", Version: "1", Digest: testDigest("a")},
		Model: "qwen/test-model", MaxOutputTokens: 1024, MaxModelCalls: 4, MaxTotalTokens: 4096,
	}
	return ResolveRuntimeConfigInput{
		ModelPolicy: policy,
		Default: PinnedRuntimeConfig{
			Label: DefaultLabel, BindingRevision: 1,
			Config: testRef("runtime-default", "1"),
			Spec: Spec{Worker: WorkerPatch{LLMGateway: LLMGatewayPatch{
				Present: true, Gateway: gatewayField(primaryRef),
			}}},
		},
		Gateways: map[contracts.LLMGatewayConfigRef]contracts.ResolvedLLMGatewayConfig{
			primaryRef: {
				Ref: primaryRef, Protocol: contracts.OpenAICompatibleProtocol,
				URL: "http://127.0.0.1:4000/v1",
			},
			secondaryRef: {
				Ref: secondaryRef, Protocol: contracts.OpenAICompatibleProtocol,
				URL: "https://gateway.example/v1",
			},
		},
		LLMCredentials:     map[string]LLMCredentialAuthorization{},
		RuntimeCredentials: map[string]contracts.RuntimeCredentialKind{},
	}
}

func authorization(
	input ResolveRuntimeConfigInput, credentialID string, gateway contracts.LLMGatewayConfigRef,
) LLMCredentialAuthorization {
	return LLMCredentialAuthorization{
		Ref: contracts.LLMCredentialRef{CredentialID: credentialID}, LLMGateway: gateway,
		ModelPolicies: []contracts.ModelPolicyRef{input.ModelPolicy.Ref}, Models: []string{input.ModelPolicy.Model},
	}
}

func testPin(label, digit string, spec Spec) PinnedRuntimeConfig {
	return PinnedRuntimeConfig{
		Label: label, BindingRevision: uint64(digit[0] - '0'),
		Config: testRef("config-"+label, digit), Spec: spec,
	}
}

func pointerPin(value PinnedRuntimeConfig) *PinnedRuntimeConfig { return &value }

func gatewayField(ref contracts.LLMGatewayConfigRef) Field[contracts.LLMGatewayConfigRef] {
	return Field[contracts.LLMGatewayConfigRef]{Present: true, Value: ref}
}

func credentialField(credentialID string) Field[string] {
	return Field[string]{Present: true, Value: credentialID}
}

func telemetryPatch(endpoint, credentialID string) AtomicPatch[TelemetryConfig] {
	return AtomicPatch[TelemetryConfig]{Present: true, Value: TelemetryConfig{
		Adapter: string(contracts.RuntimeAdapterOTLPHTTP), Endpoint: endpoint,
		Credential: credentialID, FlushTimeoutSeconds: 3,
	}}
}

func proxyPatch(credentialID string) AtomicPatch[HTTPProxyConfig] {
	return AtomicPatch[HTTPProxyConfig]{Present: true, Value: HTTPProxyConfig{
		Adapter: string(contracts.RuntimeAdapterHTTPProxy), ProxyURL: "http://127.0.0.1:8080",
		Credential: credentialID, Targets: []string{"llm-gateway", "tool-http"},
	}}
}

func caidoPatch(endpoint, credentialID string) AtomicPatch[CaidoConfig] {
	return AtomicPatch[CaidoConfig]{Present: true, Value: CaidoConfig{
		Adapter: string(contracts.RuntimeAdapterCaidoGraphQL), Endpoint: endpoint,
		Credential: credentialID, RequestTimeoutSeconds: 30,
	}}
}

func testGatewayRef(name, digit string) contracts.LLMGatewayConfigRef {
	return contracts.LLMGatewayConfigRef{GatewayID: name, Version: "1", Digest: testDigest(digit)}
}

func testDigest(digit string) string { return "sha256:" + strings.Repeat(digit, 64) }

func assertResolutionError(t *testing.T, err error, code ResolutionErrorCode, path string) {
	t.Helper()
	var resolution *ResolutionError
	if !errors.As(err, &resolution) || resolution.Code != code || resolution.Path != path {
		t.Fatalf("resolution error = %#v (%v), want %s at %s", resolution, err, code, path)
	}
}
