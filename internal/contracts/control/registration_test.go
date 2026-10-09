package control

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestPrivateRegistrationNormalizationFingerprintAndProjection(t *testing.T) {
	t.Parallel()

	registration, err := contracts.DecodePrivateStrict[AgentRegistration](
		contractstest.ReadFixture(t, "valid", "agent-registration.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	first, err := AgentRegistrationFingerprint(registration)
	if err != nil {
		t.Fatal(err)
	}
	changedSeed := registration
	changedSeed.InitialLabels = []string{"other"}
	changedSeed.ObservedState = AgentFenced
	second, err := AgentRegistrationFingerprint(changedSeed)
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatal("startup label seed or observation changed the immutable registration fingerprint")
	}
	changedCapability := registration
	changedCapability.SupportedRuntimeAdapters = []contracts.RuntimeAdapterRef{contracts.RuntimeAdapterOTLPHTTP}
	third, err := AgentRegistrationFingerprint(changedCapability)
	if err != nil {
		t.Fatal(err)
	}
	if third == first {
		t.Fatal("adapter capability change did not change the registration fingerprint")
	}
	projection := RuntimeAdapterCapabilityProjection(registration)
	if fmt.Sprint(projection) != "[caido-graphql@1 http-proxy@1 otlp-http@1]" {
		t.Fatalf("unexpected Operations projection: %v", projection)
	}
	projection[0] = "mutated@1"
	if registration.SupportedRuntimeAdapters[0] != contracts.RuntimeAdapterCaidoGraphQL {
		t.Fatal("Operations projection aliases private registration memory")
	}

	capabilities, err := contracts.DecodePrivateStrict[contracts.WorkspaceCapabilities](
		contractstest.ReadFixture(t, "valid", "workspace-capabilities.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	registration.WorkspaceCapabilities = &capabilities
	withWorkspace, err := AgentRegistrationFingerprint(registration)
	if err != nil {
		t.Fatal(err)
	}
	if withWorkspace == first {
		t.Fatal("workspace capability did not change the immutable registration fingerprint")
	}
	normalized := NormalizeAgentRegistration(registration)
	normalized.WorkspaceCapabilities.Modes[0] = contracts.WorkspaceModeOverlay
	if registration.WorkspaceCapabilities.Modes[0] != contracts.WorkspaceModeDirect {
		t.Fatal("normalized workspace capability aliases registration memory")
	}
}

func TestCompletionCapabilityCloneDoesNotGrantOrShareSupport(t *testing.T) {
	empty := NormalizeAgentRegistration(AgentRegistration{})
	if empty.Capabilities != nil {
		t.Fatal("omission invented completion capability")
	}
	original := AgentRegistration{Capabilities: &contracts.RuntimeCompletionCapabilities{CompletionContracts: []string{contracts.AuditCheckResultsV1}}}
	cloned := NormalizeAgentRegistration(original)
	cloned.Capabilities.CompletionContracts[0] = "changed"
	if original.Capabilities.CompletionContracts[0] != contracts.AuditCheckResultsV1 {
		t.Fatal("capability clone aliases mutable source")
	}
}

func TestPerformanceGoldenCapabilities(t *testing.T) {
	var cases struct {
		ValidCapabilities   []json.RawMessage `json:"validCapabilities"`
		InvalidCapabilities []json.RawMessage `json:"invalidCapabilities"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "performance-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	for _, valid := range []bool{true, false} {
		capabilities := cases.InvalidCapabilities
		if valid {
			capabilities = cases.ValidCapabilities
		}
		for _, raw := range capabilities {
			var registration map[string]json.RawMessage
			if err := json.Unmarshal(contractstest.ReadFixture(t, "valid", "agent-registration.json"), &registration); err != nil {
				t.Fatal(err)
			}
			registration["supportedPerformanceMetricsVersions"] = raw
			encoded, _ := json.Marshal(registration)
			value, err := contracts.DecodePrivateStrict[AgentRegistration](encoded)
			if (err == nil) != valid {
				t.Fatalf("capability %s validity %v: %v", raw, valid, err)
			}
			if valid && len(value.SupportedPerformanceMetricsVersions) != 0 {
				clone := NormalizeAgentRegistration(value)
				clone.SupportedPerformanceMetricsVersions[0] = 2
				if value.SupportedPerformanceMetricsVersions[0] != 1 {
					t.Fatal("normalization aliases capability")
				}
				fingerprint, _ := AgentRegistrationFingerprint(value)
				value.SupportedPerformanceMetricsVersions = nil
				withoutCapabilityFingerprint, _ := AgentRegistrationFingerprint(value)
				if fingerprint == withoutCapabilityFingerprint {
					t.Fatal("capability missing from fingerprint")
				}
			}
		}
	}
}
