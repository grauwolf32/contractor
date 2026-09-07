package contracts

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPrivateV2ValidCanonicalFixtures(t *testing.T) {
	t.Parallel()

	cases := map[string]func([]byte) ([]byte, error){
		"agent-registration.json":                privateRoundTrip[AgentRegistrationV2],
		"agent-registration-response.json":       privateRoundTrip[AgentRegistrationResponseV2],
		"runtime-settings-empty.json":            privateRoundTrip[RuntimeSettingsV2],
		"runtime-settings-telemetry.json":        privateRoundTrip[RuntimeSettingsV2],
		"runtime-settings-telemetry-export.json": privateRoundTrip[RuntimeSettingsV2],
		"runtime-settings-proxy.json":            privateRoundTrip[RuntimeSettingsV2],
		"runtime-settings-combined.json":         privateRoundTrip[RuntimeSettingsV2],
		"runtime-provenance.json":                privateRoundTrip[ResolvedRuntimeConfigProvenanceV2],
		"runtime-report.json":                    privateRoundTrip[RuntimeReportV2],
		"workspace-capabilities.json":            privateRoundTrip[WorkspaceCapabilitiesV2],
		"allocation-workspace-overlay.json":      privateRoundTrip[AllocationWorkspaceSpecV2],
	}
	for filename, roundTrip := range cases {
		filename, roundTrip := filename, roundTrip
		t.Run(filename, func(t *testing.T) {
			t.Parallel()
			raw := readPrivateFixture(t, "valid", filename)
			canonical, err := roundTrip(raw)
			if err != nil {
				t.Fatalf("decode valid private-v2 fixture: %v", err)
			}
			if !bytes.Equal(canonical, bytes.TrimSpace(raw)) {
				t.Fatalf("fixture is not the shared canonical form\n got: %s\nwant: %s", canonical, raw)
			}
		})
	}
}

func TestPrivateV2InvalidFixturesHaveSafeReasonClasses(t *testing.T) {
	t.Parallel()

	cases := []struct {
		filename string
		class    PrivateProtocolErrorClass
		reject   func([]byte) error
	}{
		{"registration-versionless.json", PrivateProtocolErrorVersion, privateReject[AgentRegistrationV2]},
		{"registration-v1.json", PrivateProtocolErrorVersion, privateReject[AgentRegistrationV2]},
		{"registration-unsorted-labels.json", PrivateProtocolErrorInvariant, privateReject[AgentRegistrationV2]},
		{"registration-unsorted-adapters.json", PrivateProtocolErrorInvariant, privateReject[AgentRegistrationV2]},
		{"runtime-settings-duplicate-key.json", PrivateProtocolErrorDuplicate, privateReject[RuntimeSettingsV2]},
		{"runtime-settings-unknown-adapter.json", PrivateProtocolErrorInvariant, privateReject[RuntimeSettingsV2]},
		{"runtime-settings-two-proxy-auth.json", PrivateProtocolErrorInvariant, privateReject[RuntimeSettingsV2]},
		{"runtime-settings-secret-error.json", PrivateProtocolErrorInvariant, privateReject[RuntimeSettingsV2]},
		{"runtime-settings-caido-secret-error.json", PrivateProtocolErrorInvariant, privateReject[RuntimeSettingsV2]},
		{"runtime-provenance-secret-field.json", PrivateProtocolErrorSchema, privateReject[ResolvedRuntimeConfigProvenanceV2]},
		{"workspace-capabilities-unsorted-modes.json", PrivateProtocolErrorInvariant, privateReject[WorkspaceCapabilitiesV2]},
		{"allocation-workspace-versionless-source.json", PrivateProtocolErrorInvariant, privateReject[AllocationWorkspaceSpecV2]},
	}
	for _, test := range cases {
		test := test
		t.Run(test.filename, func(t *testing.T) {
			t.Parallel()
			err := test.reject(readPrivateFixture(t, "invalid", test.filename))
			if err == nil {
				t.Fatal("invalid private-v2 fixture was accepted")
			}
			var privateError *PrivateProtocolError
			if !errors.As(err, &privateError) || privateError.Class != test.class {
				t.Fatalf("error class = %v, want %v", privateError, test.class)
			}
			formatted := fmt.Sprintf("%v %+v %#v", err, err, err)
			for _, canary := range []string{
				"recognizable-secret-canary", "recognizable-provenance-secret",
				"proxy-password-canary", "proxy-bearer-canary", "unknown-secret-adapter",
				"caido-invalid-secret-canary",
			} {
				if strings.Contains(formatted, canary) {
					t.Fatalf("private validation error leaked secret input: %s", formatted)
				}
			}
		})
	}
}

func TestPrivateV2AllocationSpecComposesValidatedSettingsAndProvenance(t *testing.T) {
	t.Parallel()

	active, err := DecodeStrict[AllocationSpec](readFixture(t, "valid", "allocation-spec.json"))
	if err != nil {
		t.Fatal(err)
	}
	settings, err := DecodePrivateV2Strict[RuntimeSettingsV2](
		readPrivateFixture(t, "valid", "runtime-settings-combined.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	provenance, err := DecodePrivateV2Strict[ResolvedRuntimeConfigProvenanceV2](
		readPrivateFixture(t, "valid", "runtime-provenance.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	value := AllocationSpecV2{
		APIVersion: active.APIVersion, AllocationID: active.AllocationID, RunID: active.RunID,
		StageExecutionID: active.StageExecutionID, LogicalAgentName: active.LogicalAgentName,
		Namespace: active.Namespace, WorkerSessionMode: active.WorkerSessionMode,
		RunMetadataLabels: active.RunMetadataLabels.Clone(),
		LeaseExpiresAt:    active.LeaseExpiresAt,
		AgentTemplate:     active.AgentTemplate, ResolvedSkills: active.ResolvedSkills,
		ModelPolicy:     active.ModelPolicy,
		RuntimeSettings: settings, ResolvedRuntimeConfigProvenance: provenance,
	}
	canonical, err := MarshalPrivateV2Canonical(value)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodePrivateV2Strict[AllocationSpecV2](canonical); err != nil {
		t.Fatalf("round-trip AllocationSpecV2: %v", err)
	}
	missingLabels := value
	missingLabels.RunMetadataLabels = nil
	missingCanonical, err := MarshalPrivateV2Canonical(missingLabels)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodePrivateV2Strict[AllocationSpecV2](missingCanonical); err == nil {
		t.Fatal("AllocationSpecV2 accepted missing runMetadataLabels")
	}
	if !bytes.Contains(canonical, []byte(`"resolvedSkills":[]`)) {
		t.Fatalf("private allocation omitted mandatory empty resolvedSkills: %s", canonical)
	}
	if !bytes.Contains(canonical, []byte(`"runMetadataLabels":{"eval.id":"eval_01","purpose":"eval"}`)) {
		t.Fatalf("private allocation omitted canonical Run metadata labels: %s", canonical)
	}
	formatted := fmt.Sprintf("%v %+v %#v", value.RuntimeSettings, value.RuntimeSettings, value.RuntimeSettings)
	for _, secret := range []string{"gateway-secret", "telemetry-secret", "proxy-bearer-secret", "caido-bearer-secret"} {
		if strings.Contains(formatted, secret) {
			t.Fatalf("formatted RuntimeSettingsV2 leaked %q", secret)
		}
		if !bytes.Contains(canonical, []byte(secret)) {
			t.Fatalf("private wire omitted required secret %q", secret)
		}
	}
}

func TestPrivateV2RegistrationNormalizationFingerprintAndProjection(t *testing.T) {
	t.Parallel()

	registration, err := DecodePrivateV2Strict[AgentRegistrationV2](
		readPrivateFixture(t, "valid", "agent-registration.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	first, err := AgentRegistrationFingerprintV2(registration)
	if err != nil {
		t.Fatal(err)
	}
	changedSeed := registration
	changedSeed.InitialLabels = []string{"other"}
	changedSeed.ObservedState = AgentFenced
	second, err := AgentRegistrationFingerprintV2(changedSeed)
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatal("startup label seed or observation changed the immutable registration fingerprint")
	}
	changedCapability := registration
	changedCapability.SupportedRuntimeAdapters = []RuntimeAdapterRef{RuntimeAdapterOTLPHTTP}
	third, err := AgentRegistrationFingerprintV2(changedCapability)
	if err != nil {
		t.Fatal(err)
	}
	if third == first {
		t.Fatal("adapter capability change did not change the registration fingerprint")
	}
	projection := RuntimeAdapterCapabilityProjectionV2(registration)
	if fmt.Sprint(projection) != "[caido-graphql@1 http-proxy@1 otlp-http@1]" {
		t.Fatalf("unexpected Operations projection: %v", projection)
	}
	projection[0] = "mutated@1"
	if registration.SupportedRuntimeAdapters[0] != RuntimeAdapterCaidoGraphQL {
		t.Fatal("Operations projection aliases private registration memory")
	}

	capabilities, err := DecodePrivateV2Strict[WorkspaceCapabilitiesV2](
		readPrivateFixture(t, "valid", "workspace-capabilities.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	registration.WorkspaceCapabilities = &capabilities
	withWorkspace, err := AgentRegistrationFingerprintV2(registration)
	if err != nil {
		t.Fatal(err)
	}
	if withWorkspace == first {
		t.Fatal("workspace capability did not change the immutable registration fingerprint")
	}
	normalized := NormalizeAgentRegistrationV2(registration)
	normalized.WorkspaceCapabilities.Modes[0] = WorkspaceModeOverlay
	if registration.WorkspaceCapabilities.Modes[0] != WorkspaceModeDirect {
		t.Fatal("normalized workspace capability aliases registration memory")
	}
}

func TestHTTPOriginTargetReferenceAndSecretSettingsAreStrict(t *testing.T) {
	t.Parallel()
	reference := HTTPOriginTargetRef{
		URL: "https://app.example.test/api",
		Credential: &RuntimeCredentialRefV2{
			CredentialID: "project-origin", Kind: RuntimeCredentialOriginBearer,
		},
	}
	if err := reference.Validate(); err != nil {
		t.Fatalf("valid target reference: %v", err)
	}
	invalidKind := reference
	invalidKind.Credential = &RuntimeCredentialRefV2{
		CredentialID: "project-origin", Kind: RuntimeCredentialProxyBearer,
	}
	if err := invalidKind.Validate(); err == nil {
		t.Fatal("proxy credential was accepted as an origin credential")
	}
	invalidURL := reference
	invalidURL.URL = "https://app.example.test/api?secret=x"
	if err := invalidURL.Validate(); err == nil {
		t.Fatal("target URL with query was accepted")
	}

	secret := "recognizable-project-origin-secret"
	token := NewSecretString(secret)
	settings := HTTPOriginTargetSettingsV2{
		URL: "https://app.example.test/api", BearerToken: &token,
	}
	if err := settings.Validate(); err != nil {
		t.Fatalf("valid target settings: %v", err)
	}
	if strings.Contains(fmt.Sprintf("%+v", settings), secret) {
		t.Fatal("formatted target settings exposed the bearer token")
	}
}

func privateRoundTrip[T Validatable](data []byte) ([]byte, error) {
	value, err := DecodePrivateV2Strict[T](data)
	if err != nil {
		return nil, err
	}
	return MarshalPrivateV2Canonical(value)
}

func privateReject[T Validatable](data []byte) error {
	_, err := DecodePrivateV2Strict[T](data)
	return err
}

func readPrivateFixture(t *testing.T, kind, filename string) []byte {
	t.Helper()
	path := filepath.Join("..", "..", "testdata", "contracts", "private-v2", kind, filename)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read private fixture %s: %v", path, err)
	}
	return data
}
