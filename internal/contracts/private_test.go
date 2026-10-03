package contracts

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"testing"
)

func TestPrivateValidCanonicalFixtures(t *testing.T) {
	t.Parallel()

	for filename, entry := range readFixtureIndex(t).Valid {
		roundTrip := fixtureCodecs[entry.Type].privateRoundTrip
		t.Run(filename, func(t *testing.T) {
			t.Parallel()
			raw := readFixture(t, "valid", filename)
			canonical, err := roundTrip(raw)
			if err != nil {
				t.Fatalf("decode valid private fixture: %v", err)
			}
			if !bytes.Equal(canonical, bytes.TrimSpace(raw)) {
				t.Fatalf("fixture is not the shared canonical form\n got: %s\nwant: %s", canonical, raw)
			}
		})
	}
}

func TestPrivateInvalidFixturesHaveSafeReasonClasses(t *testing.T) {
	t.Parallel()

	for filename, entry := range readFixtureIndex(t).Invalid {
		if entry.Reason == "" {
			continue
		}
		reject, class := fixtureCodecs[entry.Type].privateReject, PrivateProtocolErrorClass(entry.Reason)
		t.Run(filename, func(t *testing.T) {
			t.Parallel()
			err := reject(readFixture(t, "invalid", filename))
			if err == nil {
				t.Fatal("invalid private fixture was accepted")
			}
			var privateError *PrivateProtocolError
			if !errors.As(err, &privateError) || privateError.Class != class {
				t.Fatalf("error class = %v, want %v", privateError, class)
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

func TestPrivateAllocationSpecComposesValidatedSettingsAndProvenance(t *testing.T) {
	t.Parallel()

	active, err := DecodeStrict[AllocationSpec](readFixture(t, "valid", "allocation-spec.json"))
	if err != nil {
		t.Fatal(err)
	}
	settings, err := DecodePrivateStrict[RuntimeSettings](
		readFixture(t, "valid", "runtime-settings-combined.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	provenance, err := DecodePrivateStrict[ResolvedRuntimeConfigProvenance](
		readFixture(t, "valid", "runtime-provenance.json"),
	)
	if err != nil {
		t.Fatal(err)
	}
	value := AllocationSpec{
		APIVersion: active.APIVersion, AllocationID: active.AllocationID, RunID: active.RunID,
		StageExecutionID: active.StageExecutionID, LogicalAgentName: active.LogicalAgentName,
		Namespace: active.Namespace, WorkerSessionMode: active.WorkerSessionMode,
		RunMetadataLabels: active.RunMetadataLabels.Clone(),
		LeaseExpiresAt:    active.LeaseExpiresAt,
		AgentTemplate:     active.AgentTemplate, ResolvedSkills: active.ResolvedSkills,
		ModelPolicy:     active.ModelPolicy,
		RuntimeSettings: settings, ResolvedRuntimeConfigProvenance: provenance,
	}
	canonical, err := MarshalPrivateCanonical(value)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodePrivateStrict[AllocationSpec](canonical); err != nil {
		t.Fatalf("round-trip AllocationSpec: %v", err)
	}
	missingLabels := value
	missingLabels.RunMetadataLabels = nil
	missingCanonical, err := MarshalPrivateCanonical(missingLabels)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodePrivateStrict[AllocationSpec](missingCanonical); err == nil {
		t.Fatal("AllocationSpec accepted missing runMetadataLabels")
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
			t.Fatalf("formatted RuntimeSettings leaked %q", secret)
		}
		if !bytes.Contains(canonical, []byte(secret)) {
			t.Fatalf("private wire omitted required secret %q", secret)
		}
	}
}

func TestPrivateRegistrationNormalizationFingerprintAndProjection(t *testing.T) {
	t.Parallel()

	registration, err := DecodePrivateStrict[AgentRegistration](
		readFixture(t, "valid", "agent-registration.json"),
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
	changedCapability.SupportedRuntimeAdapters = []RuntimeAdapterRef{RuntimeAdapterOTLPHTTP}
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
	if registration.SupportedRuntimeAdapters[0] != RuntimeAdapterCaidoGraphQL {
		t.Fatal("Operations projection aliases private registration memory")
	}

	capabilities, err := DecodePrivateStrict[WorkspaceCapabilities](
		readFixture(t, "valid", "workspace-capabilities.json"),
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
	normalized.WorkspaceCapabilities.Modes[0] = WorkspaceModeOverlay
	if registration.WorkspaceCapabilities.Modes[0] != WorkspaceModeDirect {
		t.Fatal("normalized workspace capability aliases registration memory")
	}
}

func TestHTTPOriginTargetReferenceAndSecretSettingsAreStrict(t *testing.T) {
	t.Parallel()
	reference := HTTPOriginTargetRef{
		URL: "https://app.example.test/api",
		Credential: &RuntimeCredentialRef{
			CredentialID: "project-origin", Kind: RuntimeCredentialOriginBearer,
		},
	}
	if err := reference.Validate(); err != nil {
		t.Fatalf("valid target reference: %v", err)
	}
	invalidKind := reference
	invalidKind.Credential = &RuntimeCredentialRef{
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
	settings := HTTPOriginTargetSettings{
		URL: "https://app.example.test/api", BearerToken: &token,
	}
	if err := settings.Validate(); err != nil {
		t.Fatalf("valid target settings: %v", err)
	}
	if strings.Contains(fmt.Sprintf("%+v", settings), secret) {
		t.Fatal("formatted target settings exposed the bearer token")
	}
}

func TestHTTPOriginTargetRejectsUnusableBrowserURLs(t *testing.T) {
	t.Parallel()
	for _, targetURL := range []string{
		"http://target:70000/",
		"http://target:0/",
		"https://[fe80::1%25en0]/",
		"http://10.0.0.256/",
		"https://host/#",
		"http://exa<mple/",
	} {
		t.Run(targetURL, func(t *testing.T) {
			t.Parallel()
			if err := (HTTPOriginTargetRef{URL: targetURL}).Validate(); err == nil {
				t.Fatalf("unusable Project target %q was accepted", targetURL)
			}
		})
	}
	for _, targetURL := range []string{
		"https://app.example.test/api",
		"http://127.0.0.1:8080/",
		"https://[2001:db8::1]:443/",
	} {
		if err := (HTTPOriginTargetRef{URL: targetURL}).Validate(); err != nil {
			t.Fatalf("valid Project target %q was rejected: %v", targetURL, err)
		}
	}
}

func privateRoundTrip[T Validatable](data []byte) ([]byte, error) {
	value, err := DecodePrivateStrict[T](data)
	if err != nil {
		return nil, err
	}
	return MarshalPrivateCanonical(value)
}

func privateReject[T Validatable](data []byte) error {
	_, err := DecodePrivateStrict[T](data)
	return err
}
