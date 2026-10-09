package runtimesettings

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestRuntimeSettingsRedactFormattingButSerializeOnWire(t *testing.T) {
	t.Parallel()

	const token = "recognizable-secret-token"
	secret := contracts.NewSecretString(token)
	settings := RuntimeSettings{
		LLMGatewayURL:         "https://gateway.example/v1",
		LLMGatewayToken:       &secret,
		ArtifactAPIURL:        "https://server.example/private/v1",
		RequestTimeoutSeconds: 30,
	}
	formatted := fmt.Sprintf("%v %+v %#v %s", settings, settings, settings, settings.LLMGatewayToken)
	if strings.Contains(formatted, token) {
		t.Fatalf("formatted RuntimeSettings leaked token: %s", formatted)
	}
	wire, err := json.Marshal(settings)
	if err != nil {
		t.Fatalf("marshal RuntimeSettings: %v", err)
	}
	if !strings.Contains(string(wire), token) {
		t.Fatalf("wire JSON did not contain required private token: %s", wire)
	}
}

func TestRuntimeSettingsAllowsExplicitUnauthenticatedGateway(t *testing.T) {
	t.Parallel()
	settings := RuntimeSettings{
		LLMGatewayURL:         "http://127.0.0.1:4000/v1",
		LLMGatewayToken:       nil,
		ArtifactAPIURL:        "https://server.example/private/v1",
		RequestTimeoutSeconds: 30,
	}
	if err := settings.Validate(); err != nil {
		t.Fatalf("unauthenticated RuntimeSettings were rejected: %v", err)
	}
}

func TestHTTPOriginTargetReferenceAndSecretSettingsAreStrict(t *testing.T) {
	t.Parallel()
	reference := HTTPOriginTargetRef{
		URL: "https://app.example.test/api",
		Credential: &RuntimeCredentialRef{
			CredentialID: "project-origin", Kind: contracts.RuntimeCredentialOriginBearer,
		},
	}
	if err := reference.Validate(); err != nil {
		t.Fatalf("valid target reference: %v", err)
	}
	invalidKind := reference
	invalidKind.Credential = &RuntimeCredentialRef{
		CredentialID: "project-origin", Kind: contracts.RuntimeCredentialProxyBearer,
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
	token := contracts.NewSecretString(secret)
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
