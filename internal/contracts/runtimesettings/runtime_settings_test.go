package runtimesettings

import (
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// everySecretSettings sets each SecretString that RuntimeSettings can hold to
// a distinct canary. The checks below first prove by reflection that no
// secret field is left unset, so a secret field added later fails them until
// it is populated here and covered by RevealJSON and SecretValues. Mutually
// exclusive members are both set: encoding does not validate.
func everySecretSettings() RuntimeSettings {
	secret := func(value string) *contracts.SecretString {
		result := contracts.NewSecretString(value)
		return &result
	}
	return RuntimeSettings{
		LLMGatewayURL:   "https://gateway.example/v1",
		LLMGatewayToken: secret("canary-llm-token"),
		ArtifactAPIURL:  "https://server.example/private/v1",
		Telemetry: &TelemetrySettings{
			Adapter: contracts.RuntimeAdapterOTLPHTTP, Endpoint: "https://otel.example/v1/traces",
			Headers: map[string]contracts.SecretString{
				"Authorization": contracts.NewSecretString("canary-otlp-authorization"),
				"X-Scope-OrgID": contracts.NewSecretString("canary-otlp-tenant"),
			},
			FlushTimeoutSeconds: 5,
		},
		HTTPProxy: &HTTPProxySettings{
			Adapter: contracts.RuntimeAdapterHTTPProxy, ProxyURL: "http://proxy.example:3128",
			BasicAuth: &HTTPProxyBasicAuth{
				Username: contracts.NewSecretString("canary-proxy-user"),
				Password: contracts.NewSecretString("canary-proxy-password"),
			},
			BearerToken: secret("canary-proxy-bearer"),
			Targets:     []HTTPProxyTarget{ProxyTargetLLMGateway},
		},
		Caido: &CaidoSettings{
			Adapter: contracts.RuntimeAdapterCaidoGraphQL, Endpoint: "https://caido.example/graphql",
			BearerToken: secret("canary-caido-bearer"), RequestTimeoutSeconds: 30,
		},
		HTTPOriginTarget: &HTTPOriginTargetSettings{
			URL: "https://app.example/api",
			BasicAuth: &HTTPProxyBasicAuth{
				Username: contracts.NewSecretString("canary-origin-user"),
				Password: contracts.NewSecretString("canary-origin-password"),
			},
			BearerToken: secret("canary-origin-bearer"),
		},
		RequestTimeoutSeconds: 180,
	}
}

var secretStringType = reflect.TypeFor[contracts.SecretString]()

// secretTypePaths lists every place a type can hold a SecretString.
func secretTypePaths(value reflect.Type, path string) []string {
	switch {
	case value == secretStringType:
		return []string{path}
	case value.Kind() == reflect.Pointer || value.Kind() == reflect.Slice:
		return secretTypePaths(value.Elem(), path)
	case value.Kind() == reflect.Map:
		return secretTypePaths(value.Elem(), path+"[]")
	case value.Kind() == reflect.Struct:
		var paths []string
		for index := range value.NumField() {
			field := value.Field(index)
			paths = append(paths, secretTypePaths(field.Type, path+"."+field.Name)...)
		}
		return paths
	}
	return nil
}

// secretValues collects the plaintext of every SecretString a value holds,
// keyed by the secretTypePaths path that holds it.
func secretValues(value reflect.Value, path string, result map[string][]string) {
	switch {
	case value.Type() == secretStringType:
		result[path] = append(result[path], value.Interface().(contracts.SecretString).Reveal())
	case value.Kind() == reflect.Pointer:
		if !value.IsNil() {
			secretValues(value.Elem(), path, result)
		}
	case value.Kind() == reflect.Slice:
		for index := range value.Len() {
			secretValues(value.Index(index), path, result)
		}
	case value.Kind() == reflect.Map:
		for _, key := range value.MapKeys() {
			secretValues(value.MapIndex(key), path+"[]", result)
		}
	case value.Kind() == reflect.Struct:
		for index := range value.NumField() {
			secretValues(value.Field(index), path+"."+value.Type().Field(index).Name, result)
		}
	}
}

func TestRuntimeSettingsRevealSecretsOnlyThroughTheAllocationWireEncoder(t *testing.T) {
	t.Parallel()

	settings := everySecretSettings()
	held := map[string][]string{}
	secretValues(reflect.ValueOf(settings), "", held)
	var canaries []string
	for _, path := range secretTypePaths(reflect.TypeFor[RuntimeSettings](), "") {
		if len(held[path]) == 0 {
			t.Fatalf("everySecretSettings leaves the secret field %s unset", path)
		}
		canaries = append(canaries, held[path]...)
	}
	sort.Strings(canaries)

	formatted := fmt.Sprintf("%v %+v %#v %s", settings, settings, settings, settings.LLMGatewayToken)
	encoded, err := json.Marshal(settings)
	if err != nil {
		t.Fatalf("marshal RuntimeSettings: %v", err)
	}
	for _, canary := range canaries {
		if strings.Contains(formatted, canary) || strings.Contains(string(encoded), canary) {
			t.Fatalf("formatted or default-encoded RuntimeSettings leaked %q: %s %s", canary, formatted, encoded)
		}
	}
	if got := strings.Count(string(encoded), `"[REDACTED]"`); got != len(canaries) {
		t.Fatalf("default encoding redacted %d secrets, want %d: %s", got, len(canaries), encoded)
	}

	wire, err := settings.RevealJSON()
	if err != nil {
		t.Fatalf("RevealJSON: %v", err)
	}
	for _, canary := range canaries {
		if !strings.Contains(string(wire), canary) {
			t.Fatalf("allocation wire encoding omits %q: %s", canary, wire)
		}
	}
	var decoded RuntimeSettings
	decoder := json.NewDecoder(strings.NewReader(string(wire)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decoded); err != nil || !reflect.DeepEqual(decoded, settings) {
		t.Fatalf("allocation wire encoding does not round-trip: %v\n got: %#v\nwant: %#v", err, decoded, settings)
	}

	values := settings.SecretValues()
	sort.Strings(values)
	if !slices.Equal(values, canaries) {
		t.Fatalf("SecretValues = %v, want every secret %v", values, canaries)
	}
}

func TestRuntimeSettingsWireEncodingKeepsAbsentSecretsAbsent(t *testing.T) {
	t.Parallel()

	settings := RuntimeSettings{
		ArtifactAPIURL: "https://server.example/private/v1", RequestTimeoutSeconds: 30,
		Telemetry: &TelemetrySettings{
			Adapter: contracts.RuntimeAdapterOTLPHTTP, Endpoint: "https://otel.example/v1/traces",
			Headers: map[string]contracts.SecretString{}, FlushTimeoutSeconds: 5,
		},
		HTTPProxy: &HTTPProxySettings{
			Adapter: contracts.RuntimeAdapterHTTPProxy, ProxyURL: "http://proxy.example:3128",
			Targets: []HTTPProxyTarget{ProxyTargetToolHTTP},
		},
	}
	wire, err := settings.RevealJSON()
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(settings)
	if err != nil {
		t.Fatal(err)
	}
	var revealed, plain any
	if json.Unmarshal(wire, &revealed) != nil || json.Unmarshal(encoded, &plain) != nil ||
		!reflect.DeepEqual(revealed, plain) {
		t.Fatalf("settings without secrets encode differently on the wire:\n wire: %s\nplain: %s", wire, encoded)
	}
}

// An unauthenticated Gateway omits the token; an empty one is rejected like
// any other malformed value, on the Server and in the Runtime.
func TestRuntimeSettingsUnauthenticatedGatewayOmitsTheToken(t *testing.T) {
	t.Parallel()
	settings := RuntimeSettings{
		LLMGatewayURL:         "http://127.0.0.1:4000/v1",
		ArtifactAPIURL:        "https://server.example/private/v1",
		RequestTimeoutSeconds: 30,
	}
	if err := settings.Validate(); err != nil {
		t.Fatalf("unauthenticated RuntimeSettings were rejected: %v", err)
	}
	empty := contracts.NewSecretString("")
	settings.LLMGatewayToken = &empty
	if err := settings.Validate(); err == nil {
		t.Fatal("an empty llmGatewayToken was accepted")
	}
	const wire = `{"llmGatewayUrl":"http://127.0.0.1:4000/v1","llmGatewayToken":"",` +
		`"artifactApiUrl":"https://server.example/private/v1","requestTimeoutSeconds":30}`
	if _, err := contracts.DecodePrivateStrict[RuntimeSettings]([]byte(wire)); err == nil {
		t.Fatal("an empty llmGatewayToken was decoded")
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
