package runtimeconfig

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestNormalizeEquivalentDefaultsAndTargetOrder(t *testing.T) {
	t.Parallel()
	first := []byte(`{
  "apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
  "metadata":{"name":"debug","version":"1"},
  "spec":{"worker":{
    "llmGateway":{"gateway":"local-litellm@1","credential":"worker-local"},
    "telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces"},
    "httpProxy":{"adapter":"http-proxy@1","proxyUrl":"http://proxy.example:8080","targets":["tool-subprocess","llm-gateway","tool-http"]}
  }}
}`)
	second := []byte(`{
  "kind":"RuntimeConfig","apiVersion":"contractor/v1alpha1",
  "metadata":{"version":"1","name":"debug"},
  "spec":{"worker":{
    "httpProxy":{"targets":["llm-gateway","tool-http","tool-subprocess"],"proxyUrl":"http://proxy.example:8080","adapter":"http-proxy@1"},
    "telemetry":{"flushTimeoutSeconds":3,"captureContent":false,"endpoint":"https://otel.example/v1/traces","adapter":"otlp-http@1"},
    "llmGateway":{"credential":"worker-local","gateway":"local-litellm@1"}
  }}
}`)

	left, err := PreparePublication(first)
	if err != nil {
		t.Fatal(err)
	}
	right, err := PreparePublication(second)
	if err != nil {
		t.Fatal(err)
	}
	if left.RequestDigest() != right.RequestDigest() || string(left.authorCanonical) != string(right.authorCanonical) {
		t.Fatalf("equivalent author requests differ:\n%s\n%s", left.authorCanonical, right.authorCanonical)
	}
	resolver := fixedGatewayResolver("sha256:" + strings.Repeat("a", 64))
	leftVersion, err := left.Resolve(context.Background(), resolver)
	if err != nil {
		t.Fatal(err)
	}
	rightVersion, err := right.Resolve(context.Background(), resolver)
	if err != nil {
		t.Fatal(err)
	}
	if leftVersion.Ref != rightVersion.Ref || string(leftVersion.CanonicalDocument) != string(rightVersion.CanonicalDocument) {
		t.Fatalf("equivalent immutable documents differ:\n%s\n%s", leftVersion.CanonicalDocument, rightVersion.CanonicalDocument)
	}
	decoded, err := DecodeStoredDocument(leftVersion.CanonicalDocument)
	if err != nil {
		t.Fatal(err)
	}
	if !decoded.Spec.Worker.LLMGateway.Gateway.Present || decoded.Spec.Worker.Telemetry.Value.FlushTimeoutSeconds != 3 {
		t.Fatalf("decoded normalized spec = %+v", decoded.Spec)
	}
	if got := decoded.Spec.Worker.HTTPProxy.Value.Targets; strings.Join(got, ",") != "llm-gateway,tool-http,tool-subprocess" {
		t.Fatalf("normalized targets = %v", got)
	}
}

func TestNormalizeTrustedContentOptIn(t *testing.T) {
	prepared, err := PreparePublication([]byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"debug","version":"2"},"spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"http://localhost:3000/api/public/otel/v1/traces","captureContent":true}},"planner":{"telemetry":{"adapter":"otlp-http@1","endpoint":"http://localhost:3000/api/public/otel/v1/traces","captureContent":true}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	version, err := prepared.Resolve(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeStoredDocument(version.CanonicalDocument)
	if err != nil {
		t.Fatal(err)
	}
	if !decoded.Spec.Worker.Telemetry.Value.CaptureContent || !decoded.Spec.Planner.Telemetry.Value.CaptureContent {
		t.Fatal("capture opt-in lost")
	}
}

func TestExplicitNullIsPatchAndOmissionIsNot(t *testing.T) {
	t.Parallel()
	prepared, err := PreparePublication([]byte(`{
  "apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
  "metadata":{"name":"no-debug","version":"1"},
  "spec":{"worker":{"telemetry":null},"planner":{"telemetry":null}}
}`))
	if err != nil {
		t.Fatal(err)
	}
	version, err := prepared.Resolve(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !version.Spec.Worker.Telemetry.Present || !version.Spec.Worker.Telemetry.Clear ||
		!version.Spec.Planner.Telemetry.Present || !version.Spec.Planner.Telemetry.Clear {
		t.Fatalf("explicit null was not retained: %+v", version.Spec)
	}
}

func TestCaidoConfigNormalizesAsOneAtomicWorkerField(t *testing.T) {
	t.Parallel()
	document := []byte(`{
  "apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
  "metadata":{"name":"caido-lab","version":"1"},
  "spec":{"worker":{"caido":{
    "requestTimeoutSeconds":30,"credential":"caido-token",
    "endpoint":"https://caido.example/prefix","adapter":"caido-graphql@1"
  }}}
}`)
	prepared, err := PreparePublication(document)
	if err != nil {
		t.Fatal(err)
	}
	version, err := prepared.Resolve(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	caido := version.Spec.Worker.Caido
	if !caido.Present || caido.Clear || caido.Value.Adapter != "caido-graphql@1" ||
		caido.Value.Endpoint != "https://caido.example/prefix" ||
		caido.Value.Credential != "caido-token" || caido.Value.RequestTimeoutSeconds != 30 {
		t.Fatalf("normalized Caido config = %+v", caido)
	}
	decoded, err := DecodeStoredDocument(version.CanonicalDocument)
	if err != nil || decoded.Spec.Worker.Caido != caido {
		t.Fatalf("stored Caido round trip = (%+v, %v)", decoded.Spec.Worker.Caido, err)
	}

	clear, err := PreparePublication([]byte(`{
  "apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
  "metadata":{"name":"no-caido","version":"1"},
  "spec":{"worker":{"caido":null}}
}`))
	if err != nil {
		t.Fatal(err)
	}
	cleared, err := clear.Resolve(context.Background(), nil)
	if err != nil || !cleared.Spec.Worker.Caido.Present || !cleared.Spec.Worker.Caido.Clear {
		t.Fatalf("Caido clear = (%+v, %v)", cleared.Spec.Worker.Caido, err)
	}
}

func TestPlannerTelemetryAdapterPublicationUsesServerCatalog(t *testing.T) {
	t.Parallel()
	spec := Spec{Planner: PlannerPatch{Telemetry: AtomicPatch[TelemetryConfig]{
		Present: true,
		Value: TelemetryConfig{
			Adapter: "otlp-http@1", Endpoint: "https://otel.example/v1/traces",
			FlushTimeoutSeconds: 3,
		},
	}}}
	accepting := PlannerTelemetryAdapterCatalogFunc(func(ref string) bool {
		return ref == "otlp-http@1"
	})
	if err := validatePlannerTelemetryAdapter(spec, accepting); err != nil {
		t.Fatalf("registered Planner adapter rejected: %v", err)
	}
	rejecting := PlannerTelemetryAdapterCatalogFunc(func(string) bool { return false })
	if err := validatePlannerTelemetryAdapter(spec, rejecting); !errors.Is(err, ErrInvalid) {
		t.Fatalf("unregistered Planner adapter error = %v", err)
	}
	spec.Planner.Telemetry.Clear = true
	if err := validatePlannerTelemetryAdapter(spec, rejecting); err != nil {
		t.Fatalf("clear operation consulted current adapter registry: %v", err)
	}
}

func TestPreparePublicationRejectsUnsafeOrAmbiguousDocuments(t *testing.T) {
	t.Parallel()
	base := func(spec string) []byte {
		return []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"debug","version":"1"},"spec":` + spec + `}`)
	}
	tests := map[string][]byte{
		"empty":                   base(`{}`),
		"unknown":                 base(`{"worker":{"telemetry":null,"mystery":1}}`),
		"duplicate":               base(`{"worker":{"telemetry":null,"telemetry":null}}`),
		"bad endpoint":            base(`{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://user:secret@example/a?x=1"}}}`),
		"null content capture":    base(`{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://example/a","captureContent":null}}}`),
		"duplicate targets":       base(`{"worker":{"httpProxy":{"adapter":"http-proxy@1","proxyUrl":"http://proxy.example","targets":["tool-http","tool-http"]}}}`),
		"null gateway":            base(`{"worker":{"llmGateway":{"gateway":null}}}`),
		"invalid surrogate":       base(`{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://example/\ud800"}}}`),
		"Caido wrong adapter":     base(`{"worker":{"caido":{"adapter":"http-proxy@1","endpoint":"https://caido.example"}}}`),
		"Caido endpoint userinfo": base(`{"worker":{"caido":{"adapter":"caido-graphql@1","endpoint":"https://user:secret@caido.example"}}}`),
		"Caido endpoint query":    base(`{"worker":{"caido":{"adapter":"caido-graphql@1","endpoint":"https://caido.example?token=secret"}}}`),
		"Caido null credential":   base(`{"worker":{"caido":{"adapter":"caido-graphql@1","endpoint":"https://caido.example","credential":null}}}`),
		"Caido zero timeout":      base(`{"worker":{"caido":{"adapter":"caido-graphql@1","endpoint":"https://caido.example","requestTimeoutSeconds":0}}}`),
		"Caido excessive timeout": base(`{"worker":{"caido":{"adapter":"caido-graphql@1","endpoint":"https://caido.example","requestTimeoutSeconds":121}}}`),
	}
	for name, document := range tests {
		document := document
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if _, err := PreparePublication(document); !errors.Is(err, ErrInvalid) {
				t.Fatalf("PreparePublication error = %v", err)
			}
		})
	}
	if _, err := PreparePublication([]byte(strings.Repeat(" ", maxDocumentBytes+1))); !errors.Is(err, ErrInvalid) {
		t.Fatalf("oversized error = %v", err)
	}
	builtIn := []byte(BuiltInCanonicalDocument)
	if _, err := PreparePublication(builtIn); !errors.Is(err, ErrReserved) {
		t.Fatalf("built-in publication error = %v", err)
	}
}

func TestPrepareRejectsPrivateKeyPEM(t *testing.T) {
	t.Parallel()
	document := map[string]any{
		"apiVersion": APIVersion, "kind": Kind,
		"metadata": map[string]any{"name": "caido", "version": "1"},
		"spec": map[string]any{"worker": map[string]any{"httpProxy": map[string]any{
			"adapter": "http-proxy@1", "proxyUrl": "http://proxy.example", "targets": []string{"tool-http"},
			"caBundlePem": "-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n",
		}}},
	}
	encoded, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PreparePublication(encoded); !errors.Is(err, ErrInvalid) {
		t.Fatalf("private-key PEM error = %v", err)
	}
}

func TestBuiltInCanonicalDigest(t *testing.T) {
	t.Parallel()
	if got := digest([]byte(BuiltInCanonicalDocument)); got != BuiltInDigest {
		t.Fatalf("built-in digest = %s, want %s", got, BuiltInDigest)
	}
	version, err := DecodeStoredDocument([]byte(BuiltInCanonicalDocument))
	if err != nil || version.Ref.Digest != BuiltInDigest || !version.BuiltIn {
		t.Fatalf("decode built-in = (%+v, %v)", version, err)
	}
}

func fixedGatewayResolver(digest string) GatewayResolver {
	return GatewayResolverFunc(func(_ context.Context, selector string) (contracts.ResolvedLLMGatewayConfig, error) {
		id, version, ok := strings.Cut(selector, "@")
		if !ok {
			return contracts.ResolvedLLMGatewayConfig{}, errors.New("bad selector")
		}
		return contracts.ResolvedLLMGatewayConfig{
			Ref:      contracts.LLMGatewayConfigRef{GatewayID: id, Version: version, Digest: digest},
			Protocol: contracts.OpenAICompatibleProtocol, URL: "http://127.0.0.1:4000/v1",
		}, nil
	})
}
