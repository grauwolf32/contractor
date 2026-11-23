package publicclient

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/oapi-codegen/nullable"
)

// Exercise the generated typed HTTP client, not a raw-body escape hatch. Every
// request also passes the production author parser, including explicit clears.
func TestTypedRuntimeConfigRequestsPreservePatchStates(t *testing.T) {
	telemetry := publicapi.RuntimeTelemetryConfig{Adapter: "otlp-http@1", Endpoint: "https://otel.example/v1/traces"}
	proxy := publicapi.RuntimeHTTPProxyConfig{Adapter: "http-proxy@1", ProxyUrl: "http://proxy.example:8080", Targets: []publicapi.RuntimeHTTPProxyConfigTargets{publicapi.ToolHttp}}
	caido := publicapi.RuntimeCaidoConfig{Adapter: "caido-graphql@1", Endpoint: "https://caido.example/graphql"}
	for _, field := range []struct {
		name  string
		path  []string
		value any
		set   func(*publicapi.RuntimeConfigAuthorDocument, string)
	}{
		{"worker telemetry", []string{"spec", "worker", "telemetry"}, telemetry, func(d *publicapi.RuntimeConfigAuthorDocument, s string) {
			d.Spec.Worker.Telemetry = runtimeNullableState(s, telemetry)
		}},
		{"worker proxy", []string{"spec", "worker", "httpProxy"}, proxy, func(d *publicapi.RuntimeConfigAuthorDocument, s string) {
			d.Spec.Worker.HttpProxy = runtimeNullableState(s, proxy)
		}},
		{"worker Caido", []string{"spec", "worker", "caido"}, caido, func(d *publicapi.RuntimeConfigAuthorDocument, s string) {
			d.Spec.Worker.Caido = runtimeNullableState(s, caido)
		}},
		{"planner telemetry", []string{"spec", "planner", "telemetry"}, telemetry, func(d *publicapi.RuntimeConfigAuthorDocument, s string) {
			if s != "absent" {
				d.Spec.Planner = &publicapi.RuntimePlannerPatch{Telemetry: runtimeNullableState(s, telemetry)}
			}
		}},
		{"gateway credential", []string{"spec", "worker", "llmGateway", "credential"}, "worker-local", func(d *publicapi.RuntimeConfigAuthorDocument, s string) {
			d.Spec.Worker.LlmGateway.Credential = runtimeNullableState(s, "worker-local")
		}},
	} {
		for _, state := range []string{"absent", "null", "value"} {
			t.Run(field.name+"/"+state, func(t *testing.T) {
				selector := "local-litellm@1"
				document := publicapi.RuntimeConfigAuthorDocument{
					ApiVersion: "contractor/v1alpha1", Kind: "RuntimeConfig", Metadata: publicapi.RuntimeConfigMetadata{Name: "typed", Version: "1"},
					Spec: publicapi.RuntimeConfigAuthorSpec{Worker: &publicapi.RuntimeWorkerAuthorPatch{LlmGateway: &publicapi.RuntimeLLMGatewayAuthorPatch{Gateway: &selector}}},
				}
				field.set(&document, state)
				received := make(chan []byte, 1)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.Method != http.MethodPost || r.URL.Path != "/v1/operations/runtime-configs" || r.Header.Get("Content-Type") != "application/json" {
						t.Errorf("unexpected request: %s %s %v", r.Method, r.URL.Path, r.Header)
					}
					body, err := io.ReadAll(r.Body)
					if err != nil {
						t.Error(err)
					}
					received <- body
					w.WriteHeader(http.StatusCreated)
				}))
				defer server.Close()
				client, err := publicapi.NewClient(server.URL)
				if err != nil {
					t.Fatal(err)
				}
				response, err := client.PublishRuntimeConfig(t.Context(), &publicapi.PublishRuntimeConfigParams{IdempotencyKey: "typed-patch-test"}, document)
				if err != nil {
					t.Fatal(err)
				}
				response.Body.Close()
				if response.StatusCode != http.StatusCreated {
					t.Fatalf("status=%d", response.StatusCode)
				}
				body := <-received
				if _, err := runtimeconfig.PreparePublication(body); err != nil {
					t.Fatalf("typed request rejected by author parser: %s: %v", body, err)
				}
				var decoded map[string]any
				if err := json.Unmarshal(body, &decoded); err != nil {
					t.Fatal(err)
				}
				var current any = decoded
				present := true
				for _, key := range field.path {
					object, ok := current.(map[string]any)
					if !ok {
						present = false
						break
					}
					current, present = object[key]
					if !present {
						break
					}
				}
				if state == "absent" {
					if present {
						t.Fatalf("absent field was sent: %s", body)
					}
					return
				}
				if !present {
					t.Fatalf("%s field was omitted: %s", state, body)
				}
				if state == "null" {
					if current != nil {
						t.Fatalf("clear became a value: %s", body)
					}
					return
				}
				expectedJSON, err := json.Marshal(field.value)
				if err != nil {
					t.Fatal(err)
				}
				var expected any
				if err := json.Unmarshal(expectedJSON, &expected); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(current, expected) {
					t.Fatalf("value changed: got %v want %v", current, expected)
				}
			})
		}
	}
}

func runtimeNullableState[T any](state string, value T) nullable.Nullable[T] {
	switch state {
	case "absent":
		return nil
	case "null":
		return nullable.NewNullNullable[T]()
	default:
		return nullable.NewNullableWithValue(value)
	}
}
