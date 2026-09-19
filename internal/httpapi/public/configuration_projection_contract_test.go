package public

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// Exercise real catalog projections and normalized runtime documents, including
// optional fields that a minimal fixture does not contain.
func TestPublicConfigurationProjectionContracts(t *testing.T) {
	document := loadPublicOpenAPI(t)
	router, err := gorillamux.NewRouter(document)
	if err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithConfig(t, "../../../configs")

	t.Run("repository catalog", func(t *testing.T) {
		summarizers, instructions := 0, 0
		for _, kind := range []config.ConfigurationKind{
			config.ConfigurationAgentTemplates, config.ConfigurationExecutionConfigs,
			config.ConfigurationLLMGateways, config.ConfigurationModelPolicies,
		} {
			t.Run(string(kind), func(t *testing.T) {
				resources, err := fixture.configs.Configurations(kind)
				if err != nil {
					t.Fatal(err)
				}
				list := newPublicContractRequest(http.MethodGet, "/v1/configurations/"+string(kind)+"?limit=200", nil)
				if response := serveAndValidatePublicContract(t, router, fixture.handler, list, true); response.Code != http.StatusOK {
					t.Fatalf("catalog status = %d: %s", response.Code, response.Body.String())
				}
				for _, resource := range resources {
					t.Run(resource.Ref.Name+"@"+resource.Ref.Version, func(t *testing.T) {
						path := fmt.Sprintf("/v1/configurations/%s/%s/versions/%s", kind, resource.Ref.Name, resource.Ref.Version)
						response := serveAndValidatePublicContract(t, router, fixture.handler,
							newPublicContractRequest(http.MethodGet, path, nil), true)
						if response.Code != http.StatusOK {
							t.Fatalf("detail status = %d: %s", response.Code, response.Body.String())
						}
						var actual struct {
							Body struct {
								Summarizer *struct {
									Instructions map[string]string `json:"instructions"`
								} `json:"summarizer"`
							} `json:"body"`
						}
						if err := json.Unmarshal(response.Body.Bytes(), &actual); err != nil {
							t.Fatal(err)
						}
						if actual.Body.Summarizer != nil {
							summarizers++
							if ref := actual.Body.Summarizer.Instructions; ref != nil {
								instructions++
								if len(ref) != 2 || ref["ref"] == "" || ref["digest"] == "" {
									t.Fatalf("instructions must be an exact reference, got %v", ref)
								}
								var typed publicapi.ConfigurationResource
								if err := json.Unmarshal(response.Body.Bytes(), &typed); err != nil {
									t.Fatal(err)
								}
								agent, err := typed.Body.AsAgentTemplateBody()
								if err != nil || agent.Summarizer == nil || agent.Summarizer.Instructions == nil ||
									agent.Summarizer.Instructions.Ref != ref["ref"] || agent.Summarizer.Instructions.Digest != ref["digest"] {
									t.Fatalf("generated client lost summarizer instructions: %+v, %v", agent.Summarizer, err)
								}
							}
						}
					})
				}
			})
		}
		if summarizers == 0 || instructions == 0 {
			t.Fatal("catalog regression fixture must include summarizers with instructions")
		}
	})

	t.Run("telemetry retry publication and reads", func(t *testing.T) {
		for index, test := range []struct {
			name, export string
			want         *contracts.TelemetryRetrySettings
		}{
			{"legacy omission", `{}`, nil},
			{"defaults", `{"retry":{}}`, &contracts.TelemetryRetrySettings{InitialBackoffMilliseconds: 100, MaxBackoffMilliseconds: 1000}},
			{"explicit", `{"retry":{"initialBackoffMilliseconds":17,"maxBackoffMilliseconds":43}}`, &contracts.TelemetryRetrySettings{InitialBackoffMilliseconds: 17, MaxBackoffMilliseconds: 43}},
			{"minimum", `{"retry":{"initialBackoffMilliseconds":1,"maxBackoffMilliseconds":1}}`, &contracts.TelemetryRetrySettings{InitialBackoffMilliseconds: 1, MaxBackoffMilliseconds: 1}},
			{"maximum", `{"retry":{"initialBackoffMilliseconds":60000,"maxBackoffMilliseconds":60000}}`, &contracts.TelemetryRetrySettings{InitialBackoffMilliseconds: 60000, MaxBackoffMilliseconds: 60000}},
		} {
			t.Run(test.name, func(t *testing.T) {
				name := fmt.Sprintf("retry-%d", index)
				body := fmt.Sprintf(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":%q,"version":"1"},"spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://collector.example/v1/traces","export":%s}}}}`, name, test.export)
				request := newPublicContractRequest(http.MethodPost, "/v1/operations/runtime-configs", []byte(body))
				request.Header.Set("Content-Type", "application/json")
				request.Header.Set("Idempotency-Key", "publish-"+name)
				published := serveAndValidatePublicContract(t, router, fixture.handler, request, true)
				if published.Code != http.StatusCreated {
					t.Fatalf("publish status = %d: %s", published.Code, published.Body.String())
				}
				var result runtimeConfigResourceResponse
				if err := json.Unmarshal(published.Body.Bytes(), &result); err != nil {
					t.Fatal(err)
				}
				stored, err := runtimeconfig.DecodeStoredDocument(result.Document)
				if err != nil {
					t.Fatal(err)
				}
				actual := stored.Spec.Worker.Telemetry.Value.Export.Retry
				if !reflect.DeepEqual(actual, test.want) {
					t.Fatalf("normalized retry = %+v, want %+v", actual, test.want)
				}
				exportBytes, err := json.Marshal(stored.Spec.Worker.Telemetry.Value.Export)
				if err != nil {
					t.Fatal(err)
				}
				var typedExport publicapi.WorkerTelemetryExportConfig
				if err := json.Unmarshal(exportBytes, &typedExport); err != nil {
					t.Fatal(err)
				}
				if test.want == nil {
					if typedExport.Retry != nil {
						t.Fatal("generated client added retry to a legacy document")
					}
				} else if typedExport.Retry == nil || typedExport.Retry.InitialBackoffMilliseconds == nil ||
					typedExport.Retry.MaxBackoffMilliseconds == nil ||
					*typedExport.Retry.InitialBackoffMilliseconds != test.want.InitialBackoffMilliseconds ||
					*typedExport.Retry.MaxBackoffMilliseconds != test.want.MaxBackoffMilliseconds {
					t.Fatalf("generated client lost retry settings: %+v", typedExport.Retry)
				}
				for _, path := range []string{
					"/v1/operations/runtime-configs/" + name + "/versions/1",
					"/v1/operations/runtime-configs?limit=200",
				} {
					response := serveAndValidatePublicContract(t, router, fixture.handler,
						newPublicContractRequest(http.MethodGet, path, nil), true)
					if response.Code != http.StatusOK {
						t.Fatalf("read status = %d: %s", response.Code, response.Body.String())
					}
				}
			})
		}
	})

	t.Run("retry bounds and closure", func(t *testing.T) {
		schema := document.Components.Schemas["WorkerTelemetryRetryConfig"].Value
		for _, encoded := range []string{
			`null`, `{"initialBackoffMilliseconds":null}`, `{"maxBackoffMilliseconds":null}`,
			`{"initialBackoffMilliseconds":0}`, `{"maxBackoffMilliseconds":0}`,
			`{"initialBackoffMilliseconds":60001}`, `{"maxBackoffMilliseconds":60001}`,
			`{"initialBackoffMilliseconds":1.5}`, `{"initialBackoffMilliseconds":true}`, `{"unknown":1}`,
		} {
			var value any
			if err := json.Unmarshal([]byte(encoded), &value); err != nil {
				t.Fatal(err)
			}
			if err := schema.VisitJSON(value, openapi3.EnableJSONSchema2020()); err == nil {
				t.Errorf("retry schema accepted %s", encoded)
			}
		}
		// Cross-field ordering is a documented server constraint, checked after defaults.
		for _, retry := range []string{
			`{"initialBackoffMilliseconds":1001}`,
			`{"maxBackoffMilliseconds":99}`,
			`{"initialBackoffMilliseconds":43,"maxBackoffMilliseconds":17}`,
		} {
			body := fmt.Sprintf(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"invalid-retry","version":"1"},"spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://collector.example/v1/traces","export":{"retry":%s}}}}}`, retry)
			request := newPublicContractRequest(http.MethodPost, "/v1/operations/runtime-configs", []byte(body))
			request.Header.Set("Content-Type", "application/json")
			request.Header.Set("Idempotency-Key", "invalid-retry")
			response := serveAndValidatePublicContract(t, router, fixture.handler, request, true)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("inconsistent retry returned %d: %s", response.Code, response.Body.String())
			}
		}
	})
}
