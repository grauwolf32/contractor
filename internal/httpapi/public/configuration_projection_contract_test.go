package public

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// Exercise projections of manifests copied from the shipped catalog and
// normalized runtime documents, including optional fields that a minimal
// fixture does not contain.
func TestPublicConfigurationProjectionContracts(t *testing.T) {
	document := loadPublicOpenAPI(t)
	router, err := gorillamux.NewRouter(document)
	if err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithConfig(t, publicCatalogFixture)

	t.Run("summary continuation Planner compaction", func(t *testing.T) {
		workflow := catalogWorkflow(t, "likec4-from-workspace@7")
		stage := workflow.Stages[workflow.EntryStage]
		var selection resolved.ResolvedConsumerExecutionConfig
		for _, agent := range stage.ExecutionConfig.Agents {
			selection = agent
			break
		}
		selection.SummarizerModelPolicy = nil
		policy := &resolved.EscalateTransition{
			MaxAttempts: 1, Planner: &resolved.PlannerRef{PlannerID: "streamline", Version: "1"}, DeadlineSeconds: 60,
			ExecutionConfig: resolved.ResolvedEscalationExecutionConfig{Effective: resolved.ResolvedStageExecutionConfig{
				Planner: &selection, Agents: stage.ExecutionConfig.Agents,
			}},
		}
		const privateText = "private compactor instruction text"
		policy.PlannerContextManagement = &resolved.PlannerContextManagement{
			Strategy:           resolved.PlannerContextCompact,
			Instructions:       contracts.ResolvedInstructions{Ref: "instructions/planner-compaction.md", Digest: "sha256:" + strings.Repeat("a", 64), Text: privateText},
			ContextWindowRatio: 0.75, KeepRecentGroups: 4, MaxCompactions: 8,
		}
		compactor := policy.ExecutionConfig.Effective.Planner.ModelPolicy.Clone()
		compactor.Ref.PolicyID = "compactor"
		policy.ExecutionConfig.Effective.Planner.CompactorModelPolicy = &compactor
		policy.ExecutionConfig.Effective.Planner.Origins.CompactorModelPolicy = "workflow.stages.on.summarized.escalate.executionConfig"
		body, err := json.Marshal(workflowTransitionReadModel(resolved.TransitionAction{Kind: resolved.TransitionEscalate, Escalate: policy}))
		if err != nil || strings.Contains(string(body), privateText) {
			t.Fatalf("unsafe summary Planner projection: %v", err)
		}
		var value map[string]any
		if err := json.Unmarshal(body, &value); err != nil {
			t.Fatal(err)
		}
		schema := document.Components.Schemas["WorkflowSummaryTransition"].Value
		if err := schema.VisitJSON(value); err != nil {
			t.Fatalf("summary Planner compaction does not match OpenAPI: %v", err)
		}
		var typed publicapi.WorkflowSummaryTransition
		if err := json.Unmarshal(body, &typed); err != nil || typed.PlannerContextManagement == nil {
			t.Fatalf("generated client lost the continuation context policy: %v", err)
		}
		value["plannerContextManagement"] = nil
		if err := schema.VisitJSON(value); err == nil {
			t.Fatal("public schema accepted null continuation context")
		}
	})

	t.Run("removed escalation profile kind", func(t *testing.T) {
		for _, path := range []string{
			"/v1/configurations/execution-configs",
			"/v1/configurations/execution-configs/strong-routing/versions/1",
		} {
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, authenticatedRequest(http.MethodGet, path, bytes.NewReader(nil)))
			if response.Code != http.StatusBadRequest {
				t.Fatalf("removed kind %s = %d: %s", path, response.Code, response.Body.String())
			}
		}
		if err := document.Components.Schemas["ConfigurationKind"].Value.VisitJSON("execution-configs"); err == nil {
			t.Fatal("OpenAPI still accepts the removed configuration kind")
		}
	})

	t.Run("repository catalog", func(t *testing.T) {
		summarizers, instructions := 0, 0
		for _, kind := range []config.ConfigurationKind{
			config.ConfigurationAgentTemplates,
			config.ConfigurationLLMGateways, config.ConfigurationModelPolicies,
		} {
			t.Run(string(kind), func(t *testing.T) {
				discovery, err := fixture.configs.ConfigurationDiscovery(kind)
				if err != nil {
					t.Fatal(err)
				}
				resources := discovery.Resources
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
			want         *runtimesettings.TelemetryRetrySettings
		}{
			{"legacy omission", `{}`, nil},
			{"defaults", `{"retry":{}}`, &runtimesettings.TelemetryRetrySettings{InitialBackoffMilliseconds: 100, MaxBackoffMilliseconds: 1000}},
			{"explicit", `{"retry":{"initialBackoffMilliseconds":17,"maxBackoffMilliseconds":43}}`, &runtimesettings.TelemetryRetrySettings{InitialBackoffMilliseconds: 17, MaxBackoffMilliseconds: 43}},
			{"minimum", `{"retry":{"initialBackoffMilliseconds":1,"maxBackoffMilliseconds":1}}`, &runtimesettings.TelemetryRetrySettings{InitialBackoffMilliseconds: 1, MaxBackoffMilliseconds: 1}},
			{"maximum", `{"retry":{"initialBackoffMilliseconds":60000,"maxBackoffMilliseconds":60000}}`, &runtimesettings.TelemetryRetrySettings{InitialBackoffMilliseconds: 60000, MaxBackoffMilliseconds: 60000}},
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

func TestSharedPublicWorkerSessionModeContract(t *testing.T) {
	document := loadPublicOpenAPI(t)
	encoded, err := json.Marshal(document.Components.Schemas["WorkflowResource"].Value.Example)
	if err != nil {
		t.Fatal(err)
	}
	var example struct {
		Stages map[string]map[string]any `json:"stages"`
	}
	if err := json.Unmarshal(encoded, &example); err != nil {
		t.Fatal(err)
	}
	stage, ok := example.Stages["copy"]
	if !ok {
		t.Fatal("Workflow example has no copy Stage")
	}
	type sessionCase struct {
		Name  string `json:"name"`
		Omit  bool   `json:"omit"`
		Value any    `json:"value"`
	}
	var cases struct {
		Valid   []sessionCase `json:"valid"`
		Invalid []sessionCase `json:"invalid"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "worker-session-mode-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	schema := document.Components.Schemas["WorkflowStageContract"].Value
	for _, group := range []struct {
		cases []sessionCase
		valid bool
	}{{cases.Valid, true}, {cases.Invalid, false}} {
		for _, item := range group.cases {
			if item.Omit {
				delete(stage, "session")
			} else {
				stage["session"] = item.Value
			}
			if err := schema.VisitJSON(stage); (err == nil) != group.valid {
				t.Fatalf("session %s: valid=%t, want %t: %v", item.Name, err == nil, group.valid, err)
			}
		}
	}
}
