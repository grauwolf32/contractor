package public

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/getkin/kin-openapi/openapi3filter"
	"github.com/getkin/kin-openapi/routers"
	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"go.yaml.in/yaml/v4"
)

const (
	publicOpenAPIPath = "../../../api/openapi/contractor-public-v1.yaml"
	publicEventsPath  = "../../../api/events/contractor-events-v1.schema.json"
)

func TestPublicOpenAPIContractIsValidAndPolicySafe(t *testing.T) {
	document := loadPublicOpenAPI(t)
	if document.OpenAPI != "3.1.0" {
		t.Fatalf("OpenAPI version = %q, want 3.1.0", document.OpenAPI)
	}
	if document.Info.Extensions["x-contractor-api-version"] != APIVersion {
		t.Fatalf("public API compatibility version = %v", document.Info.Extensions["x-contractor-api-version"])
	}
	versionHeader := document.Components.Headers["ContractorAPIVersion"]
	if versionHeader == nil || versionHeader.Value == nil || versionHeader.Value.Schema == nil ||
		versionHeader.Value.Schema.Value == nil || versionHeader.Value.Schema.Value.Const != APIVersion {
		t.Fatalf("public API version header contract = %#v", versionHeader)
	}

	implemented := make([]string, 0)
	operationIDs := make(map[string]string)
	for path, item := range document.Paths.Map() {
		if !strings.HasPrefix(path, "/v1/") {
			t.Errorf("public path %q is outside /v1", path)
		}
		if strings.Contains(path, "/private") || strings.Contains(path, "/a2a") {
			t.Errorf("private transport path %q is exposed", path)
		}
		for method, operation := range item.Operations() {
			key := strings.ToUpper(method) + " " + path
			implementation, ok := operation.Extensions["x-contractor-implementation"].(string)
			if !ok || implementation != "implemented" && implementation != "planned" {
				t.Errorf("%s has invalid x-contractor-implementation %v", key, operation.Extensions["x-contractor-implementation"])
			}
			if implementation == "implemented" {
				implemented = append(implemented, key)
			}
			if previous, exists := operationIDs[operation.OperationID]; operation.OperationID == "" {
				t.Errorf("%s has no operationId", key)
			} else if exists {
				t.Errorf("operationId %q is shared by %s and %s", operation.OperationID, previous, key)
			} else {
				operationIDs[operation.OperationID] = key
			}
		}
	}
	sort.Strings(implemented)
	wantImplemented := []string{
		"DELETE /v1/audits/{auditId}",
		"DELETE /v1/operations/credentials/{credentialId}",
		"DELETE /v1/operations/runtime-agent-principals/{runtimeAgentId}",
		"DELETE /v1/operations/runtime-credentials/{credentialId}",
		"DELETE /v1/operations/runtime-labels/{label}",
		"DELETE /v1/projects/{projectId}",
		"DELETE /v1/runs/{runId}",
		"GET /v1/artifacts",
		"GET /v1/artifacts/{namespace}/{name}",
		"GET /v1/artifacts/{namespace}/{name}/lineage",
		"GET /v1/artifacts/{namespace}/{name}/metadata",
		"GET /v1/artifacts/{namespace}/{name}/versions",
		"GET /v1/audit-profiles",
		"GET /v1/audit-profiles/{name}/versions/{version}",
		"GET /v1/audit-standards",
		"GET /v1/audit-standards/{scheme}/versions/{version}",
		"GET /v1/audits/{auditId}",
		"GET /v1/audits/{auditId}/coverage",
		"GET /v1/audits/{auditId}/finding-proposals",
		"GET /v1/audits/{auditId}/findings",
		"GET /v1/audits/{auditId}/findings/{findingId}",
		"GET /v1/audits/{auditId}/findings/{findingId}/provenance",
		"GET /v1/audits/{auditId}/items",
		"GET /v1/audits/{auditId}/report",
		"GET /v1/audits/{auditId}/reviews",
		"GET /v1/auth/session",
		"GET /v1/configurations/agent-templates/{name}/versions/{version}/instructions",
		"GET /v1/configurations/{kind}",
		"GET /v1/configurations/{kind}/{name}/versions/{version}",
		"GET /v1/events/ws",
		"GET /v1/operations/allocations",
		"GET /v1/operations/credentials",
		"GET /v1/operations/credentials/{credentialId}",
		"GET /v1/operations/runtime-agent-principals",
		"GET /v1/operations/runtime-agent-principals/{runtimeAgentId}",
		"GET /v1/operations/runtime-agents",
		"GET /v1/operations/runtime-configs",
		"GET /v1/operations/runtime-configs/{name}/versions/{version}",
		"GET /v1/operations/runtime-credentials",
		"GET /v1/operations/runtime-credentials/{credentialId}",
		"GET /v1/operations/runtime-labels",
		"GET /v1/operations/runtime-labels/{label}",
		"GET /v1/operations/settings/scheduler",
		"GET /v1/operations/snapshot",
		"GET /v1/projects",
		"GET /v1/projects/{projectId}",
		"GET /v1/projects/{projectId}/artifacts",
		"GET /v1/projects/{projectId}/artifacts/{namespace}/{name}",
		"GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/lineage",
		"GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/metadata",
		"GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/versions",
		"GET /v1/projects/{projectId}/audits",
		"GET /v1/projects/{projectId}/runs",
		"GET /v1/queue",
		"GET /v1/queue/control",
		"GET /v1/runs",
		"GET /v1/runs/{runId}",
		"GET /v1/runs/{runId}/artifacts",
		"GET /v1/runs/{runId}/artifacts/{namespace}/{name}",
		"GET /v1/runs/{runId}/artifacts/{namespace}/{name}/lineage",
		"GET /v1/runs/{runId}/artifacts/{namespace}/{name}/metadata",
		"GET /v1/runs/{runId}/artifacts/{namespace}/{name}/versions",
		"GET /v1/runs/{runId}/finding-proposals",
		"GET /v1/runs/{runId}/outputs/{slot}",
		"GET /v1/workflows",
		"GET /v1/workflows/{name}/versions/{version}",
		"PATCH /v1/projects/{projectId}",
		"POST /v1/audits/{auditId}/cancel",
		"POST /v1/audits/{auditId}/finding-proposal-imports",
		"POST /v1/audits/{auditId}/findings/{findingId}/reviews",
		"POST /v1/audits/{auditId}/pause",
		"POST /v1/audits/{auditId}/resume",
		"POST /v1/audits/{auditId}/reviews/{requestId}/decisions",
		"POST /v1/audits/{auditId}/start",
		"POST /v1/auth/login",
		"POST /v1/auth/logout",
		"POST /v1/configurations/{kind}",
		"POST /v1/operations/credentials",
		"POST /v1/operations/runtime-configs",
		"POST /v1/operations/runtime-credentials",
		"POST /v1/projects",
		"POST /v1/projects/{projectId}/audits",
		"POST /v1/projects/{projectId}/runs",
		"POST /v1/runs",
		"POST /v1/runs/{runId}/cancel",
		"PUT /v1/artifacts/{namespace}/{name}",
		"PUT /v1/operations/runtime-agent-principals/{runtimeAgentId}/labels",
		"PUT /v1/operations/runtime-labels/{label}",
		"PUT /v1/operations/settings/scheduler",
		"PUT /v1/projects/{projectId}/artifacts/{namespace}/{name}",
		"PUT /v1/queue/control",
	}
	if !reflect.DeepEqual(implemented, wantImplemented) {
		t.Fatalf("implemented public operations = %v, want %v", implemented, wantImplemented)
	}

	websocket := document.Paths.Value("/v1/events/ws").Get
	if websocket == nil || websocket.Extensions["x-websocket-subprotocol"] != "contractor.events.v1" ||
		websocket.Extensions["x-websocket-message-schema"] != "../events/contractor-events-v1.schema.json" {
		t.Fatalf("WebSocket contract extensions = %#v", websocket)
	}
	if _, err := os.Stat(publicEventsPath); err != nil {
		t.Fatalf("linked WebSocket event schema: %v", err)
	}

	raw, err := os.ReadFile(publicOpenAPIPath)
	if err != nil {
		t.Fatal(err)
	}
	var source map[string]any
	if err := yaml.Unmarshal(raw, &source); err != nil {
		t.Fatalf("strict YAML decode: %v", err)
	}
	assertClosedObjectSchemas(t, source, "$")
	assertSafePublicSchemaFields(t, source, "$")
	assertExamplesContainNoSecrets(t, source, "$")

	labels := document.Components.Schemas["RunMetadataLabels"]
	if labels == nil || labels.Value == nil {
		t.Fatal("public API has no RunMetadataLabels schema")
	}
	if err := labels.Value.VisitJSON(map[string]any{
		"purpose": "eval", "eval.id": "eval_01", "eval.leg": "a",
	}, openapi3.EnableJSONSchema2020()); err != nil {
		t.Fatalf("RunMetadataLabels schema rejected a valid map: %v", err)
	}
	tooManyLabels := make(map[string]any)
	for index := 0; index <= runstore.MaxRunMetadataLabels; index++ {
		tooManyLabels[fmt.Sprintf("key%d", index)] = "value"
	}
	for _, invalid := range []any{
		nil,
		[]any{},
		map[string]any{"Upper": "value"},
		map[string]any{"contractor.internal": "value"},
		map[string]any{"purpose": ""},
		map[string]any{"purpose": 1},
		tooManyLabels,
	} {
		if err := labels.Value.VisitJSON(invalid, openapi3.EnableJSONSchema2020()); err == nil {
			t.Fatalf("RunMetadataLabels schema accepted invalid value %#v", invalid)
		}
	}
}

func TestPublicEventSchemaIsClosedAndExamplesValidate(t *testing.T) {
	raw, err := os.ReadFile(publicEventsPath)
	if err != nil {
		t.Fatal(err)
	}
	var source map[string]any
	if err := json.Unmarshal(raw, &source); err != nil {
		t.Fatalf("decode event JSON Schema: %v", err)
	}
	if source["$schema"] != "https://json-schema.org/draft/2020-12/schema" {
		t.Fatalf("event JSON Schema dialect = %v", source["$schema"])
	}
	assertClosedObjectSchemas(t, source, "$")
	assertExamplesContainNoSecrets(t, source, "$")

	definitions, ok := source["$defs"].(map[string]any)
	if !ok {
		t.Fatal("event JSON Schema has no $defs object")
	}
	dereferenced, err := dereferenceLocalDefinitions(source, definitions, nil)
	if err != nil {
		t.Fatalf("resolve event schema references: %v", err)
	}
	encoded, err := json.Marshal(dereferenced)
	if err != nil {
		t.Fatal(err)
	}
	var schema openapi3.Schema
	if err := json.Unmarshal(encoded, &schema); err != nil {
		t.Fatalf("decode event schema with pinned validator: %v", err)
	}
	if err := schema.Validate(t.Context(), openapi3.IsOpenAPI31OrLater(), openapi3.EnableMultiError()); err != nil {
		t.Fatalf("validate event JSON Schema: %v", err)
	}
	examples, ok := source["examples"].([]any)
	if !ok || len(examples) == 0 {
		t.Fatal("event JSON Schema has no examples")
	}
	for index, example := range examples {
		if err := schema.VisitJSON(
			example,
			openapi3.EnableJSONSchema2020(),
			openapi3.EnableFormatValidation(),
			openapi3.MultiErrors(),
		); err != nil {
			t.Errorf("event example %d: %v", index, err)
		}
	}

	invalid := cloneJSONValue(examples[0]).(map[string]any)
	invalid["unexpected"] = true
	if err := schema.VisitJSON(invalid, openapi3.EnableJSONSchema2020()); err == nil {
		t.Fatal("closed event schema accepted an unknown frame field")
	}
	missingDispatchIdentity := cloneJSONValue(examples[1]).(map[string]any)
	delete(missingDispatchIdentity["data"].(map[string]any), "callId")
	if err := schema.VisitJSON(missingDispatchIdentity, openapi3.EnableJSONSchema2020()); err == nil {
		t.Fatal("Planner event schema accepted an incomplete dispatch-selected fact")
	}
	unsafePlannerPayload := cloneJSONValue(examples[1]).(map[string]any)
	unsafePlannerPayload["data"].(map[string]any)["toolPayload"] = map[string]any{"opaque": true}
	if err := schema.VisitJSON(unsafePlannerPayload, openapi3.EnableJSONSchema2020()); err == nil {
		t.Fatal("Planner event schema accepted a raw tool payload")
	}
	schedulerSettingsWithIdentity := cloneJSONValue(examples[len(examples)-1]).(map[string]any)
	schedulerSettingsWithIdentity["data"].(map[string]any)["resourceId"] = "singleton"
	if err := schema.VisitJSON(schedulerSettingsWithIdentity, openapi3.EnableJSONSchema2020()); err == nil {
		t.Fatal("Scheduler settings event schema accepted a resource identity")
	}
}

func TestImplementedPublicHandlersConformToOpenAPI(t *testing.T) {
	document := loadPublicOpenAPI(t)
	router, err := gorillamux.NewRouter(document)
	if err != nil {
		t.Fatalf("build contract router: %v", err)
	}
	fixture := newHandlerFixture(t)
	loginRequest := newPublicContractRequest(
		http.MethodPost,
		"/v1/auth/login",
		[]byte(`{"username":"admin","password":"correct horse battery staple"}`),
	)
	loginRequest.Header.Del("Authorization")
	loginRequest.Header.Set("Content-Type", "application/json")
	loginRequest.Header.Set("Origin", testBrowserOrigin)
	loginResponse := serveAndValidatePublicContract(t, router, fixture.handler, loginRequest, true)
	if loginResponse.Code != http.StatusOK {
		t.Fatalf("browser login = %d: %s", loginResponse.Code, loginResponse.Body.String())
	}
	var browserSession authSessionResponse
	if err := json.Unmarshal(loginResponse.Body.Bytes(), &browserSession); err != nil {
		t.Fatal(err)
	}
	loginCookies := loginResponse.Result().Cookies()
	if len(loginCookies) != 1 {
		t.Fatalf("browser login cookies = %v", loginCookies)
	}
	getSession := newPublicContractRequest(http.MethodGet, "/v1/auth/session", nil)
	getSession.Header.Del("Authorization")
	getSession.AddCookie(loginCookies[0])
	getSession.Header.Set("Origin", testBrowserOrigin)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, getSession, true); response.Code != http.StatusOK {
		t.Fatalf("browser session = %d: %s", response.Code, response.Body.String())
	}
	logout := newPublicContractRequest(http.MethodPost, "/v1/auth/logout", nil)
	logout.Header.Del("Authorization")
	logout.AddCookie(loginCookies[0])
	logout.Header.Set("Origin", testBrowserOrigin)
	logout.Header.Set("X-CSRF-Token", browserSession.CSRFToken)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, logout, true); response.Code != http.StatusNoContent {
		t.Fatalf("browser logout = %d: %s", response.Code, response.Body.String())
	}

	unauthorized := newPublicContractRequest(http.MethodGet, "/v1/runs/missing", nil)
	unauthorized.Header.Del("Authorization")
	if response := serveAndValidatePublicContract(t, router, fixture.handler, unauthorized, false); response.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized response = %d: %s", response.Code, response.Body.String())
	}

	createArtifact := newPublicContractRequest(http.MethodPut, "/v1/artifacts/projects/source", []byte("first"))
	createArtifact.Header.Set("Content-Type", "text/plain")
	createArtifact.Header.Set("If-None-Match", "*")
	created := serveAndValidatePublicContract(t, router, fixture.handler, createArtifact, true)
	if created.Code != http.StatusCreated {
		t.Fatalf("create Artifact = %d: %s", created.Code, created.Body.String())
	}

	updateArtifact := newPublicContractRequest(http.MethodPut, "/v1/artifacts/projects/source", []byte("second"))
	updateArtifact.Header.Set("Content-Type", "text/plain")
	updateArtifact.Header.Set("If-Match", created.Header().Get("ETag"))
	updated := serveAndValidatePublicContract(t, router, fixture.handler, updateArtifact, true)
	if updated.Code != http.StatusOK {
		t.Fatalf("update Artifact = %d: %s", updated.Code, updated.Body.String())
	}

	listWorkflows := newPublicContractRequest(http.MethodGet, "/v1/workflows?limit=1", nil)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, listWorkflows, true); response.Code != http.StatusOK {
		t.Fatalf("list Workflows = %d: %s", response.Code, response.Body.String())
	}
	getWorkflow := newPublicContractRequest(http.MethodGet, "/v1/workflows/artifact-copy/versions/1", nil)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, getWorkflow, true); response.Code != http.StatusOK {
		t.Fatalf("get Workflow = %d: %s", response.Code, response.Body.String())
	}
	createProject := newPublicContractRequest(
		http.MethodPost, "/v1/projects",
		[]byte(`{"kind":"project","name":"Contract project","description":"Contract fixture"}`),
	)
	createProject.Header.Set("Content-Type", "application/json")
	createProject.Header.Set("Idempotency-Key", "contract-create-project")
	createdProject := serveAndValidatePublicContract(t, router, fixture.handler, createProject, true)
	if createdProject.Code != http.StatusCreated {
		t.Fatalf("create Project = %d: %s", createdProject.Code, createdProject.Body.String())
	}
	createProjectArtifact := newPublicContractRequest(
		http.MethodPut, "/v1/projects/project_fixed/artifacts/sources/service", []byte("source archive"),
	)
	createProjectArtifact.Header.Set("Content-Type", "application/zip")
	createProjectArtifact.Header.Set("If-None-Match", "*")
	if response := serveAndValidatePublicContract(
		t, router, fixture.handler, createProjectArtifact, true,
	); response.Code != http.StatusCreated {
		t.Fatalf("create Project Artifact = %d: %s", response.Code, response.Body.String())
	}
	for name, path := range map[string]string{
		"list Projects":             "/v1/projects?kind=project&limit=1",
		"get Project":               "/v1/projects/project_fixed",
		"list Project Artifacts":    "/v1/projects/project_fixed/artifacts",
		"download Project Artifact": "/v1/projects/project_fixed/artifacts/sources/service",
		"Project Artifact metadata": "/v1/projects/project_fixed/artifacts/sources/service/metadata",
		"Project Artifact versions": "/v1/projects/project_fixed/artifacts/sources/service/versions",
		"Project Artifact lineage":  "/v1/projects/project_fixed/artifacts/sources/service/lineage",
	} {
		request := newPublicContractRequest(http.MethodGet, path, nil)
		if response := serveAndValidatePublicContract(t, router, fixture.handler, request, true); response.Code != http.StatusOK {
			t.Fatalf("%s = %d: %s", name, response.Code, response.Body.String())
		}
	}
	updateProject := newPublicContractRequest(
		http.MethodPatch, "/v1/projects/project_fixed", []byte(`{"description":"Updated fixture"}`),
	)
	updateProject.Header.Set("Content-Type", "application/json")
	updateProject.Header.Set("If-Match", createdProject.Header().Get("ETag"))
	if response := serveAndValidatePublicContract(t, router, fixture.handler, updateProject, true); response.Code != http.StatusOK {
		t.Fatalf("update Project = %d: %s", response.Code, response.Body.String())
	}
	listConfigurations := newPublicContractRequest(http.MethodGet, "/v1/configurations/model-policies?limit=1", nil)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, listConfigurations, true); response.Code != http.StatusOK {
		t.Fatalf("list configurations = %d: %s", response.Code, response.Body.String())
	}
	getConfiguration := newPublicContractRequest(http.MethodGet, "/v1/configurations/model-policies/worker/versions/1", nil)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, getConfiguration, true); response.Code != http.StatusOK {
		t.Fatalf("get configuration = %d: %s", response.Code, response.Body.String())
	}
	operationsTemplate, _ := fixture.configs.Snapshot().AgentTemplate("artifact_builder@1")
	operationsGateway, _ := fixture.configs.Snapshot().LLMGateway("local-litellm@1")
	operationsNow := time.Date(2026, 8, 31, 1, 0, 0, 0, time.UTC)
	fixture.operations.set(controlplane.OperationsSnapshot{
		Cursor: controlplane.OperationsCursor{
			Generation: "operations-generation-contract", Revision: 1,
		},
		RuntimeAgents: []controlplane.RuntimeAgentObservation{{
			InstanceID: "runtime-contract", SoftwareVersion: "0.1.0",
			SupportedRuntimes: []string{"adk@1"},
			SupportedToolsets: []controlplane.RuntimeToolsetCapability{{
				Ref: "run-artifacts@1", Tools: []string{"read_artifact"},
			}},
			SupportedSandboxProfiles: []string{"local-workdir@1"},
			SupportedRuntimeAdapters: []string{},
			ObservedState:            contracts.AgentIdle, SlotState: controlplane.SlotReserved,
			LastAcceptedHeartbeat:     &operationsNow,
			AuthoritativeAllocationID: stringPointer("allocation-contract"),
		}},
		Allocations: []controlplane.AllocationObservation{{
			AllocationID: "allocation-contract", RunID: "run-contract",
			StageExecutionID: "stage-contract", RuntimeAgentInstanceID: "runtime-contract",
			LogicalWorker: "builder", AgentTemplate: operationsTemplate.Ref,
			ExecutionConfig: controlplane.AllocationExecutionConfig{
				ModelPolicy: operationsTemplate.ModelPolicy.Ref, LLMGateway: operationsGateway.Ref,
			},
			AuthoritativePhase: controlplane.AllocationPreparing,
			ObservedPhase:      controlplane.AllocationObservedAbsent,
			Metrics:            controlplane.MetricsSummary{ReportsComplete: false},
		}},
	})
	for name, path := range map[string]string{
		"Operations snapshot":    "/v1/operations/snapshot",
		"Runtime Agent page":     "/v1/operations/runtime-agents?limit=1",
		"active allocation page": "/v1/operations/allocations?limit=1",
	} {
		request := newPublicContractRequest(http.MethodGet, path, nil)
		if response := serveAndValidatePublicContract(t, router, fixture.handler, request, true); response.Code != http.StatusOK {
			t.Fatalf("%s = %d: %s", name, response.Code, response.Body.String())
		}
	}
	getSchedulerSettings := newPublicContractRequest(
		http.MethodGet, "/v1/operations/settings/scheduler", nil,
	)
	currentSchedulerSettings := serveAndValidatePublicContract(
		t, router, fixture.handler, getSchedulerSettings, true,
	)
	if currentSchedulerSettings.Code != http.StatusOK || currentSchedulerSettings.Header().Get("ETag") != `"1"` {
		t.Fatalf(
			"get Scheduler settings = %d headers=%v body=%s",
			currentSchedulerSettings.Code, currentSchedulerSettings.Header(), currentSchedulerSettings.Body.String(),
		)
	}
	putSchedulerSettings := newPublicContractRequest(
		http.MethodPut, "/v1/operations/settings/scheduler", []byte(`{"maxConcurrentRuns":2}`),
	)
	putSchedulerSettings.Header.Set("Content-Type", "application/json")
	putSchedulerSettings.Header.Set("If-Match", currentSchedulerSettings.Header().Get("ETag"))
	updatedSchedulerSettings := serveAndValidatePublicContract(
		t, router, fixture.handler, putSchedulerSettings, true,
	)
	if updatedSchedulerSettings.Code != http.StatusOK || updatedSchedulerSettings.Header().Get("ETag") != `"2"` {
		t.Fatalf(
			"put Scheduler settings = %d headers=%v body=%s",
			updatedSchedulerSettings.Code, updatedSchedulerSettings.Header(), updatedSchedulerSettings.Body.String(),
		)
	}
	publishConfiguration := newPublicContractRequest(
		http.MethodPost,
		"/v1/configurations/model-policies",
		[]byte(`{"name":"contract-policy","version":"1","modelPolicy":{"model":"contract-model"}}`),
	)
	publishConfiguration.Header.Set("Content-Type", "application/json")
	publishConfiguration.Header.Set("Idempotency-Key", "contract-publish-configuration")
	if response := serveAndValidatePublicContract(t, router, fixture.handler, publishConfiguration, true); response.Code != http.StatusCreated {
		t.Fatalf("publish configuration = %d: %s", response.Code, response.Body.String())
	}
	gateway, _ := fixture.configs.Snapshot().LLMGateway("local-litellm@1")
	modelPolicy, _ := fixture.configs.Snapshot().ModelPolicy("worker@1")
	credentialBody, err := json.Marshal(createCredentialRequest{
		CredentialID: "contract-worker", LLMGateway: gateway.Ref, Label: stringPointer("Contract worker"),
		GatewayPolicy: credentials.GatewayPolicy{
			ModelPolicies: []contracts.ModelPolicyRef{modelPolicy.Ref},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	createCredential := newPublicContractRequest(
		http.MethodPost, "/v1/operations/credentials", credentialBody,
	)
	createCredential.Header.Set("Content-Type", "application/json")
	createCredential.Header.Set("Idempotency-Key", "contract-create-credential")
	credentialCreated := serveAndValidatePublicContract(t, router, fixture.handler, createCredential, true)
	if credentialCreated.Code != http.StatusCreated || strings.Contains(credentialCreated.Body.String(), "token") {
		t.Fatalf("create credential = %d: %s", credentialCreated.Code, credentialCreated.Body.String())
	}
	for name, path := range map[string]string{
		"list credentials": "/v1/operations/credentials?limit=1",
		"get credential":   "/v1/operations/credentials/contract-worker",
	} {
		request := newPublicContractRequest(http.MethodGet, path, nil)
		if response := serveAndValidatePublicContract(t, router, fixture.handler, request, true); response.Code != http.StatusOK {
			t.Fatalf("%s = %d: %s", name, response.Code, response.Body.String())
		}
	}
	deleteCredential := newPublicContractRequest(
		http.MethodDelete, "/v1/operations/credentials/contract-worker", nil,
	)
	deleteCredential.Header.Set("Idempotency-Key", "contract-delete-credential")
	if response := serveAndValidatePublicContract(t, router, fixture.handler, deleteCredential, true); response.Code != http.StatusNoContent {
		t.Fatalf("delete credential = %d: %s", response.Code, response.Body.String())
	}
	missingCredential := newPublicContractRequest(
		http.MethodGet, "/v1/operations/credentials/contract-worker", nil,
	)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, missingCredential, true); response.Code != http.StatusNotFound {
		t.Fatalf("missing credential = %d: %s", response.Code, response.Body.String())
	}
	conflictingCredentialBody, err := json.Marshal(createCredentialRequest{
		CredentialID: "contract-conflict", LLMGateway: gateway.Ref,
		GatewayPolicy: credentials.GatewayPolicy{
			ModelPolicies: []contracts.ModelPolicyRef{modelPolicy.Ref},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	conflictingCredential := newPublicContractRequest(
		http.MethodPost, "/v1/operations/credentials", conflictingCredentialBody,
	)
	conflictingCredential.Header.Set("Content-Type", "application/json")
	conflictingCredential.Header.Set("Idempotency-Key", "contract-create-credential")
	if response := serveAndValidatePublicContract(t, router, fixture.handler, conflictingCredential, true); response.Code != http.StatusConflict {
		t.Fatalf("credential idempotency conflict = %d: %s", response.Code, response.Body.String())
	}

	createRuntimeCredential := newPublicContractRequest(
		http.MethodPost, "/v1/operations/runtime-credentials",
		[]byte(`{"credentialId":"contract-otel","kind":"otlp-headers@1","material":{"headers":{"authorization":"write-only-value"}}}`),
	)
	createRuntimeCredential.Header.Set("Content-Type", "application/json")
	createRuntimeCredential.Header.Set("Idempotency-Key", "contract-create-runtime-credential")
	if response := serveAndValidatePublicContract(t, router, fixture.handler, createRuntimeCredential, true); response.Code != http.StatusCreated {
		t.Fatalf("create Runtime credential = %d: %s", response.Code, response.Body.String())
	}
	runtimeConfigDocument := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"contract-debug","version":"1"},"spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces","credential":"contract-otel"}}}}`)
	publishRuntimeConfig := newPublicContractRequest(
		http.MethodPost, "/v1/operations/runtime-configs", runtimeConfigDocument,
	)
	publishRuntimeConfig.Header.Set("Content-Type", "application/json")
	publishRuntimeConfig.Header.Set("Idempotency-Key", "contract-publish-runtime-config")
	publishedRuntimeConfig := serveAndValidatePublicContract(t, router, fixture.handler, publishRuntimeConfig, true)
	if publishedRuntimeConfig.Code != http.StatusCreated {
		t.Fatalf("publish RuntimeConfig = %d: %s", publishedRuntimeConfig.Code, publishedRuntimeConfig.Body.String())
	}
	var runtimeResource runtimeConfigResourceResponse
	if err := json.Unmarshal(publishedRuntimeConfig.Body.Bytes(), &runtimeResource); err != nil {
		t.Fatal(err)
	}
	bindingBody, err := json.Marshal(runtimeLabelMutationRequest{Config: runtimeResource.Ref})
	if err != nil {
		t.Fatal(err)
	}
	putRuntimeLabel := newPublicContractRequest(
		http.MethodPut, "/v1/operations/runtime-labels/contract-debug", bindingBody,
	)
	putRuntimeLabel.Header.Set("Content-Type", "application/json")
	putRuntimeLabel.Header.Set("Idempotency-Key", "contract-put-runtime-label")
	putRuntimeLabel.Header.Set("If-None-Match", "*")
	if response := serveAndValidatePublicContract(t, router, fixture.handler, putRuntimeLabel, true); response.Code != http.StatusCreated {
		t.Fatalf("put Runtime label = %d: %s", response.Code, response.Body.String())
	}
	principalID := strings.Repeat("a", 64)
	fixture.runtimePrincipals.principals[principalID] = controlplane.RuntimeAgentPrincipalProjection{
		Principal: runtimeconfig.RuntimeAgentPrincipal{
			RuntimeAgentID: principalID, Labels: []string{"contract-debug"}, LabelRevision: 1,
			CreatedBy: "runtime-registration", CreatedAt: operationsNow,
			UpdatedBy: "runtime-registration", UpdatedAt: operationsNow,
		},
		RequiredRuntimeAdapters: []string{"otlp-http@1"},
		MissingRuntimeAdapters:  []string{"otlp-http@1"},
		Availability:            controlplane.PrincipalAdapterCapabilityMismatch,
		Live: &controlplane.RuntimeAgentObservation{
			InstanceID: "runtime-principal-contract", SoftwareVersion: "0.1.0",
			SupportedRuntimes: []string{"adk@1"}, SupportedToolsets: []controlplane.RuntimeToolsetCapability{},
			SupportedSandboxProfiles: []string{"local-workdir@1"}, SupportedRuntimeAdapters: []string{},
			ObservedState: contracts.AgentIdle, SlotState: controlplane.SlotIdle,
		},
	}
	for name, path := range map[string]string{
		"Runtime Agent principal page":  "/v1/operations/runtime-agent-principals?limit=1",
		"Runtime Agent principal exact": "/v1/operations/runtime-agent-principals/" + principalID,
	} {
		request := newPublicContractRequest(http.MethodGet, path, nil)
		if response := serveAndValidatePublicContract(t, router, fixture.handler, request, true); response.Code != http.StatusOK {
			t.Fatalf("%s = %d: %s", name, response.Code, response.Body.String())
		}
	}
	replacePrincipalLabels := newPublicContractRequest(
		http.MethodPut, "/v1/operations/runtime-agent-principals/"+principalID+"/labels",
		[]byte(`{"labels":[]}`),
	)
	replacePrincipalLabels.Header.Set("Content-Type", "application/json")
	replacePrincipalLabels.Header.Set("Idempotency-Key", "contract-replace-principal-labels")
	replacePrincipalLabels.Header.Set("If-Match", `"1"`)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, replacePrincipalLabels, true); response.Code != http.StatusOK {
		t.Fatalf("replace Runtime Agent labels = %d: %s", response.Code, response.Body.String())
	}
	offlinePrincipal := fixture.runtimePrincipals.principals[principalID]
	offlinePrincipal.Live = nil
	offlinePrincipal.Availability = controlplane.PrincipalOffline
	offlinePrincipal.RequiredRuntimeAdapters = []string{}
	offlinePrincipal.MissingRuntimeAdapters = []string{}
	fixture.runtimePrincipals.principals[principalID] = offlinePrincipal
	deletePrincipal := newPublicContractRequest(
		http.MethodDelete, "/v1/operations/runtime-agent-principals/"+principalID, nil,
	)
	deletePrincipal.Header.Set("Idempotency-Key", "contract-delete-principal")
	deletePrincipal.Header.Set("If-Match", `"2"`)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, deletePrincipal, true); response.Code != http.StatusNoContent {
		t.Fatalf("delete Runtime Agent principal = %d: %s", response.Code, response.Body.String())
	}
	for name, path := range map[string]string{
		"RuntimeConfig page":       "/v1/operations/runtime-configs?limit=1",
		"RuntimeConfig exact":      "/v1/operations/runtime-configs/contract-debug/versions/1",
		"Runtime label page":       "/v1/operations/runtime-labels?limit=1",
		"Runtime label exact":      "/v1/operations/runtime-labels/contract-debug",
		"Runtime credential page":  "/v1/operations/runtime-credentials?limit=1",
		"Runtime credential exact": "/v1/operations/runtime-credentials/contract-otel",
	} {
		request := newPublicContractRequest(http.MethodGet, path, nil)
		if response := serveAndValidatePublicContract(t, router, fixture.handler, request, true); response.Code != http.StatusOK {
			t.Fatalf("%s = %d: %s", name, response.Code, response.Body.String())
		}
	}
	deleteRuntimeLabel := newPublicContractRequest(
		http.MethodDelete, "/v1/operations/runtime-labels/contract-debug", nil,
	)
	deleteRuntimeLabel.Header.Set("Idempotency-Key", "contract-delete-runtime-label")
	deleteRuntimeLabel.Header.Set("If-Match", `"1"`)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, deleteRuntimeLabel, true); response.Code != http.StatusNoContent {
		t.Fatalf("delete Runtime label = %d: %s", response.Code, response.Body.String())
	}
	deleteRuntimeCredential := newPublicContractRequest(
		http.MethodDelete, "/v1/operations/runtime-credentials/contract-otel", nil,
	)
	deleteRuntimeCredential.Header.Set("Idempotency-Key", "contract-delete-runtime-credential")
	if response := serveAndValidatePublicContract(t, router, fixture.handler, deleteRuntimeCredential, true); response.Code != http.StatusNoContent {
		t.Fatalf("delete Runtime credential = %d: %s", response.Code, response.Body.String())
	}

	inUseBody, err := json.Marshal(createCredentialRequest{
		CredentialID: "contract-in-use", LLMGateway: gateway.Ref,
		GatewayPolicy: credentials.GatewayPolicy{
			ModelPolicies: []contracts.ModelPolicyRef{modelPolicy.Ref},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	createInUse := newPublicContractRequest(http.MethodPost, "/v1/operations/credentials", inUseBody)
	createInUse.Header.Set("Content-Type", "application/json")
	createInUse.Header.Set("Idempotency-Key", "contract-create-in-use")
	if response := serveAndValidatePublicContract(t, router, fixture.handler, createInUse, true); response.Code != http.StatusCreated {
		t.Fatalf("create in-use credential = %d: %s", response.Code, response.Body.String())
	}
	fixture.credentials.deleteErr = &credentials.CredentialInUseError{RunIDs: []string{"run-contract"}}
	deleteInUse := newPublicContractRequest(
		http.MethodDelete, "/v1/operations/credentials/contract-in-use", nil,
	)
	deleteInUse.Header.Set("Idempotency-Key", "contract-delete-in-use")
	if response := serveAndValidatePublicContract(t, router, fixture.handler, deleteInUse, true); response.Code != http.StatusConflict {
		t.Fatalf("credential in use = %d: %s", response.Code, response.Body.String())
	}
	fixture.credentials.deleteErr = credentials.ErrGatewayUnavailable
	deleteUnavailable := newPublicContractRequest(
		http.MethodDelete, "/v1/operations/credentials/contract-in-use", nil,
	)
	deleteUnavailable.Header.Set("Idempotency-Key", "contract-delete-unavailable")
	if response := serveAndValidatePublicContract(t, router, fixture.handler, deleteUnavailable, true); response.Code != http.StatusBadGateway {
		t.Fatalf("credential Gateway failure = %d: %s", response.Code, response.Body.String())
	}
	fixture.credentials.deleteErr = nil

	download := newPublicContractRequest(http.MethodGet, "/v1/artifacts/projects/source", nil)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, download, true); response.Code != http.StatusOK {
		t.Fatalf("download Artifact = %d: %s", response.Code, response.Body.String())
	}

	createRunBody := []byte(`{"workflow":"artifact-copy@1","parameters":{"objective":"copy"},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`)
	createRun := newPublicContractRequest(http.MethodPost, "/v1/runs", createRunBody)
	createRun.Header.Set("Content-Type", "application/json")
	createRun.Header.Set("Idempotency-Key", "contract-create-run")
	createdRun := serveAndValidatePublicContract(t, router, fixture.handler, createRun, true)
	if createdRun.Code != http.StatusAccepted {
		t.Fatalf("create Run = %d: %s", createdRun.Code, createdRun.Body.String())
	}

	for name, path := range map[string]string{
		"list UserScope Artifacts": "/v1/artifacts",
		"UserScope metadata":       "/v1/artifacts/projects/source/metadata",
		"UserScope versions":       "/v1/artifacts/projects/source/versions",
		"UserScope lineage":        "/v1/artifacts/projects/source/lineage",
		"global Run Queue":         "/v1/queue",
		"list Runs":                "/v1/runs",
		"list RunScope Artifacts":  "/v1/runs/run_fixed/artifacts",
		"download RunScope input":  "/v1/runs/run_fixed/artifacts/inputs/source",
		"RunScope metadata":        "/v1/runs/run_fixed/artifacts/inputs/source/metadata",
		"RunScope versions":        "/v1/runs/run_fixed/artifacts/inputs/source/versions",
		"RunScope lineage":         "/v1/runs/run_fixed/artifacts/inputs/source/lineage",
	} {
		request := newPublicContractRequest(http.MethodGet, path, nil)
		if response := serveAndValidatePublicContract(t, router, fixture.handler, request, true); response.Code != http.StatusOK {
			t.Fatalf("%s = %d: %s", name, response.Code, response.Body.String())
		}
	}

	getRun := newPublicContractRequest(http.MethodGet, "/v1/runs/run_fixed", nil)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, getRun, true); response.Code != http.StatusOK {
		t.Fatalf("get Run = %d: %s", response.Code, response.Body.String())
	}

	cancel := newPublicContractRequest(http.MethodPost, "/v1/runs/run_fixed/cancel", []byte(`{}`))
	cancel.Header.Set("Content-Type", "application/json")
	cancelled := serveAndValidatePublicContract(t, router, fixture.handler, cancel, true)
	if cancelled.Code != http.StatusAccepted {
		t.Fatalf("cancel Run = %d: %s", cancelled.Code, cancelled.Body.String())
	}
	terminal := fixture.runs.runs["run_fixed"]
	terminal.State = runstore.RunCancelled
	fixture.runs.runs["run_fixed"] = terminal
	repeatedCancel := newPublicContractRequest(http.MethodPost, "/v1/runs/run_fixed/cancel", []byte(`{}`))
	repeatedCancel.Header.Set("Content-Type", "application/json")
	settled := serveAndValidatePublicContract(t, router, fixture.handler, repeatedCancel, true)
	if settled.Code != http.StatusOK {
		t.Fatalf("settled cancel = %d: %s", settled.Code, settled.Body.String())
	}

	runArtifacts, err := fixture.artifacts.Run("run_fixed")
	if err != nil {
		t.Fatal(err)
	}
	output, err := runArtifacts.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "builder", Name: "result"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("finished")}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.artifacts.BindOutputExact(t.Context(), "run_fixed", "result", output.Ref, nil); err != nil {
		t.Fatal(err)
	}
	getOutput := newPublicContractRequest(http.MethodGet, "/v1/runs/run_fixed/outputs/result", nil)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, getOutput, true); response.Code != http.StatusOK {
		t.Fatalf("download Run output = %d: %s", response.Code, response.Body.String())
	}
	deleteRun := newPublicContractRequest(http.MethodDelete, "/v1/runs/run_fixed", nil)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, deleteRun, true); response.Code != http.StatusNoContent {
		t.Fatalf("delete Run = %d: %s", response.Code, response.Body.String())
	}

	badRequest := newPublicContractRequest(http.MethodGet, "/v1/runs/run_fixed?unexpected=true", nil)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, badRequest, false); response.Code != http.StatusBadRequest {
		t.Fatalf("bad request response = %d: %s", response.Code, response.Body.String())
	}
	notFound := newPublicContractRequest(http.MethodGet, "/v1/runs/not-found", nil)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, notFound, true); response.Code != http.StatusNotFound {
		t.Fatalf("not-found response = %d: %s", response.Code, response.Body.String())
	}
	conflict := newPublicContractRequest(http.MethodPut, "/v1/artifacts/projects/source", []byte("conflict"))
	conflict.Header.Set("Content-Type", "text/plain")
	conflict.Header.Set("If-Match", `"stale-revision"`)
	if response := serveAndValidatePublicContract(t, router, fixture.handler, conflict, true); response.Code != http.StatusConflict {
		t.Fatalf("conflict response = %d: %s", response.Code, response.Body.String())
	}
	tooLarge := newPublicContractRequest(http.MethodPut, "/v1/artifacts/projects/large", []byte("bounded"))
	tooLarge.Header.Set("Content-Type", "application/octet-stream")
	tooLarge.ContentLength = artifacts.MaxPayloadSize + 1
	if response := serveAndValidatePublicContract(t, router, fixture.handler, tooLarge, false); response.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("too-large response = %d: %s", response.Code, response.Body.String())
	}
}

func TestProjectRunHandlersConformToOpenAPI(t *testing.T) {
	document := loadPublicOpenAPI(t)
	router, err := gorillamux.NewRouter(document)
	if err != nil {
		t.Fatalf("build contract router: %v", err)
	}
	fixture := newHandlerFixture(t)
	createProject := newPublicContractRequest(
		http.MethodPost, "/v1/projects", []byte(`{"kind":"project","name":"Project Run fixture"}`),
	)
	createProject.Header.Set("Content-Type", "application/json")
	createProject.Header.Set("Idempotency-Key", "project-run-contract-project")
	if response := serveAndValidatePublicContract(
		t, router, fixture.handler, createProject, true,
	); response.Code != http.StatusCreated {
		t.Fatalf("create Project = %d: %s", response.Code, response.Body.String())
	}
	createArtifact := newPublicContractRequest(
		http.MethodPut, "/v1/projects/project_fixed/artifacts/sources/service", []byte("source"),
	)
	createArtifact.Header.Set("Content-Type", "text/plain")
	createArtifact.Header.Set("If-None-Match", "*")
	if response := serveAndValidatePublicContract(
		t, router, fixture.handler, createArtifact, true,
	); response.Code != http.StatusCreated {
		t.Fatalf("create Project input = %d: %s", response.Code, response.Body.String())
	}
	createRun := newPublicContractRequest(
		http.MethodPost, "/v1/projects/project_fixed/runs",
		[]byte(`{"workflow":"artifact-copy@1","artifacts":{"source":{"namespace":"sources","name":"service"}}}`),
	)
	createRun.Header.Set("Content-Type", "application/json")
	createRun.Header.Set("Idempotency-Key", "project-run-contract")
	if response := serveAndValidatePublicContract(
		t, router, fixture.handler, createRun, true,
	); response.Code != http.StatusAccepted {
		t.Fatalf("create Project Run = %d: %s", response.Code, response.Body.String())
	}
	listRuns := newPublicContractRequest(http.MethodGet, "/v1/projects/project_fixed/runs", nil)
	if response := serveAndValidatePublicContract(
		t, router, fixture.handler, listRuns, true,
	); response.Code != http.StatusOK {
		t.Fatalf("list Project Runs = %d: %s", response.Code, response.Body.String())
	}
}

func loadPublicOpenAPI(t *testing.T) *openapi3.T {
	t.Helper()
	loader := openapi3.NewLoader()
	loader.IsExternalRefsAllowed = false
	document, err := loader.LoadFromFile(publicOpenAPIPath)
	if err != nil {
		t.Fatalf("load public OpenAPI: %v", err)
	}
	if err := document.Validate(
		t.Context(),
		openapi3.EnableExamplesValidation(),
		openapi3.EnableSchemaFormatValidation(),
		openapi3.EnableSchemaPatternValidation(),
		openapi3.EnableMultiError(),
	); err != nil {
		t.Fatalf("validate public OpenAPI: %v", err)
	}
	return document
}

func newPublicContractRequest(method, path string, body []byte) *http.Request {
	request := httptest.NewRequest(method, "http://127.0.0.1:8080"+path, bytes.NewReader(body))
	request.Header.Set("Authorization", "Bearer "+testBearerToken)
	return request
}

func serveAndValidatePublicContract(
	t *testing.T,
	router routers.Router,
	handler http.Handler,
	request *http.Request,
	validateRequest bool,
) *httptest.ResponseRecorder {
	t.Helper()
	route, pathParameters, err := router.FindRoute(request)
	if err != nil {
		t.Fatalf("find contract route for %s %s: %v", request.Method, request.URL, err)
	}
	payload, err := readAndRestoreRequestBody(request)
	if err != nil {
		t.Fatal(err)
	}
	options := &openapi3filter.Options{
		AuthenticationFunc:    openapi3filter.NoopAuthenticationFunc,
		IncludeResponseStatus: true,
		MultiError:            true,
		SchemaValidationOptions: []openapi3.SchemaValidationOption{
			openapi3.EnableJSONSchema2020(),
			openapi3.EnableFormatValidation(),
			openapi3.MultiErrors(),
		},
	}
	requestInput := &openapi3filter.RequestValidationInput{
		Request: request, PathParams: pathParameters, Route: route, Options: options,
	}
	if validateRequest {
		if err := openapi3filter.ValidateRequest(t.Context(), requestInput); err != nil {
			t.Fatalf("request does not conform to %s: %v", route.Operation.OperationID, err)
		}
		restoreRequestBody(request, payload)
	}

	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	responseInput := &openapi3filter.ResponseValidationInput{
		RequestValidationInput: requestInput,
		Status:                 response.Code,
		Header:                 response.Header(),
		Options:                options,
	}
	responseInput.SetBodyBytes(response.Body.Bytes())
	if err := openapi3filter.ValidateResponse(t.Context(), responseInput); err != nil {
		t.Fatalf(
			"response does not conform to %s: status=%d headers=%v body=%q: %v",
			route.Operation.OperationID, response.Code, response.Header(), response.Body.String(), err,
		)
	}
	return response
}

func readAndRestoreRequestBody(request *http.Request) ([]byte, error) {
	if request.Body == nil || request.Body == http.NoBody {
		return nil, nil
	}
	payload, err := io.ReadAll(request.Body)
	if err != nil {
		return nil, err
	}
	restoreRequestBody(request, payload)
	return payload, nil
}

func restoreRequestBody(request *http.Request, payload []byte) {
	request.Body = http.NoBody
	if len(payload) > 0 {
		request.Body = io.NopCloser(bytes.NewReader(payload))
	}
}

func assertClosedObjectSchemas(t *testing.T, value any, path string) {
	t.Helper()
	switch current := value.(type) {
	case map[string]any:
		if schemaDeclaresObject(current["type"]) {
			if _, additional := current["additionalProperties"]; !additional {
				if _, unevaluated := current["unevaluatedProperties"]; !unevaluated {
					t.Errorf("object schema %s does not state its unknown-field policy", path)
				}
			}
		}
		for key, child := range current {
			assertClosedObjectSchemas(t, child, path+"/"+key)
		}
	case []any:
		for index, child := range current {
			assertClosedObjectSchemas(t, child, fmt.Sprintf("%s/%d", path, index))
		}
	}
}

func schemaDeclaresObject(value any) bool {
	if value == "object" {
		return true
	}
	values, ok := value.([]any)
	if !ok {
		return false
	}
	for _, candidate := range values {
		if candidate == "object" {
			return true
		}
	}
	return false
}

func assertSafePublicSchemaFields(t *testing.T, value any, path string) {
	t.Helper()
	forbidden := map[string]struct{}{
		"adminkey":             {},
		"allocationcapability": {},
		"apitoken":             {},
		"bearertoken":          {},
		"certificate":          {},
		"ciphertext":           {},
		"credentialtoken":      {},
		"llmgatewaytoken":      {},
		"modelresponse":        {},
		"nonce":                {},
		"plaintext":            {},
		"privatekey":           {},
		"prompt":               {},
		"providerbody":         {},
		"providerresponse":     {},
		"rawproviderbody":      {},
		"reasoning":            {},
		"runtimeurl":           {},
		"secret":               {},
		"tlsprincipal":         {},
		"toolarguments":        {},
		"toolpayload":          {},
		"workerendpoint":       {},
		"workerurl":            {},
	}
	switch current := value.(type) {
	case map[string]any:
		if properties, ok := current["properties"].(map[string]any); ok {
			for name := range properties {
				if _, rejected := forbidden[strings.ToLower(name)]; rejected {
					t.Errorf("private or unsafe public schema field %s/properties/%s", path, name)
				}
			}
		}
		for key, child := range current {
			assertSafePublicSchemaFields(t, child, path+"/"+key)
		}
	case []any:
		for index, child := range current {
			assertSafePublicSchemaFields(t, child, fmt.Sprintf("%s/%d", path, index))
		}
	}
}

func assertExamplesContainNoSecrets(t *testing.T, value any, path string) {
	t.Helper()
	switch current := value.(type) {
	case map[string]any:
		for key, child := range current {
			if key == "example" || key == "examples" {
				assertSafeExampleValue(t, child, path+"/"+key)
			}
			assertExamplesContainNoSecrets(t, child, path+"/"+key)
		}
	case []any:
		for index, child := range current {
			assertExamplesContainNoSecrets(t, child, fmt.Sprintf("%s/%d", path, index))
		}
	}
}

func assertSafeExampleValue(t *testing.T, value any, path string) {
	t.Helper()
	forbiddenKeys := map[string]struct{}{
		"credential": {}, "password": {}, "prompt": {}, "providerbody": {},
		"rawproviderbody": {}, "runtimeurl": {}, "token": {}, "toolarguments": {},
		"toolpayload": {}, "workerendpoint": {}, "workerurl": {},
	}
	switch current := value.(type) {
	case map[string]any:
		for key, child := range current {
			if _, rejected := forbiddenKeys[strings.ToLower(key)]; rejected {
				t.Errorf("secret-bearing example key %s/%s", path, key)
			}
			assertSafeExampleValue(t, child, path+"/"+key)
		}
	case []any:
		for index, child := range current {
			assertSafeExampleValue(t, child, fmt.Sprintf("%s/%d", path, index))
		}
	}
}

func dereferenceLocalDefinitions(value any, definitions map[string]any, stack []string) (any, error) {
	switch current := value.(type) {
	case map[string]any:
		if reference, ok := current["$ref"].(string); ok {
			const prefix = "#/$defs/"
			if !strings.HasPrefix(reference, prefix) {
				return nil, fmt.Errorf("unsupported reference %q", reference)
			}
			name := strings.TrimPrefix(reference, prefix)
			for _, active := range stack {
				if active == name {
					return nil, fmt.Errorf("recursive local definition %q", name)
				}
			}
			target, exists := definitions[name]
			if !exists {
				return nil, fmt.Errorf("unknown local definition %q", name)
			}
			resolved, err := dereferenceLocalDefinitions(target, definitions, append(stack, name))
			if err != nil {
				return nil, err
			}
			result := resolved.(map[string]any)
			for key, sibling := range current {
				if key == "$ref" {
					continue
				}
				result[key], err = dereferenceLocalDefinitions(sibling, definitions, stack)
				if err != nil {
					return nil, err
				}
			}
			return result, nil
		}
		result := make(map[string]any, len(current))
		for key, child := range current {
			resolved, err := dereferenceLocalDefinitions(child, definitions, stack)
			if err != nil {
				return nil, err
			}
			result[key] = resolved
		}
		return result, nil
	case []any:
		result := make([]any, len(current))
		for index, child := range current {
			resolved, err := dereferenceLocalDefinitions(child, definitions, stack)
			if err != nil {
				return nil, err
			}
			result[index] = resolved
		}
		return result, nil
	default:
		return current, nil
	}
}

func cloneJSONValue(value any) any {
	encoded, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	var result any
	if err := json.Unmarshal(encoded, &result); err != nil {
		panic(err)
	}
	return result
}

func TestPublicOpenAPIPathsAreRepositoryRelative(t *testing.T) {
	for _, path := range []string{publicOpenAPIPath, publicEventsPath} {
		absolute, err := filepath.Abs(path)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(absolute); err != nil {
			t.Errorf("contract path %s: %v", absolute, err)
		}
	}
}
