package public

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestWorkflowToolsetRequirementPublicProjection(t *testing.T) {
	root := catalogDiscoveryConfigRoot(t)
	path := filepath.Join(root, "agent-templates", "artifact_builder.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data = []byte(strings.Replace(string(data), "  toolsets:\n", "  toolsets:\n    - ref: mcp@1\n      tools: [search, fetch]\n", 1))
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithConfig(t, root)
	response := serveQuery(t, fixture.handler, "/v1/workflows/artifact-copy/versions/1")
	var result workflowResourceResponse
	decodeQueryResponse(t, response, &result)
	if response.Code != http.StatusOK || len(result.ToolsetConfigurationRequirements) != 1 {
		t.Fatalf("requirements: %d %s", response.Code, response.Body.String())
	}
	r := result.ToolsetConfigurationRequirements[0]
	if r.Ref != "mcp@1" || !reflect.DeepEqual(r.Tools, []string{"fetch", "search"}) || len(r.Consumers) != 1 || r.ConfigurationSchema != "mcp-streamable-http@1" {
		t.Fatalf("requirements: %+v", r)
	}
	for _, private := range []string{"endpoint", "credential", "llmGatewayUrl", "runtimeAgentId"} {
		if strings.Contains(response.Body.String(), private) {
			t.Fatalf("private field %s", private)
		}
	}
	var value any
	if err := json.Unmarshal(response.Body.Bytes(), &value); err != nil {
		t.Fatal(err)
	}
	if err := loadPublicOpenAPI(t).Components.Schemas["WorkflowResource"].Value.VisitJSON(value); err != nil {
		t.Fatal(err)
	}
	// Read and Run admission use the same traversal, including stages after the
	// entry stage and several logical consumers of one shared configuration.
	workflow, _ := fixture.configs.Workflow("artifact-copy@1")
	workflow.Stages["late"] = config.ResolvedStage{Agents: map[string]config.ResolvedAgentBinding{"reader": {Template: contracts.ResolvedAgentTemplate{Toolsets: []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "mcp", Version: "1"}, Tools: []string{"fetch"}}}}}}}
	projected, err := workflowResourceReadModel(workflow)
	if err != nil || len(projected.ToolsetConfigurationRequirements[0].Consumers) != 2 {
		t.Fatal("late consumer omitted")
	}
	admission, err := config.WorkflowToolsetRequirements(workflow)
	if err != nil || !reflect.DeepEqual(projected.ToolsetConfigurationRequirements, admission) {
		t.Fatal("read requirements diverged from Run admission")
	}
}

func TestRuntimeLabelPublicCoverageEqualClearMixedAndReplay(t *testing.T) {
	fixture := newHandlerFixture(t)
	refs := make([]runtimeconfig.Ref, 3)
	for i, name := range []string{"first", "same", "clear"} {
		toolset := `{"endpoint":"https://private-connection.example/mcp"}`
		if i == 2 {
			toolset = "null"
		}
		document := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"` + name + `","version":"1"},"spec":{"worker":{"toolsets":{"mcp@1":` + toolset + `},"telemetry":null}}}`)
		published, err := fixture.runtimeConfigs.Publish(context.Background(), document, "publish-"+name, "operator")
		if err != nil {
			t.Fatal(err)
		}
		refs[i] = published.Version.Ref
		if _, err := fixture.runtimeConfigs.CreateBinding(context.Background(), name, refs[i], "create-"+name, "operator", time.Now()); err != nil {
			t.Fatal(err)
		}
	}
	response := serveQuery(t, fixture.handler, "/v1/operations/runtime-labels?limit=200")
	var page runtimeLabelPageResponse
	decodeQueryResponse(t, response, &page)
	if response.Code != http.StatusOK || len(page.Items) != 4 {
		t.Fatalf("page: %d %s", response.Code, response.Body.String())
	}
	byLabel := map[string]runtimeLabelResponse{}
	for _, item := range page.Items {
		byLabel[item.Label] = item
	}
	if !reflect.DeepEqual(byLabel["first"].ToolsetConfigurationCoverage, byLabel["same"].ToolsetConfigurationCoverage) {
		t.Fatal("equal settings compared by document identity")
	}
	if byLabel["clear"].ToolsetConfigurationCoverage[0].State != "cleared" || len(byLabel["default"].ToolsetConfigurationCoverage) != 0 {
		t.Fatal("clear/inherit projection differs")
	}
	if !reflect.DeepEqual(byLabel["first"].ChangedSections, []string{"worker.telemetry", "worker.toolsets"}) {
		t.Fatal("mixed effects omitted")
	}
	if strings.Contains(response.Body.String(), "private-connection") || strings.Contains(response.Body.String(), "endpoint") {
		t.Fatal("label picker disclosed endpoint")
	}
	var raw any
	json.Unmarshal(response.Body.Bytes(), &raw)
	if err := loadPublicOpenAPI(t).Components.Schemas["RuntimeLabelPage"].Value.VisitJSON(raw); err != nil {
		t.Fatal(err)
	}
	// A replay returns the original config and its coverage after a later rebind.
	body, _ := json.Marshal(runtimeLabelMutationRequest{Config: refs[0]})
	put := func() *httptest.ResponseRecorder {
		r := runtimeMutationRequest(t, http.MethodPut, "/v1/operations/runtime-labels/replayed", string(body), "replayed-create", "", "*")
		w := httptest.NewRecorder()
		fixture.handler.ServeHTTP(w, r)
		return w
	}
	first := put()
	if first.Code != http.StatusCreated {
		t.Fatal(first.Body.String())
	}
	if _, err := fixture.runtimeConfigs.Rebind(context.Background(), "replayed", 1, refs[2], "rebind-replayed", "operator", time.Now()); err != nil {
		t.Fatal(err)
	}
	replay := put()
	if replay.Code != http.StatusCreated || replay.Body.String() != first.Body.String() {
		t.Fatal("replay coverage followed mutable binding")
	}
	current := serveQuery(t, fixture.handler, "/v1/operations/runtime-labels/replayed")
	var binding runtimeLabelResponse
	decodeQueryResponse(t, current, &binding)
	if binding.Revision != "2" || binding.ToolsetConfigurationCoverage[0].State != "cleared" {
		t.Fatal("retained current label did not reflect rebind")
	}
}

func TestRuntimeConfigurationIncompleteErrorIsStableAndContentFree(t *testing.T) {
	h := &handler{}
	w := httptest.NewRecorder()
	h.handleError(w, &runtimeconfig.ResolutionError{Code: runtimeconfig.ResolutionIncomplete, Path: "worker.toolsets.mcp@1"})
	var response errorResponse
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if w.Code != http.StatusBadRequest || response.Code != "runtime_config_incomplete" || strings.Contains(w.Body.String(), "worker.toolsets") {
		t.Fatal("unsafe or unstable required configuration error")
	}
}
