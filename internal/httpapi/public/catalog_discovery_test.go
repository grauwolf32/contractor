package public

import (
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
)

func TestWorkflowCatalogSearchFiltersBeforePaginationAndPinsQuery(t *testing.T) {
	root := catalogDiscoveryConfigRoot(t)
	fixture := newHandlerFixtureWithConfig(t, root)

	first := serveQuery(t, fixture.handler, "/v1/workflows?limit=1&q="+url.QueryEscape("ARTIFACT"))
	var page workflowPageResponse
	decodeQueryResponse(t, first, &page)
	if first.Code != http.StatusOK || len(page.Items) != 1 || !page.Page.HasMore || page.Page.NextCursor == nil {
		t.Fatalf("first filtered Workflow page = status %d, %+v", first.Code, page)
	}
	second := serveQuery(
		t, fixture.handler,
		"/v1/workflows?limit=1&q=artifact&cursor="+url.QueryEscape(*page.Page.NextCursor),
	)
	var next workflowPageResponse
	decodeQueryResponse(t, second, &next)
	if second.Code != http.StatusOK || len(next.Items) != 1 || next.Items[0].Ref == page.Items[0].Ref || next.Page.HasMore {
		t.Fatalf("second filtered Workflow page = status %d, %+v", second.Code, next)
	}

	decomposed := "u\u0308ber"
	unicodeMatch := serveQuery(t, fixture.handler, "/v1/workflows?q="+url.QueryEscape(decomposed))
	var matched workflowPageResponse
	decodeQueryResponse(t, unicodeMatch, &matched)
	if unicodeMatch.Code != http.StatusOK || len(matched.Items) != 1 ||
		matched.Items[0].Presentation == nil || matched.Items[0].Presentation.DisplayName != "Über Review" {
		t.Fatalf("Unicode Workflow search = status %d, %+v", unicodeMatch.Code, matched)
	}

	versions := serveQuery(t, fixture.handler, "/v1/workflows?name=artifact-copy")
	var versionPage workflowPageResponse
	decodeQueryResponse(t, versions, &versionPage)
	if versions.Code != http.StatusOK || len(versionPage.Items) != 2 ||
		versionPage.Items[0].Ref.Version != "1" || versionPage.Items[1].Ref.Version != "2" {
		t.Fatalf("exact Workflow name filter = status %d, %+v", versions.Code, versionPage)
	}

	for _, target := range []string{
		"/v1/workflows?q=" + url.QueryEscape(strings.Repeat("界", 201)),
		"/v1/workflows?q=" + url.QueryEscape(strings.Repeat(" ", 201)),
		"/v1/workflows?name=Artifact-Copy",
		"/v1/workflows?name=",
		"/v1/workflows?q=other&cursor=" + url.QueryEscape(*page.Page.NextCursor),
	} {
		response := serveQuery(t, fixture.handler, target)
		if response.Code != http.StatusBadRequest {
			t.Errorf("invalid Workflow discovery query %q = %d: %s", target, response.Code, response.Body.String())
		}
	}
}

func TestConfigurationCatalogSearchesAuthoredAgentDescriptionsAndPinsSource(t *testing.T) {
	root := catalogDiscoveryConfigRoot(t)
	fixture := newHandlerFixtureWithConfig(t, root)

	described := serveQuery(
		t, fixture.handler,
		"/v1/configurations/agent-templates?q="+url.QueryEscape("DECLARED INPUT"),
	)
	var descriptionPage configurationPageResponse
	decodeQueryResponse(t, described, &descriptionPage)
	if described.Code != http.StatusOK || len(descriptionPage.Items) != 2 {
		t.Fatalf("Agent description search = status %d, %+v", described.Code, descriptionPage)
	}

	versions := serveQuery(
		t, fixture.handler, "/v1/configurations/agent-templates?limit=1&name=artifact_builder",
	)
	var first configurationPageResponse
	decodeQueryResponse(t, versions, &first)
	if versions.Code != http.StatusOK || len(first.Items) != 1 || !first.Page.HasMore || first.Page.NextCursor == nil {
		t.Fatalf("first exact Agent version page = status %d, %+v", versions.Code, first)
	}
	wrongQuery := serveQuery(
		t, fixture.handler,
		"/v1/configurations/agent-templates?name=unused_worker&cursor="+
			url.QueryEscape(*first.Page.NextCursor),
	)
	if wrongQuery.Code != http.StatusBadRequest {
		t.Fatalf("Agent cursor reused across filters = %d: %s", wrongQuery.Code, wrongQuery.Body.String())
	}

	allPolicies := serveQuery(t, fixture.handler, "/v1/configurations/model-policies?limit=1")
	var policies configurationPageResponse
	decodeQueryResponse(t, allPolicies, &policies)
	if allPolicies.Code != http.StatusOK || policies.Page.NextCursor == nil {
		t.Fatalf("ModelPolicy cursor fixture = status %d, %+v", allPolicies.Code, policies)
	}
	maxOutput, maxModel, maxTool, maxTotal := 4096, 8, 16, 32768
	if _, err := fixture.configs.Publish(t.Context(), config.PublicationRequest{
		Kind: config.ConfigurationModelPolicies, Name: "published-later", Version: "1",
		IdempotencyKey: "catalog-source-change",
		ModelPolicy: &config.ModelPolicyPublication{
			Model: "worker-model", MaxOutputTokens: &maxOutput, MaxModelCalls: &maxModel,
			MaxToolCalls: &maxTool, MaxTotalTokens: &maxTotal,
		},
	}); err != nil {
		t.Fatal(err)
	}
	stale := serveQuery(
		t, fixture.handler,
		"/v1/configurations/model-policies?limit=1&cursor="+
			url.QueryEscape(*policies.Page.NextCursor),
	)
	if stale.Code != http.StatusBadRequest {
		t.Fatalf("configuration cursor survived source change = %d: %s", stale.Code, stale.Body.String())
	}
}

func TestAgentTemplateWorkflowBindingsAreExactSafeAndPaginated(t *testing.T) {
	root := catalogDiscoveryConfigRoot(t)
	fixture := newHandlerFixtureWithConfig(t, root)
	base := "/v1/configurations/agent-templates/artifact_builder/versions/1/workflow-bindings"

	firstResponse := serveQuery(t, fixture.handler, base+"?limit=1")
	var first agentTemplateWorkflowBindingPageResponse
	decodeQueryResponse(t, firstResponse, &first)
	if firstResponse.Code != http.StatusOK || len(first.Items) != 1 ||
		!first.Page.HasMore || first.Page.NextCursor == nil {
		t.Fatalf("first Agent usage page = status %d, %+v", firstResponse.Code, first)
	}
	if first.Items[0].Stage == "" || first.Items[0].LogicalWorker == "" ||
		strings.Contains(firstResponse.Body.String(), "Copy artifacts exactly") ||
		strings.Contains(firstResponse.Body.String(), `"objective"`) {
		t.Fatalf("unsafe or incomplete Agent usage projection: %s", firstResponse.Body.String())
	}
	secondResponse := serveQuery(
		t, fixture.handler, base+"?limit=1&cursor="+url.QueryEscape(*first.Page.NextCursor),
	)
	var second agentTemplateWorkflowBindingPageResponse
	decodeQueryResponse(t, secondResponse, &second)
	if secondResponse.Code != http.StatusOK || len(second.Items) != 1 ||
		second.Items[0].Workflow == first.Items[0].Workflow || second.Page.HasMore {
		t.Fatalf("second Agent usage page = status %d, %+v", secondResponse.Code, second)
	}

	unused := serveQuery(
		t, fixture.handler,
		"/v1/configurations/agent-templates/unused_worker/versions/1/workflow-bindings",
	)
	var empty agentTemplateWorkflowBindingPageResponse
	decodeQueryResponse(t, unused, &empty)
	if unused.Code != http.StatusOK || len(empty.Items) != 0 || empty.Page.HasMore {
		t.Fatalf("unused exact Agent usage = status %d, %+v", unused.Code, empty)
	}
	for _, target := range []string{
		"/v1/configurations/agent-templates/missing/versions/1/workflow-bindings",
		base + "?q=unexpected",
		"/v1/configurations/agent-templates/artifact_builder/versions/2/workflow-bindings?cursor=" +
			url.QueryEscape(*first.Page.NextCursor),
	} {
		response := serveQuery(t, fixture.handler, target)
		if target == "/v1/configurations/agent-templates/missing/versions/1/workflow-bindings" {
			if response.Code != http.StatusNotFound {
				t.Errorf("missing Agent usage = %d: %s", response.Code, response.Body.String())
			}
		} else if response.Code != http.StatusBadRequest {
			t.Errorf("invalid Agent usage query %q = %d: %s", target, response.Code, response.Body.String())
		}
	}
}

func catalogDiscoveryConfigRoot(t *testing.T) string {
	t.Helper()
	source := filepath.Join("..", "..", "config", "testdata", "valid")
	root := filepath.Join(t.TempDir(), "configs")
	if err := os.CopyFS(root, os.DirFS(source)); err != nil {
		t.Fatal(err)
	}

	workflowPath := filepath.Join(root, "workflows", "artifact_copy.yaml")
	workflow, err := os.ReadFile(workflowPath)
	if err != nil {
		t.Fatal(err)
	}
	versionTwo := strings.Replace(string(workflow), "version: \"1\"", "version: \"2\"", 1)
	versionTwo = strings.Replace(
		versionTwo, "spec:\n",
		"spec:\n  presentation:\n    displayName: Über Review\n    description: Artifact workflow with authored catalog metadata.\n",
		1,
	)
	if err := os.WriteFile(filepath.Join(root, "workflows", "artifact_copy_v2.yaml"), []byte(versionTwo), 0o644); err != nil {
		t.Fatal(err)
	}

	templatePath := filepath.Join(root, "agent-templates", "artifact_builder.yaml")
	template, err := os.ReadFile(templatePath)
	if err != nil {
		t.Fatal(err)
	}
	templateVersionTwo := strings.Replace(string(template), "version: \"1\"", "version: \"2\"", 1)
	if err := os.WriteFile(filepath.Join(root, "agent-templates", "artifact_builder_v2.yaml"), []byte(templateVersionTwo), 0o644); err != nil {
		t.Fatal(err)
	}
	unused := strings.Replace(string(template), "name: artifact_builder", "name: unused_worker", 1)
	unused = strings.Replace(
		unused,
		"description: Reads a declared input and writes the requested result",
		"description: Reserved test worker with no Workflow bindings",
		1,
	)
	if err := os.WriteFile(filepath.Join(root, "agent-templates", "unused_worker.yaml"), []byte(unused), 0o644); err != nil {
		t.Fatal(err)
	}

	policyPath := filepath.Join(root, "model-policies", "worker.yaml")
	policy, err := os.ReadFile(policyPath)
	if err != nil {
		t.Fatal(err)
	}
	policyVersionTwo := strings.Replace(string(policy), "version: \"1\"", "version: \"2\"", 1)
	if err := os.WriteFile(filepath.Join(root, "model-policies", "worker_v2.yaml"), []byte(policyVersionTwo), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}
