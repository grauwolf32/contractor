//go:build e2e

package e2e

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"go.yaml.in/yaml/v4"
)

// No Runtime process is started and the fixture has no model gateway. The
// public Run creation, Scheduler, domain join and ArtifactStore are real.
func TestProjectSitemapServerOperationsAcrossProductionProcesses(t *testing.T) {
	if testing.Short() || os.Getenv("CONTRACTOR_TEST_DATABASE_URL") == "" {
		t.Fatal("sitemap operation process gate requires PostgreSQL and a full run")
	}
	root := repoRoot(t)
	h := startPlacementStackWithConfiguration(t, func(target string) {
		data, err := os.ReadFile(filepath.Join(root, "internal/config/testdata/server-operations/workflows/join.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range []string{"sitemap-join", "sitemap-apply", "sitemap-apply-only"} {
			var definition map[string]any
			if err := yaml.Unmarshal(data, &definition); err != nil {
				t.Fatal(err)
			}
			definition["metadata"].(map[string]any)["name"] = name
			spec := definition["spec"].(map[string]any)
			stage := spec["stages"].(map[string]any)["join-api"].(map[string]any)
			if name != "sitemap-join" {
				stage["operation"].(map[string]any)["ref"] = contracts.SitemapApplyOpenAPIOperation
			}
			if name == "sitemap-apply-only" {
				spec["outputs"] = map[string]any{}
				stage["workflowOutputs"] = map[string]any{}
				stage["result"] = map[string]any{"artifacts": map[string]any{}}
			} else {
				stage["result"].(map[string]any)["artifacts"].(map[string]any)["report"] = map[string]any{"required": false, "mediaTypes": []string{"application/json"}, "from": map[string]any{"namespace": "join-api", "name": "report"}}
			}
			text, err := yaml.Marshal(definition)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(target, "workflows", name+".yaml"), text, 0600); err != nil {
				t.Fatal(err)
			}
		}
	})
	for _, scenario := range []string{"pure", "apply", "apply-only", "invalid"} {
		t.Run(scenario, func(t *testing.T) {
			var project projectResourceResponse
			h.decode(t, h.operations.request(http.MethodPost, "/v1/projects", map[string]any{"kind": "project", "name": "Sitemap " + scenario, "description": "Server operation fixture"}, http.StatusCreated, map[string]string{"Idempotency-Key": "operation-project-" + scenario}), &project)
			var seeded projectsitemap.Result
			h.decode(t, h.operations.request(http.MethodPost, "/v1/projects/"+project.ProjectID+"/sitemap/mutations", map[string]any{"operation": "declare", "expectedRevision": "absent", "url": "https://example.test/users/me", "method": "GET"}, http.StatusOK, map[string]string{"Idempotency-Key": "seed"}), &seeded)
			source := []byte(`{"openapi":"3.1.0","paths":{"/users/{id}":{"get":{}},"/users/me":{"get":{}},"/declared":{"post":{}}}}`)
			want := "succeeded"
			workflow := "sitemap-join@1"
			if strings.HasPrefix(scenario, "apply") {
				workflow = "sitemap-" + scenario + "@1"
			}
			if scenario == "invalid" {
				source = []byte(`{"openapi":"3.1.0","paths":{"/users":{"$ref":"https://not-contacted.invalid/api"}}}`)
				want = "failed"
			}
			openapi := uploadProjectScopeArtifact(t, h.operations.client, h.operations.baseURL, project.ProjectID, "contracts", "api", "application/json", source)
			body, _ := json.Marshal(map[string]any{"workflow": workflow, "parameters": map[string]string{"api_origin": "https://example.test"}, "artifacts": map[string]any{"sitemap": seeded.Ref, "openapi": openapi}})
			run := postProjectRun(t, h.operations.client, h.operations.baseURL, project.ProjectID, scenario, body, false)
			status := h.terminal(t, run, want)
			if len(status.Attempts) != 1 || status.Attempts[0].PlannerStartedAt != nil || len(status.Attempts[0].Resources) != 0 {
				t.Fatalf("synthetic execution: %+v", status.Attempts)
			}
			stages, err := h.store.ListStageExecutions(h.ctx, run)
			if err != nil || len(stages) != 1 || stages[0].PlannerSessionID != nil || stages[0].PlannerInvocationID != nil {
				t.Fatalf("Planner session: %+v %v", stages, err)
			}
			allocations, err := h.store.ListStageAllocations(h.ctx, stages[0].StageExecutionID)
			if err != nil || len(allocations) != 0 {
				t.Fatalf("allocated Runtime: %+v %v", allocations, err)
			}
			if scenario == "apply-only" {
				h.operations.request(http.MethodGet, "/v1/runs/"+run+"/artifacts/join-api/sitemap", nil, http.StatusNotFound, nil)
				if stages[0].AcceptedResult == nil || !strings.Contains(stages[0].AcceptedResult.Summary, "Published Project sitemap/map@") {
					t.Fatal("bounded result lost publication ref")
				}
			}
			live := h.operations.request(http.MethodGet, "/v1/projects/"+project.ProjectID+"/artifacts/sitemap/map", nil, http.StatusOK, nil)
			document, err := projectsitemap.Decode(live)
			if err != nil {
				t.Fatal(err)
			}
			if (len(document.Rules) > 0) != strings.HasPrefix(scenario, "apply") {
				t.Fatalf("unexpected live publication: %+v", document.Rules)
			}
			if want == "succeeded" && scenario != "apply-only" {
				result := h.operations.request(http.MethodGet, "/v1/runs/"+run+"/artifacts/outputs/sitemap", nil, http.StatusOK, nil)
				joined, err := projectsitemap.Decode(result)
				if err != nil || len(joined.Rules) != 3 {
					t.Fatalf("output map: %s %v", result, err)
				}
				var report struct {
					Schema  string                  `json:"schema"`
					Sitemap contracts.ExactArtifact `json:"sitemap"`
					OpenAPI contracts.ExactArtifact `json:"openapi"`
				}
				h.decode(t, h.operations.request(http.MethodGet, "/v1/runs/"+run+"/artifacts/join-api/report", nil, http.StatusOK, nil), &report)
				if report.Schema != "sitemap-openapi-comparison@1" || report.Sitemap.Ref.Namespace != "inputs" || report.OpenAPI.Ref.Namespace != "inputs" || report.OpenAPI.Validate() != nil {
					t.Fatalf("report lost exact inputs: %+v", report)
				}
			}
		})
	}
}
