//go:build e2e

package e2e

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

// The real Server/Scheduler/ArtifactStore executes this with no Runtime or model.
func TestOpenAPIPreparedWorkServerOperation(t *testing.T) {
	if testing.Short() || os.Getenv("CONTRACTOR_TEST_DATABASE_URL") == "" {
		t.Fatal("OpenAPI operation process gate requires PostgreSQL and a full run")
	}
	root := repoRoot(t)
	h := startPlacementStackWithConfiguration(t, func(target string) {
		data, err := os.ReadFile(filepath.Join(root, "internal/config/testdata/openapi-prepared-work/workflows/prepare.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(target, "workflows", "openapi_prepare.yaml"), data, 0600); err != nil {
			t.Fatal(err)
		}
	})
	for _, count := range []int{256, 257} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			var project projectResourceResponse
			h.decode(t, h.operations.request(http.MethodPost, "/v1/projects", map[string]any{"kind": "project", "name": "Prepared work", "description": "No Runtime operation fixture"}, http.StatusCreated, map[string]string{"Idempotency-Key": fmt.Sprint("prepared-", count)}), &project)
			paths := map[string]any{}
			for i := 0; i < count; i++ {
				paths[fmt.Sprintf("/items-%03d", i)] = map[string]any{"get": map[string]any{}}
			}
			data, _ := json.Marshal(map[string]any{"openapi": "3.1.0", "paths": paths})
			source := uploadProjectScopeArtifact(t, h.operations.client, h.operations.baseURL, project.ProjectID, "contracts", "api", "application/json", data)
			body, _ := json.Marshal(map[string]any{"workflow": "openapi-prepare-work@1", "parameters": map[string]string{"checking_workflow": "check", "selection": "operations", "max_items": "256"}, "artifacts": map[string]any{"openapi": source}})
			run := postProjectRun(t, h.operations.client, h.operations.baseURL, project.ProjectID, fmt.Sprint("prepare-", count), body, false)
			want := "succeeded"
			if count > 256 {
				want = "failed"
			}
			status := h.terminal(t, run, want)
			if len(status.Attempts) != 1 || status.Attempts[0].PlannerStartedAt != nil || len(status.Attempts[0].Resources) != 0 {
				t.Fatal("preparation used a Planner/Runtime")
			}
			stages, err := h.store.ListStageExecutions(h.ctx, run)
			if err != nil || len(stages) != 1 || stages[0].PlannerSessionID != nil {
				t.Fatalf("synthetic Planner: %v", err)
			}
			allocations, err := h.store.ListStageAllocations(h.ctx, stages[0].StageExecutionID)
			if err != nil || len(allocations) != 0 {
				t.Fatal("allocated a Runtime")
			}
			if count > 256 {
				h.operations.request(http.MethodGet, "/v1/runs/"+run+"/artifacts/outputs/work", nil, http.StatusNotFound, nil)
				return
			}
			payload := h.operations.request(http.MethodGet, "/v1/runs/"+run+"/artifacts/outputs/work", nil, http.StatusOK, nil)
			if _, err := auditcontract.Decode(payload, nil); err != nil {
				t.Fatal(err)
			}
			var list auditcontract.PreparedWork
			h.decode(t, payload, &list)
			if len(list.Work) != 256 {
				t.Fatalf("work=%d", len(list.Work))
			}
			seen := map[string]bool{}
			for _, item := range list.Work {
				if seen[item.Key] || item.Workflow != "check" {
					t.Fatal("lost independent work identity")
				}
				seen[item.Key] = true
				selected := item.Inputs["operation"]
				if selected.ScopeID != run || selected.Ref.Revision == nil {
					t.Fatal("unowned/inexact prepared selection")
				}
			}
			selected := list.Work[0].Inputs["operation"]
			item := h.operations.request(http.MethodGet, "/v1/runs/"+run+"/artifacts/"+selected.Ref.Namespace+"/"+selected.Ref.Name+"?revision="+*selected.Ref.Revision, nil, http.StatusOK, nil)
			if !json.Valid(item) {
				t.Fatal("prepared material is not ordinary JSON")
			}
		})
	}
}
