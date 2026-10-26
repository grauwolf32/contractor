//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"testing"
)

// Only the inputs are scripted: the Server, model-free Runtime, mTLS, catalog,
// allocation grants, Run completion and ArtifactStore all run in real processes.
func TestProjectSitemapToolsAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Fatal("the sitemap process gate cannot run in short mode")
	}
	if os.Getenv("CONTRACTOR_TEST_DATABASE_URL") == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	root := repoRoot(t)
	h := startPlacementStackWithConfiguration(t, func(target string) {
		for _, directory := range []string{"agent-templates", "workflows"} {
			entries, err := os.ReadDir(filepath.Join(root, "tests/e2e/testdata/sitemap", directory))
			if err != nil {
				t.Fatal(err)
			}
			for _, entry := range entries {
				data, err := os.ReadFile(filepath.Join(root, "tests/e2e/testdata/sitemap", directory, entry.Name()))
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(target, directory, entry.Name()), data, 0600); err != nil {
					t.Fatal(err)
				}
			}
		}
	})
	runtime := h.startRuntime("sitemap-runtime", "", false)
	project := createProjectResource(t, h.operations.client, h.operations.baseURL)
	var foreign projectResourceResponse
	h.decode(t, h.operations.request(http.MethodPost, "/v1/projects", map[string]any{"kind": "project", "name": "Foreign sitemap Project", "description": "Isolated sitemap fixture"}, http.StatusCreated, map[string]string{"Idempotency-Key": "foreign-project"}), &foreign)
	if foreign.ProjectID == project.ProjectID || foreign.ProjectID == "" {
		t.Fatal("foreign Project fixture is not distinct")
	}
	mutate := func(id, revision, path, key string) string {
		t.Helper()
		data := h.operations.request(http.MethodPost, "/v1/projects/"+id+"/sitemap/mutations", map[string]any{"operation": "declare", "expectedRevision": revision, "url": "https://example.test" + path, "method": "GET"}, 200, map[string]string{"Idempotency-Key": key})
		var result struct {
			Artifact artifactRef `json:"artifact"`
		}
		h.decode(t, data, &result)
		if result.Artifact.Revision == nil {
			t.Fatal("owner mutation returned no exact ref")
		}
		return *result.Artifact.Revision
	}
	original := mutate(project.ProjectID, "absent", "/source-assisted-canary", "seed")
	mutate(foreign.ProjectID, "absent", "/foreign-project-canary", "foreign-seed")
	create := func(workflow string, parameters map[string]string) string {
		t.Helper()
		body, err := json.Marshal(map[string]any{"workflow": workflow + "@1", "parameters": parameters, "artifacts": map[string]any{}})
		if err != nil {
			t.Fatal(err)
		}
		return postProjectRun(t, h.operations.client, h.operations.baseURL, project.ProjectID, "sitemap-"+randomHex(t, 8), body, false)
	}
	edited := create("sitemap-blackbox-edit", map[string]string{"revision": original, "url": "https://example.test/declared-by-worker"})
	status := h.terminal(t, edited, "succeeded")
	assertSitemapToolReport(t, h, status, "upsert_sitemap_entry", "")
	listed := create("sitemap-blackbox-list", map[string]string{})
	status = h.terminal(t, listed, "succeeded")
	report := assertSitemapToolReport(t, h, status, "list_sitemap_entries", "")
	if bytes.Contains(report, []byte("source-assisted-canary")) || bytes.Contains(report, []byte("foreign-project-canary")) || !bytes.Contains(report, []byte("declared-by-worker")) {
		t.Fatalf("blackbox process projection leaked or lost content: %s", report)
	}
	full := create("sitemap-source-assisted-list", map[string]string{})
	status = h.terminal(t, full, "succeeded")
	report = assertSitemapToolReport(t, h, status, "list_sitemap_entries", "")
	if !bytes.Contains(report, []byte("source-assisted-canary")) || bytes.Contains(report, []byte("foreign-project-canary")) {
		t.Fatal("trusted source mode or Project scope was not preserved")
	}
	stale := create("sitemap-blackbox-edit", map[string]string{"revision": original, "url": "https://example.test/must-not-appear"})
	status = h.terminal(t, stale, "failed")
	assertSitemapToolReport(t, h, status, "upsert_sitemap_entry", "sitemap_conflict")
	current := h.operations.request(http.MethodGet, "/v1/projects/"+project.ProjectID+"/sitemap", nil, 200, nil)
	if bytes.Contains(current, []byte("must-not-appear")) {
		t.Fatal("stale editor intent was refreshed or replayed")
	}
	// Finalization removes private authority. A fresh direct private request by
	// that Runtime certificate cannot use a settled allocation to read a map.
	allocation := onlyRunAllocation(t, h.ctx, h.store, listed)
	runtimeClient := newMTLSClient(t, h.caFile, h.identities["sitemap-runtime"])
	request, err := http.NewRequest(http.MethodGet, h.privateURL+"/private/v1/allocations/"+allocation.AllocationID+"/sitemap", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("X-Contractor-Runtime-Instance-ID", allocation.RuntimeAgentInstanceID)
	do(t, runtimeClient, request, http.StatusNotFound).Body.Close()
	if exited, err := runtime.exited(); exited {
		t.Fatalf("Runtime exited: %v", err)
	}
}

func assertSitemapToolReport(t *testing.T, h *placementProcessHarness, status runStatus, tool, errorCode string) []byte {
	t.Helper()
	var ref artifactRef
	if output, ok := status.Outputs["report"]; ok {
		ref = output
	} else {
		// Failed tool calls retain their ordinary report in their Worker namespace.
		attempts, err := h.store.ListStageExecutions(h.ctx, status.RunID)
		if err != nil || len(attempts) != 1 {
			t.Fatalf("failed tool execution: %v", err)
		}
		data := h.operations.request(http.MethodGet, "/v1/runs/"+status.RunID+"/artifacts/researcher/report", nil, 200, nil)
		var envelope struct {
			Tool        string         `json:"tool"`
			Observation map[string]any `json:"observation"`
		}
		if err := json.Unmarshal(data, &envelope); err != nil {
			t.Fatal(err)
		}
		if envelope.Tool != tool || envelope.Observation["status"] != "failed" {
			t.Fatalf("failed tool report: %s", data)
		}
		encoded, _ := json.Marshal(envelope.Observation["error"])
		if !bytes.Contains(encoded, []byte(errorCode)) {
			t.Fatalf("failure lost classified code: %s", data)
		}
		return data
	}
	data, _ := download(t, h.operations.client, h.operations.baseURL+"/v1/runs/"+status.RunID+"/artifacts/"+ref.Namespace+"/"+ref.Name+"?revision="+*ref.Revision)
	var report struct {
		Tool        string         `json:"tool"`
		Observation map[string]any `json:"observation"`
	}
	if err := json.Unmarshal(data, &report); err != nil || report.Tool != tool || report.Observation["status"] != "completed" {
		t.Fatalf("tool report: %s: %v", data, err)
	}
	if len(status.Attempts) != 1 || status.Attempts[0].Metrics == nil || status.Attempts[0].Metrics.ModelCalls != 0 {
		t.Fatal("sitemap process test unexpectedly used a model")
	}
	return data
}
