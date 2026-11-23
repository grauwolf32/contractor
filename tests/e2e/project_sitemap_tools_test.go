//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/projectsitemap"
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
				if directory == "workflows" && entry.Name() == "sitemap-reader.yaml" {
					text := strings.Replace(string(data), "inputs: {}", "inputs:\n    raw_map: {required: false, mediaTypes: [application/vnd.contractor.project-sitemap+json]}", 1)
					text = strings.Replace(text, "context: {artifacts: {}}", "context: {artifacts: {raw_map: {namespace: inputs, name: raw_map, required: false}}}", 1)
					data = []byte(text)
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
	original := mutate(project.ProjectID, "absent", "/manual-canary", "seed")
	mutate(foreign.ProjectID, "absent", "/foreign-project-canary", "foreign-seed")
	create := func(workflow string, parameters map[string]string, selected ...map[string]any) string {
		t.Helper()
		inputs := map[string]any{}
		if len(selected) > 0 {
			inputs = selected[0]
		}
		body, err := json.Marshal(map[string]any{"workflow": workflow + "@1", "parameters": parameters, "artifacts": inputs})
		if err != nil {
			t.Fatal(err)
		}
		return postProjectRun(t, h.operations.client, h.operations.baseURL, project.ProjectID, "sitemap-"+randomHex(t, 8), body, false)
	}
	edited := create("sitemap-editor", map[string]string{"revision": original, "url": "https://example.test/session/short-password"})
	status := h.terminal(t, edited, "succeeded")
	assertSitemapToolReport(t, h, status, "upsert_sitemap_entry", "")
	listed := create("sitemap-reader", map[string]string{})
	status = h.terminal(t, listed, "succeeded")
	report := assertSitemapToolReport(t, h, status, "list_sitemap_entries", "")
	if !bytes.Contains(report, []byte("manual-canary")) || bytes.Contains(report, []byte("foreign-project-canary")) || !bytes.Contains(report, []byte("/session/short-password")) {
		t.Fatalf("unified process reader leaked or lost content: %s", report)
	}
	// Owner-selected ordinary inputs are explicit context authority. They
	// retain the complete selected map, independently of later publications.
	selected := create("sitemap-reader", map[string]string{}, map[string]any{"raw_map": artifactRef{Namespace: "sitemap", Name: "map", Revision: &original}})
	status = h.terminal(t, selected, "succeeded")
	report = assertSitemapToolReport(t, h, status, "list_sitemap_entries", "")
	if !bytes.Contains(report, []byte("manual-canary")) {
		t.Fatal("ordinary input selection hid shared sitemap metadata")
	}
	selectedBytes := h.operations.request(http.MethodGet, "/v1/runs/"+selected+"/artifacts/inputs/raw_map", nil, 200, nil)
	if !bytes.Contains(selectedBytes, []byte("manual-canary")) || bytes.Contains(selectedBytes, []byte("foreign-project-canary")) {
		t.Fatal("explicit owner sitemap input lost its selected complete Project context")
	}
	assertEmptySitemapMetadataAcrossProcesses(t, h)
	stale := create("sitemap-editor", map[string]string{"revision": original, "url": "https://example.test/must-not-appear"})
	status = h.terminal(t, stale, "failed")
	assertSitemapToolReport(t, h, status, "upsert_sitemap_entry", "sitemap_conflict")
	current := h.operations.request(http.MethodGet, "/v1/projects/"+project.ProjectID+"/artifacts/sitemap/map", nil, 200, nil)
	if bytes.Contains(current, []byte("must-not-appear")) {
		t.Fatal("stale editor intent was refreshed or replayed")
	}
	// Finalization removes private authority. A fresh direct private request by
	// that Runtime certificate cannot use a settled allocation to read a map.
	allocation := onlyRunAllocation(t, h.ctx, h.store, listed)
	runtimeClient := newMTLSClient(t, h.caFile, h.identities["sitemap-runtime"])
	request, err := http.NewRequest(http.MethodGet, h.privateURL+"/private/v1/allocations/"+allocation.AllocationID+"/artifacts/sitemap/map?scope=project&operation=list_sitemap_entries", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("X-Contractor-Runtime-Instance-ID", allocation.RuntimeAgentInstanceID)
	do(t, runtimeClient, request, http.StatusNotFound).Body.Close()
	if exited, err := runtime.exited(); exited {
		t.Fatalf("Runtime exited: %v", err)
	}
}

func assertEmptySitemapMetadataAcrossProcesses(t *testing.T, h *placementProcessHarness) {
	t.Helper()
	var project projectResourceResponse
	h.decode(t, h.operations.request(http.MethodPost, "/v1/projects", map[string]any{
		"kind": "project", "name": "Empty sitemap metadata", "description": "Retained source precondition fixture",
	}, http.StatusCreated, map[string]string{"Idempotency-Key": "empty-metadata-project"}), &project)
	path := "/v1/projects/" + project.ProjectID + "/sitemap"
	replace := func(revision string, expected *projectsitemap.Source, rules []projectsitemap.EditorRule, key string) projectsitemap.Result {
		t.Helper()
		var result projectsitemap.Result
		h.decode(t, h.operations.request(http.MethodPost, path+"/mutations", projectsitemap.EditorMutation{
			Operation: "replace-source", ExpectedRevision: revision,
			Source: &projectsitemap.EditorSourceUpdate{Expected: expected, Rules: rules},
		}, http.StatusOK, map[string]string{"Idempotency-Key": key}), &result)
		if result.Ref.Revision == nil {
			t.Fatal("source replacement returned no exact revision")
		}
		return result
	}
	read := func(revision string) projectsitemap.Document {
		t.Helper()
		var document projectsitemap.Document
		endpoint := "/v1/projects/" + project.ProjectID + "/artifacts/sitemap/map?revision=" + revision
		h.decode(t, h.operations.request(http.MethodGet, endpoint, nil, http.StatusOK, nil), &document)
		return document
	}
	rule := projectsitemap.EditorRule{Origin: "https://example.test", Method: "GET", Path: "/before-clear", Parameters: []string{}, Structural: []projectsitemap.StructuralParameter{}}
	first := replace("absent", nil, []projectsitemap.EditorRule{rule}, "metadata-source-create")
	view := read(*first.Ref.Revision)
	if len(view.Sources) != 1 || view.Sources[0].Generation != 1 || len(view.Groups) != 1 {
		t.Fatalf("initial grouping source: %+v", view)
	}
	cleared := replace(*first.Ref.Revision, &view.Sources[0], []projectsitemap.EditorRule{}, "metadata-source-clear")
	view = read(*cleared.Ref.Revision)
	if len(view.Groups) != 0 || len(view.Rules) != 0 || len(view.Sources) != 1 || view.Sources[0].Generation != 2 {
		t.Fatalf("cleared source did not retain its generation in an empty map: %+v", view)
	}
	input := []byte(`{"workflow":"sitemap-metadata@1","parameters":{},"artifacts":{}}`)
	runID := postProjectRun(t, h.operations.client, h.operations.baseURL, project.ProjectID, "sitemap-metadata-"+randomHex(t, 8), input, false)
	status := h.terminal(t, runID, "succeeded")
	report := assertSitemapToolReport(t, h, status, "list_sitemap_entries", "")
	var result struct {
		Observation struct {
			Revision string `json:"revision"`
			Entries  []struct {
				Kind  string          `json:"kind"`
				Value json.RawMessage `json:"value"`
			} `json:"entries"`
			NextCursor *string `json:"nextCursor"`
		} `json:"observation"`
	}
	h.decode(t, report, &result)
	if result.Observation.Revision != *cleared.Ref.Revision || result.Observation.NextCursor != nil || len(result.Observation.Entries) != 1 || result.Observation.Entries[0].Kind != "source" {
		t.Fatalf("empty-map metadata report: %s", report)
	}
	var expected projectsitemap.Source
	h.decode(t, result.Observation.Entries[0].Value, &expected)
	if !reflect.DeepEqual(expected, view.Sources[0]) {
		t.Fatalf("Runtime metadata lost the complete source precondition: %+v", expected)
	}
	// Reuse the exact Source received from the real Runtime, without refreshing
	// its generation or policy digest from a direct store read.
	rule.Path = "/after-metadata-read"
	reconfigured := replace(result.Observation.Revision, &expected, []projectsitemap.EditorRule{rule}, "metadata-source-reconfigure")
	view = read(*reconfigured.Ref.Revision)
	if len(view.Sources) != 1 || view.Sources[0].Generation != 3 || len(view.Groups) != 1 || view.Groups[0].Route != rule.Path {
		t.Fatalf("metadata source precondition was not usable for reconfiguration: %+v", view)
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
