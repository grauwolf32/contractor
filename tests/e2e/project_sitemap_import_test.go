//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/projectsitemap"
)

// Reports are selected ordinary inputs. The real tool Worker, allocation
// authority, exact Artifact reads and Project mutation run without a scanner or
// model. A malformed report must fail its own import without damaging the map.
func TestProjectSitemapReportImportsAcrossProductionProcesses(t *testing.T) {
	if testing.Short() || os.Getenv("CONTRACTOR_TEST_DATABASE_URL") == "" {
		t.Fatal("sitemap import process gate requires PostgreSQL and a full test run")
	}
	root := repoRoot(t)
	h := startPlacementStackWithConfiguration(t, func(target string) {
		for _, directory := range []string{"agent-templates", "workflows"} {
			data, err := os.ReadFile(filepath.Join(root, "tests/e2e/testdata/sitemap", directory, "sitemap-importer.yaml"))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(target, directory, "sitemap-importer.yaml"), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
	})
	h.startRuntime("sitemap-importer-runtime", "", false)
	project := createProjectResource(t, h.operations.client, h.operations.baseURL)
	importReport := func(name, format string, payload []byte, want string) string {
		t.Helper()
		ref := uploadProjectScopeArtifact(t, h.operations.client, h.operations.baseURL, project.ProjectID, "reports", name, "application/json", payload)
		body, err := json.Marshal(map[string]any{"workflow": "sitemap-importer@1", "parameters": map[string]string{"format": format}, "artifacts": map[string]any{"source": ref}})
		if err != nil {
			t.Fatal(err)
		}
		run := postProjectRun(t, h.operations.client, h.operations.baseURL, project.ProjectID, name, body, false)
		status := h.terminal(t, run, want)
		code := ""
		if want == "failed" {
			code = "invalid_request"
		}
		assertSitemapToolReport(t, h, status, "import_sitemap_source", code)
		return run
	}
	runs := map[string]bool{}
	for _, kind := range []string{"ffuf", "katana"} {
		payload, err := os.ReadFile(filepath.Join(root, "tests/e2e/testdata/sitemap/reports", kind+"-report.json"))
		if err != nil {
			t.Fatal(err)
		}
		runs[importReport(kind, kind+"-report@1", payload, "succeeded")] = true
		if kind == "ffuf" {
			// Another Run observing the same behavior must not fragment its summary.
			runs[importReport("ffuf-again", "ffuf-report@1", payload, "succeeded")] = true
		}
	}
	readMap := func() []byte {
		return h.operations.request(http.MethodGet, "/v1/projects/"+project.ProjectID+"/artifacts/sitemap/map", nil, http.StatusOK, nil)
	}
	before := readMap()
	document, err := projectsitemap.Decode(before)
	if err != nil || len(document.Groups) != 1 || len(document.Groups[0].Members) != 1 {
		t.Fatalf("imported map: %s: %v", before, err)
	}
	member := document.Groups[0].Members[0]
	if member.Path != "/users" || len(member.Summaries) != 2 || len(member.Evidence) != 3 {
		t.Fatalf("report aggregation/evidence: %+v", member)
	}
	kinds := map[string]bool{}
	for _, summary := range member.Summaries {
		kinds[summary.Provenance.Kind] = true
		if summary.Outcome != "response" || summary.Status != 200 || summary.First == "" || summary.Last == "" || len(summary.Parameters) != 1 || summary.Parameters[0] != "expand" {
			t.Fatalf("report facts: %+v", summary)
		}
	}
	if !kinds["ffuf"] || !kinds["katana"] || bytes.Contains(before, []byte("expand=profile")) {
		t.Fatalf("source kinds or query projection: %s", before)
	}
	for _, evidence := range member.Evidence {
		if !runs[evidence.RunID] || evidence.Artifact.Ref.Namespace != "inputs" || evidence.Artifact.Ref.Name != "source" || evidence.Artifact.Ref.Revision == nil {
			t.Fatalf("evidence is not the selected exact Run input: %+v", evidence)
		}
	}
	importReport("malformed", "ffuf-report@1", []byte(`{"schemaVersion":1,"tool":"scan_ffuf","unexpected":true}`), "failed")
	if after := readMap(); !bytes.Equal(before, after) {
		t.Fatalf("failed import changed accepted map: %s", after)
	}
}
