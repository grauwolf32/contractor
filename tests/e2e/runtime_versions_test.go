//go:build e2e

package e2e

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestRuntimeManagedArtifactWritesAcrossProcesses(t *testing.T) {
	var h *findingsHarness
	args := func(path string) func(map[string]any) (map[string]any, error) {
		return fixedArguments(map[string]any{"namespace": "builder", "name": "report", "source_path": path, "media_type": "application/json"})
	}
	stale := toolGatewayStep("write_artifact", args("stale.json"))
	stale.validate = func(request map[string]any) error {
		results := findingsToolResponses(request, "write_artifact")
		if len(results) != 2 {
			return fmt.Errorf("write results=%v", results)
		}
		first, _ := results[0]["artifact"].(map[string]any)
		last, _ := results[1]["artifact"].(map[string]any)
		if first["revision"] == nil || last["revision"] == nil || first["revision"] == last["revision"] {
			return fmt.Errorf("missing create/update versions: %v", results)
		}
		pool, err := pgxpool.New(h.ctx, h.databaseURL)
		if err != nil {
			return err
		}
		defer pool.Close()
		var run string
		if err = pool.QueryRow(h.ctx, `SELECT scope_id FROM artifact_bindings WHERE scope_kind='run' AND namespace='builder' AND name='report'`).Scan(&run); err != nil {
			return err
		}
		scoped, err := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Run(run)
		if err != nil {
			return err
		}
		rev := last["revision"].(string)
		_, err = scoped.Write(h.ctx, contracts.ArtifactRef{Namespace: "builder", Name: "report"}, artifacts.Payload{MediaType: "application/json", Data: []byte(`{"external":true}`)}, &rev)
		return err
	}
	finish := finalGatewayStep("Created, updated and rejected a stale replacement", map[string]domainArtifactBinding{"report": {namespace: "builder", name: "report"}})
	finish.validate = func(request map[string]any) error {
		results := findingsToolResponses(request, "write_artifact")
		if len(results) != 3 {
			return fmt.Errorf("expected three writes: %v", results)
		}
		detail, _ := results[2]["error"].(map[string]any)
		if detail["code"] != "artifact_conflict" {
			return fmt.Errorf("stale replacement not rejected: %v", results[2])
		}
		return nil
	}
	stage := domainGatewayStage{name: "runtime-write-bases", tools: []string{"write_artifact"}, steps: []domainGatewayStep{toolGatewayStep("write_artifact", args("first.json")), toolGatewayStep("write_artifact", args("second.json")), stale, finish}}
	h = startConfiguredFindingsHarness(t, []domainGatewayStage{stage}, func(root string) {
		for source, target := range map[string]string{"agent.yaml": "agent-templates/file_publication_fixture.yaml", "workflow.yaml": "workflows/file_publication_fixture.yaml"} {
			data, err := os.ReadFile(filepath.Join(repoRoot(t), "tests/e2e/testdata/file-publication", source))
			if err != nil {
				t.Fatal(err)
			}
			if err = os.WriteFile(filepath.Join(root, target), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.WriteFile(filepath.Join(root, "instructions/file-publication-fixture.md"), []byte("Publish existing local files through write_artifact. Runtime manages write preconditions."), 0600); err != nil {
			t.Fatal(err)
		}
	})
	var archive bytes.Buffer
	writer := zip.NewWriter(&archive)
	for _, name := range []string{"first.json", "second.json", "stale.json"} {
		entry, err := writer.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = entry.Write([]byte(fmt.Sprintf(`{"file":%q}`, name))); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	input := uploadProjectArtifact(t, h.client, h.baseURL, "write-source", "application/zip", archive.Bytes())
	run := createWorkflowRun(t, h.client, h.baseURL, "file-publication-fixture@1", "runtime-write-base", input)
	waitForFindingsRun(t, h, run, "succeeded")
	actual, _ := download(t, h.client, h.baseURL+"/v1/runs/"+run+"/artifacts/builder/report")
	if string(actual) != `{"external":true}` {
		t.Fatalf("stale write changed external bytes: %s", actual)
	}
	if h.gateway.CompletedStages() != 1 || len(h.gateway.Failures()) != 0 {
		t.Fatal(strings.Join(h.gateway.Failures(), "; "))
	}
}

func TestRuntimeManagedSitemapEditsAcrossProcesses(t *testing.T) {
	var h *findingsHarness
	var project string
	edit := func(path string) domainGatewayStep {
		return toolGatewayStep("upsert_sitemap_entry", fixedArguments(map[string]any{"url": "https://example.test" + path, "method": "GET"}))
	}
	stale := edit("/stale")
	stale.validate = func(request map[string]any) error {
		pages := findingsToolResponses(request, "list_sitemap_entries")
		if len(pages) != 2 {
			return fmt.Errorf("reviewed pages=%v", pages)
		}
		body, err := json.Marshal(map[string]any{"operation": "declare", "expectedRevision": pages[1]["revision"], "url": "https://example.test/external", "method": "GET"})
		if err != nil {
			return err
		}
		req, err := http.NewRequest(http.MethodPost, h.baseURL+"/v1/projects/"+project+"/sitemap/mutations", bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Authorization", "Bearer "+publicToken)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Idempotency-Key", "external-edit")
		response, err := h.client.Do(req)
		if err != nil {
			return err
		}
		defer response.Body.Close()
		if response.StatusCode != 200 {
			return fmt.Errorf("external edit status=%d", response.StatusCode)
		}
		return nil
	}
	final := finalGatewayStep("A reviewed edit succeeded; the stale edit conflicted", nil)
	final.validate = func(request map[string]any) error {
		responses := findingsToolResponses(request, "upsert_sitemap_entry")
		if len(responses) != 2 {
			return fmt.Errorf("edit results=%v", responses)
		}
		saved, _ := responses[0]["artifact"].(map[string]any)
		detail, _ := responses[1]["error"].(map[string]any)
		if saved["revision"] == nil || detail["code"] != "sitemap_conflict" {
			return fmt.Errorf("incorrect reviewed edits: %v", responses)
		}
		return nil
	}
	stage := domainGatewayStage{name: "reviewed-sitemap", tools: []string{"list_sitemap_entries", "upsert_sitemap_entry"}, steps: []domainGatewayStep{toolGatewayStep("list_sitemap_entries", fixedArguments(map[string]any{})), edit("/selected"), toolGatewayStep("list_sitemap_entries", fixedArguments(map[string]any{"refresh": true})), stale, final}}
	h = startConfiguredFindingsHarness(t, []domainGatewayStage{stage}, func(root string) {
		files := map[string]string{
			"agent-templates/version_sitemap.yaml": `apiVersion: contractor/v1alpha1
kind: AgentTemplate
metadata: {name: version_sitemap, version: "1"}
spec:
  description: Edit a reviewed Project map
  runtime: adk@1
  instructions: {ref: instructions/version-sitemap.md}
  sitemap: {version: sitemap-access@1, collectObservations: false}
  toolsets: [{ref: sitemap@1, tools: [list_sitemap_entries, upsert_sitemap_entry]}]
  sandboxProfile: local-workdir@1
`,
			"workflows/version_sitemap.yaml": `apiVersion: contractor/v1alpha1
kind: Workflow
metadata: {name: version-sitemap, version: "1"}
spec:
  parameters: {}
  inputs: {}
  outputs: {}
  executionConfig:
    workers: {llmGateway: local-litellm@1, credential: development-worker, modelPolicy: worker@2}
  entryStage: check
  stages:
    check:
      objective: Edit a reviewed Project map
      instructions: {ref: instructions/report-planner.md}
      planner: passthrough@1
      agents: {research: {template: version_sitemap@1}}
      context: {artifacts: {}}
      result: {artifacts: {}}
      workflowOutputs: {}
      on:
        succeeded: {succeed: {}}
        failed: {fail: {}}
        interrupted: {fail: {}}
`,
			"instructions/version-sitemap.md": "Read the map before editing; Runtime supplies write preconditions. Refresh after a change.",
		}
		for name, body := range files {
			if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0600); err != nil {
				t.Fatal(err)
			}
		}
	})
	project = createProjectResource(t, h.client, h.baseURL).ProjectID
	run := postProjectRun(t, h.client, h.baseURL, project, "reviewed-edit", []byte(`{"workflow":"version-sitemap@1","parameters":{},"artifacts":{}}`), false)
	waitForFindingsRun(t, h, run, "succeeded")
	actual, _ := download(t, h.client, h.baseURL+"/v1/projects/"+project+"/artifacts/sitemap/map")
	if !bytes.Contains(actual, []byte("/selected")) || !bytes.Contains(actual, []byte("/external")) || bytes.Contains(actual, []byte("/stale")) {
		t.Fatalf("wrong current map: %s", actual)
	}
	if h.gateway.CompletedStages() != 1 || len(h.gateway.Failures()) != 0 {
		t.Fatal(h.gateway.Failures())
	}
}

func TestRuntimeManagedMaterialVersionsAcrossProcesses(t *testing.T) {
	var h *findingsHarness
	const originalTail = "old-tail\n"
	original := strings.Repeat("old-line\n", 200) + originalTail
	const newer = "new-head\nnew-tail\n"
	first := toolGatewayStep("read_artifact", fixedArguments(map[string]any{"namespace": "inputs", "name": "source"}))
	first.validate = func(request map[string]any) error {
		ref, ok := stageArtifact(request, "source")
		if !ok {
			return fmt.Errorf("missing pinned source")
		}
		pool, err := pgxpool.New(h.ctx, h.databaseURL)
		if err != nil {
			return err
		}
		defer pool.Close()
		var run string
		if err = pool.QueryRow(h.ctx, `SELECT scope_id FROM artifact_bindings WHERE scope_kind='run' AND namespace='inputs' AND name='source'`).Scan(&run); err != nil {
			return err
		}
		scoped, err := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Run(run)
		if err != nil {
			return err
		}
		rev := ref["revision"].(string)
		_, err = scoped.Write(h.ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte(newer)}, &rev)
		return err
	}
	next := toolGatewayStep("read_artifact", func(request map[string]any) (map[string]any, error) {
		responses := findingsToolResponses(request, "read_artifact")
		if len(responses) != 2 {
			return nil, fmt.Errorf("read responses=%v", responses)
		}
		return map[string]any{"cursor": responses[1]["nextCursor"]}, nil
	})
	final := finalGatewayStep("Current reads and exact continuation preserve distinct versions", nil)
	final.validate = func(request map[string]any) error {
		results := findingsToolResponses(request, "read_artifact")
		if len(results) != 4 {
			return fmt.Errorf("expected four reads: %v", results)
		}
		fresh, _ := results[0]["artifact"].(map[string]any)
		pinned, _ := results[1]["artifact"].(map[string]any)
		continued, _ := results[2]["artifact"].(map[string]any)
		if fresh["revision"] == pinned["revision"] || pinned["revision"] != continued["revision"] || results[0]["text"] != newer || results[2]["text"] != originalTail || results[3]["text"] != newer {
			return fmt.Errorf("mixed current/pinned versions: %v", results)
		}
		return nil
	}
	stage := domainGatewayStage{name: "current-and-pinned-material", tools: []string{"read_artifact"}, steps: []domainGatewayStep{first, toolGatewayStep("read_artifact", stageReadArguments("source", nil)), next, toolGatewayStep("read_artifact", fixedArguments(map[string]any{"namespace": "inputs", "name": "source"})), final}}
	h = startConfiguredFindingsHarness(t, []domainGatewayStage{stage}, func(root string) {
		workflow := strings.ReplaceAll(imageFixtureWorkflow, "image/png", "text/plain")
		files := map[string]string{"agent-templates/artifact-image-fixture.yaml": imageFixtureTemplate, "workflows/artifact-image-fixture.yaml": workflow, "model-policies/image-fixture.yaml": imageFixturePolicy, "instructions/artifact-image-fixture.md": "Read current by namespace/name; use task readCursor and nextCursor for exact sources.", "instructions/artifact-image-planner-fixture.md": "Inspect selected input."}
		for name, data := range files {
			if err := os.WriteFile(filepath.Join(root, name), []byte(data), 0600); err != nil {
				t.Fatal(err)
			}
		}
	})
	input := uploadProjectArtifact(t, h.client, h.baseURL, "material-source", "text/plain", []byte(original))
	run := createWorkflowRun(t, h.client, h.baseURL, "artifact-image-fixture@1", "current-pinned", input)
	waitForFindingsRun(t, h, run, "succeeded")
	if h.gateway.CompletedStages() != 1 || len(h.gateway.Failures()) != 0 {
		t.Fatal(h.gateway.Failures())
	}
}
