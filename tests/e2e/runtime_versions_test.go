//go:build e2e

package e2e

import (
	"archive/zip"
	"bytes"
	"fmt"
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
