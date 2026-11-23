//go:build e2e

package e2e

import (
	"archive/zip"
	"bytes"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
)

func TestArtifactFilePublicationAcrossProcesses(t *testing.T) {
	const report = "{\"message\":\"Проверено 😀\"}\r\n \t\n"
	args := fixedArguments(map[string]any{"namespace": "builder", "name": "report", "source_path": "report.json", "media_type": "application/json"})
	success := domainGatewayStage{name: "existing-file", tools: []string{"write_artifact"}, steps: []domainGatewayStep{
		toolGatewayStep("write_artifact", args),
		finalGatewayStep("Published the existing file", map[string]domainArtifactBinding{"report": {namespace: "builder", name: "report"}}),
	}}
	missing := domainGatewayStage{name: "missing-file", tools: []string{"write_artifact"}, steps: []domainGatewayStep{
		toolGatewayStep("write_artifact", args),
		{summary: "The source file is missing", validate: func(request map[string]any) error {
			responses := findingsToolResponses(request, "write_artifact")
			if len(responses) != 1 {
				return fmt.Errorf("expected one file publication error")
			}
			failure, _ := responses[0]["error"].(map[string]any)
			if failure["code"] != "tool_input_invalid" {
				return fmt.Errorf("missing source did not report its input error")
			}
			return nil
		}},
	}}
	h := startConfiguredFindingsHarness(t, []domainGatewayStage{success, missing}, func(root string) {
		for source, target := range map[string]string{"agent.yaml": "agent-templates/file_publication_fixture.yaml", "workflow.yaml": "workflows/file_publication_fixture.yaml"} {
			data, err := os.ReadFile(filepath.Join(repoRoot(t), "tests/e2e/testdata/file-publication", source))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, target), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.WriteFile(filepath.Join(root, "instructions/file-publication-fixture.md"), []byte("Publish report.json through write_artifact using only its local source_path. Do not read or transcribe the file content."), 0600); err != nil {
			t.Fatal(err)
		}
	})
	// The shipped copy executes without a model or allocation, over arbitrary bytes.
	payload := []byte("é😀\r\n \t\x00\xff")
	source := uploadProjectArtifact(t, h.client, h.baseURL, "copy-source", "text/plain", payload)
	copyRun := createWorkflowRun(t, h.client, h.baseURL, "artifact-copy@1", "server-copy", source)
	copied := waitForFindingsRun(t, h, copyRun, "succeeded")
	actual, _ := download(t, h.client, h.baseURL+"/v1/runs/"+url.PathEscape(copyRun)+"/outputs/result")
	if !bytes.Equal(actual, payload) || copied.Attempts[0].Metrics != nil || h.gateway.CompletedStages() != 0 {
		t.Fatal("Server copy used a model or changed source bytes")
	}
	pool, err := pgxpool.New(h.ctx, h.databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	var allocations int
	if err := pool.QueryRow(h.ctx, `SELECT count(*) FROM stage_allocations a JOIN stage_executions s USING (stage_execution_id) WHERE s.run_id=$1`, copyRun).Scan(&allocations); err != nil || allocations != 0 {
		t.Fatalf("copy allocations=%d: %v", allocations, err)
	}
	for _, exists := range []bool{true, false} {
		var archive bytes.Buffer
		writer := zip.NewWriter(&archive)
		name, body := "report.json", report
		if !exists {
			name, body = "other.txt", "The expected report is absent"
		}
		entry, err := writer.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = entry.Write([]byte(body)); err != nil {
			t.Fatal(err)
		}
		if err = writer.Close(); err != nil {
			t.Fatal(err)
		}
		key := fmt.Sprint("file-source-", exists)
		input := uploadProjectArtifact(t, h.client, h.baseURL, key, "application/zip", archive.Bytes())
		run := createWorkflowRun(t, h.client, h.baseURL, "file-publication-fixture@1", key, input)
		wanted := "succeeded"
		if !exists {
			wanted = "failed"
		}
		status := waitForFindingsRun(t, h, run, wanted)
		if exists {
			actual, media := download(t, h.client, h.baseURL+"/v1/runs/"+url.PathEscape(run)+"/outputs/result")
			if string(actual) != report || media != "application/json" || status.Outputs["result"].Revision == nil {
				t.Fatal("file publication changed bytes or lost its exact receipt")
			}
		} else {
			var publications int
			if err := pool.QueryRow(h.ctx, `SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind='run' AND scope_id=$1 AND namespace='builder' AND name='report'`, run).Scan(&publications); err != nil || publications != 0 {
				t.Fatalf("missing source published %d revisions: %v", publications, err)
			}
		}
	}
	if h.gateway.CompletedStages() != 2 || len(h.gateway.Failures()) != 0 {
		t.Fatal("path publication did not complete both model scripts", h.gateway.Failures())
	}
}
