//go:build e2e

package e2e

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSemgrepFindingsAcrossProductionProcesses(t *testing.T) {
	h := startConfiguredScanStackWithBudget(t, 4*time.Minute, "", func(h *scanProcessHarness, root string) {
		for _, relative := range []string{
			"agent-templates/semgrep_scan.yaml", "workflows/project_semgrep_scan.yaml",
			"instructions/scanner-findings.md",
		} {
			data, err := os.ReadFile(filepath.Join(h.repositoryRoot, "configs", relative))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, relative), data, 0o600); err != nil {
				t.Fatal(err)
			}
		}
	}, "semgrep")
	project := createProjectResource(t, h.client, h.baseURL)
	rule := []byte("rules:\n- id: python-eval\n  languages: [python]\n  severity: ERROR\n  message: Dynamic evaluation of code\n  pattern: eval(...)\n")
	rules := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "scanner-inputs", "rules", "application/yaml", rule)
	code := func(text, name string) artifactRef {
		t.Helper()
		var raw bytes.Buffer
		z := zip.NewWriter(&raw)
		file, err := z.Create("src/app.py")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := file.Write([]byte(text)); err != nil {
			t.Fatal(err)
		}
		if err := z.Close(); err != nil {
			t.Fatal(err)
		}
		return uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "scanner-inputs", name, "application/zip", raw.Bytes())
	}
	vulnerable := code("value = input()\neval(value)\n", "vulnerable")
	run := func(key string, source, pack artifactRef, state string) string {
		t.Helper()
		body, err := json.Marshal(map[string]any{"workflow": "project-semgrep-scan@1", "parameters": map[string]string{}, "artifacts": map[string]artifactRef{"source": source, "rulepack": pack}})
		if err != nil {
			t.Fatal(err)
		}
		id := postProjectRun(t, h.client, h.baseURL, project.ProjectID, key, body, false)
		h.terminal(t, id, state)
		return id
	}
	first := run("semgrep-positive", vulnerable, rules, "succeeded")
	page := scannerFindingPage(t, h, project.ProjectID)
	if len(page) != 1 || string(page[0]["analystVerdict"]) != "null" || string(page[0]["state"]) != "null" {
		t.Fatalf("Semgrep did not publish one unassessed Finding: %s", page)
	}
	candidates, _ := download(t, h.client, h.baseURL+"/v1/runs/"+first+"/artifacts/scanner/candidates")
	var batch struct {
		Candidates []struct {
			Producer string `json:"producer"`
			Location struct {
				File string `json:"file"`
			} `json:"location"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(candidates, &batch); err != nil || len(batch.Candidates) != 1 || batch.Candidates[0].Producer != "semgrep" || batch.Candidates[0].Location.File != "src/app.py" {
		t.Fatalf("source-neutral candidate/provenance invalid: %s %v", candidates, err)
	}
	evidence, _ := download(t, h.client, h.baseURL+"/v1/runs/"+first+"/artifacts/scanner/evidence")
	if !bytes.Contains(evidence, []byte("eval(value)")) || !bytes.Contains(evidence, []byte("sourceDigest")) || !bytes.Contains(evidence, []byte("rulesDigest")) {
		t.Fatalf("selected code evidence missing: %s", evidence)
	}
	// The common producer fingerprint groups repeated actual scans while their
	// individual sources and invocation receipts remain distinct.
	run("semgrep-repeat", vulnerable, rules, "succeeded")
	if page = scannerFindingPage(t, h, project.ProjectID); len(page) != 1 {
		t.Fatalf("repeated scan duplicated Finding: %s", page)
	}
	run("semgrep-safe", code("print('hello')\n", "safe"), rules, "succeeded")
	if page = scannerFindingPage(t, h, project.ProjectID); len(page) != 1 {
		t.Fatalf("safe code manufactured Findings: %s", page)
	}
	broken := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "scanner-inputs", "broken-rules", "application/yaml", []byte("rules: []\n"))
	failed := run("semgrep-invalid-pack", vulnerable, broken, "failed")
	report, _ := download(t, h.client, h.baseURL+"/v1/runs/"+failed+"/artifacts/scanner/report")
	if !strings.Contains(string(report), "semgrep_rulepack_invalid") {
		t.Fatalf("invalid pack did not fail explicitly: %s", report)
	}
	if page = scannerFindingPage(t, h, project.ProjectID); len(page) != 1 {
		t.Fatalf("invalid pack manufactured Findings: %s", page)
	}
}
