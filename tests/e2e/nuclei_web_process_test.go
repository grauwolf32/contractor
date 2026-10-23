//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

// Use the shipped Workflow, Server, Scheduler and Runtime with an installed
// Nuclei engine. Only the operator package pin changes to the local fixture.
func TestNucleiWebAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Fatal("Nuclei web process acceptance cannot run in short mode")
	}
	h := startConfiguredScanStack(t, "", func(h *scanProcessHarness, root string) {
		for _, relative := range []string{"workflows/nuclei_web.yaml", "agent-templates/nuclei_web_scan.yaml", "instructions/nuclei-web.md"} {
			data, err := os.ReadFile(filepath.Join(h.repositoryRoot, "configs", relative))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, relative), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		path := filepath.Join(h.temporaryRoot, "nuclei-templates", "fixture.yaml")
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		data = bytes.Replace(data, []byte("severity: info"), []byte("severity: info\n  tags: contractor,fixture"), 1)
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
		manifest := map[string]any{"schemaVersion": 1, "files": []map[string]any{{"path": "fixture.yaml", "digest": contentdigest.Bytes(data), "size": len(data)}}}
		canonical, err := strictjson.Canonical(manifest)
		if err != nil {
			t.Fatal(err)
		}
		replaceScanConfig(t, filepath.Join(root, "agent-templates/nuclei_web_scan.yaml"), "sha256:51a3bd4eef14e95bdeb0bad14ca4bd9bd455479a7044b244b76dee8f6dba5014", contentdigest.Bytes(canonical))
	}, "nuclei")
	for _, key := range []string{"omitted", "template_ids", "tags", "severity", "rate_limit"} {
		t.Run(key, func(t *testing.T) {
			parameters := map[string]string{"target": h.targetURL}
			switch key {
			case "template_ids":
				parameters[key] = "contractor-local-fixture"
			case "tags":
				parameters[key] = "contractor"
			case "severity":
				parameters[key] = "info"
			case "rate_limit":
				parameters[key] = "1000"
			}
			id := h.createRun(t, "nuclei-web@1", parameters, nil)
			status, report := h.completedReport(t, id, "scan_nuclei")
			h.assertExecution(t, status, "scan_nuclei", "")
			manifest, media := download(t, h.client, h.baseURL+"/v1/runs/"+id+"/outputs/template_manifest")
			var provenance struct {
				PackageDigest     string `json:"packageDigest"`
				EngineVersion     string `json:"engineVersion"`
				SelectedTemplates []struct {
					ID     string `json:"id"`
					Digest string `json:"digest"`
				} `json:"selectedTemplates"`
			}
			if err := json.Unmarshal(manifest, &provenance); err != nil {
				t.Fatal(err)
			}
			if media != "application/vnd.contractor.nuclei-manifest+json" || len(provenance.SelectedTemplates) != 1 || provenance.SelectedTemplates[0].ID != "contractor-local-fixture" || !strings.HasPrefix(provenance.EngineVersion, "v") {
				t.Fatalf("incomplete provenance: %+v", provenance)
			}
			if report.Observation["expectedPackageDigest"] != provenance.PackageDigest || report.Observation["actualPackageDigest"] != provenance.PackageDigest {
				t.Fatal("report does not match package manifest")
			}
			results, ok := report.Observation["results"].([]any)
			if !ok || len(results) != 1 {
				t.Fatalf("scan did not execute local HTTP template: %+v", report.Observation)
			}
		})
	}
	t.Run("invalid_admission", func(t *testing.T) {
		for _, params := range []map[string]string{{"target": h.targetURL, "rate_limit": "0"}, {"target": h.targetURL, "rate_limit": "1001"}, {"target": h.targetURL, "rate_limit": "010"}, {"target": h.targetURL, "severity": "bogus"}, {"target": "file:///tmp/target"}} {
			body, err := json.Marshal(map[string]any{"workflow": "nuclei-web@1", "parameters": params, "artifacts": map[string]any{}})
			if err != nil {
				t.Fatal(err)
			}
			request, err := http.NewRequest(http.MethodPost, h.baseURL+"/v1/runs", bytes.NewReader(body))
			if err != nil {
				t.Fatal(err)
			}
			request.Header.Set("Authorization", "Bearer "+publicToken)
			request.Header.Set("Content-Type", "application/json")
			request.Header.Set("Idempotency-Key", "invalid-"+randomHex(t, 8))
			response := do(t, h.client, request, http.StatusBadRequest)
			response.Body.Close()
		}
	})
	t.Run("changed_package_never_scans", func(t *testing.T) {
		before := len(scanFixtureRequests(h))
		path := filepath.Join(h.temporaryRoot, "nuclei-templates", "fixture.yaml")
		stream, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
		if err != nil {
			t.Fatal(err)
		}
		_, err = stream.WriteString("\n# content changed under the same ID\n")
		if err != nil {
			t.Fatal(err)
		}
		stream.Close()
		id := h.createRun(t, "nuclei-web@1", map[string]string{"target": h.targetURL}, nil)
		h.terminal(t, id, "failed")
		if len(scanFixtureRequests(h)) != before {
			t.Fatal("mismatched package reached the target")
		}
	})
}
