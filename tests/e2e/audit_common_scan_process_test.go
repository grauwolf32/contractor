//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/planner"
)

// Real scanner binaries, real Runtime adapters, public API and PostgreSQL.
// Only the installed Nuclei package selection is pinned to the loopback fixture.
func TestAuditCommonScanProfilesAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Fatal("common scan process gate cannot run in short mode")
	}
	h := startConfiguredScanStack(t, "", func(h *scanProcessHarness, root string) {
		stageCommonScanAuditCatalog(t, h, root)
	}, "nuclei", "sqlmap")
	for _, scanner := range []string{"sqlmap", "nuclei", "web"} {
		t.Run(scanner, func(t *testing.T) {
			var project projectResourceResponse
			scanAuditPOST(t, h, "/v1/projects", map[string]string{"kind": "project", "name": "Common scan fixture", "activeChecks": "automatic"}, 0, http.StatusCreated, &project)
			profile := "openapi-" + scanner + "-scan"
			inputs := map[string]artifactRef{}
			expectedTasks, expectedRuns := 2, 2
			if scanner == "web" {
				profile = "web-scan"
				expectedTasks, expectedRuns = 1, 1
				target, _ := json.Marshal(map[string]any{"schema": "contractor.audit.web-target.v1", "url": h.targetURL})
				settings, _ := json.Marshal(map[string]any{"schema": "contractor.audit.web-scan-settings.v1", "templateIds": "contractor-local-fixture", "tags": "", "severity": "info", "rateLimit": 5})
				inputs["target"] = uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "target", "application/json", target)
				inputs["settings"] = uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "settings", "application/json", settings)
			} else {
				root := filepath.Join(h.repositoryRoot, "configs/scan/examples/audit-openapi-scan")
				source, err := os.ReadFile(filepath.Join(root, "openapi.json"))
				if err != nil {
					t.Fatal(err)
				}
				data, err := os.ReadFile(filepath.Join(root, scanner+"-settings.json"))
				if err != nil {
					t.Fatal(err)
				}
				var settings map[string]any
				if err := json.Unmarshal(data, &settings); err != nil {
					t.Fatal(err)
				}
				settings["server"] = h.targetURL + "/api"
				data, _ = json.Marshal(settings)
				inputs["openapi"] = uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "openapi", "application/json", source)
				inputs["settings"] = uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "settings", "application/json", data)
			}
			var draft auditProgramAudit
			scanAuditPOST(t, h, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": profile, "version": "1"}, "inputs": inputs, "scope": map[string]string{"objective": "Inspect the exact controlled target"}}, 0, http.StatusCreated, &draft)
			var started struct {
				Audit auditProgramAudit `json:"audit"`
			}
			scanAuditPOST(t, h, "/v1/audits/"+draft.AuditID+"/start", nil, draft.Revision, http.StatusOK, &started)
			if started.Audit.State != "active" {
				t.Fatalf("start=%+v", started)
			}
			waitScanAudit(t, h, draft.AuditID, expectedRuns)
			var page struct {
				Items []basicTask `json:"items"`
			}
			auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+draft.AuditID+"/tasks?limit=64", &page)
			if len(page.Items) != expectedTasks {
				t.Fatalf("accepted work=%+v", page)
			}
			task := page.Items[len(page.Items)-1]
			if task.State != "settled" || task.GapCode != nil || task.Result == nil || task.Document.Hypothesis != nil || task.Attempts != 1 {
				t.Fatalf("scan Task=%+v", task)
			}
			if scanner != "web" {
				if page.Items[0].Document.Workflow != "prepare" || task.Document.Workflow != "scan" || task.Document.Inputs["operation"].ScopeKind != "audit" {
					t.Fatal("preparation bypassed common exact-material intake")
				}
			}
			data := commonScanArtifact(t, h, draft.AuditID, task)
			var report struct {
				Schema             string `json:"schema"`
				RunID              string `json:"run_id"`
				ExecutionCompleted bool   `json:"execution_completed"`
				Artifacts          []struct {
					Kind   string `json:"kind"`
					Digest string `json:"digest"`
					Data   []byte `json:"data_base64"`
				} `json:"artifacts"`
				Attempts []planner.ScanAttempt `json:"attempts"`
			}
			if err := json.Unmarshal(data, &report); err != nil || report.Schema != "contractor.scan.operation-report.v1" || !report.ExecutionCompleted || len(report.Attempts) == 0 {
				t.Fatalf("scan report=%+v error=%v", report, err)
			}
			reports := 0
			for _, artifact := range report.Artifacts {
				if contentdigest.Bytes(artifact.Data) != artifact.Digest {
					t.Fatal("retained evidence changed bytes")
				}
				if artifact.Kind == "scanner-report" {
					reports++
				}
			}
			if reports != 1 {
				t.Fatalf("scanner report count=%d", reports)
			}
			for _, member := range page.Items {
				if member.Result == nil || member.LastExecution == nil {
					t.Fatal("supporting work has no retained result")
				}
			}
			before := len(scanFixtureRequests(h))
			// Public Tasks preserve the ordinary Run association; deleting every source
			// Run must leave the exact Audit material readable without target activity.
			for _, member := range page.Items {
				var runID string
				if err := h.pool.QueryRow(h.ctx, `SELECT run_id FROM audit_executions WHERE execution_id=$1`, *member.LastExecution).Scan(&runID); err != nil {
					t.Fatal(err)
				}
				deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, runID)
			}
			if after := commonScanArtifact(t, h, draft.AuditID, task); !bytes.Equal(data, after) || before != len(scanFixtureRequests(h)) {
				t.Fatal("source Run purge changed evidence or repeated target activity")
			}
		})
	}
}
func commonScanArtifact(t *testing.T, h *scanProcessHarness, auditID string, task basicTask) []byte {
	t.Helper()
	key := "execution/" + *task.LastExecution + "/output/scan_report"
	query := url.Values{"key": {key}, "revision": {*task.Result.Ref.Revision}}
	request, err := http.NewRequestWithContext(h.ctx, http.MethodGet, h.baseURL+"/v1/audits/"+auditID+"/artifacts?"+query.Encode(), nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	response := do(t, h.client, request, http.StatusOK)
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, 8<<20))
	if err != nil {
		t.Fatal(err)
	}
	if task.Result.ScopeKind != "audit" || task.Result.ScopeID != auditID {
		t.Fatal("foreign collected material")
	}
	var material auditcontract.Material = *task.Result
	if contentdigest.Bytes(data) != material.Digest {
		t.Fatal("ordinary result digest differs")
	}
	return data
}

func stageCommonScanAuditCatalog(t *testing.T, h *scanProcessHarness, root string) {
	t.Helper()
	for _, relative := range []string{"agent-templates/audit_sqlmap_scan.yaml", "agent-templates/audit_nuclei_scan.yaml", "agent-templates/audit_web_nuclei_scan.yaml", "workflows/audit_openapi_sqlmap_scan.yaml", "workflows/audit_openapi_nuclei_scan.yaml", "workflows/audit_web_nuclei_scan.yaml", "workflows/openapi_prepare_work.yaml", "audit-profiles/openapi_sqlmap_scan.yaml", "audit-profiles/openapi_nuclei_scan.yaml", "audit-profiles/web_scan.yaml", "instructions/audit-openapi-scan.md"} {
		data, err := os.ReadFile(filepath.Join(h.repositoryRoot, "configs", relative))
		if err != nil {
			t.Fatal(err)
		}
		destination := filepath.Join(root, relative)
		if err := os.MkdirAll(filepath.Dir(destination), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(destination, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	replaceScanConfig(t, filepath.Join(root, "agent-templates/audit_nuclei_scan.yaml"), "http-missing-security-headers", "contractor-local-fixture")
	data, err := os.ReadFile(filepath.Join(h.temporaryRoot, "nuclei-templates/fixture.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := strictjson.Canonical(map[string]any{"schemaVersion": 1, "files": []map[string]any{{"path": "fixture.yaml", "digest": contentdigest.Bytes(data), "size": len(data)}}})
	if err != nil {
		t.Fatal(err)
	}
	replaceScanConfig(t, filepath.Join(root, "agent-templates/audit_web_nuclei_scan.yaml"), "sha256:51a3bd4eef14e95bdeb0bad14ca4bd9bd455479a7044b244b76dee8f6dba5014", contentdigest.Bytes(manifest))
}
