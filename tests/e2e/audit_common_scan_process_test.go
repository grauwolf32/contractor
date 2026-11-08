//go:build e2e

package e2e

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
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
			if task.State != "settled" || task.GapCode != nil || task.Findings == nil || task.Result != nil || task.Document.Hypothesis != nil || task.Attempts != 1 {
				if task.GapCode != nil {
					t.Logf("Task gap=%s", *task.GapCode)
				}
				var diagnostics []byte
				_ = h.pool.QueryRow(h.ctx, `SELECT jsonb_agg(jsonb_build_object('name',s.stage_name,'state',s.state,'reason',s.state_reason_code,'error',s.accepted_stage_result->'error','termination',s.stage_termination)) FROM stage_executions s JOIN audit_executions e USING(run_id) WHERE e.execution_id=$1`, *task.LastExecution).Scan(&diagnostics)
				t.Fatalf("scan Task=%+v stages=%s", task, diagnostics)
			}
			if scanner != "web" {
				if page.Items[0].Document.Workflow != "prepare" || task.Document.Workflow != "scan" || task.Document.Inputs["operation"].ScopeKind != "audit" {
					t.Fatal("preparation bypassed common exact-material intake")
				}
			}
			data := commonScanArtifact(t, h, draft.AuditID, task)
			var report retainedWebScanReport
			if err := json.Unmarshal(data, &report); err != nil || report.Schema != "contractor.scan.operation-report.v1" || !report.ExecutionCompleted || len(report.Attempts) == 0 {
				t.Fatalf("scan report=%+v error=%v", report, err)
			}
			reports := 0
			for _, artifact := range report.Artifacts {
				if contentdigest.Bytes(sourceScanArtifact(t, h, report.RunID, artifact.Ref)) != artifact.Digest {
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
				if member.Result == nil && member.Findings == nil || member.LastExecution == nil {
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
			var after struct {
				Items []basicTask `json:"items"`
			}
			auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+draft.AuditID+"/tasks?limit=64", &after)
			if after.Items[len(after.Items)-1].Findings == nil || before != len(scanFixtureRequests(h)) {
				t.Fatal("source purge lost canonical results or repeated target work")
			}

		})
	}
}

// Reports are read only from the concrete child Run. Audit collection contains
// canonical domain references, including an empty result, and no report export.
func commonScanArtifact(t *testing.T, h *scanProcessHarness, auditID string, task basicTask) []byte {
	t.Helper()
	if task.Result != nil || task.Findings == nil {
		t.Fatal("scanner Task substituted Artifact result")
	}
	var runID string
	var encoded []byte
	if err := h.pool.QueryRow(h.ctx, `SELECT execution.run_id,stage.accepted_stage_result->'artifacts'->'report' FROM audit_executions execution JOIN stage_executions stage USING(run_id) WHERE execution.execution_id=$1 AND stage.accepted_stage_result->'artifacts' ? 'report' ORDER BY stage.created_at DESC LIMIT 1`, *task.LastExecution).Scan(&runID, &encoded); err != nil {
		t.Fatal(err)
	}
	var ref contracts.ArtifactRef
	if err := json.Unmarshal(encoded, &ref); err != nil {
		t.Fatal(err)
	}
	var copied int
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM audit_artifact_links WHERE audit_id=$1 AND logical_key LIKE $2`, auditID, "execution/"+*task.LastExecution+"/%").Scan(&copied); err != nil || copied != 0 {
		t.Fatalf("scanner report copied into Audit: %d %v", copied, err)
	}
	return sourceScanArtifact(t, h, runID, ref)
}
func sourceScanArtifact(t *testing.T, h *scanProcessHarness, runID string, ref contracts.ArtifactRef) []byte {
	t.Helper()
	data, _ := download(t, h.client, h.baseURL+"/v1/runs/"+runID+"/artifacts/"+ref.Namespace+"/"+ref.Name+"?revision="+*ref.Revision)
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
