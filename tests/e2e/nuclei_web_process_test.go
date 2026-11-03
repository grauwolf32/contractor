//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// Use the shipped Workflow, Server, Scheduler and Runtime with an installed
// Nuclei engine. Only the operator package pin changes to the local fixture.
func TestNucleiWebAcrossProductionProcesses(t *testing.T) {
	runNucleiWebProcessAcceptance(t, false)
}

func TestNucleiWebWithDeterministicScannerAcrossProcesses(t *testing.T) {
	runNucleiWebProcessAcceptance(t, true)
}

func runNucleiWebProcessAcceptance(t *testing.T, deterministic bool) {
	if testing.Short() {
		t.Fatal("Nuclei web process acceptance cannot run in short mode")
	}
	binaries := []string{}
	if !deterministic {
		binaries = append(binaries, "nuclei")
	}
	h := startConfiguredScanStack(t, "", func(h *scanProcessHarness, root string) {
		if deterministic {
			binaryRoot := filepath.Join(h.temporaryRoot, "fixture-bin")
			if err := os.Mkdir(binaryRoot, 0700); err != nil {
				t.Fatal(err)
			}
			python := os.Getenv("CONTRACTOR_SCAN_PYTHON")
			if python == "" {
				python = filepath.Join(h.repositoryRoot, "runtime/.venv/bin/python")
			}
			source := "#!" + python + "\n" + `import json, pathlib, sys, urllib.request
args = sys.argv[1:]
if '-version' in args:
    print('Nuclei Engine Version: v3.11.1')
elif '-validate' in args:
    root = pathlib.Path(args[args.index('-t') + 1])
    if any(b'contractor-invalid-matcher' in p.read_bytes() for p in root.glob('*.yaml')):
        sys.exit(2)
elif '-tl' in args:
    root = pathlib.Path(args[args.index('-t') + 1])
    print('\n'.join(str(path.relative_to(root)) for path in sorted(root.rglob('*.yaml'))))
else:
    target = pathlib.Path(args[args.index('-l') + 1]).read_text().strip()
    with urllib.request.urlopen(target.rstrip('/') + '/nuclei', timeout=3) as response:
        body = response.read(1024)
    if b'contractor-controlled-scan-fixture' not in body:
        raise RuntimeError('unexpected loopback fixture')
    print(json.dumps({'template-id': 'contractor-local-fixture'}))
`
			if err := os.WriteFile(filepath.Join(binaryRoot, "nuclei"), []byte(source), 0700); err != nil {
				t.Fatal(err)
			}
			h.runtimePathPrefix = binaryRoot
		}
		for _, relative := range []string{"workflows/audit_web_nuclei_scan.yaml", "agent-templates/audit_web_nuclei_scan.yaml", "audit-profiles/web_scan.yaml", "instructions/audit-openapi-scan.md"} {
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
		replaceScanConfig(t, filepath.Join(root, "agent-templates/audit_web_nuclei_scan.yaml"), "sha256:51a3bd4eef14e95bdeb0bad14ca4bd9bd455479a7044b244b76dee8f6dba5014", contentdigest.Bytes(canonical))
	}, binaries...)
	t.Run("installed_rules_and_replayed_audit_admission", func(t *testing.T) {
		before := len(scanFixtureRequests(h))
		audit := startWebScanAudit(t, h, "", nil)
		waitScanAudit(t, h, audit.AuditID, 1)
		pkg := assertWebAuditRetention(t, h, audit, true)
		assertWebScanReport(t, pkg, "installed", "")
		if len(scanFixtureRequests(h)) != before+1 {
			t.Fatal("replayed admission dispatched more than one scan")
		}
	})
	t.Run("custom_project_rules_pin_inputs_without_installation", func(t *testing.T) {
		path := filepath.Join(h.temporaryRoot, "nuclei-templates")
		payload, err := os.ReadFile(filepath.Join(path, "fixture.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Rename(path, path+".saved"); err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := os.Rename(path+".saved", path); err != nil {
				t.Error(err)
			}
		}()
		audit := startWebScanAudit(t, h, "", payload)
		waitScanAudit(t, h, audit.AuditID, 1)
		pkg := assertWebAuditRetention(t, h, audit, true)
		assertWebScanReport(t, pkg, "rulepack", contentdigest.Bytes(payload))
	})
	t.Run("changed_installed_package_has_gaps_without_traffic", func(t *testing.T) {
		path := filepath.Join(h.temporaryRoot, "nuclei-templates", "fixture.yaml")
		original, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(original, []byte("\n# changed under same identity\n")...), 0600); err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := os.WriteFile(path, original, 0600); err != nil {
				t.Error(err)
			}
		}()
		before := len(scanFixtureRequests(h))
		audit := startWebScanAudit(t, h, "", nil)
		waitScanAudit(t, h, audit.AuditID, 1)
		assertWebAuditRetention(t, h, audit, false)
		if len(scanFixtureRequests(h)) != before {
			t.Fatal("changed package reached target")
		}
	})
	t.Run("invalid_custom_rules_have_gaps_without_installed_fallback", func(t *testing.T) {
		before := len(scanFixtureRequests(h))
		audit := startWebScanAudit(t, h, "", []byte("id: bad\nhttp: []\n"))
		waitScanAudit(t, h, audit.AuditID, 1)
		assertWebAuditRetention(t, h, audit, false)
		if len(scanFixtureRequests(h)) != before {
			t.Fatal("invalid rules reached target")
		}
	})
	t.Run("lost_scanner_result_retains_unknown_without_rescan", func(t *testing.T) {
		installScanFault(t, h, "web_scan_journal", "planner_sessions", "UPDATE", `IF NEW.state #>> '{scan,state,jobs,0,status}' = 'completed' THEN RAISE EXCEPTION 'controlled scanner journal acknowledgement loss'; END IF;`)
		before := len(scanFixtureRequests(h))
		audit := startWebScanAudit(t, h, "", nil)
		waitScanAudit(t, h, audit.AuditID, 1)
		pkg := assertWebAuditRetention(t, h, audit, false)
		if !slices.Contains(pkg.Limitations, "scan_outcome_unknown") || len(scanFixtureRequests(h)) != before+1 {
			t.Fatalf("lost result repeated work or lost its gap: %+v", pkg)
		}
	})
	t.Run("cancelled_child_is_not_resubmitted", func(t *testing.T) {
		audit := startWebScanAudit(t, h, "/slow", nil)
		select {
		case <-h.fixture.slowStarted:
		case <-h.ctx.Done():
			t.Fatal("scanner never reached cancellation fixture")
		}
		task := scanAuditTask(t, h, audit.AuditID)
		if task.LastExecution == nil {
			t.Fatal("missing ordinary execution")
		}
		var runID string
		if err := h.pool.QueryRow(h.ctx, `SELECT run_id FROM audit_executions WHERE execution_id=$1`, *task.LastExecution).Scan(&runID); err != nil {
			t.Fatal(err)
		}
		cancelWorkflowRun(t, h.client, h.baseURL, runID)
		waitScanAudit(t, h, audit.AuditID, 1)
		assertWebAuditRetention(t, h, audit, false)
	})
}

func startWebScanAudit(t *testing.T, h *scanProcessHarness, prefix string, rulepack []byte) auditProgramAudit {
	t.Helper()
	var project projectResourceResponse
	scanAuditPOST(t, h, "/v1/projects", map[string]string{"kind": "project", "name": "Web scan Audit fixture", "activeChecks": "automatic"}, 0, http.StatusCreated, &project)
	project = updateProjectTarget(t, h.client, h.baseURL, project.ProjectID, project.Revision, map[string]any{"url": h.targetURL + prefix}, http.StatusOK)
	target := []byte(fmt.Sprintf(`{"schema":"contractor.audit.web-target.v1","url":%q}`, h.targetURL+prefix))
	settings := []byte(`{"schema":"contractor.audit.web-scan-settings.v1","templateIds":"","tags":"","severity":"","rateLimit":10}`)
	inputs := map[string]artifactRef{
		"target":   uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "target", "application/json", target),
		"settings": uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "settings", "application/json", settings),
	}
	if rulepack != nil {
		inputs["rulepack"] = uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "rules", "application/yaml", rulepack)
	}
	var draft auditProgramAudit
	webScanReplayPOST(t, h, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": "web-scan", "version": "1"}, "inputs": inputs}, 0, http.StatusCreated, &draft)
	var started struct {
		Audit auditProgramAudit `json:"audit"`
	}
	webScanReplayPOST(t, h, "/v1/audits/"+draft.AuditID+"/start", nil, draft.Revision, http.StatusOK, &started)
	if started.Audit.State != "active" {
		t.Fatalf("unexpected Audit start: %+v", started)
	}

	// A later Project target and source advancement must not retarget accepted work.
	updateProjectTarget(t, h.client, h.baseURL, project.ProjectID, project.Revision, map[string]any{"url": "http://127.0.0.1:1/changed"}, http.StatusOK)
	for slot, ref := range inputs {
		replacement := []byte("invalid replacement")
		req, err := http.NewRequestWithContext(h.ctx, http.MethodPut, h.baseURL+"/v1/projects/"+project.ProjectID+"/artifacts/"+ref.Namespace+"/"+ref.Name, bytes.NewReader(replacement))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Authorization", "Bearer "+publicToken)
		req.Header.Set("Content-Type", "application/json")
		if slot == "rulepack" {
			req.Header.Set("Content-Type", "application/yaml")
		}
		req.Header.Set("If-Match", fmt.Sprintf("%q", *ref.Revision))
		response := do(t, h.client, req, http.StatusOK)
		response.Body.Close()
	}
	return started.Audit
}

// Discard the committed first response and recover using the identical request
// and key. Start replay must work even though its original revision is stale.
func webScanReplayPOST(t *testing.T, h *scanProcessHarness, path string, body any, revision uint64, status int, result any) {
	t.Helper()
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if body == nil {
		data = nil
	}
	key := "web-audit-" + randomHex(t, 8)
	for attempt := 0; attempt < 2; attempt++ {
		req, err := http.NewRequestWithContext(h.ctx, http.MethodPost, h.baseURL+path, bytes.NewReader(data))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Authorization", "Bearer "+publicToken)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Idempotency-Key", key)
		if revision != 0 {
			req.Header.Set("If-Match", fmt.Sprintf("%q", fmt.Sprint(revision)))
		}
		response := do(t, h.client, req, status)
		if attempt == 0 {
			io.Copy(io.Discard, response.Body)
			response.Body.Close()
			continue
		}
		if response.Header.Get("Idempotency-Replayed") != "true" {
			t.Fatal("lost reply was not reconciled")
		}
		decodeAuditProgramResponse(t, response, result)
		response.Body.Close()
	}
}

func assertWebScanReport(t *testing.T, pkg retainedWebScanReport, source, digest string) {
	t.Helper()
	found := false
	for _, artifact := range pkg.Artifacts {
		if artifact.Kind != "scanner-report" {
			continue
		}
		var report scanProcessReport
		if err := json.Unmarshal(artifact.Data, &report); err != nil {
			t.Fatal(err)
		}
		if report.Observation["source"] != source || digest != "" && report.Observation["sourceContentDigest"] != digest {
			t.Fatalf("rulepack source changed: %+v", report)
		}
		if report.Observation["templateManifest"] == nil || report.Observation["selectedTemplateCount"] != float64(1) {
			t.Fatalf("missing rule provenance: %+v", report)
		}
		found = true
	}
	if !found {
		t.Fatal("ordinary Audit report omitted scanner evidence")
	}
}
