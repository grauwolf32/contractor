//go:build e2e

package e2e

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httputil"
	neturl "net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func TestScannerFindingsAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Fatal("scanner Finding acceptance requires real processes")
	}
	h := startConfiguredScanStackWithBudget(t, 10*time.Minute, "", func(h *scanProcessHarness, root string) {
		h.targetURL = startEngagementScanTarget(t, h.fixture).URL
		for _, relative := range []string{
			"agent-templates/audit_web_nuclei_scan.yaml", "workflows/audit_web_nuclei_scan.yaml", "audit-profiles/web_scan.yaml", "instructions/audit-openapi-scan.md",
			"agent-templates/project_nuclei_scan.yaml", "agent-templates/project_sqlmap_url_scan.yaml", "agent-templates/project_sqlmap_request_scan.yaml",
			"workflows/project_nuclei_scan.yaml", "workflows/project_sqlmap_url_scan.yaml", "workflows/project_sqlmap_request_scan.yaml", "workflows/publish_findings.yaml", "instructions/scanner-findings.md",
		} {
			data, err := os.ReadFile(filepath.Join(h.repositoryRoot, "configs", relative))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.MkdirAll(filepath.Dir(filepath.Join(root, relative)), 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, relative), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		rules := []byte(`id: git-config
info:
  name: Exposed Git configuration
  author: contractor
  severity: medium
  metadata:
    product: git
http:
  - method: GET
    path: ["{{BaseURL}}/.git/config"]
    matchers-condition: and
    matchers:
      - type: status
        status: [200]
      - type: word
        words: ["[core]", "repositoryformatversion"]
        condition: and
`)
		if err := os.WriteFile(filepath.Join(h.temporaryRoot, "nuclei-templates/fixture.yaml"), rules, 0600); err != nil {
			t.Fatal(err)
		}
		manifest := map[string]any{"schemaVersion": 1, "files": []map[string]any{{"path": "fixture.yaml", "digest": contentdigest.Bytes(rules), "size": len(rules)}}}
		canonical, err := strictjson.Canonical(manifest)
		if err != nil {
			t.Fatal(err)
		}
		replaceScanConfig(t, filepath.Join(root, "agent-templates/project_nuclei_scan.yaml"), "sha256:51a3bd4eef14e95bdeb0bad14ca4bd9bd455479a7044b244b76dee8f6dba5014", contentdigest.Bytes(canonical))
		replaceScanConfig(t, filepath.Join(root, "agent-templates/audit_web_nuclei_scan.yaml"), "sha256:51a3bd4eef14e95bdeb0bad14ca4bd9bd455479a7044b244b76dee8f6dba5014", contentdigest.Bytes(canonical))
	}, "nuclei", "sqlmap")
	address := freeAddress(t)
	python := filepath.Join(h.repositoryRoot, "runtime/.venv/bin/python")
	fixture := startProcess(t, "Scanner Finding SQLite target", h.repositoryRoot, map[string]string{"PYTHONUNBUFFERED": "1"}, python, filepath.Join(h.repositoryRoot, "tests/e2e/testdata/scanner-findings/target.py"), "--listen", address)
	backend := "http://" + address
	waitForHTTP(t, h.ctx, fixture, h.client, backend+"/readyz", http.StatusOK)
	backendURL, err := neturl.Parse(backend)
	if err != nil {
		t.Fatal(err)
	}
	// The fixture remains loopback-only; engagement addresses this owned
	// private-interface HTTP endpoint through the production transport.
	target := startEngagementScanTarget(t, httputil.NewSingleHostReverseProxy(backendURL)).URL
	project := createProjectResource(t, h.client, h.baseURL)
	run := func(workflow, key, url string, inputs map[string]artifactRef) string {
		t.Helper()
		parameters := map[string]string{}
		if url != "" {
			parameters["target"] = url
		}
		if inputs == nil {
			inputs = map[string]artifactRef{}
		}
		body, err := json.Marshal(map[string]any{"workflow": workflow, "parameters": parameters, "artifacts": inputs})
		if err != nil {
			t.Fatal(err)
		}
		id := postProjectRun(t, h.client, h.baseURL, project.ProjectID, key, body, false)
		if workflow == "publish-findings@1" {
			status := waitForRunTerminalAcross(t, h.ctx, h.server, nil, h.client, h.baseURL, id)
			if status.State != "succeeded" {
				t.Fatalf("publication state=%s", status.State)
			}
		} else {
			h.terminal(t, id, "succeeded")
		}
		return id
	}
	positive := run("project-nuclei-scan@1", "nuclei-positive", target, nil)
	page := scannerFindingPage(t, h, project.ProjectID)
	if len(page) != 1 {
		t.Fatalf("real git-config detection did not publish one Finding: %s", page)
	}
	if string(page[0]["state"]) != "null" || string(page[0]["analystVerdict"]) != "null" {
		t.Fatalf("scanner supplied analyst verdict: %s", page)
	}
	_ = run("project-nuclei-scan@1", "nuclei-fixed", target+"/fixed", nil)
	if items := scannerFindingPage(t, h, project.ProjectID); len(items) != 1 {
		t.Fatalf("fixed Nuclei control created Findings: %s", items)
	}
	rawReport, _ := download(t, h.client, h.baseURL+"/v1/runs/"+positive+"/artifacts/scanner/report")
	var report scanProcessReport
	if err := json.Unmarshal(rawReport, &report); err != nil || report.Observation["status"] != "completed" {
		t.Fatalf("original report is unavailable: %v", err)
	}
	// Exact prepared request: one real vulnerable parameter, no invented tested siblings.
	request := []byte(fmt.Sprintf(`{"schemaVersion":1,"method":"GET","url":%q,"headers":[],"body":"","testParameters":["id"]}`, target+"/vulnerable?id=1"))
	input := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "scanner-inputs", "request", "application/vnd.contractor.http-request+json", request)
	sql := run("project-sqlmap-request-scan@1", "sqlmap-positive", "", map[string]artifactRef{"request": input})
	if items := scannerFindingPage(t, h, project.ProjectID); len(items) != 2 {
		t.Fatalf("real SQLMap injection did not publish exactly one additional Finding: %s", items)
	}
	fixed := bytes.Replace(request, []byte("/vulnerable?"), []byte("/fixed?"), 1)
	fixedInput := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "scanner-inputs", "fixed-request", "application/vnd.contractor.http-request+json", fixed)
	_ = run("project-sqlmap-request-scan@1", "sqlmap-fixed", "", map[string]artifactRef{"request": fixedInput})
	if items := scannerFindingPage(t, h, project.ProjectID); len(items) != 2 {
		t.Fatalf("fixed SQLMap control created a Finding: %s", items)
	}
	// URL mode uses the same adapter and canonical parameter identity, while a
	// new actual scan contributes its own independent source.
	_ = run("project-sqlmap-url-scan@1", "sqlmap-url-positive", target+"/vulnerable?id=1", nil)
	if items := scannerFindingPage(t, h, project.ProjectID); len(items) != 2 {
		t.Fatalf("URL mode lost canonical equality: %s", items)
	}
	// Real Audit children publish canonical Findings without an Artifact result,
	// including a truthful empty result for the fixed target.
	engagementInput := uploadScanEngagement(t, h.client, h.baseURL, project.ProjectID, target, false)
	auditFor := func(key, url string) string {
		targetData, _ := json.Marshal(map[string]any{"schema": "contractor.audit.web-target.v1", "url": url})
		targetInput := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "scanner-inputs", key+"-target", "application/json", targetData)
		var draft auditProgramAudit
		scanAuditPOST(t, h, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": "web-scan", "version": "1"}, "inputs": map[string]artifactRef{"target": targetInput, "engagement": engagementInput}, "parameters": map[string]string{"template_ids": "git-config", "tags": "", "severity": "medium"}, "scope": map[string]string{"objective": "Check the exact fixture"}}, 0, http.StatusCreated, &draft)
		scanAuditPOST(t, h, "/v1/audits/"+draft.AuditID+"/start", nil, draft.Revision, http.StatusOK, nil)
		waitScanAudit(t, h, draft.AuditID, 1)
		task := scanAuditTask(t, h, draft.AuditID)
		expected := 1
		if key == "fixed" {
			expected = 0
		}
		if task.Findings == nil || len(task.Findings.IDs) != expected || task.Result != nil || task.GapCode != nil {
			t.Fatalf("canonical scanner Audit task=%+v", task)
		}
		_ = commonScanArtifact(t, h, draft.AuditID, task)
		return draft.AuditID
	}
	auditPositive := auditFor("positive", target)
	auditEmpty := auditFor("fixed", target+"/fixed")
	// A genuine information-only custom rule enriches sitemap independently.
	infoRules := []byte(`id: nginx-info
info:
  name: Nginx fingerprint
  author: contractor
  severity: info
  metadata:
    product: nginx
http:
  - method: GET
    path: ["{{BaseURL}}/tech"]
    matchers:
      - type: word
        words: ["nginx/1.24.0"]
    extractors:
      - type: regex
        regex: ['1\.24\.0']
`)
	infoInput := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "scanner-inputs", "nginx-rule", "application/yaml", infoRules)
	_ = run("project-nuclei-scan@1", "informational-fingerprint", target, map[string]artifactRef{"rulepack": infoInput})
	sitemap, _ := download(t, h.client, h.baseURL+"/v1/projects/"+project.ProjectID+"/artifacts/sitemap/map")
	if !bytes.Contains(sitemap, []byte("Technology: nginx; reported versions: 1.24.0")) {
		t.Fatal("real informational Nuclei metadata missing from sitemap")
	}
	if items := scannerFindingPage(t, h, project.ProjectID); len(items) != 2 {
		t.Fatal("informational match fabricated a Finding")
	}
	assertScannerLargeRulepackFinding(t, h, target)

	// Current public source detail exposes actual scan and publishing Stage provenance.
	var receiptID string
	if err := h.pool.QueryRow(h.ctx, `SELECT receipt_id FROM finding_proposal_receipts WHERE run_id=$1`, sql).Scan(&receiptID); err != nil {
		t.Fatal(err)
	}
	data := scannerFindingGET(t, h, "/v1/projects/"+project.ProjectID+"/finding-sources/"+receiptID)
	var receipt findingintake.Receipt
	if err := json.Unmarshal(data, &receipt); err != nil || receipt.Origin.Scan == nil || receipt.Origin.Scan.RunID != sql || receipt.Origin.Operation != "publish-findings@1" || receipt.Origin.AllocationID != "" || len(receipt.ProjectEvidence) != 2 {
		t.Fatalf("operation source detail: %+v %v", receipt.Origin, err)
	}
	for _, sourceRun := range []string{positive, sql} {
		var allocations, models int
		if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM stage_allocations a JOIN stage_executions s USING(stage_execution_id) WHERE s.run_id=$1 AND s.stage_spec_snapshot ? 'operation'`, sourceRun).Scan(&allocations); err != nil || allocations != 0 {
			t.Fatalf("publication allocated Runtime: %d %v", allocations, err)
		}
		if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM stage_executions WHERE run_id=$1 AND stage_spec_snapshot ? 'operation' AND planner_session_id IS NOT NULL`, sourceRun).Scan(&models); err != nil || models != 0 {
			t.Fatalf("publication recorded a Planner: %d %v", models, err)
		}
	}
	// Project-retained common evidence is an ordinary explicit replay input.
	selected := map[string]artifactRef{}
	for _, evidence := range receipt.ProjectEvidence {
		key := "evidence"
		if evidence.MediaType == findingintake.CandidateMediaType {
			key = "candidates"
		}
		encoded, _ := json.Marshal(evidence.Ref)
		var ref artifactRef
		if err := json.Unmarshal(encoded, &ref); err != nil {
			t.Fatal(err)
		}
		selected[key] = ref
	}
	var sourcesBefore int
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM project_finding_sources WHERE project_id=$1`, project.ProjectID).Scan(&sourcesBefore); err != nil || sourcesBefore != 4 {
		t.Fatalf("independent source count=%d %v", sourcesBefore, err)
	}
	h.runtime.stop(t) // Publication works while the actual Runtime is unavailable.
	beforeRequest, _ := http.NewRequest(http.MethodGet, target+"/requests", nil)
	beforeResponse := do(t, h.client, beforeRequest, http.StatusOK)
	var before struct {
		Requests int `json:"requests"`
	}
	json.NewDecoder(beforeResponse.Body).Decode(&before)
	beforeResponse.Body.Close()
	replay := run("publish-findings@1", "publication-only", "", selected)
	var allocations, models int
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM stage_allocations JOIN stage_executions USING(stage_execution_id) WHERE run_id=$1`, replay).Scan(&allocations); err != nil || allocations != 0 {
		t.Fatalf("replay Runtime allocation=%d %v", allocations, err)
	}
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM planner_sessions JOIN stage_executions USING(stage_execution_id) WHERE run_id=$1`, replay).Scan(&models); err != nil || models != 0 {
		t.Fatalf("replay model session=%d %v", models, err)
	}
	afterResponse := do(t, h.client, beforeRequest, http.StatusOK)
	var after struct {
		Requests int `json:"requests"`
	}
	json.NewDecoder(afterResponse.Body).Decode(&after)
	afterResponse.Body.Close()
	var sourcesAfter int
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM project_finding_sources WHERE project_id=$1`, project.ProjectID).Scan(&sourcesAfter); err != nil || sourcesAfter != sourcesBefore || before.Requests != after.Requests {
		t.Fatalf("replay added source/target traffic: %d->%d, %d->%d %v", sourcesBefore, sourcesAfter, before.Requests, after.Requests, err)
	}
	if os.Getenv("CONTRACTOR_SCAN_BROWSER") == "1" {
		runScanBrowserSpec(t, h, "e2e/scanner-findings.spec.ts", map[string]string{"CONTRACTOR_SCAN_FINDING_PROJECT": project.ProjectID, "CONTRACTOR_SCAN_FINDING_RUN": positive, "CONTRACTOR_SCAN_FINDING_AUDIT": auditPositive, "CONTRACTOR_SCAN_EMPTY_AUDIT": auditEmpty})
	}
	deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, sql)
	replayAfterPurge := run("publish-findings@1", "publication-after-purge", "", selected)
	_ = replayAfterPurge
	if items := scannerFindingPage(t, h, project.ProjectID); len(items) != 2 {
		t.Fatal("source Run purge lost Findings")
	}

}

func assertScannerLargeRulepackFinding(t *testing.T, h *scanProcessHarness, target string) {
	t.Helper()
	create, err := http.NewRequest(http.MethodPost, h.baseURL+"/v1/projects", strings.NewReader(`{"kind":"project","name":"Large Nuclei rulepack"}`))
	if err != nil {
		t.Fatal(err)
	}
	create.Header.Set("Authorization", "Bearer "+publicToken)
	create.Header.Set("Content-Type", "application/json")
	create.Header.Set("Idempotency-Key", "scanner-large-rulepack-project")
	response := do(t, h.client, create, http.StatusCreated)
	defer response.Body.Close()
	var project projectResourceResponse
	decodeResponse(t, response, &project)
	var archive bytes.Buffer
	pack := zip.NewWriter(&archive)
	matchedRule := ""
	for index := 0; index < 1000; index++ {
		rule := fmt.Sprintf("rule-%04d-", index) + strings.Repeat("y", 100)
		name := fmt.Sprintf("rule-%04d-", index) + strings.Repeat("x", 64) + ".yaml"
		word := fmt.Sprintf("absent-rule-%04d", index)
		if index == 0 {
			matchedRule, word = rule, "repositoryformatversion"
		}
		entry, err := pack.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		data := fmt.Sprintf("id: %s\ninfo:\n  name: Rule fixture\n  author: contractor\n  severity: medium\nhttp:\n  - method: GET\n    path: [\"{{BaseURL}}/.git/config\"]\n    matchers:\n      - type: word\n        words: [%q]\n", rule, word)
		if _, err := entry.Write([]byte(data)); err != nil {
			t.Fatal(err)
		}
	}
	if err := pack.Close(); err != nil {
		t.Fatal(err)
	}
	input := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "scanner-inputs", "large-rulepack", "application/vnd.contractor.nuclei-rulepack+zip", archive.Bytes())
	body, err := json.Marshal(map[string]any{"workflow": "project-nuclei-scan@1", "parameters": map[string]string{"target": target}, "artifacts": map[string]artifactRef{"rulepack": input}})
	if err != nil {
		t.Fatal(err)
	}
	run := postProjectRun(t, h.client, h.baseURL, project.ProjectID, "large-rulepack", body, false)
	h.terminal(t, run, "succeeded")
	if items := scannerFindingPage(t, h, project.ProjectID); len(items) != 1 {
		t.Fatalf("large valid rulepack did not publish one Finding: %s", items)
	}
	manifest, _ := download(t, h.client, h.baseURL+"/v1/runs/"+run+"/artifacts/scanner/template-manifest")
	if len(manifest)*4/3 <= 512<<10 {
		t.Fatalf("manifest does not reproduce Base64 overflow: %d bytes", len(manifest))
	}
	evidence, _ := download(t, h.client, h.baseURL+"/v1/runs/"+run+"/artifacts/scanner/evidence")
	var proof struct {
		Inputs []struct {
			Kind     string `json:"kind"`
			Manifest struct {
				Digest    string `json:"digest"`
				SizeBytes int    `json:"sizeBytes"`
			} `json:"manifest"`
			Templates []struct {
				ID string `json:"id"`
			} `json:"templates"`
		} `json:"inputs"`
	}
	if err := json.Unmarshal(evidence, &proof); err != nil || len(evidence) > 512<<10 || len(proof.Inputs) != 1 {
		t.Fatalf("invalid bounded evidence: bytes=%d %v", len(evidence), err)
	}
	selected := proof.Inputs[0]
	if selected.Kind != "nuclei-manifest-selection" || selected.Manifest.Digest != contentdigest.Bytes(manifest) || selected.Manifest.SizeBytes != len(manifest) || len(selected.Templates) != 1 || selected.Templates[0].ID != matchedRule {
		t.Fatalf("selected manifest evidence lost exact origin: %+v", selected)
	}
	t.Logf("large rulepack: 1000 selected templates, manifest=%d bytes, evidence=%d bytes, one real Finding", len(manifest), len(evidence))
}

func scannerFindingGET(t *testing.T, h *scanProcessHarness, path string) []byte {
	t.Helper()
	request, err := http.NewRequest(http.MethodGet, h.baseURL+path, nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	response := do(t, h.client, request, http.StatusOK)
	defer response.Body.Close()
	var raw json.RawMessage
	if err := json.NewDecoder(response.Body).Decode(&raw); err != nil {
		t.Fatal(err)
	}
	return raw
}
func scannerFindingPage(t *testing.T, h *scanProcessHarness, project string) []map[string]json.RawMessage {
	t.Helper()
	data := scannerFindingGET(t, h, "/v1/projects/"+project+"/findings?limit=100")
	var page struct {
		Items []map[string]json.RawMessage `json:"items"`
	}
	if err := json.Unmarshal(data, &page); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "secret-canary") {
		t.Fatal("credentials leaked in Finding metadata")
	}
	return page.Items
}
