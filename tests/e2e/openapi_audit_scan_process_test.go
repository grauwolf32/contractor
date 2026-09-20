//go:build e2e

package e2e

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/planner"
)

// This gate uses the public Audit API and real Server, Scheduler, Runtime and
// scanners. Database faults affect persistence only; they never emulate a scan.
func TestOpenAPIAuditScanAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Fatal("the OpenAPI Audit release gate cannot run in short mode")
	}
	h := startConfiguredScanStack(t, "", func(h *scanProcessHarness, root string) {
		for _, relative := range []string{
			"agent-templates/audit_sqlmap_scan.yaml", "agent-templates/audit_nuclei_scan.yaml",
			"workflows/audit_openapi_sqlmap_scan.yaml", "workflows/audit_openapi_nuclei_scan.yaml",
			"audit-profiles/openapi_sqlmap_scan.yaml", "audit-profiles/openapi_nuclei_scan.yaml",
			"instructions/audit-openapi-scan.md",
		} {
			data, err := os.ReadFile(filepath.Join(h.repositoryRoot, "configs", relative))
			if err != nil {
				t.Fatal(err)
			}
			target := filepath.Join(root, relative)
			if err := os.MkdirAll(filepath.Dir(target), 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(target, data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		replaceScanConfig(t, filepath.Join(root, "agent-templates/audit_nuclei_scan.yaml"), "http-missing-security-headers", "contractor-local-fixture")
	}, "nuclei", "sqlmap")

	t.Run("sqlmap_exact_request_and_retained_package", func(t *testing.T) {
		before := scanFixtureRequests(h)
		audit := startOpenAPIScanAudit(t, h, "sqlmap", "", nil)
		waitScanAudit(t, h, audit.AuditID, 1)
		pkg := assertScanAuditRetention(t, h, audit, "sqlmap", true, 1)
		journal, _ := pkg.Package.MemberByID("scan-journal")
		if bytes.Contains(journal.Data(), []byte("local-fixture-token")) {
			t.Fatal("credentials copied into retained assignment")
		}
		requests := scanFixtureRequests(h)[len(before):]
		exact, probed := false, false
		for _, request := range requests {
			if request.method != "POST" || request.authorization != "Bearer local-fixture-token" || !strings.HasPrefix(request.uri, "/api/pets/7?") {
				t.Fatalf("scanner changed assigned method/path/auth: %+v", request)
			}
			if request.uri == "/api/pets/7?search=Milo" && request.body == `{"name":"Milo"}` {
				exact = true
			}
			if request.uri != "/api/pets/7?search=Milo" || request.body != `{"name":"Milo"}` {
				probed = true
			}
		}
		if !exact || !probed {
			t.Fatalf("exact request/probing = %v/%v (%d requests)", exact, probed, len(requests))
		}
	})
	t.Run("nuclei_url_coverage_and_pinned_template", func(t *testing.T) {
		audit := startOpenAPIScanAudit(t, h, "nuclei", "", nil)
		waitScanAudit(t, h, audit.AuditID, 1)
		pkg := assertScanAuditRetention(t, h, audit, "nuclei", true, 1)
		for _, gap := range []string{"url_template_scan_only", "http_method_not_replayed", "request_body_not_replayed", "authentication_not_applied"} {
			if !slices.Contains(pkg.Results.Results[0].Coverage.Gaps, gap) {
				t.Fatalf("missing Nuclei limitation %s: %+v", gap, pkg.Results.Results[0])
			}
		}
		found := false
		for _, evidence := range pkg.Evidence.Evidence {
			if evidence.Kind != "scanner-report" {
				continue
			}
			member, _ := pkg.Package.MemberByID(evidence.ContentMemberID)
			var report scanProcessReport
			if err := json.Unmarshal(member.Data(), &report); err != nil {
				t.Fatal(err)
			}
			results, ok := report.Observation["results"].([]any)
			if !ok || len(results) != 1 || results[0].(map[string]any)["template-id"] != "contractor-local-fixture" {
				t.Fatalf("Nuclei template evidence = %+v", report)
			}
			found = true
		}
		if !found {
			t.Fatal("missing retained Nuclei report")
		}
	})
	t.Run("missing_concrete_input_never_dispatches", func(t *testing.T) {
		before := len(scanFixtureRequests(h))
		audit := startOpenAPIScanAudit(t, h, "sqlmap", "", func(settings map[string]any) { delete(settings, "authentication") })
		waitScanAudit(t, h, audit.AuditID, 1)
		pkg := assertScanAuditRetention(t, h, audit, "sqlmap", false, 1)
		if len(scanFixtureRequests(h)) != before || len(pkg.Results.Results[0].Coverage.Gaps) == 0 {
			t.Fatal("unprepared operation dispatched or lost its gap")
		}
	})
	t.Run("lost_journal_ack_does_not_repeat_scan", func(t *testing.T) {
		installScanFault(t, h, "scan_journal", "planner_sessions", "UPDATE", `IF NEW.state #>> '{scan,state,jobs,0,status}' = 'completed' THEN RAISE EXCEPTION 'controlled completed journal acknowledgement loss'; END IF;`)
		audit := startOpenAPIScanAudit(t, h, "nuclei", "", nil)
		waitScanAudit(t, h, audit.AuditID, 1)
		pkg := assertScanAuditRetention(t, h, audit, "nuclei", false, 1)
		if !slices.Contains(pkg.Results.Results[0].Coverage.Gaps, "scan_outcome_unknown") {
			t.Fatalf("lost result was not unknown: %+v", pkg.Results)
		}
	})
	t.Run("publication_and_collection_faults_survive_restart", func(t *testing.T) {
		installScanFault(t, h, "scan_publication", "artifact_bindings", "INSERT", `IF NEW.scope_kind = 'run' AND NEW.namespace = 'scan-results' AND NEW.name LIKE 'audit-result.%' THEN RAISE EXCEPTION 'controlled result publication failure'; END IF;`)
		removeReceiptFault := installScanFault(t, h, "scan_collection", "audit_collection_receipts", "INSERT", `RAISE EXCEPTION 'controlled collection commit failure';`)
		audit := startOpenAPIScanAudit(t, h, "nuclei", "", nil)
		waitScanCondition(t, h, "uncommitted retained recovery package", func() bool {
			var count int
			err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind = 'project' AND scope_id = $1 AND name LIKE 'scan-recovery-%'`, audit.ProjectID).Scan(&count)
			if err != nil || count != 1 {
				return false
			}
			var called bool
			if err := h.pool.QueryRow(h.ctx, `SELECT is_called FROM scan_collection_hits`).Scan(&called); err != nil || !called {
				return false
			}
			var receipts int
			if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM audit_collection_receipts WHERE audit_id=$1`, audit.AuditID).Scan(&receipts); err != nil {
				t.Fatal(err)
			}
			if receipts != 0 {
				t.Fatal("collection fault unexpectedly committed a receipt")
			}
			return true
		})
		before := len(scanFixtureRequests(h))
		h.restartServer()
		removeReceiptFault()
		waitScanAudit(t, h, audit.AuditID, 1)
		pkg := assertScanAuditRetention(t, h, audit, "nuclei", true, 1)
		if len(scanFixtureRequests(h)) != before || len(pkg.Results.Results[0].Coverage.Completed) != 1 {
			t.Fatal("collection replay repeated the scan or lost completed coverage")
		}
	})
	t.Run("known_failure_has_bounded_cross_run_retries", func(t *testing.T) {
		template := filepath.Join(h.temporaryRoot, "nuclei-templates", "fixture.yaml")
		original, err := os.ReadFile(template)
		if err != nil {
			t.Fatal(err)
		}
		// A real scanner rejects this template. Its known failure may be retried,
		// while successful and unknown scans in the other cases must not be repeated.
		if err := os.WriteFile(template, []byte("id: contractor-local-fixture\ninfo: [invalid]\n"), 0600); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := os.WriteFile(template, original, 0600); err != nil {
				t.Error(err)
			}
		})
		before := len(scanFixtureRequests(h))
		audit := startOpenAPIScanAudit(t, h, "nuclei", "", nil)
		waitScanAudit(t, h, audit.AuditID, 3)
		pkg := assertScanAuditRetention(t, h, audit, "nuclei", false, 3)
		if len(pkg.Results.Results[0].Coverage.Gaps) == 0 || before != len(scanFixtureRequests(h)) {
			t.Fatal("invalid template lost failure or reached target")
		}
	})
	t.Run("cancelled_scan_is_not_resubmitted", func(t *testing.T) {
		audit := startOpenAPIScanAudit(t, h, "nuclei", "/slow", nil)
		select {
		case <-h.fixture.slowStarted:
		case <-h.ctx.Done():
			t.Fatal("scanner never reached cancellation fixture")
		}
		item := scanAuditItem(t, h, audit.AuditID)
		if len(item.Attempts) != 1 || item.Attempts[0].RunID == "" {
			t.Fatalf("missing running attempt: %+v", item)
		}
		cancelWorkflowRun(t, h.client, h.baseURL, item.Attempts[0].RunID)
		waitScanAudit(t, h, audit.AuditID, 1)
		assertScanAuditRetention(t, h, audit, "nuclei", false, 1)
	})
}

func scanFixtureRequests(h *scanProcessHarness) []scanHTTPRequest {
	h.fixture.mu.Lock()
	defer h.fixture.mu.Unlock()
	return append([]scanHTTPRequest(nil), h.fixture.requests...)
}

func startOpenAPIScanAudit(t *testing.T, h *scanProcessHarness, scanner, prefix string, mutate func(map[string]any)) auditProgramAudit {
	t.Helper()
	var project projectResourceResponse
	scanAuditPOST(t, h, "/v1/projects", map[string]string{"kind": "project", "name": "Local scan fixture", "description": "Controlled acceptance target"}, 0, http.StatusCreated, &project)
	root := filepath.Join(h.repositoryRoot, "configs", "scan", "examples", "audit-openapi-scan")
	source, err := os.ReadFile(filepath.Join(root, "openapi.json"))
	if err != nil {
		t.Fatal(err)
	}
	settingsBytes, err := os.ReadFile(filepath.Join(root, scanner+"-settings.json"))
	if err != nil {
		t.Fatal(err)
	}
	var settings map[string]any
	if err := json.Unmarshal(settingsBytes, &settings); err != nil {
		t.Fatal(err)
	}
	settings["server"] = h.targetURL + prefix + "/api"
	if mutate != nil {
		mutate(settings)
	}
	settingsBytes, err = json.Marshal(settings)
	if err != nil {
		t.Fatal(err)
	}
	inputs := map[string]artifactRef{
		"openapi":  uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "openapi", "application/json", source),
		"settings": uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "settings", "application/json", settingsBytes),
	}
	var draft auditProgramAudit
	scanAuditPOST(t, h, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{
		"profile": map[string]string{"name": "openapi-" + scanner + "-scan", "version": "1"}, "inputs": inputs,
		"scope": map[string]string{"objective": "Bounded local Audit scan acceptance"},
	}, 0, http.StatusCreated, &draft)
	var started struct {
		Audit auditProgramAudit  `json:"audit"`
		Items []auditProgramItem `json:"items"`
	}
	scanAuditPOST(t, h, "/v1/audits/"+draft.AuditID+"/start", nil, draft.Revision, http.StatusOK, &started)
	if started.Audit.State != "active" || started.Audit.SubmittedRunCount != 0 || len(started.Items) != 1 {
		t.Fatalf("unexpected pre-approval Audit: %+v", started)
	}
	var reviews struct {
		Items []auditProgramReview `json:"items"`
	}
	auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+draft.AuditID+"/reviews?limit=100", &reviews)
	if len(reviews.Items) != 1 || reviews.Items[0].Kind != "active-check-approval" || reviews.Items[0].State != "pending" {
		t.Fatalf("active-check approval missing: %+v", reviews)
	}
	review := reviews.Items[0]
	scanAuditPOST(t, h, "/v1/audits/"+draft.AuditID+"/reviews/"+review.RequestID+"/decisions", map[string]string{"action": "approve", "rationale": "Permit only the controlled loopback fixture."}, review.Revision, http.StatusOK, nil)
	return started.Audit
}

func scanAuditPOST(t *testing.T, h *scanProcessHarness, path string, body any, revision uint64, status int, result any) {
	t.Helper()
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	if body == nil {
		data = nil
	}
	request, err := http.NewRequestWithContext(h.ctx, http.MethodPost, h.baseURL+path, bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", "scan-audit-"+randomHex(t, 8))
	if revision != 0 {
		request.Header.Set("If-Match", fmt.Sprintf("\"%d\"", revision))
	}
	response := do(t, h.client, request, status)
	defer response.Body.Close()
	if result != nil {
		decodeAuditProgramResponse(t, response, result)
	}
}

func waitScanCondition(t *testing.T, h *scanProcessHarness, description string, ready func() bool) {
	t.Helper()
	ctx, cancel := context.WithTimeout(h.ctx, 150*time.Second)
	defer cancel()
	ticker := time.NewTicker(150 * time.Millisecond)
	defer ticker.Stop()
	for !ready() {
		select {
		case <-ctx.Done():
			t.Fatalf("waiting for %s: %v\nserver: %s\nruntime: %s", description, ctx.Err(), h.server.logs.redacted(publicToken), h.runtime.logs.redacted(publicToken))
		case <-ticker.C:
		}
	}
}

func waitScanAudit(t *testing.T, h *scanProcessHarness, auditID string, runs int) {
	t.Helper()
	waitScanCondition(t, h, "completed Audit "+auditID, func() bool {
		var audit auditProgramAudit
		if auditProgramTryGET(h.ctx, h.client, h.baseURL+"/v1/audits/"+auditID, &audit) != nil {
			return false
		}
		if audit.State == "failed" || audit.State == "cancelled" {
			t.Fatalf("unexpected Audit terminal state: %+v", audit)
		}
		if audit.State != "completed" {
			return false
		}
		if audit.SubmittedRunCount != runs || audit.OutstandingRuns != 0 {
			t.Fatalf("unexpected cross-Run counters: %+v", audit)
		}
		return true
	})
}

func scanAuditItem(t *testing.T, h *scanProcessHarness, auditID string) auditProgramItem {
	t.Helper()
	var page struct {
		Items []auditProgramItem `json:"items"`
	}
	auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+auditID+"/items?limit=100", &page)
	if len(page.Items) != 1 {
		t.Fatalf("Audit items = %+v", page)
	}
	return page.Items[0]
}

func assertScanAuditRetention(t *testing.T, h *scanProcessHarness, audit auditProgramAudit, scanner string, completed bool, runs int) auditdomain.CheckResultPackage {
	t.Helper()
	item := scanAuditItem(t, h, audit.AuditID)
	if len(item.Attempts) != runs {
		t.Fatalf("unexpected Audit item attempts: %+v", item)
	}
	var retained []byte
	if err := h.pool.QueryRow(h.ctx, `SELECT retained_refs FROM audit_collection_receipts WHERE audit_id = $1 ORDER BY created_at DESC LIMIT 1`, audit.AuditID).Scan(&retained); err != nil {
		t.Fatal(err)
	}
	var links []auditstore.ArtifactLink
	if err := json.Unmarshal(retained, &links); err != nil {
		t.Fatal(err)
	}
	var resultLink auditstore.ArtifactLink
	for _, link := range links {
		if strings.HasPrefix(link.LogicalKey, "result/") {
			resultLink = link
		}
	}
	if resultLink.Artifact.Ref.Revision == nil {
		t.Fatalf("receipt has no retained package: %s", retained)
	}
	scope, err := artifacts.NewService(artifacts.NewPostgresRepository(h.pool)).Project(audit.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	read, err := scope.Read(h.ctx, resultLink.Artifact.Ref)
	if err != nil {
		t.Fatal(err)
	}
	pkg, err := auditdomain.DecodeCheckResultPackage(read.Payload.Data)
	if err != nil {
		t.Fatal(err)
	}
	if len(pkg.Results.Results) != 1 {
		t.Fatalf("unexpected canonical results: %+v", pkg.Results)
	}
	journalMember, exists := pkg.Package.MemberByID("scan-journal")
	if !exists {
		t.Fatal("canonical result omitted durable scan history")
	}
	var journal struct {
		Task     auditdomain.ItemTask  `json:"task"`
		Attempts []planner.ScanAttempt `json:"attempts"`
	}
	if err := json.Unmarshal(journalMember.Data(), &journal); err != nil {
		t.Fatal(err)
	}
	task := journal.Task
	if task.Scan == nil || task.Scan.Scanner != scanner || task.Scan.Operation != "#/paths/~1pets~1{id}/post" || task.SourceRef.Revision == nil || task.Scan.Settings.Ref.Revision == nil || !strings.HasPrefix(task.SourceContentDigest, "sha256:") || !strings.HasPrefix(task.Scan.Settings.Digest, "sha256:") || !strings.HasPrefix(task.Scan.PreparationDigest, "sha256:") {
		t.Fatalf("incomplete exact scan provenance: %+v", task)
	}
	jobs := 0
	for _, attempt := range journal.Attempts {
		for _, job := range attempt.State.Jobs {
			if job.Status != planner.ScanJobPending {
				jobs++
			}
		}
	}
	if runs == 1 && jobs > 1 {
		t.Fatalf("scanner dispatched %d jobs across Stage attempts", jobs)
	}
	if completed && jobs != 1 {
		t.Fatalf("completed scanner has %d journal jobs", jobs)
	}
	result := pkg.Results.Results[0]
	requirement := "sqlmap-request-scan"
	if scanner == "nuclei" {
		requirement = "nuclei-url-template-scan"
	}
	if result.ItemKey != item.ItemKey || !slices.Equal(result.Coverage.Requested, []string{requirement}) {
		t.Fatalf("incorrect assigned coverage: %+v", result)
	}
	if completed != slices.Contains(result.Coverage.Completed, requirement) {
		t.Fatalf("completed=%v; coverage=%+v", completed, result.Coverage)
	}
	if result.Assessment != "inconclusive" && result.Assessment != "not-tested" {
		t.Fatalf("scanner fabricated a security verdict: %+v", result)
	}
	var report auditProgramReport
	auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+audit.AuditID+"/report", &report)
	if len(report.Machine) == 0 {
		t.Fatal("Audit report missing before Run deletion")
	}
	before := len(scanFixtureRequests(h))
	for _, attempt := range item.Attempts {
		deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, attempt.RunID)
	}
	deleted := scanAuditItem(t, h, audit.AuditID)
	for _, attempt := range deleted.Attempts {
		if !attempt.RunDeleted {
			t.Fatalf("Run tombstone missing: %+v", deleted)
		}
	}
	after, err := scope.Read(h.ctx, resultLink.Artifact.Ref)
	if err != nil || !bytes.Equal(read.Payload.Data, after.Payload.Data) {
		t.Fatalf("retained package changed after Run deletion: %v", err)
	}
	var afterReport auditProgramReport
	auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+audit.AuditID+"/report", &afterReport)
	if !bytes.Equal(report.Machine, afterReport.Machine) || before != len(scanFixtureRequests(h)) {
		t.Fatal("Run deletion changed report or repeated scanner action")
	}
	var receipts int
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM audit_collection_receipts WHERE audit_id=$1`, audit.AuditID).Scan(&receipts); err != nil || receipts != runs {
		t.Fatalf("collection receipts=%d want %d: %v", receipts, runs, err)
	}
	return pkg
}

func installScanFault(t *testing.T, h *scanProcessHarness, name, table, event, body string) func() {
	t.Helper()
	statement := fmt.Sprintf(`CREATE SEQUENCE %s_hits; CREATE FUNCTION %s() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN PERFORM nextval('%s_hits'); %s RETURN NEW; END $$; CREATE TRIGGER %s BEFORE %s ON %s FOR EACH ROW EXECUTE FUNCTION %s()`, name, name, name, body, name, event, table, name)
	if _, err := h.pool.Exec(h.ctx, statement); err != nil {
		t.Fatal(err)
	}
	removed := false
	remove := func() {
		if removed {
			return
		}
		if _, err := h.pool.Exec(context.Background(), fmt.Sprintf(`DROP TRIGGER %s ON %s; DROP FUNCTION %s(); DROP SEQUENCE %s_hits`, name, table, name, name)); err != nil {
			t.Error(err)
		}
		removed = true
	}
	t.Cleanup(remove)
	return remove
}
