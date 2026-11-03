//go:build e2e

package e2e

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/planner"
)

func scanFixtureRequests(h *scanProcessHarness) []scanHTTPRequest {
	h.fixture.mu.Lock()
	defer h.fixture.mu.Unlock()
	return append([]scanHTTPRequest(nil), h.fixture.requests...)
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

type retainedWebScanReport struct {
	Schema             string                `json:"schema"`
	RunID              string                `json:"run_id"`
	Scanner            string                `json:"scanner"`
	PreparationDigest  string                `json:"preparation_digest"`
	ExecutionCompleted bool                  `json:"execution_completed"`
	Limitations        []string              `json:"limitations"`
	Attempts           []planner.ScanAttempt `json:"attempts"`
	Artifacts          []struct {
		Kind   string `json:"kind"`
		Digest string `json:"digest"`
		Data   []byte `json:"data_base64"`
	} `json:"artifacts"`
}

func scanAuditTask(t *testing.T, h *scanProcessHarness, auditID string) basicTask {
	t.Helper()
	var page struct {
		Items []basicTask `json:"items"`
	}
	auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+auditID+"/tasks?limit=64", &page)
	if len(page.Items) != 1 {
		t.Fatalf("web target Tasks=%+v", page)
	}
	return page.Items[0]
}

func assertWebAuditRetention(t *testing.T, h *scanProcessHarness, audit auditProgramAudit, completed bool) retainedWebScanReport {
	t.Helper()
	task := scanAuditTask(t, h, audit.AuditID)
	if task.Attempts != 1 || task.LastExecution == nil || task.State != "settled" || task.Document.Hypothesis != nil {
		t.Fatalf("ordinary target Task=%+v", task)
	}
	var runID string
	if err := h.pool.QueryRow(h.ctx, `SELECT run_id FROM audit_executions WHERE execution_id=$1`, *task.LastExecution).Scan(&runID); err != nil {
		t.Fatal(err)
	}
	if task.Result == nil {
		if completed || task.GapCode == nil {
			t.Fatalf("missing retained report or explicit gap: %+v", task)
		}
		deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, runID)
		return retainedWebScanReport{Limitations: []string{*task.GapCode}}
	}
	for _, slot := range []string{"target", "settings"} {
		material := task.Document.Inputs[slot]
		if material.ScopeKind != "audit" || material.ScopeID != audit.AuditID || material.Ref.Revision == nil || !strings.HasPrefix(material.Digest, "sha256:") {
			t.Fatalf("exact assigned %s lost: %+v", slot, material)
		}
	}
	data := commonScanArtifact(t, h, audit.AuditID, task)
	var report retainedWebScanReport
	if err := json.Unmarshal(data, &report); err != nil || report.Schema != "contractor.scan.operation-report.v1" || report.Scanner != "nuclei" || report.RunID != runID || completed != report.ExecutionCompleted {
		t.Fatalf("ordinary scan report=%+v %v", report, err)
	}
	jobs := 0
	for _, attempt := range report.Attempts {
		for _, job := range attempt.State.Jobs {
			if job.Status != planner.ScanJobPending {
				jobs++
			}
		}
	}
	if jobs > 1 || completed && jobs != 1 {
		t.Fatalf("scanner jobs=%d completed=%v", jobs, completed)
	}
	for _, artifact := range report.Artifacts {
		if contentdigest.Bytes(artifact.Data) != artifact.Digest {
			t.Fatal("retained report bytes changed")
		}
	}
	if completed {
		var workflow string
		var allocations int
		if err := h.pool.QueryRow(h.ctx, `SELECT r.workflow_name,count(a.allocation_id) FROM workflow_runs r JOIN stage_executions s USING(run_id) LEFT JOIN stage_allocations a USING(stage_execution_id) WHERE r.run_id=$1 GROUP BY r.workflow_name`, runID).Scan(&workflow, &allocations); err != nil || workflow != "audit-web-nuclei-scan" || allocations == 0 {
			t.Fatalf("ordinary Workflow allocation: %s/%d %v", workflow, allocations, err)
		}
	}
	before := len(scanFixtureRequests(h))
	deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, runID)
	if after := commonScanArtifact(t, h, audit.AuditID, task); !bytes.Equal(data, after) || before != len(scanFixtureRequests(h)) {
		t.Fatal("source Run deletion changed exact report or repeated target work")
	}
	var receipts int
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM audit_executions WHERE audit_id=$1 AND collection_receipt_id IS NOT NULL`, audit.AuditID).Scan(&receipts); err != nil || receipts != 1 {
		t.Fatalf("collection receipts=%d %v", receipts, err)
	}
	return report
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
