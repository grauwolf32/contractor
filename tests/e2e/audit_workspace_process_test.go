//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Exercise the existing Project publication owner over a collected ordinary
// Task output. No fixture SQL manufactures outputs, Tasks or receipts.
func TestAuditWorkspaceAcrossProductionProcesses(t *testing.T) {
	h := startConfiguredFindingsHarness(t, []domainGatewayStage{basicReportStage(), basicConclusionStage(false)}, func(root string) { installBasicAuditProgram(t, root) })
	project := createProjectResource(t, h.client, h.baseURL)
	source := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "source", "application/zip", auditProgramZip(t, map[string][]byte{"app.py": []byte(findingsSource)}))
	report := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "report", "text/plain", []byte("Inspect the selected source and author a review report.\n"))
	draft, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": "basic-process", "version": "1"}, "inputs": map[string]artifactRef{"source": source, "report": report}}, "draft", 0, http.StatusCreated)
	id := draft["auditId"].(string)
	blocked := h.gateway.blockNextRequest()
	basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/start", nil, "start", uint64(draft["revision"].(float64)), http.StatusOK)
	select {
	case <-blocked:
	case <-h.ctx.Done():
		t.Fatal("ordinary producer did not reach Runtime")
	}
	current := basicAuditGet(t, h, id)
	basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/pause", nil, "pause", uint64(current["revision"].(float64)), http.StatusOK)
	h.gateway.releaseBlockedRequest()
	first := waitBasicTask(t, h, id, 1, "settled")[0]
	if first.Result == nil {
		t.Fatal("ordinary output missing")
	}
	result := first.Result
	revision := result.Ref.Revision
	request := auditservice.MaterialPublicationRequest{AuditID: id, Source: contracts.ArtifactRef{Namespace: result.Ref.Namespace, Name: result.Ref.Name, Revision: revision}, Target: contracts.ArtifactRef{Namespace: "materials", Name: "selected"}}
	published := commonWorkspacePublication(t, h, project.ProjectID, "publish", request, nil, http.StatusCreated)
	original := published
	pool, err := pgxpool.New(h.ctx, h.databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	t.Run("handoff", func(t *testing.T) {
		current = basicAuditGet(t, h, id)
		work := map[string]any{"work": []auditcontract.PreparedSelection{{Key: "explicit-consumer", Workflow: "check", Objective: "Read the exact retained output", Inputs: map[string]auditcontract.Material{"material": *result}}}}
		basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/tasks", work, "consumer", uint64(current["revision"].(float64)), http.StatusOK)
		current = basicAuditGet(t, h, id)
		basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/resume", nil, "resume", uint64(current["revision"].(float64)), http.StatusOK)
		tasks := waitBasicTask(t, h, id, 2, "settled")
		if tasks[1].Result == nil || tasks[1].GapCode != nil || h.gateway.CompletedStages() != 2 {
			t.Fatal("ordinary exact-input consumer failed")
		}
		if !bytes.Equal(basicTaskArtifact(t, h, id, first, "material"), []byte(findingsEvidence)) {
			t.Fatal("handoff changed bytes")
		}
	})
	t.Run("replay", func(t *testing.T) {
		replay := commonWorkspacePublication(t, h, project.ProjectID, "publish", request, nil, http.StatusOK)
		if !replay.Artifact.SameExact(original.Artifact) {
			t.Fatal("replay changed publication identity")
		}
	})
	t.Run("cas", func(t *testing.T) {
		commonWorkspacePublication(t, h, project.ProjectID, "collision", request, nil, http.StatusConflict)
		before := published.Artifact.Revision
		published = commonWorkspacePublication(t, h, project.ProjectID, "replace", request, before, http.StatusOK)
		commonWorkspacePublication(t, h, project.ProjectID, "stale", request, before, http.StatusConflict)
	})
	t.Run("authority", func(t *testing.T) {
		foreign := request
		foreign.AuditID = "foreign-audit"
		commonWorkspacePublication(t, h, project.ProjectID, "foreign", foreign, nil, http.StatusNotFound)
		missing := "missing"
		foreign = request
		foreign.Source.Revision = &missing
		commonWorkspacePublication(t, h, project.ProjectID, "missing", foreign, nil, http.StatusNotFound)
	})
	t.Run("publication", func(t *testing.T) {
		rows, err := pool.Query(h.ctx, "SELECT run_id FROM audit_executions WHERE audit_id=$1", id)
		if err != nil {
			t.Fatal(err)
		}
		var runs []string
		for rows.Next() {
			var run string
			if err := rows.Scan(&run); err != nil {
				t.Fatal(err)
			}
			runs = append(runs, run)
		}
		rows.Close()
		if rows.Err() != nil {
			t.Fatal(rows.Err())
		}
		for _, run := range runs {
			deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, run)
		}
		current = basicAuditGet(t, h, id)
		req, _ := http.NewRequestWithContext(h.ctx, http.MethodDelete, h.baseURL+"/v1/audits/"+id, nil)
		req.Header.Set("Authorization", "Bearer "+publicToken)
		req.Header.Set("If-Match", fmt.Sprintf("\"%d\"", uint64(current["revision"].(float64))))
		req.Header.Set("Idempotency-Key", "delete-common-workspace")
		response := do(t, h.client, req, http.StatusAccepted)
		response.Body.Close()
		for {
			var n int
			if err := pool.QueryRow(h.ctx, "SELECT count(*) FROM audits WHERE audit_id=$1", id).Scan(&n); err != nil {
				t.Fatal(err)
			}
			if n == 0 {
				break
			}
			basicPoll(t, h)
		}
		replay := commonWorkspacePublication(t, h, project.ProjectID, "publish", request, nil, http.StatusOK)
		if !replay.Artifact.SameExact(original.Artifact) {
			t.Fatal("Audit purge changed original Project receipt")
		}
		data, _ := download(t, h.client, fmt.Sprintf("%s/v1/projects/%s/artifacts/%s/%s?revision=%s", h.baseURL, project.ProjectID, published.Artifact.Namespace, published.Artifact.Name, *published.Artifact.Revision))
		if !bytes.Equal(data, []byte(findingsEvidence)) {
			t.Fatal("Project material lost its independent bytes")
		}
	})
}

func commonWorkspacePublication(t *testing.T, h *findingsHarness, project, key string, request auditservice.MaterialPublicationRequest, expected *string, status int) auditservice.MaterialPublication {
	t.Helper()
	data, _ := json.Marshal(request)
	req, _ := http.NewRequestWithContext(h.ctx, http.MethodPost, h.baseURL+"/v1/projects/"+project+"/audit-material-publications", bytes.NewReader(data))
	req.Header.Set("Authorization", "Bearer "+publicToken)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Idempotency-Key", key)
	if expected != nil {
		req.Header.Set("If-Match", fmt.Sprintf("\"%s\"", *expected))
	}
	response := do(t, h.client, req, status)
	defer response.Body.Close()
	var result auditservice.MaterialPublication
	if status < 300 {
		if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
			t.Fatal(err)
		}
	}
	return result
}
