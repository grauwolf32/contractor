//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
	"go.yaml.in/yaml/v4"
)

// One ordinary model Worker is a deterministic latch before the ordinary
// preparation operation. Pause therefore precedes publication without a new
// Audit waiting state or a timing assertion.
func TestAuditCommonPrepared256PauseRestartAndCancelAcrossProcesses(t *testing.T) {
	h := startConfiguredFindingsHarness(t, []domainGatewayStage{basicReportStage()}, func(root string) { installCommonPrepared256Program(t, root) })
	pool, err := pgxpool.New(h.ctx, h.databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	project := createProjectResource(t, h.client, h.baseURL)
	paths := map[string]any{}
	for index := range 256 {
		paths[fmt.Sprintf("/resource-%03d", index)] = map[string]any{"get": map[string]any{"operationId": fmt.Sprintf("getResource%03d", index), "responses": map[string]any{"200": map[string]string{"description": "Selected resource"}}}}
	}
	contract, _ := json.Marshal(map[string]any{"openapi": "3.1.0", "info": map[string]string{"title": "Bounded work fixture", "version": "1"}, "paths": paths})
	inputs := map[string]artifactRef{
		"openapi": uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "openapi", "application/json", contract),
		"signal":  uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "signal", "text/plain", []byte("Ordinary Worker completed before preparation")),
		"archive": uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "archive", "application/zip", projectSourceArchive(t)),
	}
	draft, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": "prepared-256", "version": "1"}, "inputs": inputs, "scope": map[string]string{"objective": "Accept bounded operation work during Pause"}}, "draft", 0, http.StatusCreated)
	id := draft["auditId"].(string)
	blocked := h.gateway.blockNextRequest()
	basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/start", nil, "start", uint64(draft["revision"].(float64)), http.StatusOK)
	select {
	case <-blocked:
	case <-h.ctx.Done():
		t.Fatal("preparation Worker never reached the latch")
	}
	audit := basicAuditGet(t, h, id)
	basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/pause", nil, "pause", uint64(audit["revision"].(float64)), http.StatusOK)
	h.gateway.releaseBlockedRequest()
	for {
		var total, ready, settled int
		err := pool.QueryRow(h.ctx, `SELECT count(*),count(*) FILTER(WHERE state='ready'),count(*) FILTER(WHERE state='settled') FROM audit_tasks WHERE audit_id=$1`, id).Scan(&total, &ready, &settled)
		if err != nil {
			t.Fatal(err)
		}
		if total == 257 && ready == 256 && settled == 1 {
			break
		}
		if len(h.gateway.Failures()) > 0 {
			t.Fatalf("ordinary producer failed: %v", h.gateway.Failures())
		}
		basicPoll(t, h)
	}
	tasks := commonPreparedTaskPages(t, h, id)
	if len(tasks) != 257 || tasks[0].Result == nil || tasks[0].GapCode != nil {
		t.Fatalf("accepted count=%d producer=%+v", len(tasks), tasks[0])
	}
	identities := map[string]bool{}
	for index, task := range tasks {
		if identities[task.Document.TaskID] || task.Document.Hypothesis != nil {
			t.Fatal("duplicate or invented hypothesis")
		}
		identities[task.Document.TaskID] = true
		if index > 0 && (task.State != "ready" || task.Attempts != 0 || task.LastExecution != nil || task.Document.Inputs["operation"].ScopeID != id || task.Document.Inputs["operation"].ScopeKind != "audit") {
			t.Fatalf("paused work dispatched or lacks exact local operation: %+v", task)
		}
	}
	work := basicTaskArtifact(t, h, id, tasks[0], "work")
	var list struct {
		Schema string            `json:"schema"`
		Work   []json.RawMessage `json:"work"`
	}
	if err := json.Unmarshal(work, &list); err != nil || len(list.Work) != 256 || list.Schema != "contractor.audit.prepared-work.v1" {
		t.Fatalf("ordinary bounded work=%d %v", len(list.Work), err)
	}
	h.restartServer()
	audit = basicAuditGet(t, h, id)
	if audit["state"] != "paused" || int(audit["submittedRunCount"].(float64)) != 1 || len(commonPreparedTaskPages(t, h, id)) != 257 {
		t.Fatal("restart duplicated intake or resumed dispatch")
	}
	basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/cancel", nil, "cancel", uint64(audit["revision"].(float64)), http.StatusAccepted)
	for {
		audit = basicAuditGet(t, h, id)
		if audit["state"] == "cancelled" {
			break
		}
		basicPoll(t, h)
	}
	after := commonPreparedTaskPages(t, h, id)
	if len(after) != 257 || int(audit["submittedRunCount"].(float64)) != 1 || h.gateway.CompletedStages() != 1 {
		t.Fatal("cancellation lost accepted history or dispatched operation Runs")
	}
	for index, task := range after {
		if task.State != "settled" || !identities[task.Document.TaskID] || index > 0 && task.Result != nil {
			t.Fatal("cancellation invented a result or changed Task identity")
		}
	}
	var runID string
	if err := pool.QueryRow(h.ctx, `SELECT run_id FROM audit_executions WHERE execution_id=$1`, *tasks[0].LastExecution).Scan(&runID); err != nil {
		t.Fatal(err)
	}
	deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, runID)
	if retained := basicTaskArtifact(t, h, id, tasks[0], "work"); !bytes.Equal(retained, work) {
		t.Fatal("producer purge changed the retained intake source")
	}
	assertCommonAuditEventContract(t, h, id)
}
func commonPreparedTaskPages(t *testing.T, h *findingsHarness, id string) []basicTask {
	t.Helper()
	result := []basicTask{}
	cursor := ""
	for {
		path := "/v1/audits/" + id + "/tasks?limit=64"
		if cursor != "" {
			path += "&cursor=" + url.QueryEscape(cursor)
		}
		response, _ := basicAuditRequest(t, h, http.MethodGet, path, nil, "", 0, http.StatusOK)
		var page struct {
			Items []basicTask `json:"items"`
			Page  struct {
				NextCursor *string `json:"nextCursor"`
			} `json:"page"`
		}
		if err := json.Unmarshal(mustBasicJSON(t, response), &page); err != nil {
			t.Fatal(err)
		}
		result = append(result, page.Items...)
		if len(result) > 257 {
			t.Fatal("prepared work exceeds its bound")
		}
		if page.Page.NextCursor == nil {
			return result
		}
		if *page.Page.NextCursor == cursor {
			t.Fatal("Task page cursor did not advance")
		}
		cursor = *page.Page.NextCursor
	}
}
func installCommonPrepared256Program(t *testing.T, root string) {
	t.Helper()
	installBasicAuditProgram(t, root)
	read := func(relative string) map[string]any {
		data, err := os.ReadFile(filepath.Join(root, relative))
		if err != nil {
			t.Fatal(err)
		}
		var value map[string]any
		if err := yaml.Unmarshal(data, &value); err != nil {
			t.Fatal(err)
		}
		return value
	}
	copy := read("workflows/basic_copy.yaml")
	operation := read("workflows/openapi_prepare_work.yaml")
	copy["metadata"].(map[string]any)["name"] = "prepared-256-workflow"
	spec, opSpec := copy["spec"].(map[string]any), operation["spec"].(map[string]any)
	spec["parameters"] = opSpec["parameters"]
	spec["inputs"].(map[string]any)["openapi"] = map[string]any{"required": true, "mediaTypes": []string{"application/json"}}
	outputs := spec["outputs"].(map[string]any)
	outputs["result"].(map[string]any)["primary"] = false
	outputs["work"] = opSpec["outputs"].(map[string]any)["work"]
	stages := spec["stages"].(map[string]any)
	stages["copy"].(map[string]any)["on"].(map[string]any)["succeeded"] = map[string]any{"next": "prepare"}
	stages["prepare"] = opSpec["stages"].(map[string]any)["prepare"]
	data, err := yaml.Marshal(copy)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "workflows/prepared_256.yaml"), data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "audit-profiles/prepared_256.yaml"), []byte(commonPrepared256Program), 0600); err != nil {
		t.Fatal(err)
	}
}

const commonPrepared256Program = `apiVersion: contractor/v1alpha1
kind: AuditProfile
metadata: {name: prepared-256, version: "1"}
spec:
  schema: contractor.audit.program.v1
  inputs:
    archive: {required: true, media_types: [application/zip]}
    openapi: {required: true, media_types: [application/json]}
    signal: {required: true, media_types: [text/plain]}
  workflows:
    prepare:
      ref: prepared-256-workflow@1
      inputs: {source: {source: audit-input, name: signal}, openapi: {source: audit-input, name: openapi}}
      parameters: {checking_workflow: inspect, selection: operations, max_items: "256"}
      outputs: {work: work}
    inspect:
      ref: audit-openapi-operation-observe@1
      inputs: {source: {source: audit-input, name: archive}, openapi: {source: task-input}, operation: {source: task-input}, context: {source: task-context}}
      parameters: {}
      outputs: {result: result}
  hypotheses: {}
  initial_work: [{key: prepare, workflow: prepare, objective: Prepare the selected bounded operation list}]
  limits:
    max_tasks: 257
    max_iterations: 1
    max_tasks_per_iteration: 257
    batch_size: 1
    max_submitted_runs: 257
    max_attempts: 2
    deadline_seconds: 600
    max_evidence_bytes: 67108864
    max_hypotheses: 1
    max_hypothesis_revisions: 1
    max_hypothesis_bytes: 32768
  policy: {active_operations: prohibited}
`
