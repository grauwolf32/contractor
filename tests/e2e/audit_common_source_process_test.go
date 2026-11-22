//go:build e2e

package e2e

import (
	"encoding/json"
	"github.com/jackc/pgx/v5/pgxpool"
	"net/http"
	"os/exec"
	"testing"
)

func TestAuditCommonSourcePreparationAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Fatal("source preparation process gate cannot run in short mode")
	}
	if _, err := exec.LookPath("sqlmap"); err != nil {
		t.Fatal("real sqlmap is required", err)
	}
	target := startEngagementScanTarget(t, &scanHTTPFixture{})
	defer target.Close()
	stages := domainGatewayStages()[:4]
	// The shipped source program needs no authored OpenAPI. Exercise the ordinary
	// builder's supported creation tool before its normal edits and validation.
	stages[2].steps[3] = toolGatewayStep("initialize_openapi", fixedArguments(map[string]any{"title": "Widget Service", "version": "1.0.0", "target_name": "openapi", "expected_revision": nil}))
	original := stages[2].steps[5].arguments
	stages[2].steps[5].arguments = func(request map[string]any) (map[string]any, error) {
		arguments, err := original(request)
		if err != nil {
			return nil, err
		}
		get := arguments["path_item"].(map[string]any)["get"].(map[string]any)
		get["parameters"] = append(get["parameters"].([]any), map[string]any{"name": "search", "in": "query", "schema": map[string]any{"type": "string"}})
		return arguments, nil
	}
	h := startConfiguredFindingsHarness(t, stages, nil, func(h *findingsHarness) {
		h.runtimeExtraArguments = []string{"--allowed-target-network", "127.0.0.0/8"}
	})
	project, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/projects", map[string]any{"kind": "project", "name": "Source preparation fixture", "activeChecks": "automatic"}, "project", 0, http.StatusCreated)
	projectID := project["projectId"].(string)
	source := uploadProjectScopeArtifact(t, h.client, h.baseURL, projectID, "fixture", "source", "application/zip", projectSourceArchive(t))
	values, _ := json.Marshal(map[string]any{"#/paths/~1widgets~1{widget_id}/get": map[string]any{"parameters": map[string]any{"path:widget_id": "7", "query:search": "Milo"}}})
	options := uploadProjectScopeArtifact(t, h.client, h.baseURL, projectID, "fixture", "request-values", "application/json", values)
	engagement := uploadScanEngagement(t, h.client, h.baseURL, projectID, target.URL, false)
	parameters := map[string]string{"server": target.URL, "operations": `["#/paths/~1widgets~1{widget_id}/get"]`, "test_parameters": `["search"]`}
	draft, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/projects/"+projectID+"/audits", map[string]any{"profile": map[string]string{"name": "source-openapi-sqlmap-scan", "version": "3"}, "inputs": map[string]artifactRef{"source": source, "request_values": options, "engagement": engagement}, "parameters": parameters, "scope": map[string]string{"objective": "Generate and validate the contract, then scan its selected concrete operation"}}, "draft", 0, http.StatusCreated)
	id := draft["auditId"].(string)
	basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/start", nil, "start", uint64(draft["revision"].(float64)), http.StatusOK)
	tasks := waitBasicTask(t, h, id, 2, "settled")
	if tasks[0].Document.Workflow != "prepare" || tasks[0].GapCode != nil || tasks[1].Document.Workflow != "scan" || tasks[1].GapCode != nil || tasks[1].Findings == nil || tasks[1].Result != nil || tasks[1].Document.Hypothesis != nil {
		t.Fatalf("source preparation work=%+v", tasks)
	}
	for {
		audit := basicAuditGet(t, h, id)
		if audit["state"] == "completed" {
			if int(audit["submittedRunCount"].(float64)) != 2 {
				t.Fatal("source preparation/scan repeated")
			}
			break
		}
		basicPoll(t, h)
	}
	// The scan report is an ordinary supporting material; it has execution facts
	// and exact embedded evidence, with no mandatory security verdict.
	var report struct {
		Schema    string `json:"schema"`
		Completed bool   `json:"execution_completed"`
		Artifacts []struct {
			Kind string `json:"kind"`
		} `json:"artifacts"`
	}
	pool, err := pgxpool.New(h.ctx, h.databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	var runID string
	var encoded []byte
	if err := pool.QueryRow(h.ctx, `SELECT e.run_id,s.accepted_stage_result->'artifacts'->'report' FROM audit_executions e JOIN stage_executions s USING(run_id) WHERE e.execution_id=$1 AND s.accepted_stage_result->'artifacts' ? 'report' LIMIT 1`, *tasks[1].LastExecution).Scan(&runID, &encoded); err != nil {
		t.Fatal(err)
	}
	var ref artifactRef
	if err := json.Unmarshal(encoded, &ref); err != nil {
		t.Fatal(err)
	}
	data, _ := download(t, h.client, h.baseURL+"/v1/runs/"+runID+"/artifacts/"+ref.Namespace+"/"+ref.Name+"?revision="+*ref.Revision)
	if err := json.Unmarshal(data, &report); err != nil || report.Schema != "contractor.scan.operation-report.v1" || !report.Completed || len(report.Artifacts) != 2 {
		t.Fatalf("source-derived scan=%+v error=%v", report, err)
	}
	if h.gateway.CompletedStages() != 4 || len(h.gateway.Failures()) != 0 {
		t.Fatalf("generation reused ordinary Workflow stages incorrectly: %v", h.gateway.Failures())
	}
}
