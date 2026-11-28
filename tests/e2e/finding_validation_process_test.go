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
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestFindingTargetValidationAcrossProcesses(t *testing.T) {
	// Execute the very same lookup function delivered to the source Worker,
	// using real SQLite behind HTTP rather than scripting probe responses.
	root := t.TempDir()
	for name, body := range map[string]string{"app.py": findingsSource, "target.py": findingValidationTarget} {
		if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	address := freeAddress(t)
	targetURL := "http://" + address
	target := startProcess(t, "Finding validation target", root, nil, filepath.Join(repoRoot(t), "runtime", ".venv", "bin", "python"), "target.py", address, "app.py")
	waitForHTTP(t, t.Context(), target, &http.Client{}, targetURL+"/health", http.StatusOK)
	var h *findingsHarness
	var findingPath string
	conditions := []string{"the selected deployment, unauthenticated GET /items, SQLite fixture records 1 and 2"}
	stage := domainGatewayStage{name: "finding-http-validation", tools: []string{"read_audit_task", "evidence", "submit_result", "submit_results", "http_request", "http_read_body", "http_history", "http_session_set", "http_session_get", "http_session_clear", "read_artifact"}, steps: []domainGatewayStep{
		toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
		toolGatewayStep("read_artifact", fixedArguments(map[string]any{"namespace": "inputs", "name": "finding"})),
		toolGatewayStep("http_request", fixedArguments(map[string]any{"url": targetURL + "/items/1", "follow_redirects": false})),
		toolGatewayStep("http_request", fixedArguments(map[string]any{"url": targetURL + "/items/" + url.PathEscape("x' OR '1'='1"), "follow_redirects": false})),
		toolGatewayStep("evidence", func(request map[string]any) (map[string]any, error) {
			exchanges := findingsToolResponses(request, "http_request")
			if len(exchanges) != 2 {
				return nil, fmt.Errorf("expected a real control and probe")
			}
			sources := []any{}
			for _, exchange := range exchanges {
				if exchange["status"] != json.Number("200") {
					return nil, fmt.Errorf("HTTP probe failed: %v", exchange)
				}
				sources = append(sources, map[string]any{"kind": "http", "request_id": exchange["request_id"]})
			}
			return auditResultArguments(map[string]any{"sources": sources, "interpretation": "The control returns record 1; the boolean SQL probe returns both records", "tested_conditions": conditions, "limitations": []string{}})(request)
		}),
		toolGatewayStep("submit_result", func(request map[string]any) (map[string]any, error) {
			// The launch pin remains revision 1 while the analyst changes the card.
			basicAuditRequest(t, h, http.MethodPatch, findingPath, map[string]any{"verdict": "true_positive", "severity": "high", "rationale": "Independent analyst decision"}, "finding-check-analyst", 1, http.StatusOK)
			return auditResultArguments(map[string]any{"verdict": "supported", "rationale": "The admitted control/probe captures demonstrate SQL query manipulation under the tested conditions", "limitations": []string{}})(request)
		}),
		finalGatewayStep("Validation recorded", nil),
	}}
	h = startFindingsHarness(t, []domainGatewayStage{ordinaryFindingsProducerStage("validation-source"), stage})
	project := createProjectResource(t, h.client, h.baseURL)
	project = updateProjectTarget(t, h.client, h.baseURL, project.ProjectID, project.Revision, map[string]any{"url": targetURL}, http.StatusOK)
	source := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "sources", "validation-source", "application/zip", auditProgramZip(t, map[string][]byte{"app.py": []byte(findingsSource)}))
	body, _ := json.Marshal(map[string]any{"workflow": "fixture-ordinary-findings@1", "artifacts": map[string]artifactRef{"source": source}})
	sourceRun := postProjectRun(t, h.client, h.baseURL, project.ProjectID, "validation-source", body, false)
	waitForFindingsRun(t, h, sourceRun, "succeeded")
	receipts := findingsReceipts(t, h, sourceRun)
	if len(receipts) != 1 || receipts[0].FindingID == nil {
		t.Fatal("source Worker produced no Finding")
	}
	check := contracts.FindingCheck{FindingID: *receipts[0].FindingID, ReceiptID: receipts[0].ReceiptID, Revision: 1, Assertion: "The deployed GET /items endpoint permits boolean SQL query manipulation and disclosure of another item"}
	findingPath = "/v1/projects/" + project.ProjectID + "/findings/" + check.FindingID
	launch := func(selection any) map[string]any {
		return map[string]any{"workflow": "finding-http-validation@1", "findingCheck": selection}
	}
	route := "/v1/projects/" + project.ProjectID + "/runs"
	for _, test := range []struct {
		name   string
		body   map[string]any
		status int
	}{
		{"missing-binding", map[string]any{"workflow": "finding-http-validation@1"}, 400},
		{"null-binding", launch(nil), 400},
		{"foreign-source", launch(contracts.FindingCheck{FindingID: check.FindingID, ReceiptID: "foreign-source", Revision: 1, Assertion: check.Assertion}), 404},
		{"stale-revision", launch(contracts.FindingCheck{FindingID: check.FindingID, ReceiptID: check.ReceiptID, Revision: 2, Assertion: check.Assertion}), 409},
		{"incompatible-workflow", map[string]any{"workflow": "fixture-ordinary-findings@1", "findingCheck": check}, 400},
		{"target-override", map[string]any{"workflow": "finding-http-validation@1", "findingCheck": check, "parameters": map[string]string{"target": "https://other.invalid"}}, 400},
	} {
		t.Run(test.name, func(t *testing.T) {
			basicAuditRequest(t, h, http.MethodPost, route, test.body, test.name, 0, test.status)
		})
	}
	created, _ := basicAuditRequest(t, h, http.MethodPost, route, launch(check), "validation-launch", 0, http.StatusAccepted)
	runID := created["runId"].(string)
	waitForFindingsRun(t, h, runID, "succeeded")
	if len(findingsReceipts(t, h, runID)) != 0 {
		t.Fatal("validation republished a Finding")
	}
	result, _ := download(t, h.client, h.baseURL+"/v1/runs/"+runID+"/outputs/result")
	var document map[string]any
	if json.Unmarshal(result, &document) != nil {
		t.Fatal("invalid deterministic result")
	}
	item := document["items"].([]any)[0].(map[string]any)
	origin := document["origin"].(map[string]any)
	if item["verdict"] != "supported" || origin["audit_id"] != nil || origin["task_id"] != nil || origin["finding"].(map[string]any)["revision"] != float64(1) {
		t.Fatalf("wrong result authority: %s", result)
	}
	current, _ := basicAuditRequest(t, h, http.MethodGet, findingPath, nil, "", 0, http.StatusOK)
	if current["revision"] != float64(2) || current["analystVerdict"] != "true_positive" || current["rationale"] != "Independent analyst decision" {
		t.Fatalf("validation replaced analyst assessment: %v", current)
	}
	// A lost launch reply reconciles before consulting the changed Finding or target.
	project = updateProjectTarget(t, h.client, h.baseURL, project.ProjectID, project.Revision, nil, http.StatusOK)
	replayed, headers := basicAuditRequest(t, h, http.MethodPost, route, launch(check), "validation-launch", 0, http.StatusAccepted)
	if replayed["runId"] != runID || headers.Get("Idempotency-Replayed") != "true" {
		t.Fatal("launch response-loss replay changed Run")
	}
	changed := check
	changed.Assertion += " changed"
	basicAuditRequest(t, h, http.MethodPost, route, launch(changed), "validation-launch", 0, http.StatusConflict)
	basicAuditRequest(t, h, http.MethodPost, route, launch(check), "missing-target", 0, http.StatusBadRequest)
	stats, _ := download(t, h.client, targetURL+"/stats")
	if strings.TrimSpace(string(stats)) != "2" || h.gateway.CompletedStages() != 2 || len(h.gateway.Failures()) != 0 {
		t.Fatalf("target/model work repeated: requests=%s stages=%d errors=%v", stats, h.gateway.CompletedStages(), h.gateway.Failures())
	}
	historyRoute := fmt.Sprintf("/v1/projects/%s/finding-sources/%s/history?limit=64", project.ProjectID, check.ReceiptID)
	history, _ := basicAuditRequest(t, h, http.MethodGet, historyRoute, nil, "", 0, http.StatusOK)
	if len(history["items"].([]any)) != 3 {
		t.Fatal("missing launch/evidence/result history")
	}
	deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, sourceRun)
	deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, runID)
	for _, raw := range history["items"].([]any) {
		meta := raw.(map[string]any)
		ref := meta["artifact"].(map[string]any)
		data, _ := download(t, h.client, fmt.Sprintf("%s/v1/projects/%s/artifacts/%s/%s?revision=%s", h.baseURL, project.ProjectID, ref["namespace"], ref["name"], url.QueryEscape(ref["revision"].(string))))
		if !bytes.Contains(data, []byte(runID)) {
			t.Fatal("retained history lost Run provenance")
		}
	}
	after, _ := basicAuditRequest(t, h, http.MethodGet, historyRoute, nil, "", 0, http.StatusOK)
	if len(after["items"].([]any)) != 3 {
		t.Fatal("producer purge lost validation history")
	}
}

const findingValidationTarget = `import ast,json,sqlite3,sys
from http.server import BaseHTTPRequestHandler,HTTPServer
from urllib.parse import unquote
database=sqlite3.connect(":memory:")
database.executescript("CREATE TABLE items(id TEXT, secret TEXT);INSERT INTO items VALUES('1','alpha'),('2','beta');")
tree=ast.parse(open(sys.argv[2]).read())
scope={"database":database}
exec(compile(ast.Module(body=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="lookup")],type_ignores=[]),sys.argv[2],"exec"),scope)
count=0
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  global count
  status=200
  if self.path=="/health":body=b"ready"
  elif self.path=="/stats":body=str(count).encode()
  elif self.path.startswith("/items/"):
   count+=1
   body=json.dumps(scope["lookup"](unquote(self.path[len("/items/"):])) .fetchall()).encode()
  else:status,body=404,b"missing"
  self.send_response(status);self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)
 def log_message(self,*args):pass
host,port=sys.argv[1].rsplit(":",1)
HTTPServer((host,int(port)),Handler).serve_forever()
`
