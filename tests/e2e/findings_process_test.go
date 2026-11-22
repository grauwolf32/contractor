//go:build e2e

package e2e

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/localpki"
)

type findingsHarness struct {
	browser                          *researchBrowserTransport
	runtimeExtraArguments            []string
	runtimeControlPlaneURL           string
	proxyPrivateAPI                  func(func(*http.Response) error) string
	restartServer                    func()
	ctx                              context.Context
	server, runtimeProcess           *childProcess
	gateway                          *domainGateway
	client                           *http.Client
	baseURL, databaseURL, configRoot string
}

func startFindingsHarness(t *testing.T, stages []domainGatewayStage, beforeRuntime ...func(*findingsHarness)) *findingsHarness {
	t.Helper()
	return startConfiguredFindingsHarness(t, stages, nil, beforeRuntime...)
}

func startConfiguredFindingsHarness(t *testing.T, stages []domainGatewayStage, configure func(string), beforeRuntime ...func(*findingsHarness)) *findingsHarness {
	t.Helper()
	return startFindingsHarnessForBrowser(t, stages, false, configure, beforeRuntime...)
}

func startFindingsHarnessForBrowser(t *testing.T, stages []domainGatewayStage, browser bool, configure func(string), beforeRuntime ...func(*findingsHarness)) *findingsHarness {
	t.Helper()
	if testing.Short() {
		t.Fatal("findings process checks cannot run in short mode")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	repositoryRoot := repoRoot(t)
	temporaryRoot := t.TempDir()
	budget := 4 * time.Minute
	if browser {
		budget = 8 * time.Minute
	}
	ctx, cancel := context.WithTimeout(context.Background(), budget)
	t.Cleanup(cancel)

	isolateURL := isolatedDatabase(t, ctx, databaseURL)
	serverBinary := sharedServerBinary(t)
	runChecked(t, repositoryRoot, map[string]string{
		"CONTRACTOR_DATABASE_URL": isolateURL,
	}, serverBinary, "migrate")

	pkiRoot := filepath.Join(temporaryRoot, "pki")
	generator := localpki.Generator{}
	caPaths, err := generator.InitCA(pkiRoot, false)
	if err != nil {
		t.Fatalf("initialize test CA: %v", err)
	}
	leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	controlPlanePaths, err := generator.IssueControlPlane(pkiRoot, localpki.ControlPlaneOptions{
		LeafOptions: leaf, URI: "urn:contractor:control-plane:findings-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	agentPaths, err := generator.IssueAgent(pkiRoot, "findings-agent", leaf)
	if err != nil {
		t.Fatalf("issue Runtime Agent certificate: %v", err)
	}

	gateway := newBlockedDomainGateway(llmGatewayToken, stages)
	t.Cleanup(gateway.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	installOrdinaryFindingFixture(t, configRoot, "audit_openapi_operation_tracer", "audit_openapi_operation_trace", "fixture-ordinary-findings")
	if configure != nil {
		configure(configRoot)
	}
	publicAddress, privateAddress, runtimeAddress := freeAddress(t), freeAddress(t), freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	runtimeBaseURL := "https://" + runtimeAddress
	userID := "findings-user-" + randomHex(t, 8)
	localAuthFile := writeE2ELocalAuth(t, temporaryRoot, userID)
	serverEnvironment := map[string]string{
		"CONTRACTOR_DATABASE_URL":            isolateURL,
		"CONTRACTOR_OPERATOR_CONFIG_ROOT":    configRoot,
		"CONTRACTOR_PUBLIC_LISTEN":           publicAddress,
		"CONTRACTOR_PRIVATE_LISTEN":          privateAddress,
		"CONTRACTOR_PRIVATE_URL":             privateBaseURL,
		"CONTRACTOR_CA_FILE":                 caPaths.Certificate,
		"CONTRACTOR_CONTROL_PLANE_CERT_FILE": controlPlanePaths.Certificate,
		"CONTRACTOR_CONTROL_PLANE_KEY_FILE":  controlPlanePaths.PrivateKey,
		"CONTRACTOR_LLM_GATEWAY_TOKEN":       llmGatewayToken,
		"CONTRACTOR_PUBLIC_BEARER_TOKEN":     publicToken,
		"CONTRACTOR_LOCAL_AUTH_FILE":         localAuthFile,
		"CONTRACTOR_BROWSER_ORIGINS":         "https://ui.contractor.invalid",
	}
	var transport *researchBrowserTransport
	if browser {
		transport = configureResearchBrowser(t, publicBaseURL, temporaryRoot)
		serverEnvironment["CONTRACTOR_BROWSER_ORIGINS"] = transport.baseURL
	}
	startServer := func() *childProcess {
		return startProcess(t, "Go Server", repositoryRoot, serverEnvironment, serverBinary, "serve")
	}
	server := startServer()

	client := &http.Client{Timeout: 8 * time.Second}
	waitForHTTP(t, ctx, server, client, publicBaseURL+"/readyz", http.StatusOK)
	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	prepared := &findingsHarness{ctx: ctx, server: server, gateway: gateway, client: client, baseURL: publicBaseURL, databaseURL: isolateURL, configRoot: configRoot}
	prepared.runtimeControlPlaneURL = privateBaseURL
	prepared.proxyPrivateAPI = func(modify func(*http.Response) error) string {
		address := startPrivateAPIResponseProxy(t, privateBaseURL, caPaths, controlPlanePaths, agentPaths, modify)
		// Allocation Artifact clients use the Server-published private URL,
		// independently of the Runtime registration endpoint. Publish the test
		// proxy through the ordinary Server setting before any allocation exists.
		server.stop(t)
		serverEnvironment["CONTRACTOR_PRIVATE_URL"] = address
		server = startServer()
		prepared.server = server
		waitForHTTP(t, ctx, server, client, publicBaseURL+"/readyz", http.StatusOK)
		return address
	}
	for _, prepare := range beforeRuntime {
		prepare(prepared)
	}
	workRoot := filepath.Join(temporaryRoot, "runtime-work")
	runtimeArguments := []string{"-m", "contractor_runtime",
		"--control-plane-url", prepared.runtimeControlPlaneURL,
		"--advertised-control-url", runtimeBaseURL,
		"--advertised-a2a-url", runtimeBaseURL,
		"--ca-file", caPaths.Certificate,
		"--certificate-file", agentPaths.Certificate,
		"--private-key-file", agentPaths.PrivateKey,
		"--listen", runtimeAddress,
		"--work-root", workRoot,
		"--workspace-storage", "local",
		"--workspace-work-root", filepath.Join(temporaryRoot, "workspace"),
		"--request-timeout-seconds", "12",
		"--shutdown-grace-seconds", "5",
	}
	runtimeArguments = append(runtimeArguments, prepared.runtimeExtraArguments...)
	runtimeProcess := startProcess(t, "Python Runtime Agent", filepath.Join(repositoryRoot, "runtime"), map[string]string{"PYTHONUNBUFFERED": "1"}, python, runtimeArguments...)
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(t, ctx, runtimeProcess, controlClient, runtimeBaseURL+"/healthz", http.StatusOK)
	waitForProcessLog(t, ctx, runtimeProcess, "runtime agent registered")
	t.Cleanup(func() {
		if t.Failed() {
			t.Logf("findings server:\n%s\nruntime:\n%s\ngateway: %v", server.logs.redacted(publicToken, llmGatewayToken), runtimeProcess.logs.redacted(publicToken, llmGatewayToken), gateway.Failures())
		}
	})

	gateway.releaseBlockedRequest()
	return &findingsHarness{browser: transport, ctx: ctx, server: server, runtimeProcess: runtimeProcess,
		gateway: gateway, client: client, baseURL: publicBaseURL, databaseURL: isolateURL, configRoot: configRoot,
		restartServer: func() {
			server.stop(t)
			server = startServer()
			waitForHTTP(t, ctx, server, client, publicBaseURL+"/readyz", http.StatusOK)
		},
	}
}

const findingsSource = `from fastapi import FastAPI
app = FastAPI()

def lookup(item_id):
    query = "SELECT * FROM items WHERE id = '" + item_id + "'"
    return database.execute(query)

@app.get("/items/{item_id}")
def get_item(item_id: str):
    return lookup(item_id)
`
const findingsOpenAPI = `openapi: 3.0.3
info: {title: Findings fixture, version: '1'}
paths:
  /items/{item_id}:
    get:
      operationId: getItem
      parameters:
        - {name: item_id, in: path, required: true, schema: {type: string}}
      responses:
        '200': {description: Item}
`
const findingsEvidence = "app.py:4-6: lookup concatenates item_id into a query; app.py:8-10 passes the route parameter. Reproduction is source-inferred and unexecuted.\n"

func TestFindingsProducerAndReaderAcrossProcesses(t *testing.T) {
	h := startFindingsHarness(t, []domainGatewayStage{findingsProducerStage("trace"), findingsReaderStage(1)})
	project := createProjectResource(t, h.client, h.baseURL)
	source := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID,
		"sources", "findings-source", "application/zip", auditProgramZip(t, map[string][]byte{"app.py": []byte(findingsSource)}))
	openAPI := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID,
		"openapi", "findings-openapi", "application/yaml", []byte(findingsOpenAPI))
	auditID, tasks := runCommonFindingAudit(t, h, project.ProjectID, map[string]artifactRef{"source": source, "openapi": openAPI}, 1)
	runID := commonFindingTaskRun(t, h, auditID, tasks[1])
	receipts := findingsReceipts(t, h, runID)
	if len(receipts) != 1 {
		t.Fatalf("producer receipts = %+v", receipts)
	}
	receipt := receipts[0]
	if receipt.Origin.Audit == nil || receipt.Origin.Audit.AuditID != auditID ||
		receipt.Document.Hypothesis != "" || receipt.Document.Subject != nil || len(receipt.Evidence) != 3 {
		t.Fatalf("generic direct finding origin/content = %+v", receipt)
	}
	assertFindingsResultReceipt(t, h, runID, receipt)
	var before map[string]any
	auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+auditID+"/findings?limit=100", &before)
	collection := publishFindingsCollection(t, h, "analysis-snapshot", []findingintake.CollectionSelection{{
		Kind: "audit", ID: auditID, ReceiptIDs: []string{receipt.ReceiptID}, Findings: []findingintake.CollectionFindingSelection{},
	}}, false)
	raw, mime := download(t, h.client, h.baseURL+"/v1/artifacts/"+collection.Artifact.Ref.Namespace+"/"+
		collection.Artifact.Ref.Name+"?revision="+url.QueryEscape(*collection.Artifact.Ref.Revision))
	decoded, _, err := auditdomain.DecodeFindingCollectionPackage(raw)
	if err != nil || mime != resolved.FindingCollectionMediaType || len(decoded.Entries) != 1 ||
		decoded.Entries[0].ReceiptID != receipt.ReceiptID || decoded.Entries[0].Finding == nil || decoded.Entries[0].Finding.State != nil {
		t.Fatalf("published collection = %+v, %v", decoded, err)
	}
	readerRun := runFindingsReader(t, h, "findings-analysis", collection.Artifact.Ref)
	report, mediaType := download(t, h.client, h.baseURL+"/v1/runs/"+readerRun+"/outputs/report")
	if mediaType != "text/markdown" || !strings.Contains(string(report), receipt.ReceiptID) ||
		!strings.Contains(string(report), "source-inferred and unexecuted") ||
		!strings.Contains(string(report), "app.py") {
		t.Fatalf("reader report = %s (%s)", report, mediaType)
	}
	if len(findingsReceipts(t, h, readerRun)) != 0 {
		t.Fatal("reader created proposals")
	}
	var after map[string]any
	auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+auditID+"/findings?limit=100", &after)
	// asOf identifies each read snapshot; all other fields describe review state.
	for _, snapshot := range []map[string]any{before, after} {
		asOf, ok := snapshot["asOf"].(string)
		if !ok {
			t.Fatalf("Findings snapshot timestamp is missing or not a string: %+v", snapshot["asOf"])
		}
		if timestamp, err := time.Parse(time.RFC3339Nano, asOf); err != nil || timestamp.IsZero() {
			t.Fatalf("invalid Findings snapshot timestamp %q: %v", asOf, err)
		}
		delete(snapshot, "asOf")
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("collection analysis mutated Audit review state: before=%+v after=%+v", before, after)
	}
	if h.gateway.CompletedStages() != 2 || len(h.gateway.Failures()) != 0 {
		t.Fatalf("gateway completed %d stages: %v", h.gateway.CompletedStages(), h.gateway.Failures())
	}
}

func findingsProducerStage(name string) domainGatewayStage {
	tools := append([]string{"list_skills", "load_skill", "load_skill_resource",
		"read_audit_task", "submit_check_result", "submit_check_results", "ls", "glob", "grep", "read_file",
		"write_text_artifact", "finding", "annotate_trace", "annotate_validate", "annotate_sink",
		"changed_paths", "diff", "rollback_changes"}, completeCodeAnalysisTools...)
	final := finalGatewayStep("All assigned results recorded", nil)
	final.validate = requireRecordedAuditResults(1)
	return domainGatewayStage{name: "findings/" + name, tools: withMemoryTools(tools), steps: []domainGatewayStep{
		toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
		toolGatewayStep("graph_summary", fixedArguments(map[string]any{})),
		toolGatewayStep("read_file", func(request map[string]any) (map[string]any, error) {
			if !hasGraphSummary(request) {
				return nil, fmt.Errorf("producer did not inspect its source graph")
			}
			return map[string]any{"path": "app.py", "start_line": 1, "max_lines": 100}, nil
		}),
		toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{
			"name": "source-trace", "text": findingsEvidence, "media_type": "text/plain",
		})),
		toolGatewayStep("finding", func(request map[string]any) (map[string]any, error) {
			evidence, ok := lastExactArtifact(request, "audit-check", "source-trace")
			if !ok {
				return nil, fmt.Errorf("finding called without a prior exact evidence receipt")
			}
			return map[string]any{
				"title":       "Query concatenation in lookup",
				"description": "GET /items/{item_id} passes its input to lookup (app.py:4-10). The query uses concatenation. Reproduce by supplying a quote in item_id and checking query parsing; source-inferred and unexecuted, database availability and surrounding controls remain unverified.",
				"file":        "app.py", "range": map[string]int{"start_line": 4, "end_line": 6},
				"evidence_refs": []any{materialSelector(evidence)},
			}, nil
		}),
		toolGatewayStep("submit_check_result", func(request map[string]any) (map[string]any, error) {
			responses := findingsToolResponses(request, "finding")
			if len(responses) == 0 || responses[0]["receipt_id"] == nil || responses[0]["proposal_id"] == nil {
				return nil, fmt.Errorf("canonical result has no successful finding receipt: %+v", responses)
			}
			for _, response := range responses[1:] {
				if !reflect.DeepEqual(response, responses[0]) {
					return nil, fmt.Errorf("identical finding retry changed receipt")
				}
			}
			return auditResultArguments(map[string]any{"rationale": "Operation mapped to lookup; query concatenation proposal recorded.", "tested_conditions": []string{"operation-resolution"}, "limitations": []string{"Source inferred; no live exploit executed"}, "evidence": []string{"evidence-000", "evidence-001"}})(request)
		}),
		final,
	}}
}

func findingsReaderStage(count int, pageSizes ...int) domainGatewayStage {
	pageSize := 100
	if len(pageSizes) > 0 {
		pageSize = pageSizes[0]
	}
	steps := []domainGatewayStep{}
	for page := 0; page < max(1, (count+pageSize-1)/pageSize); page++ {
		steps = append(steps, toolGatewayStep("list_findings", func(request map[string]any) (map[string]any, error) {
			arguments := map[string]any{"limit": pageSize}
			if page > 0 {
				pages := findingsToolResponses(request, "list_findings")
				if len(pages) != page || pages[page-1]["next_cursor"] == nil {
					return nil, fmt.Errorf("pagination ended before complete inventory")
				}
				arguments["cursor"] = pages[page-1]["next_cursor"]
			}
			return arguments, nil
		}))
	}
	for i := 0; i < count; i++ {
		steps = append(steps, toolGatewayStep("read_artifact", findingsDocumentArguments(i, false)),
			toolGatewayStep("read_artifact", findingsDocumentArguments(i, true)))
	}
	steps = append(steps, toolGatewayStep("write_text_artifact", func(request map[string]any) (map[string]any, error) {
		pages := findingsToolResponses(request, "list_findings")
		if len(pages) == 0 || pages[len(pages)-1]["next_cursor"] != nil {
			return nil, fmt.Errorf("reader has not enumerated its collection")
		}
		items := findingsPageItems(pages)
		if len(items) != count {
			return nil, fmt.Errorf("reader count = %+v, want %d", pages, count)
		}
		reads := findingsToolResponses(request, "read_artifact")
		if len(reads) != 2*count {
			return nil, fmt.Errorf("reader did not open proposals and selected text evidence")
		}
		report := fmt.Sprintf("# Findings analysis\n\nCollection: %d proposals.\n", count)
		for i, raw := range items {
			item := raw.(map[string]any)
			proposalData, err := artifactTextBytes(reads[2*i]["text"])
			if err != nil {
				return nil, err
			}
			proposal, err := auditdomain.DecodeFindingProposal(proposalData)
			if err != nil || proposal.Subject != nil || item["has_hypothesis"] != (proposal.Hypothesis != "") {
				return nil, fmt.Errorf("reader did not read generic proposal: %v", err)
			}
			evidence, err := artifactTextBytes(reads[2*i+1]["text"])
			if err != nil || string(evidence) != findingsEvidence {
				return nil, fmt.Errorf("reader evidence bytes differ")
			}
			report += fmt.Sprintf("\nReceipt %s, proposal %s, Run %s, %s/%s.\n", item["receipt_id"], item["proposal_id"], item["run_id"], "source", proposal.Locations[0].File)
			proposalRef, _ := json.Marshal(item["proposal"])
			report += fmt.Sprintf("Exact proposal and provenance: %s\n%s\n", proposalRef, evidence)
		}
		report += "\nGrouping is advisory; underlying proposals and captured reviews are unchanged.\n"
		return map[string]any{"name": "report", "media_type": "text/markdown", "text": report}, nil
	}), finalGatewayStep("Finding collection analysis published", map[string]domainArtifactBinding{
		"report": {namespace: "findings-review", name: "report"},
	}))
	return domainGatewayStage{name: "findings/reader", tools: withMemoryTools([]string{"list_findings", "read_artifact", "write_text_artifact"}), steps: steps}
}

func findingsDocumentArguments(index int, evidence bool) func(map[string]any) (map[string]any, error) {
	return func(request map[string]any) (map[string]any, error) {
		pages := findingsToolResponses(request, "list_findings")
		if len(pages) == 0 {
			return nil, fmt.Errorf("reader has no list response")
		}
		items := findingsPageItems(pages)
		if index >= len(items) {
			return nil, fmt.Errorf("reader item is missing")
		}
		item := items[index].(map[string]any)
		document := item["proposal"].(map[string]any)
		if evidence {
			values, ok := item["evidence"].([]any)
			if !ok {
				return nil, fmt.Errorf("reader evidence is missing")
			}
			var selected map[string]any
			for _, value := range values {
				candidate := value.(map[string]any)
				if candidate["media_type"] == "text/plain" {
					if selected != nil {
						return nil, fmt.Errorf("reader text evidence is ambiguous")
					}
					selected = candidate
				}
			}
			if selected == nil {
				return nil, fmt.Errorf("reader text evidence is missing")
			}
			document = selected
		}
		ref, ok := document["ref"].(map[string]any)
		if !ok || !strings.HasPrefix(fmt.Sprint(ref["namespace"]), "findings-") || ref["revision"] == nil {
			return nil, fmt.Errorf("reader attempted a non-consumer or versionless ref")
		}
		return ref, nil
	}
}

func findingsToolResponses(request map[string]any, name string) []map[string]any {
	var result []map[string]any
	messages, _ := request["messages"].([]any)
	for _, raw := range messages {
		message, _ := raw.(map[string]any)
		calls, _ := message["tool_calls"].([]any)
		for _, rawCall := range calls {
			call, _ := rawCall.(map[string]any)
			function, _ := call["function"].(map[string]any)
			if function["name"] != name {
				continue
			}
			id, _ := call["id"].(string)
			value, ok := toolResponse(request, id)
			if !ok {
				continue
			}
			object, ok := value.(map[string]any)
			if !ok {
				continue
			}
			if inner, wrapped := object["result"].(map[string]any); wrapped {
				object = inner
			}
			result = append(result, object)
		}
	}
	return result
}

func findingsReceipts(t *testing.T, h *findingsHarness, runID string) []findingintake.Receipt {
	t.Helper()
	var page struct {
		Items []findingintake.Receipt `json:"items"`
	}
	auditProgramGET(t, h.client, h.baseURL+"/v1/runs/"+url.PathEscape(runID)+"/finding-proposals?limit=100", &page)
	return page.Items
}

func assertFindingsResultReceipt(t *testing.T, h *findingsHarness, runID string, receipt findingintake.Receipt) {
	t.Helper()
	if receipt.Origin.Audit == nil {
		t.Fatal("missing Audit origin")
	}
	var page struct {
		Items []basicTask `json:"items"`
	}
	auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+receipt.Origin.Audit.AuditID+"/tasks?limit=128", &page)
	var payload []byte
	for _, task := range page.Items {
		if task.Document.TaskID == receipt.Origin.Audit.TaskID {
			payload = basicTaskArtifact(t, h, receipt.Origin.Audit.AuditID, task, "result")
		}
	}
	var result struct {
		Schema string `json:"schema"`
		Items  []struct {
			FindingReceipts  []string `json:"finding_receipts"`
			TestedConditions []string `json:"tested_conditions"`
			Origin           struct {
				TaskID string `json:"task_id"`
			} `json:"origin"`
		} `json:"items"`
	}
	if err := json.Unmarshal(payload, &result); err != nil || result.Schema != "contractor.audit.result.v1" || len(result.Items) != 1 {
		t.Fatalf("canonical common result: %+v %v", result, err)
	}
	if !reflect.DeepEqual(result.Items[0].FindingReceipts, []string{receipt.ReceiptID}) || receipt.Origin.Audit == nil || result.Items[0].Origin.TaskID != receipt.Origin.Audit.TaskID || !reflect.DeepEqual(result.Items[0].TestedConditions, []string{"operation-resolution"}) {
		t.Fatalf("result lost finding/Task correlation: %+v versus %+v", result, receipt)
	}

}

func publishFindingsCollection(t *testing.T, h *findingsHarness, key string, sources []findingintake.CollectionSelection, replay bool) findingintake.PublishedCollection {
	t.Helper()
	body, err := json.Marshal(findingintake.PublishCollectionRequest{ClientKey: key, Sources: sources})
	if err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(http.MethodPost, h.baseURL+"/v1/finding-collections", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	expected := http.StatusCreated
	if replay {
		expected = http.StatusOK
	}
	response := do(t, h.client, request, expected)
	defer response.Body.Close()
	var value findingintake.PublishedCollection
	decodeAuditProgramResponse(t, response, &value)
	if value.Replayed != replay || value.Artifact.Ref.Revision == nil {
		t.Fatalf("collection response = %+v", value)
	}
	return value
}

func runFindingsReader(t *testing.T, h *findingsHarness, key string, input any) string {
	t.Helper()
	runID := createFindingsReader(t, h, key, input)
	waitForFindingsRun(t, h, runID, "succeeded")
	return runID
}

func createFindingsReader(t *testing.T, h *findingsHarness, key string, input any) string {
	t.Helper()
	body, err := json.Marshal(map[string]any{"workflow": "findings-review@2", "artifacts": map[string]any{"findings": input}})
	if err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(http.MethodPost, h.baseURL+"/v1/runs", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", key)
	response := do(t, h.client, request, http.StatusAccepted)
	var value runCreateResponse
	decodeResponse(t, response, &value)
	response.Body.Close()
	return value.RunID
}

func waitForFindingsRun(t *testing.T, h *findingsHarness, runID, wanted string) runStatus {
	t.Helper()
	ticker := time.NewTicker(150 * time.Millisecond)
	defer ticker.Stop()
	for {
		var status runStatus
		if err := auditProgramTryGET(h.ctx, h.client, h.baseURL+"/v1/runs/"+url.PathEscape(runID), &status); err == nil {
			switch status.State {
			case "succeeded", "failed", "cancelled":
				if status.State != wanted {
					t.Fatalf("Run %s reached %s, want %s: %+v", runID, status.State, wanted, status)
				}
				return status
			}
		}
		for _, process := range []*childProcess{h.server, h.runtimeProcess} {
			if exited, err := process.exited(); exited {
				t.Fatalf("%s exited: %v", process.name, err)
			}
		}
		if failures := h.gateway.Failures(); len(failures) != 0 {
			t.Fatalf("findings gateway: %v", failures)
		}
		select {
		case <-h.ctx.Done():
			t.Fatalf("wait for findings Run: %v", h.ctx.Err())
		case <-ticker.C:
		}
	}
}

func findingsPageItems(pages []map[string]any) []any {
	var items []any
	for _, page := range pages {
		values, _ := page["items"].([]any)
		items = append(items, values...)
	}
	return items
}

func ordinaryFindingsProducerStage(name string) domainGatewayStage {
	stage := findingsProducerStage(name)
	tools := make([]string, 0, len(stage.tools))
	for _, tool := range stage.tools {
		if tool != "read_audit_task" && tool != "submit_check_result" && tool != "submit_check_results" &&
			tool != "annotate_trace" && tool != "annotate_validate" && tool != "annotate_sink" &&
			tool != "changed_paths" && tool != "diff" && tool != "rollback_changes" {
			tools = append(tools, tool)
		}
	}
	stage.tools = tools
	stage.steps = append(stage.steps[1:5:5],
		toolGatewayStep("write_text_artifact", func(request map[string]any) (map[string]any, error) {
			receipts := findingsToolResponses(request, "finding")
			if len(receipts) == 0 || receipts[0]["receipt_id"] == nil {
				return nil, fmt.Errorf("ordinary finding receipt is missing")
			}
			for _, receipt := range receipts[1:] {
				if !reflect.DeepEqual(receipt, receipts[0]) {
					return nil, fmt.Errorf("identical finding retry changed receipt")
				}
			}
			return map[string]any{"name": "result", "text": fmt.Sprint(receipts[0]["receipt_id"]), "media_type": "text/plain"}, nil
		}),
		finalGatewayStep("Ordinary finding recorded", map[string]domainArtifactBinding{"result": {namespace: "audit-check", name: "result"}}),
	)
	return stage
}

func runCommonFindingAudit(t *testing.T, h *findingsHarness, projectID string, inputs map[string]artifactRef, operations int) (string, []basicTask) {
	t.Helper()
	draft, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/projects/"+projectID+"/audits", map[string]any{"profile": map[string]string{"name": "openapi-operation-trace", "version": "3"}, "inputs": inputs, "scope": map[string]string{"objective": "Trace exact operations and retain their independent finding sources"}}, "finding-draft", 0, http.StatusCreated)
	id := draft["auditId"].(string)
	basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/start", nil, "finding-start", uint64(draft["revision"].(float64)), http.StatusOK)
	tasks := waitBasicTask(t, h, id, operations+1, "settled")
	for _, task := range tasks {
		if task.GapCode != nil || task.Result == nil {
			t.Fatalf("finding task did not collect: %+v", task)
		}
	}
	for basicAuditGet(t, h, id)["state"] != "completed" {
		basicPoll(t, h)
	}
	return id, tasks
}
func commonFindingTaskRun(t *testing.T, h *findingsHarness, auditID string, task basicTask) string {
	t.Helper()
	var result struct {
		Items []struct {
			Origin struct {
				RunID string `json:"run_id"`
			} `json:"origin"`
		} `json:"items"`
	}
	if err := json.Unmarshal(basicTaskArtifact(t, h, auditID, task, "result"), &result); err != nil || len(result.Items) != 1 || result.Items[0].Origin.RunID == "" {
		t.Fatalf("missing ordinary Run provenance: %+v %v", result, err)
	}
	return result.Items[0].Origin.RunID
}
