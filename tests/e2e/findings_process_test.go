//go:build e2e

package e2e

import (
	"bytes"
	"context"
	"encoding/base64"
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
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/localpki"
)

type findingsHarness struct {
	ctx                              context.Context
	server, runtimeProcess           *childProcess
	gateway                          *domainGateway
	client                           *http.Client
	baseURL, databaseURL, configRoot string
}

func startFindingsHarness(t *testing.T, stages []domainGatewayStage, beforeRuntime ...func(*findingsHarness)) *findingsHarness {
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
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	t.Cleanup(cancel)

	isolateURL := isolatedDatabase(t, ctx, databaseURL)
	serverBinary := filepath.Join(temporaryRoot, "contractor-server")
	runChecked(t, repositoryRoot, nil, "go", "build", "-o", serverBinary, "./cmd/contractor-server")
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
	publicAddress, privateAddress, runtimeAddress := freeAddress(t), freeAddress(t), freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	runtimeBaseURL := "https://" + runtimeAddress
	userID := "findings-user-" + randomHex(t, 8)
	localAuthFile := writeE2ELocalAuth(t, temporaryRoot, userID)
	serverEnvironment := map[string]string{
		"CONTRACTOR_DATABASE_URL":            isolateURL,
		"CONTRACTOR_CONFIG_ROOT":             configRoot,
		"CONTRACTOR_PUBLIC_LISTEN":           publicAddress,
		"CONTRACTOR_PRIVATE_LISTEN":          privateAddress,
		"CONTRACTOR_PRIVATE_URL":             privateBaseURL,
		"CONTRACTOR_CA_FILE":                 caPaths.Certificate,
		"CONTRACTOR_CONTROL_PLANE_CERT_FILE": controlPlanePaths.Certificate,
		"CONTRACTOR_CONTROL_PLANE_KEY_FILE":  controlPlanePaths.PrivateKey,
		"CONTRACTOR_LLM_GATEWAY_TOKEN":       llmGatewayToken,
		"CONTRACTOR_PUBLIC_USER_ID":          userID,
		"CONTRACTOR_PUBLIC_BEARER_TOKEN":     publicToken,
		"CONTRACTOR_LOCAL_AUTH_FILE":         localAuthFile,
		"CONTRACTOR_BROWSER_ORIGINS":         "https://ui.contractor.invalid",
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
	for _, prepare := range beforeRuntime {
		prepare(&findingsHarness{ctx: ctx, server: server, gateway: gateway, client: client,
			baseURL: publicBaseURL, databaseURL: isolateURL, configRoot: configRoot})
	}
	workRoot := filepath.Join(temporaryRoot, "runtime-work")
	runtimeProcess := startProcess(
		t, "Python Runtime Agent", filepath.Join(repositoryRoot, "runtime"),
		map[string]string{"PYTHONUNBUFFERED": "1"},
		python, "-m", "contractor_runtime",
		"--control-plane-url", privateBaseURL,
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
	)
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(t, ctx, runtimeProcess, controlClient, runtimeBaseURL+"/healthz", http.StatusOK)
	waitForProcessLog(t, ctx, runtimeProcess, "runtime agent registered")
	t.Cleanup(func() {
		if t.Failed() {
			t.Logf("findings server:\n%s\nruntime:\n%s\ngateway: %v", server.logs.redacted(publicToken, llmGatewayToken), runtimeProcess.logs.redacted(publicToken, llmGatewayToken), gateway.Failures())
		}
	})

	gateway.releaseBlockedRequest()
	return &findingsHarness{ctx: ctx, server: server, runtimeProcess: runtimeProcess,
		gateway: gateway, client: client, baseURL: publicBaseURL, databaseURL: isolateURL, configRoot: configRoot}
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
	audit := runAuditProgram(t, h.ctx, h.server, h.runtimeProcess, h.gateway, h.client,
		h.baseURL, project.ProjectID, "openapi-operation-trace@3",
		map[string]artifactRef{"source": source, "openapi": openAPI}, 1, 1, 0, "")
	var items struct {
		Items []auditProgramItem `json:"items"`
	}
	auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+audit.AuditID+"/items?limit=100", &items)
	if len(items.Items) != 1 || len(items.Items[0].Attempts) != 1 {
		t.Fatalf("Audit items = %+v", items)
	}
	runID := items.Items[0].Attempts[0].RunID
	receipts := findingsReceipts(t, h, runID)
	if len(receipts) != 1 {
		t.Fatalf("producer receipts = %+v", receipts)
	}
	receipt := receipts[0]
	if receipt.Origin.Audit == nil || receipt.Origin.Audit.AuditID != audit.AuditID ||
		receipt.Document.Hypothesis != "" || receipt.Document.Subject.Kind != "function" || len(receipt.Evidence) != 1 {
		t.Fatalf("generic direct finding origin/content = %+v", receipt)
	}
	assertFindingsResultReceipt(t, h, runID, receipt)
	var before map[string]any
	auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+audit.AuditID+"/findings?limit=100", &before)
	collection := publishFindingsCollection(t, h, "analysis-snapshot", []findingintake.CollectionSelection{{
		Kind: "audit", ID: audit.AuditID, ReceiptIDs: []string{receipt.ReceiptID}, Findings: []findingintake.CollectionFindingSelection{},
	}}, false)
	raw, mime := download(t, h.client, h.baseURL+"/v1/artifacts/"+collection.Artifact.Ref.Namespace+"/"+
		collection.Artifact.Ref.Name+"?revision="+url.QueryEscape(*collection.Artifact.Ref.Revision))
	decoded, _, err := auditdomain.DecodeFindingCollectionPackage(raw)
	if err != nil || mime != auditdomain.FindingCollectionMediaType || len(decoded.Entries) != 1 ||
		decoded.Entries[0].ReceiptID != receipt.ReceiptID || len(decoded.Entries[0].Reviews) != 1 {
		t.Fatalf("published collection = %+v, %v", decoded, err)
	}
	readerRun := runFindingsReader(t, h, "findings-analysis", collection.Artifact.Ref)
	report, mediaType := download(t, h.client, h.baseURL+"/v1/runs/"+readerRun+"/outputs/report")
	if mediaType != "text/markdown" || !strings.Contains(string(report), receipt.ReceiptID) ||
		!strings.Contains(string(report), "source-inferred and unexecuted") ||
		!strings.Contains(string(report), "function/lookup") {
		t.Fatalf("reader report = %s (%s)", report, mediaType)
	}
	if len(findingsReceipts(t, h, readerRun)) != 0 {
		t.Fatal("reader created proposals")
	}
	var after map[string]any
	auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+audit.AuditID+"/findings?limit=100", &after)
	if !reflect.DeepEqual(before, after) {
		t.Fatal("collection analysis mutated Audit review state")
	}
	if h.gateway.CompletedStages() != 2 || len(h.gateway.Failures()) != 0 {
		t.Fatalf("gateway completed %d stages: %v", h.gateway.CompletedStages(), h.gateway.Failures())
	}
}

func findingsProducerStage(name string) domainGatewayStage {
	tools := append([]string{"list_skills", "load_skill", "load_skill_resource", "read_artifact",
		"read_audit_task", "submit_check_result", "ls", "glob", "grep", "read_file",
		"write_text_artifact", "finding"}, completeCodeAnalysisTools...)
	return domainGatewayStage{name: "findings/" + name, tools: tools, steps: []domainGatewayStep{
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
				"client_key": "query-concatenation", "title": "Query concatenation in lookup",
				"description":   "GET /items/{item_id} passes its input to lookup (app.py:4-10). The query uses concatenation. Reproduce by supplying a quote in item_id and checking query parsing; source-inferred and unexecuted, database availability and surrounding controls remain unverified.",
				"subject":       map[string]string{"kind": "function", "key": "lookup"},
				"evidence_refs": []any{evidence}, "severity_suggestion": "high",
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
			return map[string]any{"assessment": "supported", "summary": "Operation mapped to lookup; query concatenation proposal recorded.",
				"completed": []string{"operation-resolution"}, "gaps": []string{},
				"evidence":      []map[string]string{{"kind": "source", "summary": findingsEvidence}},
				"proposal_keys": []string{"query-concatenation"}}, nil
		}),
		finalGatewayStep("Canonical operation result and finding receipt published", map[string]domainArtifactBinding{
			"result": {namespace: "audit-check", name: "result"},
		}),
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
			return nil, fmt.Errorf("reader did not open all full documents")
		}
		report := fmt.Sprintf("# Findings analysis\n\nCollection: %d proposals.\n", count)
		for i, raw := range items {
			item := raw.(map[string]any)
			proposalData, err := base64.StdEncoding.DecodeString(fmt.Sprint(reads[2*i]["dataBase64"]))
			if err != nil {
				return nil, err
			}
			proposal, err := auditdomain.DecodeFindingProposal(proposalData)
			if err != nil || proposal.Subject.Kind != "function" || item["has_hypothesis"] != (proposal.Hypothesis != "") {
				return nil, fmt.Errorf("reader did not read generic proposal: %v", err)
			}
			evidence, err := base64.StdEncoding.DecodeString(fmt.Sprint(reads[2*i+1]["dataBase64"]))
			if err != nil || string(evidence) != findingsEvidence {
				return nil, fmt.Errorf("reader evidence bytes differ")
			}
			report += fmt.Sprintf("\nReceipt %s, proposal %s, Run %s, %s/%s.\n", item["receipt_id"], item["proposal_id"], item["run_id"], proposal.Subject.Kind, proposal.Subject.Key)
			proposalRef, _ := json.Marshal(item["proposal"])
			report += fmt.Sprintf("Exact proposal and provenance: %s\n%s\n", proposalRef, evidence)
		}
		report += "\nGrouping is advisory; underlying proposals and captured reviews are unchanged.\n"
		return map[string]any{"name": "report", "media_type": "text/markdown", "text": report}, nil
	}), finalGatewayStep("Finding collection analysis published", map[string]domainArtifactBinding{
		"report": {namespace: "findings-review", name: "report"},
	}))
	return domainGatewayStage{name: "findings/reader", tools: []string{"list_findings", "read_artifact", "write_text_artifact"}, steps: steps}
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
			if !ok || len(values) != 1 {
				return nil, fmt.Errorf("reader evidence is missing")
			}
			document = values[0].(map[string]any)
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
	payload, _ := download(t, h.client, h.baseURL+"/v1/runs/"+runID+"/outputs/result")
	result, err := auditdomain.DecodeCheckResultPackage(payload)
	if err != nil || len(result.Results.Results) != 1 {
		t.Fatalf("canonical result: %+v %v", result, err)
	}
	selections := result.Results.Results[0].Proposals
	if len(selections) != 1 || selections[0].ClientKey != receipt.ClientKey ||
		selections[0].InvocationID != receipt.Origin.InvocationID {
		t.Fatalf("canonical result lost receipt correlation: %+v versus %+v", selections, receipt)
	}
	if !reflect.DeepEqual(result.Results.Results[0].Coverage.Completed, []string{"operation-resolution"}) {
		t.Fatal("finding emission changed operation-resolution coverage")
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
	body, err := json.Marshal(map[string]any{"workflow": "findings-review@1", "artifacts": map[string]any{"findings": input}})
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
