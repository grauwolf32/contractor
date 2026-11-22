//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

// Public API, real Server/Runtime/PostgreSQL. Only model judgment is scripted.
// No SQL creates Tasks, advances Runs or supplies result/publication receipts.
func TestAuditBasicExactHandoffAcrossProductionProcesses(t *testing.T) {
	for _, withFinding := range []bool{false, true} {
		t.Run(fmt.Sprintf("finding_%t", withFinding), func(t *testing.T) {
			if testing.Short() {
				t.Fatal("the basic Audit gate cannot run in short mode")
			}
			stages := []domainGatewayStage{basicReportStage(), basicConclusionStage(withFinding)}
			h := startConfiguredFindingsHarness(t, stages, func(root string) { installBasicAuditProgram(t, root) })
			project := createProjectResource(t, h.client, h.baseURL)
			sourceBytes := auditProgramZip(t, map[string][]byte{"app.py": []byte(findingsSource)})
			source := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "source", "application/zip", sourceBytes)
			report := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "report", "text/plain", []byte("Inspect the selected source and author a review report.\n"))
			draft, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{
				"profile": map[string]string{"name": "basic-process", "version": "1"}, "inputs": map[string]artifactRef{"source": source, "report": report}, "scope": map[string]string{"objective": "Inspect exact retained source across two ordinary Workflows"},
			}, "draft", 0, http.StatusCreated)
			auditID := draft["auditId"].(string)
			blocked := h.gateway.blockNextRequest()
			started, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+auditID+"/start", nil, "start", uint64(draft["revision"].(float64)), http.StatusOK)
			if started["audit"].(map[string]any)["state"] != "active" {
				t.Fatalf("start: %v", started)
			}
			select {
			case <-blocked:
			case <-h.ctx.Done():
				t.Fatal("first ordinary Run did not reach Runtime")
			}
			current := basicAuditGet(t, h, auditID)
			paused, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+auditID+"/pause", nil, "pause", uint64(current["revision"].(float64)), http.StatusOK)
			if paused["state"] != "paused" {
				t.Fatalf("pause: %v", paused)
			}
			h.gateway.releaseBlockedRequest()
			first := waitBasicTask(t, h, auditID, 1, "settled")[0]
			if first.Result == nil || first.Result.ScopeKind != "audit" || first.Document.Hypothesis != nil || first.GapCode != nil {
				t.Fatalf("supporting Task: %+v", first)
			}
			current = basicAuditGet(t, h, auditID)
			if current["state"] != "paused" || int(current["submittedRunCount"].(float64)) != 1 {
				t.Fatalf("paused Audit completed or dispatched: %v", current)
			}
			// Exact output is readable through the ordinary Audit Artifact link.
			payload := basicTaskArtifact(t, h, auditID, first, "material")
			if !bytes.Equal(payload, []byte(findingsEvidence)) {
				t.Fatal("ordinary output changed selected bytes")
			}
			h.restartServer()
			current = basicAuditGet(t, h, auditID)
			work := map[string]any{"work": []auditcontract.PreparedSelection{{Key: "explicit-second", Workflow: "check", Objective: "Inspect the exact copied source", Inputs: map[string]auditcontract.Material{"material": *first.Result}}}}
			revision := uint64(current["revision"].(float64))
			accepted, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+auditID+"/tasks", work, "second", revision, http.StatusOK)
			replayed, header := basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+auditID+"/tasks", work, "second", revision, http.StatusOK)
			if header.Get("Idempotency-Replayed") != "true" || !bytes.Equal(mustBasicJSON(t, accepted), mustBasicJSON(t, replayed)) {
				t.Fatal("lost Task acceptance acknowledgement changed the receipt")
			}
			tasks := waitBasicTask(t, h, auditID, 2, "ready")
			if tasks[1].LastExecution != nil {
				t.Fatal("paused intake dispatched a Run")
			}
			current = basicAuditGet(t, h, auditID)
			if current["state"] != "paused" || int(current["submittedRunCount"].(float64)) != 1 {
				t.Fatal("paused intake changed dispatch or completed the Audit")
			}
			basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+auditID+"/resume", nil, "resume", uint64(current["revision"].(float64)), http.StatusOK)
			tasks = waitBasicTask(t, h, auditID, 2, "settled")
			second := tasks[1]
			if second.Result == nil || second.Document.Hypothesis != nil || second.GapCode != nil || second.LastExecution == nil {
				t.Fatalf("second task: %+v", second)
			}
			var result struct {
				Schema string `json:"schema"`
				Items  []struct {
					Verdict         *string  `json:"verdict"`
					Rationale       string   `json:"rationale"`
					FindingReceipts []string `json:"finding_receipts"`
					Origin          struct {
						AuditID string `json:"audit_id"`
						TaskID  string `json:"task_id"`
						RunID   string `json:"run_id"`
					} `json:"origin"`
				} `json:"items"`
			}
			if err := json.Unmarshal(basicTaskArtifact(t, h, auditID, second, "result"), &result); err != nil {
				t.Fatal(err)
			}
			if result.Schema != "contractor.audit.result.v1" || len(result.Items) != 1 || result.Items[0].Verdict != nil || result.Items[0].Origin.TaskID != second.Document.TaskID || result.Items[0].Origin.AuditID != auditID {
				t.Fatalf("exploration result: %+v", result)
			}
			if (len(result.Items[0].FindingReceipts) > 0) != withFinding {
				t.Fatalf("optional finding receipt association: %+v", result.Items[0])
			}
			if withFinding {
				receipts := findingsReceipts(t, h, result.Items[0].Origin.RunID)
				if len(receipts) != 1 || receipts[0].FindingID == nil || receipts[0].Origin.Audit == nil || receipts[0].Origin.Audit.TaskID != second.Document.TaskID {
					t.Fatalf("Project finding origin: %+v", receipts)
				}
			}
			for {
				current = basicAuditGet(t, h, auditID)
				if current["state"] == "completed" {
					break
				}
				if current["state"] == "failed" {
					t.Fatalf("Audit failed: %v", current)
				}
				basicPoll(t, h)
			}
			if int(current["submittedRunCount"].(float64)) != 2 || h.gateway.CompletedStages() != 2 || len(h.gateway.Failures()) != 0 {
				t.Fatalf("duplicate or failed ordinary Run: %v gateway=%v", current, h.gateway.Failures())
			}
			// Retained evidence remains readable after permitted producer purge.
			// The returned Run aliases are informational; Audit sources own bytes.
			producer := result.Items[0].Origin.RunID
			deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, producer)
			materials, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/audits/"+auditID+"/materials?limit=100", nil, "", 0, http.StatusOK)
			if materials["page"].(map[string]any)["hasMore"].(bool) {
				t.Fatal("small retention fixture exceeded its bounded material page")
			}
			retained := 0
			for _, raw := range materials["items"].([]any) {
				entry := raw.(map[string]any)
				material := entry["material"].(map[string]any)
				ref := material["ref"].(map[string]any)
				if !strings.HasPrefix(ref["name"].(string), "evidence-") && !strings.HasPrefix(ref["name"].(string), "source-") {
					continue
				}
				query := url.Values{"key": {entry["key"].(string)}, "revision": {ref["revision"].(string)}}
				data, _ := download(t, h.client, h.baseURL+"/v1/audits/"+auditID+"/artifacts?"+query.Encode())
				if len(data) == 0 {
					t.Fatal("producer purge lost accepted evidence")
				}
				retained++
			}
			if retained < 2 {
				t.Fatal("selected source and annotation were not retained")
			}
			assertCommonAuditEventContract(t, h, auditID)
		})
	}
}

type basicTask struct {
	Findings      *auditstore.CanonicalFindingResult `json:"findings"`
	Document      auditcontract.Task                 `json:"document"`
	State         string                             `json:"state"`
	Attempts      int                                `json:"attempts"`
	LastExecution *string                            `json:"last_execution_id"`
	Result        *auditcontract.Material            `json:"result"`
	Stop          *auditcontract.TaskStop            `json:"stop"`
	GapCode       *string                            `json:"gap_code"`
}

func basicReportStage() domainGatewayStage {
	stage := artifactReportGatewayStage("basic-report")
	stage.steps[1] = toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{
		"name": "copied", "media_type": "text/plain", "text": findingsEvidence,
	}))
	return stage
}

func basicConclusionStage(withFinding bool) domainGatewayStage {
	stage := domainGatewayStage{name: "basic-check", tools: []string{"list_artifacts", "read_artifact", "write_text_artifact", "finding", "read_file", "read_audit_task", "submit_result", "submit_results", "evidence"}, steps: []domainGatewayStep{
		toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
		toolGatewayStep("read_artifact", fixedArguments(map[string]any{"namespace": "inputs", "name": "material"})),
		toolGatewayStep("read_file", fixedArguments(map[string]any{"path": "app.py", "start_line": 1, "max_lines": 100})),
	}}
	stage.steps = append(stage.steps, toolGatewayStep("evidence", auditResultArguments(map[string]any{
		"sources":           []any{map[string]any{"kind": "artifact", "namespace": "inputs", "name": "material"}},
		"interpretation":    "Selected source supports only the stated source observation.",
		"tested_conditions": []string{"source inspected; no live target action"},
		"limitations":       []string{"Source reasoning does not verify deployment behavior"},
	})))
	if withFinding {
		stage.steps = append(stage.steps, toolGatewayStep("finding", func(request map[string]any) (map[string]any, error) {
			receipts := findingsToolResponses(request, "evidence")
			if len(receipts) != 1 || receipts[0]["persistence"] != "accepted" {
				return nil, fmt.Errorf("evidence was not Server accepted: %v", receipts)
			}
			selected, ok := receipts[0]["sourceSelectors"].([]any)
			if !ok || len(selected) != 1 {
				return nil, fmt.Errorf("missing exact reusable source selector")
			}

			return map[string]any{"title": "Query concatenation in lookup", "description": findingsEvidence, "file": "app.py", "range": map[string]int{"start_line": 4, "end_line": 10}, "cwe": "CWE-89", "sources": selected}, nil
		}))
	}
	stage.steps = append(stage.steps, toolGatewayStep("submit_result", auditResultArguments(map[string]any{"rationale": findingsEvidence, "limitations": []string{"No live exploit was executed"}})), finalGatewayStep("Recorded exploration", nil))
	return stage
}

func basicAuditRequest(t *testing.T, h *findingsHarness, method, path string, payload any, key string, revision uint64, status int) (map[string]any, http.Header) {
	t.Helper()
	var body io.Reader
	if payload != nil {
		body = bytes.NewReader(mustBasicJSON(t, payload))
	}
	request, err := http.NewRequestWithContext(h.ctx, method, h.baseURL+path, body)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	if payload != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	if key != "" {
		request.Header.Set("Idempotency-Key", "basic-"+key)
	}
	if revision > 0 {
		request.Header.Set("If-Match", strconv.Quote(strconv.FormatUint(revision, 10)))
	}
	response := do(t, h.client, request, status)
	defer response.Body.Close()
	var value map[string]any
	if err := json.NewDecoder(response.Body).Decode(&value); err != nil {
		t.Fatal(err)
	}
	return value, response.Header.Clone()
}

func basicAuditGet(t *testing.T, h *findingsHarness, id string) map[string]any {
	value, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/audits/"+id, nil, "", 0, http.StatusOK)
	return value
}
func mustBasicJSON(t *testing.T, value any) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
func basicPoll(t *testing.T, h *findingsHarness) {
	t.Helper()
	select {
	case <-time.After(100 * time.Millisecond):
	case <-h.ctx.Done():
		t.Fatal("Audit process scenario timed out")
	}
}
func waitBasicTask(t *testing.T, h *findingsHarness, auditID string, count int, lastState string) []basicTask {
	t.Helper()
	for {
		value, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/audits/"+auditID+"/tasks?limit=64", nil, "", 0, http.StatusOK)
		var page struct {
			Items []basicTask `json:"items"`
		}
		if err := json.Unmarshal(mustBasicJSON(t, value), &page); err != nil {
			t.Fatal(err)
		}
		if len(page.Items) == count && page.Items[count-1].State == lastState {
			return page.Items
		}
		if lastState == "settled" {
			for _, task := range page.Items {
				if task.GapCode != nil || task.State == "ready" && task.Attempts > 0 {
					t.Fatalf("ordinary work failed before the requested result: %+v; Gateway failures: %v", task, h.gateway.Failures())
				}
			}
		}
		if current := basicAuditGet(t, h, auditID); current["state"] == "failed" {
			t.Fatalf("Audit failed before task result: %v", current)
		}
		basicPoll(t, h)
	}
}
func basicTaskArtifact(t *testing.T, h *findingsHarness, auditID string, task basicTask, logical string) []byte {
	t.Helper()
	if task.LastExecution == nil || task.Result == nil || task.Result.Ref.Revision == nil {
		t.Fatal("task has no exact collected output")
	}
	key := "task/" + task.Document.TaskID + "/result"
	path := "/v1/audits/" + auditID + "/artifacts?" + url.Values{"key": {key}, "revision": {*task.Result.Ref.Revision}}.Encode()
	for {
		request, err := http.NewRequestWithContext(h.ctx, http.MethodGet, h.baseURL+path, nil)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+publicToken)
		response, err := h.client.Do(request)
		if err != nil {
			t.Fatal(err)
		}
		data, readErr := io.ReadAll(io.LimitReader(response.Body, 8<<20))
		response.Body.Close()
		if readErr != nil {
			t.Fatal(readErr)
		}
		// The existing read fence reports concurrent owner revision changes.
		// Retry only the same exact immutable selection, never its current ref.
		if response.StatusCode == http.StatusConflict || response.StatusCode == http.StatusServiceUnavailable {
			basicPoll(t, h)
			continue
		}
		if response.StatusCode != http.StatusOK {
			t.Fatalf("exact Task output read HTTP %d: %s", response.StatusCode, data)
		}
		return data
	}
}

func installBasicAuditProgram(t *testing.T, root string) {
	t.Helper()
	// Source ZIP is ordinary input/output material, never an Audit task package.
	copyWorkflow, err := os.ReadFile(filepath.Join(root, "workflows", "artifact_report_memory.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	copyText := replaceAuditFixture(t, string(copyWorkflow), "name: artifact-report-memory-fixture", "name: basic-copy")

	files := map[string]string{"instructions/basic-process-checker.md": "Read the assigned Audit tasks, inspect selected source and record each check with exact evidence. Publish an evidenced problem with finding; finish after all assigned results are recorded.\n", "workflows/basic_copy.yaml": copyText, "agent-templates/basic_check.yaml": basicCheckTemplate, "workflows/basic_check.yaml": basicCheckWorkflow, "audit-profiles/basic_process.yaml": basicProcessProgram}
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(root, name), []byte(content), 0600); err != nil {
			t.Fatal(err)
		}
	}
}

const basicCheckTemplate = `apiVersion: contractor/v1alpha1
kind: AgentTemplate
metadata: {name: basic_check, version: "1"}
spec:
  description: Inspect selected exact source and record ordinary conclusions
  runtime: adk@1
  instructions: {ref: instructions/basic-process-checker.md}
  toolsets:
    - {ref: run-artifacts@1, tools: [list_artifacts, read_artifact]}
    - {ref: text-artifacts@1, tools: [write_text_artifact]}
    - {ref: security-findings-code@1, tools: [finding]}
    - {ref: filesystem@1, tools: [read_file]}
    - {ref: audit-results@2, tools: [read_audit_task, submit_result, submit_results, evidence]}
  sandboxProfile: local-workdir@1
`
const basicCheckWorkflow = `apiVersion: contractor/v1alpha1
kind: Workflow
metadata: {name: basic-check, version: "1"}
spec:
  parameters: {}
  inputs:
    source: {required: true, mediaTypes: [application/zip]}
    material: {required: true, mediaTypes: [text/plain]}
    context: {required: true, mediaTypes: [application/json]}
  outputs:
    result: {required: true, primary: true, mediaTypes: [application/json]}
  executionConfig:
    workers: {llmGateway: local-litellm@1, credential: development-worker, modelPolicy: worker@2}
  entryStage: check
  stages:
    check:
      objective: Inspect the exact selected source and record bounded exploration
      instructions: {ref: instructions/basic-process-checker.md}
      planner: passthrough@1
      agents:
        checker: {template: basic_check@1, namespace: checker}
      context:
        artifacts:
          source: {namespace: inputs, name: source, required: true}
          material: {namespace: inputs, name: material, required: true}
          context: {namespace: inputs, name: context, required: true}
        workspace:
          mode: overlay
          sources: [{artifact: source, target: ""}]
      result:
        artifacts:
          result: {required: true, mediaTypes: [application/json], from: {namespace: checker, name: result}}
      workflowOutputs: {result: result}
      on: {succeeded: {succeed: {}}, failed: {fail: {}}, interrupted: {fail: {}}}
`

// Both direct and checklist-seeded scenarios share this exact source fragment.
const basicInitialWork = "  initial_work: [{key: copy-source, workflow: copy, objective: Author a source review as an ordinary report}]"

func replaceAuditFixture(t *testing.T, source, old, replacement string) string {
	t.Helper()
	if count := strings.Count(source, old); count != 1 {
		t.Fatalf("expected one Audit fixture match for %q, found %d", old, count)
	}
	return strings.Replace(source, old, replacement, 1)
}

func basicSeededProcessProgram(t *testing.T) string {
	t.Helper()
	program := replaceAuditFixture(t, basicProcessProgram, "    report: {required: true, media_types: [text/plain]}", "    report: {required: true, media_types: [text/plain]}\n    checklist: {required: true, media_types: [application/json]}")
	program = replaceAuditFixture(t, program, "material: {source: task-input}", "material: {source: audit-input, name: report}")
	program = replaceAuditFixture(t, program, basicInitialWork, "  initial_work: []\n  seed_source: {input: checklist, workflow: check}")
	for _, pair := range [][2]string{{"max_tasks: 2", "max_tasks: 3"}, {"max_tasks_per_iteration: 2", "max_tasks_per_iteration: 3"}, {"batch_size: 1", "batch_size: 3"}, {"max_hypotheses: 1", "max_hypotheses: 3"}, {"max_hypothesis_revisions: 1", "max_hypothesis_revisions: 8"}} {
		program = replaceAuditFixture(t, program, pair[0], pair[1])
	}
	return program
}

const basicProcessProgram = `apiVersion: contractor/v1alpha1
kind: AuditProfile
metadata: {name: basic-process, version: "1"}
spec:
  schema: contractor.audit.program.v1
  inputs:
    source: {required: true, media_types: [application/zip]}
    report: {required: true, media_types: [text/plain]}
  workflows:
    copy:
      ref: basic-copy@1
      inputs: {source: {source: audit-input, name: report}}
      parameters: {}
      outputs: {material: result}
    check:
      ref: basic-check@1
      inputs: {source: {source: audit-input, name: source}, material: {source: task-input}, context: {source: task-context}}
      parameters: {}
      outputs: {result: result}
  hypotheses: {}
` + basicInitialWork + `
  limits:
    max_tasks: 2
    max_iterations: 1
    max_tasks_per_iteration: 2
    batch_size: 1
    max_submitted_runs: 4
    max_attempts: 2
    deadline_seconds: 600
    max_evidence_bytes: 1048576
    max_hypotheses: 1
    max_hypothesis_revisions: 1
    max_hypothesis_bytes: 32768
  policy: {active_operations: prohibited}
`

func assertCommonAuditEventContract(t *testing.T, h *findingsHarness, auditID string) {
	t.Helper()
	document, err := openapi3.NewLoader().LoadFromFile(filepath.Join(repoRoot(t), "api/openapi/contractor-public-v1.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	schema := document.Components.Schemas["AuditEventPage"].Value
	next := ""
	seen := map[string]bool{}
	count := 0
	for {
		query := url.Values{"limit": {"200"}}
		if next != "" {
			query.Set("cursor", next)
		}
		page, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/audits/"+auditID+"/events?"+query.Encode(), nil, "", 0, http.StatusOK)
		if err := schema.VisitJSON(page, openapi3.EnableJSONSchema2020()); err != nil {
			t.Fatalf("common Audit event page violates public contract: %v", err)
		}
		count += len(page["items"].([]any))
		continuation := page["page"].(map[string]any)
		if !continuation["hasMore"].(bool) {
			break
		}
		next = continuation["nextCursor"].(string)
		if seen[next] {
			t.Fatal("repeated Audit event cursor")
		}
		seen[next] = true
	}
	if count == 0 {
		t.Fatal("the real common Audit emitted no event history")
	}
}

func assignedAuditKeys(request map[string]any) ([]string, error) {
	reads := findingsToolResponses(request, "read_audit_task")
	if len(reads) == 0 {
		return nil, fmt.Errorf("assigned Audit tasks were not read")
	}
	tasks, ok := reads[len(reads)-1]["tasks"].([]any)
	if !ok {
		return nil, fmt.Errorf("assigned Audit tasks have no task array")
	}
	keys := make([]string, 0, len(tasks))
	for _, raw := range tasks {
		task, ok := raw.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("assigned Audit member has an invalid shape")
		}
		key, ok := task["item_key"].(string)
		if !ok || key == "" {
			return nil, fmt.Errorf("assigned Audit member has no item_key")
		}
		keys = append(keys, key)
	}
	return keys, nil
}

// The scripted model explicitly selects this trusted source for the indexed item.
// Task reading alone performs no evidence association.
func auditSourceEvidenceArguments(index int, conditions []string) func(map[string]any) (map[string]any, error) {
	return func(request map[string]any) (map[string]any, error) {
		keys, err := assignedAuditKeys(request)
		if err != nil {
			return nil, err
		}
		if index < 0 || index >= len(keys) {
			return nil, fmt.Errorf("missing assigned evidence item")
		}
		return map[string]any{"item_key": keys[index], "sources": []any{map[string]any{"kind": "artifact", "namespace": "inputs", "name": "source"}}, "interpretation": "The selected source supports only this scoped fixture interpretation; deployment behavior is independently unverified.", "tested_conditions": conditions, "limitations": []string{}}, nil
	}
}

func selectedAuditMaterialKey(t *testing.T, h *findingsHarness, auditID string, material contracts.MaterialDescriptor) string {
	t.Helper()
	page, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/audits/"+auditID+"/materials?limit=100", nil, "", 0, http.StatusOK)
	if page["page"].(map[string]any)["hasMore"].(bool) {
		t.Fatal("small source fixture exceeded its material page")
	}
	wanted, err := json.Marshal(material)
	if err != nil {
		t.Fatal(err)
	}
	var expected map[string]any
	if err := json.Unmarshal(wanted, &expected); err != nil {
		t.Fatal(err)
	}
	for _, raw := range page["items"].([]any) {
		row := raw.(map[string]any)
		if reflect.DeepEqual(row["material"], expected) {
			return row["key"].(string)
		}
	}
	t.Fatal("accepted exact material has no ordinary Audit link")
	return ""
}

func auditHTTPSourceEvidenceArguments(ids []int, conditions []string) func(map[string]any) (map[string]any, error) {
	sources := []any{}
	for _, id := range ids {
		sources = append(sources, map[string]any{"kind": "http", "request_id": id})
	}
	return auditResultArguments(map[string]any{"sources": sources, "interpretation": "Actual captured requests support only the stated fixture observations, not a target vulnerability verdict.", "tested_conditions": conditions, "limitations": []string{}})
}

func auditResultArguments(value map[string]any) func(map[string]any) (map[string]any, error) {
	return func(request map[string]any) (map[string]any, error) {
		args := make(map[string]any, len(value)+1)
		for key, item := range value {
			args[key] = item
		}
		if _, selected := args["item_key"]; selected {
			return args, nil
		}
		keys, err := assignedAuditKeys(request)
		if err != nil {
			return nil, err
		}
		if len(keys) != 1 {
			return nil, fmt.Errorf("a scalar fixture must select its assigned item explicitly")
		}
		args["item_key"] = keys[0]
		return args, nil
	}
}

func requireRecordedAuditResults(count int) func(map[string]any) error {
	return func(request map[string]any) error {
		responses := findingsToolResponses(request, "submit_result")
		if len(responses) != count {
			return fmt.Errorf("scalar result responses = %d, want %d", len(responses), count)
		}
		for _, response := range responses {
			if response["status"] != "recorded" {
				return fmt.Errorf("intended scalar result was not recorded")
			}
		}
		return nil
	}
}
