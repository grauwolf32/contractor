//go:build e2e

package e2e

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

// Fixed assertions reach the real Runtime without a generation Run. Only the
// model's bounded judgments are scripted, as in the ordinary process proof.
func TestAuditFixedHypothesesAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Fatal("the assertion process proof cannot run in short mode")
	}
	stage := basicConclusionStage(false)
	results := []any{}
	for _, verdict := range []string{"supported", "refuted", "inconclusive"} {
		results = append(results, map[string]any{"verdict": verdict, "rationale": "Judgment applies only to the selected fixture", "limitations": []string{"No global claim follows from this sample"}})
	}
	stage.steps = []domainGatewayStep{toolGatewayStep("read_audit_task", fixedArguments(map[string]any{}))}
	for index := range results {
		stage.steps = append(stage.steps, toolGatewayStep("evidence", auditSourceEvidenceArguments(index, []string{"Selected fixture and actor only"})))
	}
	stage.steps = append(stage.steps, toolGatewayStep("submit_results", func(request map[string]any) (map[string]any, error) {
		keys, err := assignedAuditKeys(request)
		if err != nil {
			return nil, err
		}
		if len(keys) != len(results) {
			return nil, fmt.Errorf("assigned batch size differs")
		}
		items := make([]any, 0, len(keys))
		for index := len(keys) - 1; index >= 0; index-- {
			item := results[index].(map[string]any)
			item["item_key"] = keys[index]
			items = append(items, item)
		}
		return map[string]any{"items": items}, nil
	}), finalGatewayStep("Recorded scoped conclusions", nil))
	h := startConfiguredFindingsHarness(t, []domainGatewayStage{stage}, func(root string) {
		installBasicAuditProgram(t, root)
		program := basicSeededProcessProgram(t)
		if err := os.WriteFile(filepath.Join(root, "audit-profiles/basic_process.yaml"), []byte(program), 0600); err != nil {
			t.Fatal(err)
		}
	})
	project := createProjectResource(t, h.client, h.baseURL)
	source := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "source", "application/zip", auditProgramZip(t, map[string][]byte{"app.py": []byte(findingsSource)}))
	report := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "report", "text/plain", []byte(findingsEvidence))
	entries := []any{}
	for index, statement := range []string{"A reader can access another owner's record", "Only the owner can access this record", "An unobserved path may bypass the owner check"} {
		entries = append(entries, map[string]any{"key": auditcontract.ExecutionItemKey(index), "version": "1", "statement": statement, "scope": map[string]string{"conditions": "Selected fixture and actor only"}, "grounds": "An explicit fixed source assertion", "applicability": "selected fixture", "allowed_methods": []string{"static"}, "required_evidence": []string{"Exact source and tested conditions"}})
	}
	checklist := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "checklist", "application/json", mustBasicJSON(t, map[string]any{"schema": "contractor.audit.assertions.v1", "items": entries}))
	draft, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": "basic-process", "version": "1"}, "inputs": map[string]artifactRef{"source": source, "report": report, "checklist": checklist}}, "assertion-draft", 0, http.StatusCreated)
	id := draft["auditId"].(string)
	blocked := h.gateway.blockNextRequest()
	basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/start", nil, "assertion-start", uint64(draft["revision"].(float64)), http.StatusOK)
	select {
	case <-blocked:
	case <-h.ctx.Done():
		t.Fatal("fixed checks did not reach Runtime")
	}
	registry, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/audits/"+id+"/hypotheses?limit=64", nil, "", 0, http.StatusOK)
	rows := registry["items"].([]any)
	if len(rows) != 3 || registry["fixed_denominator"].(float64) != 3 {
		t.Fatalf("fixed source membership=%v", registry)
	}
	first := rows[0].(map[string]any)
	document := first["document"].(map[string]any)
	current := basicAuditGet(t, h, id)
	body := map[string]any{"key": first["source_key"], "statement": "A later edited assertion in another scope", "scope": document["scope"], "grounds": "An explicit human edit after Task acceptance"}
	revision := uint64(current["revision"].(float64))
	path := "/v1/audits/" + id + "/hypotheses/" + document["id"].(string)
	basicAuditRequest(t, h, http.MethodPatch, path, body, "edit-running-assertion", revision, http.StatusOK)
	_, headers := basicAuditRequest(t, h, http.MethodPatch, path, body, "edit-running-assertion", revision, http.StatusOK)
	if headers.Get("Idempotency-Replayed") != "true" {
		t.Fatal("hypothesis edit lost its receipt")
	}
	h.gateway.releaseBlockedRequest()
	tasks := waitBasicTask(t, h, id, 3, "settled")
	if h.gateway.CompletedStages() != 1 || len(h.gateway.Failures()) != 0 {
		t.Fatalf("fixed seeding introduced Runs or failed: stages=%d failures=%v", h.gateway.CompletedStages(), h.gateway.Failures())
	}
	var output struct {
		Items []struct {
			Verdict string `json:"verdict"`
			Origin  struct {
				TaskID     string                      `json:"task_id"`
				Hypothesis auditcontract.HypothesisRef `json:"hypothesis"`
			} `json:"origin"`
		} `json:"items"`
	}
	if err := json.Unmarshal(basicTaskArtifact(t, h, id, tasks[0], "result"), &output); err != nil {
		t.Fatal(err)
	}
	if len(output.Items) != 3 {
		t.Fatalf("batch result=%+v", output)
	}
	for index, task := range tasks {
		if task.Document.Hypothesis == nil || task.Document.Hypothesis.Revision != 1 || output.Items[index].Origin.TaskID != task.Document.TaskID || output.Items[index].Origin.Hypothesis != *task.Document.Hypothesis || output.Items[index].Verdict != []string{"supported", "refuted", "inconclusive"}[index] {
			t.Fatalf("per-item pin changed: task=%+v result=%+v", task, output.Items[index])
		}
	}
	findings, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/projects/"+project.ProjectID+"/findings?limit=64", nil, "", 0, http.StatusOK)
	if len(findings["items"].([]any)) != 0 {
		t.Fatal("verdict polarity published a Project Finding")
	}
	registry, _ = basicAuditRequest(t, h, http.MethodGet, "/v1/audits/"+id+"/hypotheses?limit=64", nil, "", 0, http.StatusOK)
	if registry["fixed_denominator"].(float64) != 3 || registry["items"].([]any)[0].(map[string]any)["document"].(map[string]any)["revision"].(float64) != 2 {
		t.Fatal("edit changed fixed coverage or did not append a revision")
	}
}
