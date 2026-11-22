//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

func TestProjectFactsAcrossProcesses(t *testing.T) {
	tools := withMemoryTools([]string{"finding", "record_fact", "list_project_findings", "read_project_finding", "select_project_findings", "read_finding_selection", "read_artifact", "write_text_artifact"})
	producer := domainGatewayStage{name: "facts/producer", tools: tools}
	for index, severity := range []string{"informational", "informational", "low", ""} {
		name := fmt.Sprintf("proof-%d", index)
		producer.steps = append(producer.steps, toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{"name": name, "text": fmt.Sprintf("Source %d reports nginx; independently unverified.", index), "media_type": "text/plain"})))
		tool := "finding"
		if index == 0 {
			tool = "record_fact"
		}
		producer.steps = append(producer.steps, toolGatewayStep(tool, func(request map[string]any) (map[string]any, error) {
			proof, ok := lastExactArtifact(request, "findings-review", name)
			if !ok {
				return nil, fmt.Errorf("missing exact source")
			}
			args := map[string]any{"title": fmt.Sprintf("Observation %d", index), "description": "Reported observation from selected material; not independently verified.", "evidence_refs": []any{materialSelector(proof)}}
			if tool == "finding" && severity != "" {
				args["severity_suggestion"] = severity
			}
			return args, nil
		}))
	}
	producer.steps = append(producer.steps, toolGatewayStep("finding", func(request map[string]any) (map[string]any, error) {
		proof, ok := lastExactArtifact(request, "findings-review", "proof-0")
		if !ok {
			return nil, fmt.Errorf("missing original fact evidence")
		}
		return map[string]any{"title": "Another title over the same source", "description": "Independent informational submission", "severity_suggestion": "informational", "evidence_refs": []any{materialSelector(proof)}}, nil
	}))
	producer.steps = append(producer.steps, toolGatewayStep("list_project_findings", fixedArguments(map[string]any{"limit": 100})), toolGatewayStep("select_project_findings", func(request map[string]any) (map[string]any, error) {
		pages := findingsToolResponses(request, "list_project_findings")
		if len(pages) != 1 {
			return nil, fmt.Errorf("missing live Project findings")
		}
		rows := pages[0]["items"].([]any)
		if len(rows) != 4 {
			return nil, fmt.Errorf("expected four unassessed records, got %d", len(rows))
		}
		for _, raw := range rows {
			row := raw.(map[string]any)
			if row["severitySuggestion"] == "informational" {
				return map[string]any{"selection_key": "selected-findings", "findings": []any{map[string]any{"findingId": row["findingId"], "revision": row["revision"]}}}, nil
			}
		}
		return nil, fmt.Errorf("missing informational record")
	}), toolGatewayStep("read_finding_selection", func(request map[string]any) (map[string]any, error) {
		responses := findingsToolResponses(request, "select_project_findings")
		if len(responses) != 1 {
			return nil, fmt.Errorf("no selection: %+v", responses)
		}
		artifact, ok := responses[0]["artifact"].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("selection failed: %+v", responses)
		}
		return map[string]any{"collection_ref": artifact["ref"]}, nil
	}), toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{"name": "report", "text": "Facts and issues retained with an exact selected material.", "media_type": "text/markdown"})), finalGatewayStep("Retained", map[string]domainArtifactBinding{"report": {namespace: "findings-review", name: "report"}}))
	rejected := domainGatewayStage{name: "facts/project-required", tools: tools, steps: []domainGatewayStep{
		toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{"name": "proof", "text": "nginx source", "media_type": "text/plain"})),
	}}
	for _, tool := range []string{"finding", "record_fact"} {
		rejected.steps = append(rejected.steps, toolGatewayStep(tool, func(request map[string]any) (map[string]any, error) {
			proof, ok := lastExactArtifact(request, "findings-review", "proof")
			if !ok {
				return nil, fmt.Errorf("missing source")
			}
			return map[string]any{"title": "nginx", "description": "Reported source", "evidence_refs": []any{materialSelector(proof)}}, nil
		}))
	}
	rejected.steps = append(rejected.steps, toolGatewayStep("write_text_artifact", func(request map[string]any) (map[string]any, error) {
		for _, tool := range []string{"finding", "record_fact"} {
			responses := findingsToolResponses(request, tool)
			body, _ := json.Marshal(responses)
			if len(responses) != 1 || !strings.Contains(string(body), "invalid_finding_request") {
				return nil, fmt.Errorf("projectless %s did not return a permanent error: %s", tool, body)
			}
		}
		return map[string]any{"name": "report", "text": "Project required by both tools.", "media_type": "text/markdown"}, nil
	}), finalGatewayStep("Rejected", map[string]domainArtifactBinding{"report": {namespace: "findings-review", name: "report"}}))
	h := startConfiguredFindingsHarness(t, []domainGatewayStage{producer, rejected}, func(root string) {
		p := filepath.Join(root, "agent-templates", "project_findings_analyst.yaml")
		data, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		data = []byte(strings.Replace(string(data), "tools: [list_project_findings, read_project_finding, select_project_findings, read_finding_selection]", "tools: [finding, record_fact, list_project_findings, read_project_finding, select_project_findings, read_finding_selection]", 1))
		if err := os.WriteFile(p, data, 0600); err != nil {
			t.Fatal(err)
		}
	})
	project := createProjectResource(t, h.client, h.baseURL)
	body, _ := json.Marshal(map[string]any{"workflow": "project-findings-review@1", "artifacts": map[string]any{}})
	run := postProjectRun(t, h.client, h.baseURL, project.ProjectID, "facts", body, false)
	waitForFindingsRun(t, h, run, "succeeded")
	receipts := findingsReceipts(t, h, run)
	if len(receipts) != 5 {
		t.Fatalf("receipts=%d", len(receipts))
	}
	for _, receipt := range receipts {
		if receipt.FindingID == nil || receipt.ProjectID == nil || receipt.ProjectProposal == nil {
			t.Fatalf("not admitted: %+v", receipt)
		}
	}
	var original, duplicate string
	for _, receipt := range receipts {
		if receipt.Document.Title == "Observation 0" {
			original = *receipt.FindingID
		}
		if receipt.Document.Title == "Another title over the same source" {
			duplicate = *receipt.FindingID
		}
	}
	if original == "" || original != duplicate {
		t.Fatal("equal evidence did not reuse canonical Finding")
	}
	var facts, needsReview struct {
		Items []auditstore.ProjectFindingSummary `json:"items"`
	}
	auditProgramGET(t, h.client, h.baseURL+"/v1/findings?severity=informational", &facts)
	auditProgramGET(t, h.client, h.baseURL+"/v1/findings?verdict=needs_review", &needsReview)
	if len(facts.Items) != 2 || len(needsReview.Items) != 2 {
		t.Fatalf("facts=%d needsReview=%d", len(facts.Items), len(needsReview.Items))
	}
	for _, row := range append(facts.Items, needsReview.Items...) {
		if row.AnalystVerdict != nil || row.AnalystSeverity != nil {
			t.Fatal("producer fabricated analyst assessment")
		}
	}
	selection, _ := download(t, h.client, h.baseURL+"/v1/runs/"+run+"/outputs/selection")
	if len(selection) == 0 {
		t.Fatal("missing exact selection")
	}
	// Ordinary projectless Runs are still allowed, but both intake tools reject.
	request, err := http.NewRequest(http.MethodPost, h.baseURL+"/v1/runs", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", "facts-without-project")
	result := do(t, h.client, request, http.StatusAccepted)
	var response runCreateResponse
	decodeResponse(t, result, &response)
	result.Body.Close()
	waitForFindingsRun(t, h, response.RunID, "succeeded")
	if len(findingsReceipts(t, h, response.RunID)) != 0 {
		t.Fatal("projectless call wrote a receipt")
	}
	if h.gateway.CompletedStages() != 2 || len(h.gateway.Failures()) != 0 {
		t.Fatal(h.gateway.Failures())
	}
}
