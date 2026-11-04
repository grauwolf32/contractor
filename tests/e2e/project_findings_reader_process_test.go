//go:build e2e

package e2e

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// This is the ordinary Workflow production path: real Server, Runtime and
// PostgreSQL, with a scripted provider choosing exact tools and inputs.
func TestProjectFindingsReaderAcrossProcesses(t *testing.T) {
	seed := domainGatewayStage{name: "findings/project-seed", tools: withMemoryTools([]string{"finding", "read_artifact", "write_text_artifact"})}
	seed.steps = append(seed.steps, toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{"name": "proof", "text": "Project proof under selected conditions", "media_type": "text/plain"})))
	for index := range 101 {
		seed.steps = append(seed.steps, toolGatewayStep("finding", func(request map[string]any) (map[string]any, error) {
			exact, ok := lastExactArtifact(request, "findings-review", "proof")
			if !ok {
				return nil, fmt.Errorf("missing exact proof")
			}
			return map[string]any{"title": fmt.Sprintf("Project problem %03d", index), "description": "An evidenced observed problem in the selected conditions", "evidence_refs": []any{exact}, "publish_problem": true}, nil
		}))
	}
	seed.steps = append(seed.steps, toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{"name": "report", "text": "Seeded Project problems", "media_type": "text/markdown"})), finalGatewayStep("Project problems retained", map[string]domainArtifactBinding{"report": {namespace: "findings-review", name: "report"}}))
	reader := domainGatewayStage{name: "findings/project-reader", tools: withMemoryTools([]string{"list_project_findings", "read_project_finding", "select_project_findings", "read_finding_selection", "read_artifact", "write_text_artifact"})}
	reader.steps = append(reader.steps, toolGatewayStep("list_project_findings", fixedArguments(map[string]any{"limit": 100})), toolGatewayStep("list_project_findings", func(request map[string]any) (map[string]any, error) {
		pages := findingsToolResponses(request, "list_project_findings")
		if len(pages) != 1 || len(pages[0]["items"].([]any)) != 100 || pages[0]["nextCursor"] == nil {
			return nil, fmt.Errorf("first Project page incomplete")
		}
		return map[string]any{"limit": 100, "cursor": pages[0]["nextCursor"]}, nil
	}), toolGatewayStep("read_project_finding", func(request map[string]any) (map[string]any, error) {
		pages := findingsToolResponses(request, "list_project_findings")
		if len(pages) != 2 || len(pages[1]["items"].([]any)) != 1 || pages[1]["nextCursor"] != nil {
			return nil, fmt.Errorf("did not finish Project discovery")
		}
		item := pages[0]["items"].([]any)[0].(map[string]any)
		return map[string]any{"finding_id": item["findingId"], "revision": item["revision"]}, nil
	}), toolGatewayStep("select_project_findings", func(request map[string]any) (map[string]any, error) {
		pages := findingsToolResponses(request, "list_project_findings")
		item := pages[0]["items"].([]any)[0].(map[string]any)
		return map[string]any{"selection_key": "selected-findings", "findings": []any{map[string]any{"findingId": item["findingId"], "revision": item["revision"]}}}, nil
	}), toolGatewayStep("read_finding_selection", func(request map[string]any) (map[string]any, error) {
		selected := findingsToolResponses(request, "select_project_findings")
		if len(selected) != 1 {
			return nil, fmt.Errorf("no selected snapshot")
		}
		return map[string]any{"collection_ref": selected[0]["artifact"].(map[string]any)["ref"]}, nil
	}))
	for _, evidence := range []bool{false, true} {
		reader.steps = append(reader.steps, toolGatewayStep("read_artifact", func(request map[string]any) (map[string]any, error) {
			pages := findingsToolResponses(request, "read_finding_selection")
			if len(pages) != 1 || pages[0]["next_cursor"] != nil {
				return nil, fmt.Errorf("snapshot not fully enumerated")
			}
			item := pages[0]["items"].([]any)[0].(map[string]any)
			exact := item["proposal"].(map[string]any)
			if evidence {
				exact = item["evidence"].([]any)[0].(map[string]any)
			}
			return exact["ref"].(map[string]any), nil
		}))
	}
	reader.steps = append(reader.steps, toolGatewayStep("read_artifact", stageRefArguments("check_results", nil)), toolGatewayStep("write_text_artifact", func(request map[string]any) (map[string]any, error) {
		reads := findingsToolResponses(request, "read_artifact")
		if len(reads) != 3 {
			return nil, fmt.Errorf("missing full selected evidence or negative result")
		}
		proof, err := base64.StdEncoding.DecodeString(fmt.Sprint(reads[1]["dataBase64"]))
		if err != nil || string(proof) != "Project proof under selected conditions" {
			return nil, fmt.Errorf("exact proof changed")
		}
		negative, err := base64.StdEncoding.DecodeString(fmt.Sprint(reads[2]["dataBase64"]))
		if err != nil || !strings.Contains(string(negative), "refuted") {
			return nil, fmt.Errorf("negative check lost")
		}
		selection, _ := json.Marshal(findingsToolResponses(request, "select_project_findings")[0])
		return map[string]any{"name": "report", "media_type": "text/markdown", "text": fmt.Sprintf("101 live Project findings; 1 explicit exact snapshot.\n%s\n%s\n%s", selection, proof, negative)}, nil
	}), finalGatewayStep("Project analysis retained", map[string]domainArtifactBinding{"report": {namespace: "findings-review", name: "report"}}))
	h := startConfiguredFindingsHarness(t, []domainGatewayStage{seed, reader}, func(root string) {
		source, err := os.ReadFile(filepath.Join(root, "workflows", "project_findings_review.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		workflow := strings.Replace(string(source), "name: project-findings-review", "name: project-findings-seed", 1)
		workflow = strings.Replace(workflow, "template: project_findings_analyst@1", "template: project_findings_seed@1", 1)
		if err := os.WriteFile(filepath.Join(root, "workflows", "project_findings_seed.yaml"), []byte(workflow), 0600); err != nil {
			t.Fatal(err)
		}
		source, err = os.ReadFile(filepath.Join(root, "agent-templates", "project_findings_analyst.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		template := strings.Replace(string(source), "name: project_findings_analyst", "name: project_findings_seed", 1)
		template = strings.Replace(template, "tools: [list_project_findings, read_project_finding, select_project_findings, read_finding_selection]", "tools: [finding]", 1)
		if err := os.WriteFile(filepath.Join(root, "agent-templates", "project_findings_seed.yaml"), []byte(template), 0600); err != nil {
			t.Fatal(err)
		}
	})
	t.Cleanup(func() {
		t.Logf("Project reader gateway calls=%d completed=%d", h.gateway.Calls(), h.gateway.CompletedStages())
	})
	project := createProjectResource(t, h.client, h.baseURL)
	seedBody, _ := json.Marshal(map[string]any{"workflow": "project-findings-seed@1", "artifacts": map[string]any{}})
	seedRun := postProjectRun(t, h.client, h.baseURL, project.ProjectID, "project-seed", seedBody, false)
	waitForFindingsRun(t, h, seedRun, "succeeded")
	negative := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "results", "negative", "application/json", []byte(`{"verdict":"refuted","tested_conditions":["selected baseline"],"finding":null}`))
	body, _ := json.Marshal(map[string]any{"workflow": "project-findings-review@1", "artifacts": map[string]any{"check_results": negative}})
	run := postProjectRun(t, h.client, h.baseURL, project.ProjectID, "project-review", body, false)
	waitForFindingsRun(t, h, run, "succeeded")
	report, _ := download(t, h.client, h.baseURL+"/v1/runs/"+run+"/outputs/report")
	if !strings.Contains(string(report), "101 live Project findings") || !strings.Contains(string(report), "refuted") || !strings.Contains(string(report), "selected-findings") {
		t.Fatalf("report lost exact context: %s", report)
	}
	selectedBytes, _ := download(t, h.client, h.baseURL+"/v1/runs/"+run+"/outputs/selection")
	snapshot, _, err := auditdomain.DecodeFindingCollectionPackage(selectedBytes)
	if err != nil || len(snapshot.Entries) != 1 || snapshot.Entries[0].Finding == nil {
		t.Fatalf("selection was not an ordinary exact Workflow output: %+v %v", snapshot, err)
	}
	if len(findingsReceipts(t, h, run)) != 0 {
		t.Fatal("reader created problem findings")
	}
	if len(h.gateway.Failures()) != 0 || h.gateway.CompletedStages() != 2 {
		t.Fatal(h.gateway.Failures())
	}
}
