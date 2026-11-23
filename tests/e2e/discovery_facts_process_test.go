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
)

func TestDiscoveryFactsAcrossProcesses(t *testing.T) {
	// Keep the production discovery template, instructions, workspace and tool
	// selection. Only shorten the Workflow to its existing discovery boundary.
	stage := func(name string, publish, project bool) domainGatewayStage {
		tools := domainGatewayStages()[1].tools
		steps := []domainGatewayStep{
			toolGatewayStep("read_artifact", stageReadArguments("dependency_report", nil)),
			toolGatewayStep("read_file", fixedArguments(map[string]any{"path": "app.py", "start_line": 1, "max_lines": 100})),
		}
		if publish {
			steps = append(steps, toolGatewayStep("fact", fixedArguments(map[string]any{
				"title":       "Widget lookup calls an external inventory service",
				"description": "The source-defined widget handler calls inventory.example through httpx. Deployment reachability and credentials are untested.",
				"sources":     []any{map[string]any{"kind": "code", "file": "app.py", "start_line": 15, "end_line": 17, "workspace": true}},
			})))
		}
		steps = append(steps, toolGatewayStep("write_text_artifact", func(request map[string]any) (map[string]any, error) {
			responses := findingsToolResponses(request, "fact")
			report := "# Project Structure and Runtime Inventory\n\nNo reusable facts selected.\n"
			if publish {
				if len(responses) != 1 {
					return nil, fmt.Errorf("expected one fact response: %+v", responses)
				}
				if project {
					if responses[0]["receipt_id"] == nil {
						return nil, fmt.Errorf("fact was not saved: %+v", responses)
					}
					report = "# Project Structure and Runtime Inventory\n\nSaved source-derived inventory integration fact; live behavior untested.\n"
				} else {
					data, _ := json.Marshal(responses)
					if !strings.Contains(string(data), "invalid_finding_request") {
						return nil, fmt.Errorf("missing permanent Project denial: %s", data)
					}
					report = "# Project Structure and Runtime Inventory\n\nProject publication denied; no fact saved.\n"
				}
			} else if len(responses) != 0 {
				return nil, fmt.Errorf("zero-fact investigation published a fact")
			}
			return map[string]any{"name": "project", "text": report, "media_type": "text/markdown"}, nil
		}), finalGatewayStep("Discovery report published", map[string]domainArtifactBinding{"project_report": {namespace: "analysis", name: "project"}}))
		return domainGatewayStage{name: name, tools: tools, steps: steps}
	}
	h := startConfiguredFindingsHarness(t, []domainGatewayStage{
		stage("discovery/project", true, true), stage("discovery/projectless", true, false), stage("discovery/zero", false, true),
	}, func(root string) {
		data, err := os.ReadFile(filepath.Join(repoRoot(t), "tests/e2e/testdata/discovery-facts/workflows/discovery.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "workflows/discovery_facts.yaml"), data, 0600); err != nil {
			t.Fatal(err)
		}
	})
	project := createProjectResource(t, h.client, h.baseURL)
	// Each Run selects exact materials from its own authorized input scope.
	source := uploadProjectArtifact(t, h.client, h.baseURL, "discovery-source", "application/zip", projectSourceArchive(t))
	dependencies := uploadProjectArtifact(t, h.client, h.baseURL, "discovery-dependencies", "text/markdown", []byte(dependencyReport))
	projectSource := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "sources", "discovery-source", "application/zip", projectSourceArchive(t))
	projectDependencies := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "sources", "discovery-dependencies", "text/markdown", []byte(dependencyReport))
	for index, bound := range []bool{true, false, true} {
		inputs := map[string]artifactRef{"source": source, "dependency_report": dependencies}
		if bound {
			inputs = map[string]artifactRef{"source": projectSource, "dependency_report": projectDependencies}
		}
		body, err := json.Marshal(map[string]any{"workflow": "discovery-facts@1", "artifacts": inputs})
		if err != nil {
			t.Fatal(err)
		}
		var run string
		if bound {
			run = postProjectRun(t, h.client, h.baseURL, project.ProjectID, fmt.Sprintf("discovery-%d", index), body, false)
		} else {
			request, err := http.NewRequest(http.MethodPost, h.baseURL+"/v1/runs", bytes.NewReader(body))
			if err != nil {
				t.Fatal(err)
			}
			request.Header.Set("Authorization", "Bearer "+publicToken)
			request.Header.Set("Content-Type", "application/json")
			request.Header.Set("Idempotency-Key", "projectless-discovery")
			response := do(t, h.client, request, http.StatusAccepted)
			var created runCreateResponse
			decodeResponse(t, response, &created)
			response.Body.Close()
			run = created.RunID
		}
		waitForFindingsRun(t, h, run, "succeeded")
		report, _ := download(t, h.client, h.baseURL+"/v1/runs/"+run+"/outputs/report")
		if !strings.Contains(string(report), "# Project Structure and Runtime Inventory") {
			t.Fatal("discovery report missing")
		}
		receipts := findingsReceipts(t, h, run)
		if index != 0 {
			if len(receipts) != 0 {
				t.Fatalf("unexpected publication: %+v", receipts)
			}
			continue
		}
		if len(receipts) != 1 || receipts[0].FindingID == nil || receipts[0].ProjectID == nil || *receipts[0].ProjectID != project.ProjectID || receipts[0].Document.SeveritySuggestion != "informational" || receipts[0].Origin.RunID != run || receipts[0].Origin.Audit != nil {
			t.Fatalf("discovery fact not retained with Project/Run origin: %+v", receipts)
		}
		proofFound := false
		for _, evidence := range receipts[0].Evidence {
			if evidence.MediaType != "application/json" {
				continue
			}
			// Read the retained exact proof through the normal Run artifact API.
			ref := evidence.Ref
			payload, _ := download(t, h.client, h.baseURL+"/v1/runs/"+run+"/artifacts/"+ref.Namespace+"/"+ref.Name+"?revision="+url.QueryEscape(*ref.Revision))
			if strings.Contains(string(payload), "contractor.code-source-evidence.v1") && strings.Contains(string(payload), "inventory.example") {
				proofFound = true
			}
		}
		if !proofFound {
			t.Fatal("fact lost exact source region evidence")
		}
	}
	if h.gateway.CompletedStages() != 3 || len(h.gateway.Failures()) != 0 {
		t.Fatalf("gateway completed %d stages: %v", h.gateway.CompletedStages(), h.gateway.Failures())
	}
}
