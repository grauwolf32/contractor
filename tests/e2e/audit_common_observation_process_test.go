//go:build e2e

package e2e

import (
	"encoding/json"
	"net/http"
	"testing"
)

func TestAuditCommonObservationProfilesAcrossProductionProcesses(t *testing.T) {
	for _, mode := range []string{"observe", "trace"} {
		t.Run(mode, func(t *testing.T) {
			tools := []string{"list_skills", "load_skill", "load_skill_resource", "ls", "glob", "read_file", "grep", "search_def", "list_symbols", "graph_summary", "find_symbol", "find_callers", "find_callees", "paths_between", "entrypoint_paths_to", "attack_surface", "complexity_hotspots", "functions_that_raise", "read_audit_task", "submit_check_result", "submit_check_results"}
			version := "4"
			if mode == "trace" {
				version = "5"
				tools = append(tools, "write_text_artifact", "finding", "annotate_trace", "annotate_validate", "annotate_sink", "changed_paths", "diff", "rollback_changes")
			}
			stage := domainGatewayStage{name: "common-operation-" + mode, tools: withMemoryTools(tools), steps: []domainGatewayStep{
				toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
				toolGatewayStep("load_skill", fixedArguments(map[string]any{"skill_name": "trace"})),
				toolGatewayStep("read_file", fixedArguments(map[string]any{"path": "app.py", "start_line": 1, "max_lines": 100})),
				toolGatewayStep("submit_check_result", auditResultArguments(map[string]any{"rationale": "The selected operation and local source were inspected within the assigned scope.", "tested_conditions": []string{"GET widget operation against the exact source snapshot"}, "limitations": []string{"No external target activity or exploit was attempted"}, "evidence": []string{"evidence-000", "evidence-002"}})),
				finalGatewayStep("Recorded bounded source exploration", nil),
			}}
			h := startConfiguredFindingsHarness(t, []domainGatewayStage{stage}, nil)
			project := createProjectResource(t, h.client, h.baseURL)
			source := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "source", "application/zip", projectSourceArchive(t))
			contract := []byte(`{"openapi":"3.1.0","info":{"title":"Widget fixture","version":"1"},"paths":{"/widgets/{widget_id}":{"get":{"operationId":"getWidget","parameters":[{"name":"widget_id","in":"path","required":true,"schema":{"type":"string"}}],"responses":{"200":{"description":"Widget availability"}}}}}}`)
			api := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "openapi", "application/json", contract)
			draft, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": "openapi-operation-" + mode, "version": version}, "inputs": map[string]artifactRef{"source": source, "openapi": api}, "scope": map[string]string{"objective": "Explore one exact operation against source"}}, "draft", 0, http.StatusCreated)
			auditID := draft["auditId"].(string)
			basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+auditID+"/start", nil, "start", uint64(draft["revision"].(float64)), http.StatusOK)
			tasks := waitBasicTask(t, h, auditID, 2, "settled")
			if tasks[0].Document.Workflow != "prepare" || tasks[0].Result == nil || tasks[1].Document.Workflow != "trace" || tasks[1].GapCode != nil || tasks[1].Document.Hypothesis != nil || tasks[1].Result == nil {
				t.Fatalf("separate preparation/operation work=%+v", tasks)
			}
			var result struct {
				Schema string `json:"schema"`
				Items  []struct {
					Verdict          *string  `json:"verdict"`
					TestedConditions []string `json:"tested_conditions"`
					Origin           struct {
						TaskID string `json:"task_id"`
					} `json:"origin"`
				} `json:"items"`
			}
			if err := json.Unmarshal(basicTaskArtifact(t, h, auditID, tasks[1], "result"), &result); err != nil || result.Schema != "contractor.audit.result.v1" || len(result.Items) != 1 || result.Items[0].Verdict != nil || len(result.Items[0].TestedConditions) != 1 || result.Items[0].Origin.TaskID != tasks[1].Document.TaskID {
				t.Fatalf("ordinary exploratory result=%+v error=%v", result, err)
			}
			for {
				current := basicAuditGet(t, h, auditID)
				if current["state"] == "completed" {
					if int(current["submittedRunCount"].(float64)) != 2 {
						t.Fatal("preparation or operation duplicated")
					}
					break
				}
				basicPoll(t, h)
			}
			if h.gateway.CompletedStages() != 1 || len(h.gateway.Failures()) != 0 {
				t.Fatalf("unexpected model activity: %v", h.gateway.Failures())
			}
		})
	}
}
