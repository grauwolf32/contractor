//go:build e2e

package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestFactExtractionAuditEvalAcrossProcesses(t *testing.T) {
	stage := func(name string) domainGatewayStage {
		return domainGatewayStage{name: name, tools: []string{"read_text_artifact", "record_fact", "read_audit_task", "submit_check_result"}, steps: []domainGatewayStep{
			toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
			toolGatewayStep("read_text_artifact", stageRefArguments("memories", nil)),
			toolGatewayStep("record_fact", func(request map[string]any) (map[string]any, error) {
				tasks := findingsToolResponses(request, "read_audit_task")
				if len(tasks) != 1 {
					return nil, fmt.Errorf("expected assigned Audit task")
				}
				members, ok := tasks[0]["tasks"].([]any)
				if !ok || len(members) != 1 {
					return nil, fmt.Errorf("expected singleton Audit task")
				}
				member, ok := members[0].(map[string]any)
				if !ok {
					return nil, fmt.Errorf("invalid task")
				}
				return map[string]any{"title": "Reported nginx", "description": "The original memory reports nginx; no version or independent verification is claimed.", "memory_name": "banner", "item_key": member["item_key"]}, nil
			}),
			toolGatewayStep("submit_check_result", fixedArguments(map[string]any{"rationale": "Selected the useful reported technology observation", "tested_conditions": []string{"Read the exact exported memories"}, "limitations": []string{"No independent probe"}, "evidence": []string{}})),
			finalGatewayStep("Recorded memory observations", nil),
		}}
	}
	root := filepath.Join(repoRoot(t), "configs", "evals", "project-facts")
	source, err := os.ReadFile(filepath.Join(repoRoot(t), "tests", "eval", "project_facts", "cases", "technology.json"))
	if err != nil {
		t.Fatal(err)
	}
	runFindingEvalNavigation(t, stage, root, source, filepath.Join(repoRoot(t), "tests", "eval", "project_facts", "experiment-template.json"))
}
