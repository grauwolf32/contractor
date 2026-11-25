//go:build e2e

package uistack

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/configload"
	"go.yaml.in/yaml/v4"
)

const managedEvalWorkerSummary = "Managed Evals deterministic output retained"

// Derive the next response from this request's tool history, so independent
// members and retries never share a global script cursor.
func managedEvalGatewayMessage(request map[string]any) (map[string]any, string, string, error) {
	model, _ := request["model"].(string)
	tools, err := requestToolNames(request)
	if err != nil {
		return nil, "", model, err
	}
	audit := false
	for _, name := range tools {
		audit = audit || name == "read_audit_task"
	}
	completed := map[string]bool{}
	for _, value := range request["messages"].([]any) {
		message, _ := value.(map[string]any)
		if message["role"] != "assistant" {
			continue
		}
		calls, _ := message["tool_calls"].([]any)
		for _, value := range calls {
			call, _ := value.(map[string]any)
			function, _ := call["function"].(map[string]any)
			name, _ := function["name"].(string)
			completed[name] = true
		}
	}
	call := func(name string, arguments map[string]any) (map[string]any, string, string, error) {
		return toolCallMessage("managed-"+name, name, arguments), "tool_calls", model, nil
	}
	if audit {
		if !completed["read_audit_task"] {
			return call("read_audit_task", map[string]any{})
		}
		if !completed["submit_results"] {
			var tasks []any
			acknowledged := map[string]bool{}
			walkJSON(request, func(object map[string]any) {
				if value, ok := object["tasks"].([]any); ok {
					tasks = value
				}
				if key, ok := object["itemKey"].(string); ok && object["persistence"] == "accepted" {
					acknowledged[key] = true
				}
			})
			if len(tasks) == 0 {
				return nil, "", model, errors.New("Audit tasks missing")
			}
			results := make([]any, 0, len(tasks))
			for _, raw := range tasks {
				task, ok := raw.(map[string]any)
				if !ok {
					return nil, "", model, errors.New("Audit task missing")
				}
				key, ok := task["item_key"].(string)
				if !ok || key == "" {
					return nil, "", model, errors.New("Audit task key missing")
				}
				if !acknowledged[key] {
					materials, _ := task["materials"].([]any)
					if len(materials) == 0 {
						return nil, "", model, errors.New("Audit ground materials missing")
					}
					material, _ := materials[0].(map[string]any)
					source, ok := material["source"].(map[string]any)
					if !ok {
						return nil, "", model, errors.New("Audit source selector missing")
					}
					return toolCallMessage("managed-evidence-"+key, "evidence", map[string]any{
						"item_key": key, "sources": []any{source},
						"interpretation":    "Deterministic fixture source, applicable only to this assigned check",
						"tested_conditions": []string{"Selected authored fixture and retained source only"},
						"limitations":       []string{},
					}), "tool_calls", model, nil
				}
				result := map[string]any{
					"item_key": key, "rationale": "Deterministic retained fixture evidence",
					"limitations": []string{},
				}
				if task["hypothesis"] != nil {
					result["verdict"] = "supported"
				}
				results = append(results, result)
			}
			return call("submit_results", map[string]any{"items": results})
		}
	} else {
		if !completed["read_artifact"] {
			return call("read_artifact", map[string]any{"namespace": "inputs", "name": "source"})
		}
		if !completed["write_text_artifact"] {
			text, found := lastStringValue(request, "text")
			if !found {
				return nil, "", model, errors.New("Workflow source bytes missing")
			}
			return call("write_text_artifact", map[string]any{"name": "copied",
				"media_type": "text/plain", "text": text})
		}
	}
	return map[string]any{"role": "assistant", "content": managedEvalWorkerSummary}, "stop", model, nil
}

func stageManagedEvalConfiguration(t *testing.T, root, target string) {
	t.Helper()
	copyVariant := func(source, directory, oldName, newName string) {
		raw, err := os.ReadFile(filepath.Join(root, source))
		if err != nil {
			t.Fatal(err)
		}
		if directory == "workflows" {
			raw = adaptUIStackCopyWorkflow(t, raw)
		}
		var document map[string]any
		if err := yaml.Unmarshal(raw, &document); err != nil {
			t.Fatal(err)
		}
		metadata, ok := document["metadata"].(map[string]any)
		if !ok || metadata["name"] != oldName {
			t.Fatalf("unexpected source identity for %s", newName)
		}
		metadata["name"], metadata["version"] = newName, "1"
		encoded, err := yaml.Marshal(document)
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(target, directory, newName+".yaml"), encoded, 0600); err != nil {
			t.Fatal(err)
		}
	}
	for _, arm := range []string{"a", "b"} {
		copyVariant("configs/e2e/workflows/artifact_report.yaml", "workflows", "artifact-report-fixture", "eval-copy-"+arm)
		// Keep the current Program's two-assertion batch, including both exact
		// item results in the ordinary Audit-owned child Run.
		copyVariant("configs/audit-profiles/source_checklist.yaml", "audit-profiles", "source-checklist", "eval-audit-"+arm)
	}
	// The fixture uses no Skills uploaded by the Operations browser journey.
	path := filepath.Join(target, "agent-templates", "audit_source_checker.yaml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var document map[string]any
	if err := yaml.Unmarshal(raw, &document); err != nil {
		t.Fatal(err)
	}
	document["spec"].(map[string]any)["skills"] = []any{}
	encoded, err := yaml.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, encoded, 0600); err != nil {
		t.Fatal(err)
	}
}

func TestManagedEvalGatewayHasIndependentHistories(t *testing.T) {
	request := map[string]any{"model": "fixture", "messages": []any{}, "tools": []any{
		map[string]any{"function": map[string]any{"name": "read_artifact"}},
	}}
	for range 2 {
		message, reason, _, err := managedEvalGatewayMessage(request)
		if err != nil || reason != "tool_calls" {
			t.Fatalf("response %v %s", err, reason)
		}
		raw, _ := json.Marshal(message)
		if !strings.Contains(string(raw), "read_artifact") {
			t.Fatal(fmt.Sprint(message))
		}
	}
}

func TestUIStackConfigurationClosure(t *testing.T) {
	root := repoRoot(t)
	fixture, _, err := configload.LoadUnionReadOnly(filepath.Join(root, "configs", "e2e"), filepath.Join(t.TempDir(), "fixture-managed"))
	if err != nil {
		t.Fatal(err)
	}
	worker, err := fixture.ModelPolicy("worker@1")
	if err != nil {
		t.Fatal(err)
	}
	production, _, err := configload.LoadUnionReadOnly(filepath.Join(root, "configs"), filepath.Join(t.TempDir(), "production-managed"))
	if err != nil {
		t.Fatal(err)
	}
	productionWorker, err := production.ModelPolicy("worker@1")
	if err != nil {
		t.Fatal(err)
	}
	summarizer, err := production.ModelPolicy("summarizer@1")
	if err != nil {
		t.Fatal(err)
	}
	for _, managedEvals := range []bool{false, true} {
		t.Run(fmt.Sprintf("managed_evals_%t", managedEvals), func(t *testing.T) {
			target := stageUIStackConfiguration(t, root, filepath.Join(t.TempDir(), "configs"),
				"http://127.0.0.1:9999/v1", "http://127.0.0.1:9998")
			if managedEvals {
				stageManagedEvalConfiguration(t, root, target)
			}
			snapshot, _, err := configload.LoadUnionReadOnly(target, filepath.Join(t.TempDir(), "managed"))
			if err != nil {
				t.Fatal(err)
			}
			stagedWorker, err := snapshot.ModelPolicy("ui-worker@1")
			expectedWorker := worker
			expectedWorker.Ref = stagedWorker.Ref
			if err != nil || stagedWorker.Ref.PolicyID != "ui-worker" || !reflect.DeepEqual(stagedWorker, expectedWorker) {
				t.Fatalf("staged Worker fixture model or budgets changed: %v", err)
			}
			stagedProductionWorker, err := snapshot.ModelPolicy("worker@1")
			if err != nil || !reflect.DeepEqual(stagedProductionWorker, productionWorker) {
				t.Fatalf("staging replaced the production Worker policy: %v", err)
			}
			fixtureWorkflows := []string{"streamline-copy@1"}
			for _, selector := range []string{"streamline-copy@1", "openapi-from-workspace@1"} {
				if _, err := snapshot.Workflow(selector); err != nil {
					t.Fatal(err)
				}
			}
			if managedEvals {
				for _, arm := range []string{"a", "b"} {
					fixtureWorkflows = append(fixtureWorkflows, "eval-copy-"+arm+"@1")
					profile, err := snapshot.AuditProfile("eval-audit-" + arm + "@1")
					if err != nil {
						t.Fatal(err)
					}
					if profile.Program.Limits.BatchSize != 2 {
						t.Fatal("managed Eval fixture must retain the current two-assertion batch")
					}
				}
			}
			for _, selector := range fixtureWorkflows {
				workflow, err := snapshot.Workflow(selector)
				if err != nil {
					t.Fatal(err)
				}
				selection := workflow.Stages["copy"].ExecutionConfig.Agents["builder"]
				if !reflect.DeepEqual(selection.ModelPolicy, stagedWorker) {
					t.Fatalf("%s must keep its exact fixture Worker model and budgets", selector)
				}
				if selection.SummarizerModelPolicy == nil || !reflect.DeepEqual(*selection.SummarizerModelPolicy, summarizer) {
					t.Fatalf("%s must keep the adapted template's original summarizer model and budgets", selector)
				}
			}
		})
	}
}
