//go:build e2e

package uistack

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
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
		if !completed["submit_check_result"] {
			var coverage []any
			walkJSON(request, func(object map[string]any) {
				if value, ok := object["requestedCoverage"].([]any); ok {
					coverage = value
				}
			})
			if len(coverage) == 0 {
				return nil, "", model, errors.New("Audit task coverage missing")
			}
			results := make([]any, 0, len(coverage))
			for _, requested := range coverage {
				results = append(results, map[string]any{"assessment": "satisfied",
					"summary": "Deterministic retained fixture evidence", "completed": requested,
					"gaps": []string{}, "evidence": []map[string]string{{"kind": "source-trace", "summary": "Fixture evidence"}}})
			}
			return call("submit_check_result", map[string]any{"results": results})
		}
	} else {
		if !completed["read_artifact"] {
			return call("read_artifact", map[string]any{"namespace": "inputs", "name": "source", "revision": nil})
		}
		if !completed["write_artifact"] {
			data, found := lastStringValue(request, "dataBase64")
			if !found {
				return nil, "", model, errors.New("Workflow source bytes missing")
			}
			return call("write_artifact", map[string]any{"namespace": "builder", "name": "copied",
				"media_type": "text/plain", "data_base64": data, "expected_revision": nil})
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
		value := strings.Replace(string(raw), "name: "+oldName, "name: "+newName, 1)
		value = strings.ReplaceAll(value, "artifact_builder@1", "artifact_builder@2")
		if directory == "audit-profiles" {
			value = strings.Replace(value, "batchSize: 2", "batchSize: 1", 1)
		}
		if err = os.WriteFile(filepath.Join(target, directory, newName+".yaml"), []byte(value), 0600); err != nil {
			t.Fatal(err)
		}
	}
	for _, arm := range []string{"a", "b"} {
		copyVariant("configs/e2e/workflows/artifact_copy.yaml", "workflows", "artifact-copy", "eval-copy-"+arm)
		copyVariant("configs/audit-profiles/source_checklist.yaml", "audit-profiles", "source-checklist", "eval-audit-"+arm)
	}
	// The fixture uses no Skills uploaded by the Operations browser journey.
	path := filepath.Join(target, "agent-templates", "audit_source_checker.yaml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	value := strings.Replace(string(raw), "  skills:\n    - namespace: skills\n      name: trace", "  skills: []", 1)
	if string(raw) == value {
		t.Fatal("Audit fixture skill block changed")
	}
	if err = os.WriteFile(path, []byte(value), 0600); err != nil {
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
	for _, managedEvals := range []bool{false, true} {
		t.Run(fmt.Sprintf("managed_evals_%t", managedEvals), func(t *testing.T) {
			root := repoRoot(t)
			target := stageUIStackConfiguration(t, root, filepath.Join(t.TempDir(), "configs"),
				"http://127.0.0.1:9999/v1", "http://127.0.0.1:9998")
			if managedEvals {
				stageManagedEvalConfiguration(t, root, target)
			}
			snapshot, err := config.Load(target, config.MVPDescriptors())
			if err != nil {
				t.Fatal(err)
			}
			for _, selector := range []string{"streamline-copy@1", "openapi-from-workspace@7"} {
				if _, err := snapshot.Workflow(selector); err != nil {
					t.Fatal(err)
				}
			}
			if managedEvals {
				for _, arm := range []string{"a", "b"} {
					if _, err := snapshot.Workflow("eval-copy-" + arm + "@1"); err != nil {
						t.Fatal(err)
					}
					if _, err := snapshot.AuditProfile("eval-audit-" + arm + "@1"); err != nil {
						t.Fatal(err)
					}
				}
			}
		})
	}
}
