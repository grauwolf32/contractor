//go:build e2e

package uistack

import (
	"bytes"
	"encoding/json"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func TestManagedEvalAssertionsUseCurrentDomainContract(t *testing.T) {
	producer := filepath.Join(repoRoot(t), "tests", "ui-stack", "evals_producer.py")
	command := exec.Command("python3", "-I", "-S", "-B", "-c", `
import runpy, sys
fixture = runpy.run_path(sys.argv[1])
print(fixture["encoded"]({"assertions": fixture["audit_assertion_source"](), "members": fixture["MEMBERS"]}).decode())
`, producer)
	raw, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("read producer assertion fixture: %v\n%s", err, raw)
	}
	var fixture struct {
		Assertions json.RawMessage `json:"assertions"`
		Members    int             `json:"members"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	plan, err := auditdomain.ChecklistAssertions(fixture.Assertions, "application/json", "check")
	if err != nil {
		t.Fatal(err)
	}
	if fixture.Members != 8 || len(plan.Work) != 2 || len(plan.Hypotheses) != 2 {
		t.Fatalf("fixture changed its Eval membership or assertion count: members=%d plan=%+v", fixture.Members, plan)
	}
	for _, key := range []string{"check-one", "check-two"} {
		seed := plan.Hypotheses[key]
		if seed.Grounds == "" || seed.Scope["source_version"] != "1" || seed.Scope["allowed_methods"] != `["static-trace"]` || seed.Scope["required_evidence"] != `["source-trace"]` {
			t.Fatalf("%s lost its source contract: %+v", key, seed)
		}
	}
	t.Run("owned-task-evidence", testManagedEvalProducerTaskEvidence)
}

func testManagedEvalProducerTaskEvidence(t *testing.T) {
	root := repoRoot(t)
	command := exec.Command("python3", "-I", "-S", "-B",
		filepath.Join(root, "tests", "ui-stack", "evals_producer_test.py"),
		filepath.Join(root, "tests", "ui-stack", "evals_producer.py"), root)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("producer exact Task evidence regression: %v\n%s", err, output)
	}
	var fixture struct {
		Result       json.RawMessage `json:"result"`
		Context      json.RawMessage `json:"context"`
		Document     json.RawMessage `json:"document"`
		Rejections   int             `json:"rejections"`
		Dataset      json.RawMessage `json:"dataset"`
		Registration json.RawMessage `json:"registration"`
	}
	if err := json.Unmarshal(output, &fixture); err != nil {
		t.Fatal(err)
	}
	if _, err := auditcontract.Decode(fixture.Context, nil); err != nil {
		t.Fatalf("invalid trusted Task context: %v", err)
	}
	if _, err := auditcontract.Decode(fixture.Document, fixture.Context); err != nil {
		t.Fatalf("Task evidence differs from owning result/context contract: %v", err)
	}
	if _, err := evaldomain.Freeze("ResultInput", fixture.Result); err != nil {
		t.Fatalf("producer emitted invalid current Eval result: %v", err)
	}
	if _, err := evaldomain.Freeze("DatasetInput", fixture.Dataset); err != nil {
		t.Fatalf("current native Audit dataset/case contract rejected: %v", err)
	}
	if _, err := evaldomain.Freeze("ExternalRegistration", fixture.Registration); err != nil {
		t.Fatalf("current external Audit manifest/recipe/mapping contract rejected: %v", err)
	}
	if fixture.Rejections != 11 {
		t.Fatalf("producer skipped an invalid-evidence control: %d", fixture.Rejections)
	}
}

func TestManagedEvalGatewayUsesCurrentCheckerContract(t *testing.T) {
	tasks := []any{
		map[string]any{"item_key": "check-one", "hypothesis": map[string]any{"id": "hyp-one", "revision": 1},
			"evidence": []any{map[string]any{"handle": "evidence-000", "material": map[string]any{"kind": "artifact"}}}},
		map[string]any{"item_key": "exploration", "evidence": []any{map[string]any{"handle": "evidence-001", "material": map[string]any{"kind": "artifact"}}}},
	}
	response, _ := json.Marshal(map[string]any{"batchSize": 2, "tasks": tasks})
	request := map[string]any{
		"model": "fixture", "messages": []any{
			toolCallMessage("read", "read_audit_task", map[string]any{}),
			map[string]any{"role": "tool", "tool_call_id": "read", "content": string(response)},
		},
		"tools": []any{map[string]any{"function": map[string]any{"name": "read_audit_task"}}},
	}
	message, reason, _, err := managedEvalGatewayMessage(request)
	if err != nil || reason != "tool_calls" {
		t.Fatalf("checker fixture response: %v %s %v", message, reason, err)
	}
	call := message["tool_calls"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if call["name"] != "submit_check_result" {
		t.Fatalf("unexpected checker call: %v", call)
	}
	var arguments struct {
		Results []map[string]any `json:"results"`
	}
	if err := json.Unmarshal([]byte(call["arguments"].(string)), &arguments); err != nil {
		t.Fatal(err)
	}
	if len(arguments.Results) != 2 || arguments.Results[0]["verdict"] != "supported" || arguments.Results[1]["verdict"] != nil {
		t.Fatalf("fixture lost assignment order or invented an exploration verdict: %+v", arguments.Results)
	}
	for index, handle := range []string{"evidence-000", "evidence-001"} {
		if !reflect.DeepEqual(arguments.Results[index]["evidence"], []any{handle}) {
			t.Fatalf("fixture changed the assigned evidence handles: %+v", arguments.Results)
		}
	}
	// Use the same strict Runtime normalization as SubmitCheckResultTool rather
	// than a Go copy of its accepted argument fields and enum values.
	root := repoRoot(t)
	command := exec.Command(filepath.Join(root, "runtime", ".venv", "bin", "python"), "-I", "-B", "-c", `
import json, sys
sys.path.insert(0, sys.argv[1])
from contractor_runtime.toolsets.audit_results.v2 import _normalize
arguments = json.load(sys.stdin)
for key, value in zip(("check-one", "exploration"), arguments["results"], strict=True):
    _normalize(key, value)
`, filepath.Join(root, "runtime", "src"))
	command.Stdin = bytes.NewBufferString(call["arguments"].(string))
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("Runtime rejected checker fixture arguments: %v\n%s", err, output)
	}
}
