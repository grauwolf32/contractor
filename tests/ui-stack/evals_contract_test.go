//go:build e2e

package uistack

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
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
		Result             json.RawMessage   `json:"result"`
		Context            json.RawMessage   `json:"context"`
		Document           json.RawMessage   `json:"document"`
		Rejections         int               `json:"rejections"`
		Dataset            json.RawMessage   `json:"dataset"`
		Registration       json.RawMessage   `json:"registration"`
		Wrapper            json.RawMessage   `json:"wrapper"`
		RetainedContext    json.RawMessage   `json:"retainedContext"`
		InvalidConclusions []json.RawMessage `json:"invalidConclusions"`
		InvalidWrappers    []json.RawMessage `json:"invalidWrappers"`
		Tasks              []struct {
			Document   json.RawMessage        `json:"document"`
			Hypothesis json.RawMessage        `json:"hypothesis_snapshot"`
			Result     auditcontract.Material `json:"result"`
		} `json:"tasks"`
	}
	if err := json.Unmarshal(output, &fixture); err != nil {
		t.Fatal(err)
	}
	decodeContext := func(data []byte) ([]byte, error) {
		var header struct {
			Schema string `json:"schema"`
		}
		if err := json.Unmarshal(data, &header); err != nil || header.Schema != "contractor.audit.context.v1" {
			return nil, fmt.Errorf("expected exact trusted context kind, got %q", header.Schema)
		}
		return auditcontract.Decode(data, nil)
	}
	decodeConclusions := func(data, context []byte) ([]byte, error) {
		if err := auditcontract.ValidateResultShape("contractor.audit.conclusions.v1", data); err != nil {
			return nil, err
		}
		return auditcontract.Decode(data, context)
	}
	decodeWrapper := func(data, context []byte) ([]byte, error) {
		if err := auditcontract.ValidateResultShape("contractor.audit.result.v1", data); err != nil {
			return nil, err
		}
		return auditcontract.Decode(data, context)
	}
	if _, err := decodeContext(fixture.Context); err != nil {
		t.Fatalf("invalid trusted Task context: %v", err)
	}
	if _, err := decodeConclusions(fixture.Document, fixture.Context); err != nil {
		t.Fatalf("raw checking conclusions differ from trusted Run context: %v", err)
	}
	if _, err := decodeContext(fixture.RetainedContext); err != nil {
		t.Fatalf("invalid retained Task context: %v", err)
	}
	if _, err := decodeWrapper(fixture.Wrapper, fixture.RetainedContext); err != nil {
		t.Fatalf("accepted Task wrapper differs from retained evidence context: %v", err)
	}
	for index, invalid := range fixture.InvalidConclusions {
		if _, err := decodeConclusions(invalid, fixture.Context); err == nil {
			t.Errorf("raw conclusions owner accepted invalid boundary case %d", index)
		}
	}
	for index, invalid := range fixture.InvalidWrappers {
		if _, err := decodeWrapper(invalid, fixture.RetainedContext); err == nil {
			t.Errorf("accepted wrapper owner accepted invalid provenance case %d", index)
		}
	}
	if len(fixture.Tasks) != 2 || len(fixture.InvalidConclusions) != 4 || len(fixture.InvalidWrappers) != 5 {
		t.Fatal("missing an exact Task or owner-decoder negative control")
	}
	for _, task := range fixture.Tasks {
		if _, err := auditcontract.DecodeTask(task.Document); err != nil {
			t.Fatalf("current accepted Task document rejected: %v", err)
		}
		if _, err := auditcontract.DecodeHypothesis(task.Hypothesis); err != nil {
			t.Fatalf("current Task hypothesis snapshot rejected: %v", err)
		}
		if err := task.Result.Validate(); err != nil {
			t.Fatalf("current exact Task result descriptor rejected: %v", err)
		}
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
	if fixture.Rejections != 24 {
		t.Fatalf("producer skipped an invalid-evidence control: %d", fixture.Rejections)
	}
}

func TestManagedEvalGatewayUsesCurrentCheckerContract(t *testing.T) {
	source := map[string]any{"kind": "artifact", "namespace": "inputs", "name": "source"}
	tasks := []any{
		map[string]any{"item_key": "check-one", "hypothesis": map[string]any{"id": "hyp-one", "revision": 1},
			"materials": []any{map[string]any{"source": source}}},
		map[string]any{"item_key": "exploration", "materials": []any{map[string]any{"source": source}}},
	}
	response, _ := json.Marshal(map[string]any{"batchSize": 2, "tasks": tasks})
	request := map[string]any{
		"model": "fixture", "messages": []any{
			toolCallMessage("read", "read_audit_task", map[string]any{}),
			map[string]any{"role": "tool", "tool_call_id": "read", "content": string(response)},
		},
		"tools": []any{map[string]any{"function": map[string]any{"name": "read_audit_task"}}},
	}
	next := func() (map[string]any, map[string]any) {
		message, reason, _, err := managedEvalGatewayMessage(request)
		if err != nil || reason != "tool_calls" {
			t.Fatalf("checker fixture response: %v %s %v", message, reason, err)
		}
		return message, message["tool_calls"].([]any)[0].(map[string]any)["function"].(map[string]any)
	}
	for _, key := range []string{"check-one", "exploration"} {
		message, call := next()
		var args map[string]any
		if err := json.Unmarshal([]byte(call["arguments"].(string)), &args); err != nil {
			t.Fatal(err)
		}
		if call["name"] != "evidence" || args["item_key"] != key || len(args["sources"].([]any)) != 1 {
			t.Fatalf("fixture did not explicitly select the item's evidence: %+v", call)
		}
		selected := args["sources"].([]any)[0].(map[string]any)
		if len(selected) != 3 || selected["kind"] != "artifact" || selected["namespace"] != "inputs" || selected["name"] != "source" {
			t.Fatalf("fixture source is not the declared revision-free selector: %+v", selected)
		}
		request["messages"] = append(request["messages"].([]any), message)
		// Calling the tool is not an acknowledgement. The fixture must still
		// wait for a receipt before it can submit the judgment.
		_, repeated := next()
		if repeated["name"] != "evidence" {
			t.Fatal("fixture treated an unconfirmed call as durable evidence")
		}
		receipt, _ := json.Marshal(map[string]any{"status": "recorded", "persistence": "accepted", "itemKey": key})
		request["messages"] = append(request["messages"].([]any), map[string]any{
			"role": "tool", "tool_call_id": "managed-evidence-" + key, "content": string(receipt),
		})
	}
	_, call := next()
	if call["name"] != "submit_results" {
		t.Fatalf("unexpected checker call: %v", call)
	}
	var arguments struct {
		Items []map[string]any `json:"items"`
	}
	if err := json.Unmarshal([]byte(call["arguments"].(string)), &arguments); err != nil {
		t.Fatal(err)
	}
	if len(arguments.Items) != 2 || arguments.Items[0]["verdict"] != "supported" || arguments.Items[1]["verdict"] != nil {
		t.Fatalf("fixture lost assignment order or invented an exploration verdict: %+v", arguments.Items)
	}
	for index, key := range []string{"check-one", "exploration"} {
		item := arguments.Items[index]
		size := 3
		if index == 0 {
			size = 4
		}
		if item["item_key"] != key || len(item) != size {
			t.Fatalf("fixture changed the thin judgment: %+v", item)
		}
	}
	// Use the actual shared Runtime entry point for argument normalization.
	root := repoRoot(t)
	command := exec.Command(filepath.Join(root, "runtime", ".venv", "bin", "python"), "-I", "-B", "-c", `
import json, sys
sys.path.insert(0, sys.argv[1])
from contractor_runtime.toolsets.audit_results.v2 import _prepare_batch
_prepare_batch(json.load(sys.stdin)["items"])
`, filepath.Join(root, "runtime", "src"))
	command.Stdin = bytes.NewBufferString(call["arguments"].(string))
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("Runtime rejected checker fixture arguments: %v\n%s", err, output)
	}
}
