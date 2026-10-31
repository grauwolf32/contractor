package config

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"go.yaml.in/yaml/v4"
)

func TestServerOperationWorkflowUsesNoPlannerOrWorker(t *testing.T) {
	snapshot := mustLoad(t, copyCoreFixture(t, "server-operations"))
	workflow, err := snapshot.ResolveRunWorkflow(context.Background(), "sitemap-join@1", nil)
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	if stage.Operation == nil || stage.Planner != (resolved.PlannerRef{}) || len(stage.Agents) != 0 || stage.Instructions.Text != "" || stage.ExecutionConfig.Planner != nil || resolved.StageLaneClass(stage) != resolved.LaneClassTool {
		t.Fatalf("unexpected executor: %+v", stage)
	}
	encoded, err := resolved.EncodeResolvedWorkflowSnapshot(workflow)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := resolved.DecodeResolvedWorkflowSnapshot(encoded); err != nil {
		t.Fatal(err)
	}
	copy := workflow.Clone()
	copy.Stages[workflow.EntryStage].Operation.Arguments["contract"] = stage.Operation.Arguments["origin"]
	if workflow.Stages[workflow.EntryStage].Operation.Arguments["contract"].Source != "literal" {
		t.Fatal("clone changed pinned arguments")
	}
	stageBytes, _ := json.Marshal(stage)
	for _, member := range []string{`"planner":null`, `"planner":{}`, `"instructions":null`, `"operation":null`, `"Operation":{}`} {
		body := append(append([]byte{}, stageBytes[:len(stageBytes)-1]...), []byte(","+member+"}")...)
		var decoded resolved.ResolvedStage
		if err := strictjson.Decode(body, &decoded); err == nil {
			t.Fatalf("accepted %s", member)
		}
	}
}

func TestSharedServerOperationStageCases(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Name  string
			Path  []string
			Value any
			Valid bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "server-operation-stage-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			var source map[string]any
			if err := yaml.Unmarshal(readFile(t, "testdata/server-operations/workflows/join.yaml"), &source); err != nil {
				t.Fatal(err)
			}
			stage := source["spec"].(map[string]any)["stages"].(map[string]any)["join-api"].(map[string]any)
			for _, part := range tc.Path[:len(tc.Path)-1] {
				stage = stage[part].(map[string]any)
			}
			stage[tc.Path[len(tc.Path)-1]] = tc.Value
			data, err := yaml.Marshal(source)
			if err != nil {
				t.Fatal(err)
			}
			root := copyCoreFixture(t)
			writeFile(t, filepath.Join(root, "workflows", "join.yaml"), data)
			_, err = LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
			if (err == nil) != tc.Valid {
				t.Fatalf("valid=%t, want %t: %v", err == nil, tc.Valid, err)
			}
		})
	}
}

func TestServerOperationRejectsConflictingOrUnboundConfiguration(t *testing.T) {
	valid := string(readFile(t, "testdata/server-operations/workflows/join.yaml"))
	for _, tc := range []struct{ name, from, to string }{
		{"planner", "      operation:", "      planner: passthrough@1\n      operation:"},
		{"null planner", "      operation:", "      planner: null\n      operation:"},
		{"null instructions", "      operation:", "      instructions: null\n      operation:"},
		{"unknown operation", "sitemap-openapi-join@1", "arbitrary-command@1"},
		{"unbound parameter", "name: api_origin", "name: missing"},
		{"artifact argument", "source: parameter", "source: artifact"},
		{"unknown argument", "          origin:", "          another:"},
		{"optional input", "name: openapi, required: true", "name: openapi, required: false"},
		{"missing output binding", "            from: {namespace: join-api, name: sitemap}", ""},
		{"workspace", "      context:", "      context:\n        workspace: {mode: direct, sources: []}"},
		{"shared session", "      operation:", "      session: shared\n      operation:"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := copyCoreFixture(t)
			writeFile(t, filepath.Join(root, "workflows", "join.yaml"), []byte(strings.Replace(valid, tc.from, tc.to, 1)))
			if _, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed")); err == nil {
				t.Fatal("accepted invalid operation")
			}
		})
	}
}
