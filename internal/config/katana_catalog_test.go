package config

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestKatanaCatalogResolvesBoundedDiscoveryWithoutModels(t *testing.T) {
	snapshot := mustLoad(t, "../../configs/scan", MVPDescriptors())
	if counts := snapshot.Counts(); counts.ModelPolicies != 0 || counts.LLMGateways != 0 {
		t.Fatalf("Katana must not require model services: %+v", counts)
	}
	workflow, err := snapshot.ResolveRunWorkflow(context.Background(), "katana-discovery@1", ExecutionConfigPatch{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	worker := stage.Agents["scanner"]
	if len(workflow.Stages) != 1 || stage.Planner != (PlannerRef{PlannerID: "passthrough", Version: "1"}) || len(stage.Agents) != 1 || !worker.Template.IsToolWorker() || stage.ExecutionConfig.Planner != nil || !stage.ExecutionConfig.Agents["scanner"].ModelPolicy.IsZero() || stage.ScanPlan != nil {
		t.Fatalf("discovery must invoke one model-free Worker without dispatching scans: %+v", stage)
	}
	if len(workflow.Parameters) != 1 || !workflow.Parameters["target"].Required || len(workflow.Inputs) != 0 {
		t.Fatalf("Katana must require only its seed target: %+v", workflow)
	}
	execution := worker.Template.Execution
	if execution.Tool != "scan_katana" || execution.Arguments["url"] != (contracts.ToolArgumentBinding{Source: "parameter", Name: "target"}) {
		t.Fatalf("Katana seed must be a typed target parameter: %+v", execution)
	}
	for name, want := range map[string]float64{"max_depth": 2, "max_pages": 100, "rate_limit": 10, "timeout_seconds": 60} {
		binding := execution.Arguments[name]
		if binding.Source != "literal" || binding.Value != want {
			t.Fatalf("%s must be the template-owned bound %v: %+v", name, want, binding)
		}
	}
	if execution.TimeoutSeconds != 90 || execution.ResultArtifact != "report" || len(workflow.Outputs) != 2 {
		t.Fatalf("unexpected Worker deadline or discovery outputs: %+v", execution)
	}
	for name, mediaType := range map[string]string{"report": "application/json", "targets": "text/vnd.contractor.target-list"} {
		output, result := workflow.Outputs[name], stage.Result.Artifacts[name]
		if !output.Required || output.Primary != (name == "report") || !reflect.DeepEqual(output.MediaTypes, []string{mediaType}) || stage.WorkflowOutputs[name] != name || !result.Required || !reflect.DeepEqual(result.MediaTypes, []string{mediaType}) || result.From == nil || result.From.Namespace != worker.Namespace || result.From.Name != name {
			t.Fatalf("%s must be published from the same Worker: output=%+v result=%+v", name, output, result)
		}
	}
	if stage.On.Succeeded.Kind != TransitionSucceed || stage.On.Failed.Kind != TransitionFail || stage.On.Interrupted.Kind != TransitionFail {
		t.Fatal("discovery must stop after its bounded execution")
	}
	encoded, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := DecodeResolvedWorkflowSnapshot(encoded)
	if err != nil {
		t.Fatalf("Katana contract did not survive persisted snapshot validation: %v", err)
	}
	if restored.Stages["discover"].Agents["scanner"].Template.Ref != worker.Template.Ref || !reflect.DeepEqual(restored.Outputs, workflow.Outputs) {
		t.Fatal("persisted Katana template or output identity changed")
	}
}
