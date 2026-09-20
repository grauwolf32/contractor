package config

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestFFUFCatalogResolvesArtifactInputsWithoutModels(t *testing.T) {
	snapshot := mustLoad(t, "../../configs/scan", MVPDescriptors())
	if counts := snapshot.Counts(); counts.ModelPolicies != 0 || counts.LLMGateways != 0 {
		t.Fatalf("standalone scan catalog must not require model services: %+v", counts)
	}
	workflow, err := snapshot.ResolveRunWorkflow(context.Background(), "ffuf-wordlist@1", ExecutionConfigPatch{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	worker := stage.Agents["scanner"]
	if stage.Planner != (PlannerRef{PlannerID: "passthrough", Version: "1"}) || len(stage.Agents) != 1 || !worker.Template.IsToolWorker() || stage.ExecutionConfig.Planner != nil || !stage.ExecutionConfig.Agents["scanner"].ModelPolicy.IsZero() {
		t.Fatalf("ffuf must route one invocation through an ordinary model-free Worker: %+v", stage)
	}
	if len(workflow.Parameters) != 1 || !workflow.Parameters["target"].Required || len(workflow.Inputs) != 1 || !workflow.Inputs["wordlist"].Required || !reflect.DeepEqual(workflow.Inputs["wordlist"].MediaTypes, []string{"text/vnd.contractor.wordlist", "text/plain"}) {
		t.Fatalf("ffuf must require one target and uploaded text wordlist: %+v", workflow)
	}
	if stage.Context.Artifacts["wordlist"] != (ContextArtifact{Namespace: "inputs", Name: "wordlist", Required: true}) {
		t.Fatalf("wordlist must come from the selected Run input: %+v", stage.Context)
	}
	execution := worker.Template.Execution
	if execution.Tool != "scan_ffuf" || execution.Arguments["url"] != (contracts.ToolArgumentBinding{Source: "parameter", Name: "target"}) || execution.Arguments["wordlist_ref"] != (contracts.ToolArgumentBinding{Source: "artifact", Name: "wordlist"}) {
		t.Fatalf("scanner must receive the target and exact Artifact binding: %+v", execution)
	}
	for _, name := range []string{"rate", "timeout_seconds", "match_status"} {
		if execution.Arguments[name].Source != "literal" {
			t.Fatalf("%s must be a template-owned bound", name)
		}
	}
	if execution.Arguments["rate"].Value != float64(10) || execution.Arguments["timeout_seconds"].Value != float64(240) || execution.Arguments["match_status"].Value != "all" || execution.TimeoutSeconds != 300 {
		t.Fatalf("unexpected scan rate/deadline/matcher: %+v", execution)
	}
	if len(workflow.Outputs) != 1 || !workflow.Outputs["report"].Required || !workflow.Outputs["report"].Primary || !reflect.DeepEqual(workflow.Outputs["report"].MediaTypes, []string{"application/json"}) || stage.WorkflowOutputs["report"] != "report" || stage.Result.Artifacts["report"].From == nil || stage.Result.Artifacts["report"].From.Namespace != worker.Namespace || stage.Result.Artifacts["report"].From.Name != execution.ResultArtifact {
		t.Fatalf("report is not routed from the selected Worker to the primary output: %+v", workflow.Outputs)
	}
	if stage.On.Succeeded.Kind != TransitionSucceed || stage.On.Failed.Kind != TransitionFail || stage.On.Interrupted.Kind != TransitionFail {
		t.Fatal("failed or interrupted scans must not be retried or promoted to success")
	}
	encoded, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := DecodeResolvedWorkflowSnapshot(encoded)
	if err != nil {
		t.Fatalf("ffuf contract did not survive persisted snapshot validation: %v", err)
	}
	if restored.Stages["scan"].Agents["scanner"].Template.Ref != worker.Template.Ref || restored.Stages["scan"].Context.Artifacts["wordlist"] != stage.Context.Artifacts["wordlist"] {
		t.Fatal("persisted ffuf template/input identity changed")
	}
}

func TestFFUFCatalogRejectsMissingDynamicInputs(t *testing.T) {
	for _, missing := range []string{"target", "wordlist"} {
		t.Run(missing, func(t *testing.T) {
			workflow, err := mustLoad(t, "../../configs/scan", MVPDescriptors()).ResolveRunWorkflow(context.Background(), "ffuf-wordlist@1", ExecutionConfigPatch{}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if missing == "target" {
				delete(workflow.Parameters, "target")
			} else {
				delete(workflow.Stages["scan"].Context.Artifacts, "wordlist")
			}
			if err := validateWorkflowExecutionConfigs(workflow); err == nil {
				t.Fatalf("catalog accepted ffuf without its required %s binding", missing)
			}
		})
	}
}
