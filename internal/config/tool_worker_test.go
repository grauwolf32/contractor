package config

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/control"
)

func TestToolWorkersResolveWithoutModelCatalogs(t *testing.T) {
	snapshot := mustLoad(t, scanFixtureRoot)
	if snapshot.Counts().ModelPolicies != 0 {
		t.Fatal("scan fixtures must have no model policies")
	}
	for _, name := range []string{"nuclei-target@1", "naabu-host@1", "sqlmap-request@1"} {
		workflow, err := snapshot.ResolveRunWorkflow(context.Background(), name, resolved.ExecutionConfigPatch{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		stage := workflow.Stages["scan"]
		template := stage.Agents["scanner"].Template
		if err := template.Validate(); err != nil {
			t.Fatal(err)
		}
		if !template.IsToolWorker() || stage.ExecutionConfig.Planner != nil || !stage.ExecutionConfig.Agents["scanner"].ModelPolicy.IsZero() {
			t.Fatal("model leaked into tool workflow")
		}
		encoded, err := json.Marshal(workflow)
		if err != nil {
			t.Fatal(err)
		}
		if restored, err := resolved.DecodeResolvedWorkflowSnapshot(encoded); err != nil || !reflect.DeepEqual(restored, workflow) {
			t.Fatalf("%s did not survive persisted snapshot validation: %v", name, err)
		}
		wire, err := json.Marshal(template)
		if err != nil {
			t.Fatal(err)
		}
		var fields map[string]json.RawMessage
		if json.Unmarshal(wire, &fields) != nil || fields["modelPolicy"] != nil || fields["instructions"] != nil || fields["execution"] == nil {
			t.Fatal(string(wire))
		}
		cloned := template.Clone()
		cloned.Execution.Arguments["injected"] = contracts.ToolArgumentBinding{Source: "literal", Value: true}
		if _, found := template.Execution.Arguments["injected"]; found {
			t.Fatal("execution map is not owned")
		}
	}
}

func TestSQLMapWorkerBindsPreparedRequestArtifact(t *testing.T) {
	snapshot := mustLoad(t, scanFixtureRoot)
	workflow, err := snapshot.ResolveRunWorkflow(context.Background(), "sqlmap-request@1", resolved.ExecutionConfigPatch{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["scan"]
	execution := stage.Agents["scanner"].Template.Execution
	if execution.Tool != "scan_sqlmap" || execution.Arguments["request_ref"] != (contracts.ToolArgumentBinding{Source: "artifact", Name: "request"}) {
		t.Fatalf("prepared request is not passed as an Artifact ref: %+v", execution)
	}
	if stage.Context.Artifacts["request"] != (resolved.ContextArtifact{Namespace: "inputs", Name: "request", Required: true}) {
		t.Fatalf("unexpected request context: %+v", stage.Context.Artifacts)
	}
	if !workflow.Inputs["request"].Required || !reflect.DeepEqual(workflow.Inputs["request"].MediaTypes, []string{"application/json", "application/vnd.contractor.http-request+json"}) {
		t.Fatalf("prepared request input must require JSON: %+v", workflow.Inputs)
	}
	delete(stage.Context.Artifacts, "request")
	if resolved.ValidateWorkflowExecutionConfigs(workflow) == nil {
		t.Fatal("missing prepared request context accepted")
	}
}

func TestSharedToolWorkerGoldenDigest(t *testing.T) {
	var spec control.AllocationSpec
	data := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "valid", "allocation-spec-tool.json")
	if err := json.Unmarshal(data, &spec); err != nil {
		t.Fatal(err)
	}
	if err := spec.Validate(); err != nil {
		t.Fatal(err)
	}
	got, err := agentTemplateDigest(resolved.Selector{ID: spec.AgentTemplate.Ref.TemplateID, Version: spec.AgentTemplate.Ref.Version}, spec.AgentTemplate)
	if err != nil || got != spec.AgentTemplate.Ref.Digest {
		t.Fatalf("digest %s, error %v", got, err)
	}
}

func TestToolWorkerRejectsModelAndInvalidBindings(t *testing.T) {
	snapshot := mustLoad(t, scanFixtureRoot)
	original, _ := snapshot.AgentTemplate("nuclei-scan@1")
	for _, change := range []func(*contracts.ResolvedAgentTemplate){
		func(t *contracts.ResolvedAgentTemplate) { t.Execution = nil },
		func(t *contracts.ResolvedAgentTemplate) { t.Instructions.Text = "unused" },
		func(t *contracts.ResolvedAgentTemplate) { t.ModelPolicy.Model = "unused" },
		func(t *contracts.ResolvedAgentTemplate) { t.Execution.Tool = "scan_naabu" },
		func(t *contracts.ResolvedAgentTemplate) {
			t.Execution.Arguments["url"] = contracts.ToolArgumentBinding{Source: "parameter"}
		},
		func(t *contracts.ResolvedAgentTemplate) {
			t.Execution.Arguments["url"] = contracts.ToolArgumentBinding{Source: "literal", Value: []string{"target"}}
		},
	} {
		value := original.Clone()
		change(&value)
		if value.Validate() == nil {
			t.Fatal("invalid tool template accepted")
		}
	}
	workflow, _ := snapshot.Workflow("nuclei-target@1")
	delete(workflow.Parameters, "target")
	if resolved.ValidateWorkflowExecutionConfigs(workflow) == nil {
		t.Fatal("missing parameter accepted")
	}
}

func TestUserRulepackScopeIsNarrowAndSurvivesSnapshots(t *testing.T) {
	snapshot := mustLoad(t, scanFixtureRoot)
	for _, variant := range []string{"valid", "required", "wrong_slot", "wrong_media", "missing_binding", "required_context", "output", "stage_result"} {
		t.Run(variant, func(t *testing.T) {
			workflow, err := snapshot.Workflow("nuclei-target@1")
			if err != nil {
				t.Fatal(err)
			}
			workflow.Inputs["rulepack"] = resolved.ArtifactSlot{SourceScope: "user", MediaTypes: []string{"application/yaml"}}
			stage := workflow.Stages["scan"]
			stage.Context.Artifacts["rulepack"] = resolved.ContextArtifact{Namespace: "inputs", Name: "rulepack"}
			agent := stage.Agents["scanner"]
			agent.Template.Execution.Arguments["rulepack_ref"] = contracts.ToolArgumentBinding{Source: "artifact", Name: "rulepack", Optional: true}
			switch variant {
			case "required":
				slot := workflow.Inputs["rulepack"]
				slot.Required = true
				workflow.Inputs["rulepack"] = slot
			case "wrong_slot":
				workflow.Inputs["source"] = workflow.Inputs["rulepack"]
				delete(workflow.Inputs, "rulepack")
			case "wrong_media":
				slot := workflow.Inputs["rulepack"]
				slot.MediaTypes = []string{"application/zip"}
				workflow.Inputs["rulepack"] = slot
			case "missing_binding":
				delete(agent.Template.Execution.Arguments, "rulepack_ref")
			case "required_context":
				value := stage.Context.Artifacts["rulepack"]
				value.Required = true
				stage.Context.Artifacts["rulepack"] = value
			case "output":
				workflow.Outputs["rulepack"] = workflow.Inputs["rulepack"]
			case "stage_result":
				stage.Result.Artifacts["rulepack"] = workflow.Inputs["rulepack"]
			}
			stage.Agents["scanner"] = agent
			workflow.Stages["scan"] = stage
			err = resolved.ValidateWorkflowExecutionConfigs(workflow)
			if (err == nil) != (variant == "valid") {
				t.Fatalf("validation: %v", err)
			}
			if variant == "valid" {
				encoded, err := json.Marshal(workflow)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := resolved.DecodeResolvedWorkflowSnapshot(encoded); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}
