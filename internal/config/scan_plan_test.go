package config

import (
	"context"
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"go.yaml.in/yaml/v4"
)

func scanPlanWorkflow(t *testing.T, name string) ResolvedWorkflow {
	t.Helper()
	workflow, err := mustLoad(t, "../../configs/scan", MVPDescriptors()).ResolveRunWorkflow(context.Background(), name, ExecutionConfigPatch{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return workflow
}

func TestScanPlanCatalogModelFreeAndFixedWorkers(t *testing.T) {
	for _, name := range []string{"request-set-scan@1", "target-scan-plan@1"} {
		workflow := scanPlanWorkflow(t, name)
		stage := workflow.Stages["scan"]
		if stage.ScanPlan == nil || !IsModelFreePlanner(stage.Planner) || stage.ExecutionConfig.Planner != nil || len(stage.Agents) != len(stage.ScanPlan.Tools) {
			t.Fatalf("invalid model-free stage: %+v", stage)
		}
		for _, tool := range stage.ScanPlan.Tools {
			binding := stage.Agents[tool.Worker]
			if !binding.Template.IsToolWorker() || binding.Template.Execution.ResultArtifact != "report" || stage.ExecutionConfig.Agents[tool.Worker] != (ResolvedConsumerExecutionConfig{}) {
				t.Fatalf("model or dynamic Worker leaked into scan policy: %+v", tool)
			}
		}
		if err := ValidateWorkflowGraph(workflow); err != nil {
			t.Fatal(err)
		}
		stage.ScanPlan.MaxTotalSeconds = 1
		for i := range stage.ScanPlan.Tools {
			stage.ScanPlan.Tools[i].MaxTotalSeconds = 1
		}
		if err := ValidateScanPlanStage(stage); err != nil {
			t.Fatalf("policy with insufficient time for any job must remain valid: %v", err)
		}
	}
}

func TestScanPlanRejectsInvalidRoutingAndPolicy(t *testing.T) {
	original := scanPlanWorkflow(t, "request-set-scan@1")
	tests := []struct {
		name   string
		change func(*ResolvedStage)
	}{
		{"missing policy", func(s *ResolvedStage) { s.ScanPlan = nil }},
		{"wrong planner", func(s *ResolvedStage) { s.Planner = PlannerRef{PlannerID: "passthrough", Version: "1"} }},
		{"unknown planner version", func(s *ResolvedStage) { s.Planner.Version = "2" }},
		{"zero max inputs", func(s *ResolvedStage) { s.ScanPlan.MaxInputs = 0 }},
		{"max inputs exceeded", func(s *ResolvedStage) { s.ScanPlan.MaxInputs = 1001 }},
		{"zero max jobs", func(s *ResolvedStage) { s.ScanPlan.MaxJobs = 0 }},
		{"max jobs exceeded", func(s *ResolvedStage) { s.ScanPlan.MaxJobs = 101 }},
		{"max seconds exceeded", func(s *ResolvedStage) { s.ScanPlan.MaxTotalSeconds = 86401 }},
		{"unknown input", func(s *ResolvedStage) { s.ScanPlan.InputArtifact = "missing" }},
		{"optional input", func(s *ResolvedStage) {
			a := s.Context.Artifacts["requests"]
			a.Required = false
			s.Context.Artifacts["requests"] = a
		}},
		{"unknown worker", func(s *ResolvedStage) { s.ScanPlan.Tools[0].Worker = "missing" }},
		{"empty fixed workers", func(s *ResolvedStage) { s.Agents = map[string]ResolvedAgentBinding{} }},
		{"unlisted worker", func(s *ResolvedStage) { s.Agents["extra"] = s.Agents["sqlmap"] }},
		{"duplicate worker policy", func(s *ResolvedStage) { s.ScanPlan.Tools = append(s.ScanPlan.Tools, s.ScanPlan.Tools[0]) }},
		{"duplicate scanner", func(s *ResolvedStage) {
			b := s.Agents["sqlmap"]
			b.Namespace = "other"
			s.Agents["other"] = b
			tool := s.ScanPlan.Tools[0]
			tool.Worker = "other"
			s.ScanPlan.Tools = append(s.ScanPlan.Tools, tool)
		}},
		{"model worker", func(s *ResolvedStage) {
			b := s.Agents["sqlmap"]
			b.Template.Runtime.RuntimeID = "adk"
			s.Agents["sqlmap"] = b
		}},
		{"wrong toolset", func(s *ResolvedStage) {
			b := s.Agents["sqlmap"]
			b.Template.Toolsets[0].Ref.ToolsetID = "other"
			s.Agents["sqlmap"] = b
		}},
		{"missing test parameters", func(s *ResolvedStage) { s.ScanPlan.Tools[0].TestParameters = nil }},
		{"wordlist on sqlmap", func(s *ResolvedStage) { s.ScanPlan.Tools[0].WordlistArtifact = "requests" }},
		{"wrong request binding", func(s *ResolvedStage) {
			s.Agents["sqlmap"].Template.Execution.Arguments["request_ref"] = contracts.ToolArgumentBinding{Source: "artifact", Name: "requests"}
		}},
		{"literal request binding", func(s *ResolvedStage) {
			s.Agents["sqlmap"].Template.Execution.Arguments["request_ref"] = contracts.ToolArgumentBinding{Source: "literal", Value: "literal"}
		}},
		{"additional dynamic argument", func(s *ResolvedStage) {
			s.Agents["sqlmap"].Template.Execution.Arguments["level"] = contracts.ToolArgumentBinding{Source: "parameter", Name: "level"}
		}},
		{"mixed sqlmap mode", func(s *ResolvedStage) {
			s.Agents["sqlmap"].Template.Execution.Arguments["url"] = contracts.ToolArgumentBinding{Source: "literal", Value: "https://example.test"}
		}},
		{"wrong worker result slot", func(s *ResolvedStage) { s.Agents["sqlmap"].Template.Execution.ResultArtifact = "other" }},
		{"planner model config", func(s *ResolvedStage) { s.ExecutionConfig.Planner = &ResolvedConsumerExecutionConfig{} }},
		{"workspace", func(s *ResolvedStage) { s.Context.Workspace = &WorkspaceContext{} }},
		{"extra result", func(s *ResolvedStage) { s.Result.Artifacts["extra"] = s.Result.Artifacts["report"] }},
		{"optional report", func(s *ResolvedStage) {
			a := s.Result.Artifacts["report"]
			a.Required = false
			s.Result.Artifacts["report"] = a
		}},
		{"unbound report", func(s *ResolvedStage) {
			a := s.Result.Artifacts["report"]
			a.From = nil
			s.Result.Artifacts["report"] = a
		}},
		{"wrong report MIME", func(s *ResolvedStage) {
			a := s.Result.Artifacts["report"]
			a.MediaTypes = []string{"text/plain"}
			s.Result.Artifacts["report"] = a
		}},
		{"worker report namespace", func(s *ResolvedStage) { s.Result.Artifacts["report"].From.Namespace = "sqlmap" }},
		{"reserved report namespace", func(s *ResolvedStage) { s.Result.Artifacts["report"].From.Namespace = "inputs" }},
		{"reserved report name", func(s *ResolvedStage) { s.Result.Artifacts["report"].From.Name = "tool-invocation.hidden" }},
		{"long report prefix", func(s *ResolvedStage) { s.Result.Artifacts["report"].From.Name = strings.Repeat("a", 101) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			workflow := cloneWorkflow(original)
			stage := workflow.Stages["scan"]
			tt.change(&stage)
			workflow.Stages["scan"] = stage
			if ValidateScanPlanStage(stage) == nil || ValidateWorkflowGraph(workflow) == nil {
				t.Fatal("invalid scan contract accepted")
			}
		})
	}
}

func TestScanPlanInputMIMEAndOrdinaryToolValidationRemainStrict(t *testing.T) {
	original := scanPlanWorkflow(t, "request-set-scan@1")
	for _, media := range []string{"application/json", "text/plain", "application/octet-stream"} {
		workflow := cloneWorkflow(original)
		slot := workflow.Inputs["requests"]
		slot.MediaTypes = []string{media}
		workflow.Inputs["requests"] = slot
		if err := ValidateWorkflowGraph(workflow); err == nil {
			t.Fatalf("unsupported input MIME %q accepted", media)
		}
	}
	workflow := cloneWorkflow(original)
	stage := workflow.Stages["scan"]
	stage.Planner = PlannerRef{PlannerID: "passthrough", Version: "1"}
	stage.ScanPlan = nil
	workflow.Stages["scan"] = stage
	if err := ValidateWorkflowGraph(workflow); err == nil {
		t.Fatal("ordinary tool Worker bypassed undeclared per-job input validation")
	}
}

func TestScanPlanFFUFRequiresBoundWordlist(t *testing.T) {
	workflow := scanPlanWorkflow(t, "target-scan-plan@1")
	stage := workflow.Stages["scan"]
	binding := stage.Agents["nuclei"]
	binding.Template.Toolsets[0].Tools = []string{"scan_ffuf"}
	binding.Template.Execution.Tool = "scan_ffuf"
	binding.Template.Execution.Arguments = map[string]contracts.ToolArgumentBinding{
		"url":          {Source: "parameter", Name: "target"},
		"wordlist_ref": {Source: "artifact", Name: "wordlist"},
		"rate":         {Source: "literal", Value: 10},
	}
	stage.Agents = map[string]ResolvedAgentBinding{"ffuf": binding}
	stage.ScanPlan.Tools = []contracts.ScanToolPolicy{{Worker: "ffuf", MaxJobs: 2, MaxTotalSeconds: 600, WordlistArtifact: "words"}}
	stage.Context.Artifacts["words"] = ContextArtifact{Namespace: "inputs", Name: "words", Required: true}
	if err := ValidateScanPlanStage(stage); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*ResolvedStage){
		func(s *ResolvedStage) { s.ScanPlan.Tools[0].WordlistArtifact = "" },
		func(s *ResolvedStage) { delete(s.Context.Artifacts, "words") },
		func(s *ResolvedStage) {
			s.Agents["ffuf"].Template.Execution.Arguments["wordlist_ref"] = contracts.ToolArgumentBinding{Source: "artifact", Name: "words"}
		},
		func(s *ResolvedStage) { s.ScanPlan.Tools[0].TestParameters = []string{"query"} },
	} {
		invalid := cloneStage(stage)
		change(&invalid)
		if ValidateScanPlanStage(invalid) == nil {
			t.Fatal("invalid ffuf policy accepted")
		}
	}
}

func TestScanPlanPolicySnapshotIsDetachedAndRevalidated(t *testing.T) {
	snapshot := mustLoad(t, "../../configs/scan", MVPDescriptors())
	workflow, _ := snapshot.Workflow("request-set-scan@1")
	stage := workflow.Stages["scan"]
	stage.ScanPlan.Tools[0].TestParameters[0] = "changed"
	stage.ScanPlan.Tools[0].MaxJobs = 99
	stage.ScanPlan.MaxInputs = 99
	again, _ := snapshot.Workflow("request-set-scan@1")
	if again.Stages["scan"].ScanPlan.Tools[0].TestParameters[0] != "query" || again.Stages["scan"].ScanPlan.Tools[0].MaxJobs != 3 || again.Stages["scan"].ScanPlan.MaxInputs != 100 {
		t.Fatal("scan policy mutation leaked into immutable snapshot")
	}
	data, err := json.Marshal(again)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeResolvedWorkflowSnapshot(data)
	if err != nil || !reflect.DeepEqual(again, decoded) {
		t.Fatalf("workflow roundtrip failed: %v", err)
	}
	stage = again.Stages["scan"]
	stageData, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	decodedStage, err := DecodeResolvedStageSnapshot(stageData)
	if err != nil || !reflect.DeepEqual(stage, decodedStage) {
		t.Fatalf("stage roundtrip failed: %v", err)
	}
	for _, replace := range []struct{ before, after string }{
		{`"maxInputs":100`, `"maxInputs":1001`},
		{`"testParameters":["query"]`, `"testParameters":[]`},
		{`"inputArtifact":"requests"`, `"inputArtifact":"missing"`},
		{`"maxJobs":3`, `"unknown":3`},
	} {
		if _, err := DecodeResolvedStageSnapshot([]byte(strings.Replace(string(stageData), replace.before, replace.after, 1))); err == nil {
			t.Fatalf("tampered stage policy accepted: %s", replace.after)
		}
		if _, err := DecodeResolvedWorkflowSnapshot([]byte(strings.Replace(string(data), replace.before, replace.after, 1))); err == nil {
			t.Fatalf("tampered workflow policy accepted: %s", replace.after)
		}
	}
}

func TestScanPlanModelOverridesAndAuthoringUnknownFieldsAreRejected(t *testing.T) {
	workflow := scanPlanWorkflow(t, "request-set-scan@1")
	stage := workflow.Stages["scan"]
	l := loader{}
	if err := l.applyPlannerSelection(&stage, ExecutionSelectionPatch{}, "test"); err == nil || !strings.Contains(err.Error(), "does not accept Planner model") {
		t.Fatalf("direct model override error=%v", err)
	}
	patch := ExecutionConfigPatch{Planner: &ExecutionSelectionPatch{}}
	if err := l.applyExecutionConfigPatch(&workflow, patch, "test"); err == nil || !strings.Contains(err.Error(), "do not apply") {
		t.Fatalf("workflow model override error=%v", err)
	}
	valid := string(readFile(t, filepath.Join("..", "..", "configs", "scan", "workflows", "request_set_scan.yaml")))
	for _, invalid := range []string{
		strings.Replace(valid, "maxInputs: 100", "maxInputs: 100\n        unknown: true", 1),
		strings.Replace(valid, "worker: sqlmap", "worker: sqlmap\n            unknown: true", 1),
	} {
		var doc workflowDocument
		decoder := yaml.NewDecoder(strings.NewReader(invalid))
		decoder.KnownFields(true)
		if decoder.Decode(&doc) == nil {
			t.Fatal("unknown scan policy authoring field accepted")
		}
	}
}
