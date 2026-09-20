package config

import (
	"fmt"
	"slices"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// IsModelFreePlanner classifies exact implementations, independently of any
// authored model selection. Workflow-wide defaults skip these implementations.
func IsModelFreePlanner(ref PlannerRef) bool {
	return ref == (PlannerRef{PlannerID: "passthrough", Version: "1"}) ||
		ref == (PlannerRef{PlannerID: "scan-plan", Version: "1"})
}

func cloneScanPlanPolicy(source *contracts.ScanPlanPolicy) *contracts.ScanPlanPolicy {
	if source == nil {
		return nil
	}
	result := *source
	result.Tools = append([]contracts.ScanToolPolicy(nil), source.Tools...)
	for i := range result.Tools {
		result.Tools[i].TestParameters = append([]string(nil), source.Tools[i].TestParameters...)
	}
	return &result
}

// ValidateScanPlanStage validates both authored and durable routing authority.
// The exception to ordinary tool@1 input/result bindings applies only after
// this contract has validated every member of the fixed Worker set.
func ValidateScanPlanStage(stage ResolvedStage) error {
	if stage.Planner != (PlannerRef{PlannerID: "scan-plan", Version: "1"}) {
		if stage.ScanPlan != nil || stage.AuditScan != nil {
			return fmt.Errorf("scanPlan is only supported by scan-plan@1")
		}
		return nil
	}
	if stage.ScanPlan == nil {
		return fmt.Errorf("scan-plan@1 requires scanPlan")
	}
	if err := validateAuditScanStage(stage); err != nil {
		return err
	}
	if err := stage.ScanPlan.Validate(); err != nil {
		return fmt.Errorf("scanPlan: %w", err)
	}
	if stage.ExecutionConfig.Planner != nil {
		return fmt.Errorf("scan-plan@1 must not have Planner executionConfig")
	}
	if stage.Context.Workspace != nil {
		return fmt.Errorf("scan-plan@1 does not support project workspace")
	}
	if err := scanPlanContextArtifact(stage, stage.ScanPlan.InputArtifact); err != nil {
		return err
	}
	if len(stage.Agents) != len(stage.ScanPlan.Tools) {
		return fmt.Errorf("scanPlan.tools must cover exactly the fixed logical Worker set")
	}
	seenTools := map[string]bool{}
	for _, tool := range stage.ScanPlan.Tools {
		if err := validateMapKey("scanPlan Worker", tool.Worker); err != nil {
			return err
		}
		binding, exists := stage.Agents[tool.Worker]
		if !exists || !binding.Template.IsToolWorker() {
			return fmt.Errorf("scanPlan.tools must select existing tool@1 Workers")
		}
		if err := validateArtifactComponent("scanPlan Worker namespace", binding.Namespace); err != nil {
			return err
		}
		if artifactpolicy.IsPurposeReservedNamespace(binding.Namespace) {
			return fmt.Errorf("scanPlan Worker namespace is reserved")
		}
		if err := binding.Template.Validate(); err != nil {
			return fmt.Errorf("scanPlan Worker template: %w", err)
		}
		execution := binding.Template.Execution
		if binding.Template.Toolsets[0].Ref != (contracts.ToolsetRef{ToolsetID: "scan", Version: "1"}) || execution.ResultArtifact != "report" {
			return fmt.Errorf("scanPlan Workers require scan@1 and the report result slot")
		}
		if seenTools[execution.Tool] {
			return fmt.Errorf("scanPlan permits only one Worker per scanner")
		}
		seenTools[execution.Tool] = true
		expected := map[string]contracts.ToolArgumentBinding{}
		switch execution.Tool {
		case "scan_nuclei", "scan_ffuf":
			expected["url"] = contracts.ToolArgumentBinding{Source: "parameter", Name: "target"}
		case "scan_naabu":
			expected["host"] = contracts.ToolArgumentBinding{Source: "parameter", Name: "target"}
		case "scan_sqlmap":
			expected["request_ref"] = contracts.ToolArgumentBinding{Source: "artifact", Name: "request"}
		default:
			return fmt.Errorf("scanPlan Worker selects an unsupported scanner")
		}
		if execution.Tool == "scan_sqlmap" {
			if len(tool.TestParameters) == 0 && stage.AuditScan == nil {
				return fmt.Errorf("scanPlan SQLMap policy requires testParameters")
			}
			if _, exists := execution.Arguments["url"]; exists {
				return fmt.Errorf("scanPlan SQLMap Worker must use request artifact mode")
			}
		} else if len(tool.TestParameters) != 0 {
			return fmt.Errorf("scanPlan testParameters are only supported by SQLMap")
		}
		if execution.Tool == "scan_ffuf" {
			if tool.WordlistArtifact == "" {
				return fmt.Errorf("scanPlan ffuf policy requires wordlistArtifact")
			}
			if err := scanPlanContextArtifact(stage, tool.WordlistArtifact); err != nil {
				return err
			}
			expected["wordlist_ref"] = contracts.ToolArgumentBinding{Source: "artifact", Name: "wordlist"}
		} else if tool.WordlistArtifact != "" {
			return fmt.Errorf("scanPlan wordlistArtifact is only supported by ffuf")
		}
		for name, source := range expected {
			actual, exists := execution.Arguments[name]
			if !exists || actual.Source != source.Source || actual.Name != source.Name || actual.Value != nil {
				return fmt.Errorf("scanPlan Worker has an invalid dynamic input binding")
			}
		}
		for name, source := range execution.Arguments {
			if _, dynamic := expected[name]; !dynamic && source.Source != "literal" {
				return fmt.Errorf("scanPlan additional Worker arguments must be literals")
			}
		}
	}
	if len(stage.Result.Artifacts) != 1 {
		return fmt.Errorf("scan-plan@1 requires exactly one aggregate report result")
	}
	report, exists := stage.Result.Artifacts["report"]
	reportMedia := "application/json"
	if stage.AuditScan != nil {
		reportMedia = "application/zip"
	}
	if !exists || !report.Required || report.From == nil || len(report.MediaTypes) != 1 || report.MediaTypes[0] != reportMedia {
		return fmt.Errorf("scan-plan@1 requires a bound application/json aggregate report")
	}
	if err := validateArtifactComponent("scanPlan report namespace", report.From.Namespace); err != nil {
		return err
	}
	if err := validateArtifactComponent("scanPlan report name", report.From.Name); err != nil {
		return err
	}
	if len(report.From.Name) > 100 {
		return fmt.Errorf("scanPlan aggregate report name prefix must contain at most 100 bytes")
	}
	if artifactpolicy.IsPurposeReservedNamespace(report.From.Namespace) || artifactpolicy.IsReservedMemoryBinding(report.From.Namespace, report.From.Name) || strings.HasPrefix(report.From.Name, "tool-invocation.") {
		return fmt.Errorf("scanPlan aggregate report must not use a reserved binding")
	}
	for _, binding := range stage.Agents {
		if report.From.Namespace == binding.Namespace {
			return fmt.Errorf("scanPlan aggregate report requires a namespace separate from Workers")
		}
	}
	return nil
}

func scanPlanContextArtifact(stage ResolvedStage, name string) error {
	if err := validateMapKey("scanPlan context artifact", name); err != nil {
		return err
	}
	artifact, exists := stage.Context.Artifacts[name]
	if !exists || !artifact.Required {
		return fmt.Errorf("scanPlan requires a declared required context artifact")
	}
	if err := validateArtifactComponent("scanPlan context namespace", artifact.Namespace); err != nil {
		return err
	}
	if err := validateArtifactComponent("scanPlan context name", artifact.Name); err != nil {
		return err
	}
	if artifactpolicy.IsReservedMemoryBinding(artifact.Namespace, artifact.Name) {
		return fmt.Errorf("scanPlan context must not select reserved Memory")
	}
	return nil
}

func validateScanPlanInputMedia(workflow ResolvedWorkflow, stage ResolvedStage) error {
	validate := func(name string, allowed []string) error {
		binding := stage.Context.Artifacts[name]
		if binding.Namespace != "inputs" {
			// Inputs produced by earlier Stages are inspected at their exact
			// pinned revision when planning begins.
			return nil
		}
		slot, exists := workflow.Inputs[binding.Name]
		if !exists || !slot.Required || len(slot.MediaTypes) == 0 {
			return fmt.Errorf("scanPlan context must select a declared required Workflow input")
		}
		for _, media := range slot.MediaTypes {
			if !slices.Contains(allowed, media) {
				return fmt.Errorf("scanPlan Workflow input declares an unsupported media type")
			}
		}
		return nil
	}
	inputMedia := []string{contracts.HTTPRequestSetMediaType, "text/vnd.contractor.target-list"}
	if stage.AuditScan != nil {
		inputMedia = []string{"application/json", "application/yaml"}
		for name, media := range map[string][]string{
			stage.AuditScan.SettingsArtifact: {"application/json"},
			stage.AuditScan.TaskArtifact:     {"application/zip"},
			stage.AuditScan.ManifestArtifact: {"application/json"},
		} {
			if err := validate(name, media); err != nil {
				return err
			}
		}
	}
	if err := validate(stage.ScanPlan.InputArtifact, inputMedia); err != nil {
		return err
	}
	for _, tool := range stage.ScanPlan.Tools {
		if tool.WordlistArtifact != "" {
			if err := validate(tool.WordlistArtifact, []string{"text/vnd.contractor.wordlist", "text/plain"}); err != nil {
				return err
			}
		}
	}
	return nil
}
