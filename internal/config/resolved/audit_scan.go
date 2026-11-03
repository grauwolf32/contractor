package resolved

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
)

// ScanPreparationConfig prepares one fixed operation or target using exact
// ordinary Workflow artifacts. It grants no Audit identity or dispatch authority.
type ScanPreparationConfig struct {
	Kind              string `json:"kind" yaml:"kind"`
	Scanner           string `json:"scanner" yaml:"scanner"`
	SettingsArtifact  string `json:"settingsArtifact" yaml:"settingsArtifact"`
	OperationArtifact string `json:"operationArtifact,omitempty" yaml:"operationArtifact,omitempty"`
	UnknownOutcome    string `json:"unknownOutcome" yaml:"unknownOutcome"`
}

func cloneScanPreparation(value *ScanPreparationConfig) *ScanPreparationConfig {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func validateScanPreparationStage(stage ResolvedStage) error {
	value := stage.ScanPreparation
	if value == nil {
		return nil
	}
	if value.Scanner != "sqlmap" && value.Scanner != "nuclei" {
		return fmt.Errorf("scanPreparation scanner must be sqlmap or nuclei")
	}
	if value.UnknownOutcome != "retain-gap" {
		return fmt.Errorf("scanPreparation requires unknownOutcome retain-gap")
	}
	if value.Kind != "openapi-operation" && value.Kind != "web-target" {
		return fmt.Errorf("unsupported scanPreparation kind")
	}
	if (value.Kind == "openapi-operation") != (value.OperationArtifact != "") {
		return fmt.Errorf("OpenAPI preparation requires its exact operation artifact")
	}
	if stage.ScanPlan == nil || len(stage.ScanPlan.Tools) != 1 || len(stage.Agents) != 1 || stage.ScanPlan.MaxInputs != 1 || stage.ScanPlan.MaxJobs != 1 {
		return fmt.Errorf("scanPreparation requires one Worker and one input/job")
	}
	tool := stage.ScanPlan.Tools[0]
	binding, ok := stage.Agents[tool.Worker]
	if !ok || binding.Template.Execution == nil || binding.Template.Execution.Tool != "scan_"+value.Scanner || tool.MaxJobs != 1 || len(tool.TestParameters) != 0 || tool.WordlistArtifact != "" {
		return fmt.Errorf("scanPreparation Worker must match the exact scanner settings")
	}
	if value.Scanner == "nuclei" {
		templates, ok := binding.Template.Execution.Arguments["template_ids"]
		text, isString := templates.Value.(string)
		if !ok || !(templates.Source == "literal" && isString && text != "" || scan.IsNucleiSelectionBinding("template_ids", templates)) {
			return fmt.Errorf("Nuclei preparation requires a literal or exact-settings template selection")
		}
	}
	if value.Kind == "web-target" {
		if value.Scanner != "nuclei" {
			return fmt.Errorf("web target preparation requires Nuclei")
		}
		for _, name := range []string{"template_ids", "tags", "severity", "rate_limit"} {
			if !scan.IsNucleiSelectionBinding(name, binding.Template.Execution.Arguments[name]) {
				return fmt.Errorf("web target preparation requires exact-settings selection bindings")
			}
		}
		pin := binding.Template.Execution.Arguments["templates_digest"]
		digest, ok := pin.Value.(string)
		if pin.Source != "literal" || !ok || contracts.ValidateDigest("templates_digest", digest) != nil {
			return fmt.Errorf("web target preparation requires a pinned installed package")
		}
	}
	return validateScanPreparationContext(stage)
}

func validateScanPreparationContext(stage ResolvedStage) error {
	value := stage.ScanPreparation
	slots := []string{stage.ScanPlan.InputArtifact, value.SettingsArtifact}
	if value.OperationArtifact != "" {
		slots = append(slots, value.OperationArtifact)
	}
	for _, agent := range stage.Agents {
		if binding, exists := agent.Template.Execution.Arguments["rulepack_ref"]; exists && scan.IsNucleiSelectionBinding("rulepack_ref", binding) {
			slots = append(slots, binding.Name)
		}
	}
	seen := map[string]bool{}
	for _, name := range slots {
		if seen[name] {
			return fmt.Errorf("scanPreparation artifact slots must be distinct")
		}
		seen[name] = true
		input, exists := stage.Context.Artifacts[name]
		if !exists {
			return fmt.Errorf("scanPreparation context input is absent")
		}
		if name == "rulepack" && !input.Required {
			if ValidateArtifactComponent("rulepack context", input.Name) != nil {
				return fmt.Errorf("invalid rulepack context")
			}
		} else if err := scanPlanContextArtifact(stage, name); err != nil {
			return err
		}
		if input.Namespace != "inputs" {
			return fmt.Errorf("scanPreparation requires immutable Workflow inputs")
		}
	}
	if len(stage.Context.Artifacts) != len(slots) || stage.Context.Workspace != nil {
		return fmt.Errorf("scanPreparation accepts only its declared source, operation, settings and rulepack")
	}
	return nil
}
