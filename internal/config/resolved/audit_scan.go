package resolved

import (
	"fmt"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
)

// AuditScanConfig opts scan-plan@1 into trusted single-item preparation and
// result assembly. Settings and manifest bytes are pinned Audit inputs.
type AuditScanConfig struct {
	Scanner          string `json:"scanner" yaml:"scanner"`
	SettingsArtifact string `json:"settingsArtifact" yaml:"settingsArtifact"`
	TaskArtifact     string `json:"taskArtifact" yaml:"taskArtifact"`
	ManifestArtifact string `json:"manifestArtifact" yaml:"manifestArtifact"`
	UnknownOutcome   string `json:"unknownOutcome" yaml:"unknownOutcome"`
}

func cloneAuditScan(value *AuditScanConfig) *AuditScanConfig {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func validateAuditScanStage(stage ResolvedStage) error {
	value := stage.AuditScan
	if value == nil {
		return nil
	}
	if value.Scanner != "sqlmap" && value.Scanner != "nuclei" {
		return fmt.Errorf("auditScan scanner must be sqlmap or nuclei")
	}
	if value.UnknownOutcome != "retain-gap" {
		return fmt.Errorf("auditScan requires unknownOutcome retain-gap")
	}
	if stage.ScanPlan == nil || len(stage.ScanPlan.Tools) != 1 || len(stage.Agents) != 1 ||
		stage.ScanPlan.MaxInputs != 1 || stage.ScanPlan.MaxJobs != 1 {
		return fmt.Errorf("auditScan requires one Worker and one input/job per assigned item")
	}
	tool := stage.ScanPlan.Tools[0]
	binding, ok := stage.Agents[tool.Worker]
	if !ok || binding.Template.Execution == nil || binding.Template.Execution.Tool != "scan_"+value.Scanner ||
		tool.MaxJobs != 1 || len(tool.TestParameters) != 0 || tool.WordlistArtifact != "" {
		return fmt.Errorf("auditScan Worker must match scanner; test parameters come from exact settings")
	}
	if value.Scanner == "nuclei" {
		templates, ok := binding.Template.Execution.Arguments["template_ids"]
		text, isString := templates.Value.(string)
		if !ok || !(templates.Source == "literal" && isString && text != "" || scan.IsNucleiSelectionBinding("template_ids", templates)) {
			return fmt.Errorf("auditScan Nuclei requires a literal or exact-settings template selection")
		}
	}
	return validateAuditScanContext(stage)
}

func validateAuditScanContext(stage ResolvedStage) error {
	value := stage.AuditScan
	slots := []string{stage.ScanPlan.InputArtifact, value.SettingsArtifact, value.TaskArtifact, value.ManifestArtifact}
	for _, agent := range stage.Agents {
		if binding, exists := agent.Template.Execution.Arguments["rulepack_ref"]; exists && scan.IsNucleiSelectionBinding("rulepack_ref", binding) {
			slots = append(slots, binding.Name)
		}
	}
	seen := map[string]bool{}
	for _, name := range slots {
		if seen[name] {
			return fmt.Errorf("auditScan artifact slots must be distinct")
		}
		seen[name] = true
		context, exists := stage.Context.Artifacts[name]
		if !exists {
			return fmt.Errorf("auditScan context input is absent")
		}
		if name == "rulepack" && !context.Required {
			if ValidateArtifactComponent("rulepack context", context.Name) != nil {
				return fmt.Errorf("invalid rulepack context")
			}
		} else {
			if err := scanPlanContextArtifact(stage, name); err != nil {
				return err
			}
		}
		if stage.Context.Artifacts[name].Namespace != "inputs" {
			return fmt.Errorf("auditScan requires direct immutable Workflow inputs")
		}
	}
	if len(stage.Context.Artifacts) != len(slots) || stage.Context.Workspace != nil {
		return fmt.Errorf("auditScan accepts only source, settings, task and execution manifest context")
	}
	return nil
}
