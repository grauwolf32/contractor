package config

import "fmt"

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
		if !ok || templates.Source != "literal" || !isString || text == "" {
			return fmt.Errorf("auditScan Nuclei requires a fixed literal template_ids selection")
		}
	}
	return validateAuditScanContext(stage)
}

func validateAuditScanContext(stage ResolvedStage) error {
	value := stage.AuditScan
	slots := []string{stage.ScanPlan.InputArtifact, value.SettingsArtifact, value.TaskArtifact, value.ManifestArtifact}
	seen := map[string]bool{}
	for _, name := range slots {
		if seen[name] {
			return fmt.Errorf("auditScan artifact slots must be distinct")
		}
		seen[name] = true
		if err := scanPlanContextArtifact(stage, name); err != nil {
			return err
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

func validateAuditScanMappings(profile ResolvedAuditProfile, binding ResolvedAuditWorkflowBinding, stage ResolvedStage) error {
	if profile.Inventory.Source == nil || profile.Inventory.Settings == nil || *profile.Inventory.Source == *profile.Inventory.Settings {
		return fmt.Errorf("scan task requires distinct source and settings")
	}
	checks := []struct {
		slot    string
		mapping AuditWorkflowInputMapping
	}{
		{stage.ScanPlan.InputArtifact, *profile.Inventory.Source},
		{stage.AuditScan.SettingsArtifact, *profile.Inventory.Settings},
		{stage.AuditScan.TaskArtifact, AuditWorkflowInputMapping{Source: AuditInputFromItemPackage}},
		{stage.AuditScan.ManifestArtifact, AuditWorkflowInputMapping{Source: AuditInputFromExecutionManifest}},
	}
	for _, check := range checks {
		mapping, exists := binding.Inputs[stage.Context.Artifacts[check.slot].Name]
		if !exists || mapping != check.mapping {
			return fmt.Errorf("auditScan input mapping differs from inventory authority")
		}
	}
	if output, exists := binding.Outputs["result"]; !exists || stage.WorkflowOutputs[output] != "report" {
		return fmt.Errorf("auditScan must publish its canonical result package")
	}
	output := binding.Outputs["result"]
	for name, other := range binding.Workflow.Stages {
		if name != binding.Workflow.AuditTask.Stage {
			if _, publishes := other.WorkflowOutputs[output]; publishes {
				return fmt.Errorf("auditTask canonical result has another Stage producer")
			}
		}
	}
	return nil
}
