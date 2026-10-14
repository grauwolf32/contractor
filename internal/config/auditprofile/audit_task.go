package auditprofile

import (
	"fmt"
	"maps"
	"slices"

	"github.com/grauwolf32/contractor/internal/config/resolved"
)

// ValidateAuditTaskProfile connects a producer's output contract to a consumer's
// accepted contract. Inventory identifiers never select Workflow graph shapes.
func ValidateAuditTaskProfile(profile ResolvedAuditProfile) error {
	producer, exists := auditInventories[profile.Inventory.Implementation]
	if !exists {
		return fmt.Errorf("unsupported inventory implementation %q", profile.Inventory.Implementation)
	}
	if _, exists := profile.Workflows[profile.Inventory.ItemWorkflowRole]; !exists {
		return fmt.Errorf("inventory names an absent item Workflow role")
	}
	for _, role := range slices.Sorted(maps.Keys(profile.Workflows)) {
		binding := profile.Workflows[role]
		if err := resolved.ValidateAuditTaskWorkflow(binding.Workflow); err != nil {
			return fmt.Errorf("Workflow role %q: %w", role, err)
		}
		execution := binding.Workflow.AuditTask
		if execution == nil {
			if role == profile.Inventory.ItemWorkflowRole && producer.taskContract != "" {
				return fmt.Errorf("inventory produces %s tasks; item Workflow must declare auditTask", producer.taskContract)
			}
			continue
		}
		if role == profile.Inventory.ItemWorkflowRole && execution.Contract != producer.taskContract {
			return fmt.Errorf("inventory task contract %q differs from Workflow contract %q", producer.taskContract, execution.Contract)
		}
		if binding.Kind != AuditWorkflowCheck || binding.WorkerCompletion != nil {
			return fmt.Errorf("auditTask requires a check role with executor-owned completion")
		}
		if err := validateAuditTaskBinding(profile, binding); err != nil {
			return err
		}
	}
	return nil
}

func validateAuditTaskBinding(profile ResolvedAuditProfile, binding ResolvedAuditWorkflowBinding) error {
	execution := binding.Workflow.AuditTask
	stage := binding.Workflow.Stages[execution.Stage]
	switch execution.Contract {
	case resolved.AuditTaskOpenAPIScanV1:
		// This executor handles one item and one scanner invocation. The
		// restriction belongs to the consumer, not to its inventory producer.
		if profile.Execution.BatchSize != 1 {
			return fmt.Errorf("%s executor requires batchSize 1", execution.Contract)
		}
		if profile.Interaction.ActiveChecks == AuditActiveChecksProhibited {
			return fmt.Errorf("%s executor requires active checks", execution.Contract)
		}
		return validateAuditScanMappings(profile, binding, stage)
	default:
		return fmt.Errorf("unsupported auditTask contract %q", execution.Contract)
	}
}

func validateAuditScanMappings(profile ResolvedAuditProfile, binding ResolvedAuditWorkflowBinding, stage resolved.ResolvedStage) error {
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
