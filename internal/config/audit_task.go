package config

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// AuditTaskExecution declares which task contract a Workflow accepts and which
// Stage owns its execution/result. Surrounding Stages remain ordinary Workflow
// graph nodes; the task Stage need not be the entry Stage.
type AuditTaskExecution struct {
	Contract contracts.AuditTaskContract `json:"contract" yaml:"contract"`
	Stage    string                      `json:"stage" yaml:"stage"`
}

func cloneAuditTask(value *AuditTaskExecution) *AuditTaskExecution {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

// ValidateAuditTaskWorkflow checks executor capabilities without consulting any
// inventory or profile. A declaration alone cannot grant an unsupported one.
func ValidateAuditTaskWorkflow(workflow ResolvedWorkflow) error {
	execution := workflow.AuditTask
	for name, stage := range workflow.Stages {
		if stage.AuditScan != nil && (execution == nil || name != execution.Stage) {
			return fmt.Errorf("Stage %q must be the declared auditTask executor", name)
		}
	}
	if execution == nil {
		return nil
	}
	stage, exists := workflow.Stages[execution.Stage]
	if !exists {
		return fmt.Errorf("auditTask.stage names an absent Stage")
	}
	switch execution.Contract {
	case contracts.AuditTaskOpenAPIScanV1:
		if stage.AuditScan == nil {
			return fmt.Errorf("%s requires an auditScan executor", execution.Contract)
		}
		if err := ValidateScanPlanStage(stage); err != nil {
			return err
		}
		return validateScanPlanInputMedia(workflow, stage)
	default:
		return fmt.Errorf("unsupported auditTask contract %q", execution.Contract)
	}
}

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
	for _, role := range sortedMapKeys(profile.Workflows) {
		binding := profile.Workflows[role]
		if err := ValidateAuditTaskWorkflow(binding.Workflow); err != nil {
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
	case contracts.AuditTaskOpenAPIScanV1:
		// This executor handles one item and one scanner invocation. The
		// restriction belongs to the consumer, not to its inventory producer.
		if profile.Execution.BatchSize != 1 {
			return fmt.Errorf("%s executor requires batchSize 1", execution.Contract)
		}
		if profile.Interaction.ActiveChecks != AuditActiveChecksApprovalRequired {
			return fmt.Errorf("%s executor requires approval-required active checks", execution.Contract)
		}
		return validateAuditScanMappings(profile, binding, stage)
	default:
		return fmt.Errorf("unsupported auditTask contract %q", execution.Contract)
	}
}
