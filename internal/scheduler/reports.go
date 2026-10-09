package scheduler

import (
	"context"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

func (s *Scheduler) persistReports(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	execution runstore.StageExecution,
	reservations []controlplane.Reservation,
	reports map[string]reporting.AllocationFinalReport,
) {
	// A crash can resume directly in finalizing/aborting without recreating the
	// Planner instance. Ensure its durable session still has a report record;
	// an already persisted complete report wins over this placeholder.
	s.persistPlannerReport(ctx, execution, nil, nil)
	listContext, cancelList := s.terminalOperationContext(ctx)
	allocations, err := s.store.ListStageAllocations(listContext, execution.StageExecutionID)
	cancelList()
	if err != nil {
		s.options.Logger.Warn("list allocations for reports failed", "stage_execution_id", execution.StageExecutionID)
		return
	}
	if len(allocations) == 0 {
		return
	}
	byName := make(map[string]controlplane.Reservation, len(reservations))
	for _, reservation := range reservations {
		byName[reservation.Grant.LogicalAgentName] = reservation
	}
	finishedAt := s.now().UTC().Round(0)
	startedAt := execution.CreatedAt.UTC().Round(0)
	if execution.PlannerStartedAt != nil {
		startedAt = execution.PlannerStartedAt.UTC().Round(0)
	}
	if startedAt.IsZero() || startedAt.After(finishedAt) {
		startedAt = finishedAt
	}
	for _, allocation := range allocations {
		report, ok := reports[allocation.LogicalAgentName]
		reservation, live := byName[allocation.LogicalAgentName]
		validReport := ok && report.AllocationID == allocation.AllocationID &&
			(!live || reservation.Grant.AllocationID == allocation.AllocationID)
		if !validReport {
			retryable := true
			report = reporting.AllocationFinalReport{
				ReportID:     "allocation-final-missing-" + allocation.AllocationID,
				AllocationID: allocation.AllocationID,
				StartedAt:    startedAt,
				FinishedAt:   finishedAt,
				Worker: reporting.ExecutionReport{
					ReportID:  "worker-missing-" + allocation.AllocationID,
					Complete:  false,
					Metrics:   reporting.ExecutionMetrics{Tools: map[string]reporting.ToolMetrics{}},
					ToolCalls: []reporting.ToolCallRecord{},
					Errors: []reporting.ExecutionError{{
						Code:      "allocation_report_unavailable",
						Message:   "Runtime Agent did not return an execution report before lifecycle completion",
						Retryable: &retryable,
					}},
				},
				Runtime: reporting.RuntimeReport{Complete: false},
			}
		}
		secrets := s.telemetrySecrets()
		if live && validReport {
			resolutionContext, cancelResolution := s.terminalOperationContext(ctx)
			secrets = s.allocationReportSecrets(resolutionContext, run, workflow, reservation)
			cancelResolution()
		}
		operationContext, cancel := s.terminalOperationContext(ctx)
		err := s.store.RecordStageExecutionReport(operationContext, runstore.RecordStageExecutionReportParams{
			StageExecutionID: execution.StageExecutionID, AllocationID: allocation.AllocationID,
			LogicalAgentName: allocation.LogicalAgentName, ReportSchemaVersion: contracts.APIVersion,
			Report: report, PerformanceCollectionPolicy: allocation.PerformanceCollectionPolicy,
			Secrets: secrets,
		})
		cancel()
		if err != nil {
			s.options.Logger.Warn(
				"execution report persistence failed",
				"stage_execution_id", execution.StageExecutionID,
				"logical_agent_name", allocation.LogicalAgentName,
			)
		}
	}
	s.rebuildStageMetrics(ctx, execution.StageExecutionID)
}

func (s *Scheduler) persistPlannerReport(
	ctx context.Context,
	execution runstore.StageExecution,
	instance planner.Planner,
	exportResult *telemetry.PlannerExportResult,
) {
	if execution.PlannerSessionID == nil || execution.PlannerInvocationID == nil {
		return
	}
	report := reporting.ExecutionReport{
		ReportID:  "planner-missing-" + *execution.PlannerSessionID,
		Complete:  false,
		Metrics:   reporting.ExecutionMetrics{Tools: map[string]reporting.ToolMetrics{}},
		ToolCalls: []reporting.ToolCallRecord{},
		Errors:    []reporting.ExecutionError{},
	}
	if provider, ok := instance.(planner.ReportProvider); ok {
		if provided, available := provider.ExecutionReport(); available {
			report = provided
		}
	}
	applyPlannerTelemetryResult(&report, exportResult)
	operationContext, cancel := s.terminalOperationContext(ctx)
	startedAt := execution.CreatedAt.UTC().Round(0)
	if execution.PlannerStartedAt != nil {
		startedAt = execution.PlannerStartedAt.UTC().Round(0)
	}
	if startedAt.IsZero() {
		startedAt = s.now().UTC().Round(0)
	}
	finishedAt := startedAt
	if report.Metrics.DurationMS != nil {
		finishedAt = startedAt.Add(time.Duration(*report.Metrics.DurationMS) * time.Millisecond)
	}
	err := s.store.RecordPlannerExecutionReport(operationContext, runstore.RecordPlannerExecutionReportParams{
		StageExecutionID: execution.StageExecutionID,
		SessionID:        *execution.PlannerSessionID, InvocationID: *execution.PlannerInvocationID,
		StartedAt: startedAt, FinishedAt: finishedAt,
		ReportSchemaVersion: contracts.APIVersion, Report: report,
		Secrets: s.telemetrySecrets(),
	})
	cancel()
	if err != nil {
		s.options.Logger.Warn(
			"Planner report persistence failed",
			"stage_execution_id", execution.StageExecutionID,
		)
		return
	}
	s.rebuildStageMetrics(ctx, execution.StageExecutionID)
}

func applyPlannerTelemetryResult(
	report *reporting.ExecutionReport,
	result *telemetry.PlannerExportResult,
) {
	if report == nil || result == nil || !result.Attempted {
		return
	}
	if report.Metrics.Tools == nil {
		report.Metrics.Tools = make(map[string]reporting.ToolMetrics)
	}
	calls, succeeded, failed := int64(1), int64(0), int64(1)
	outcome := reporting.ToolCallFailed
	var executionError *reporting.ExecutionError
	if result.Succeeded {
		succeeded, failed = 1, 0
		outcome = reporting.ToolCallSucceeded
	} else {
		retryable := false
		executionError = &reporting.ExecutionError{
			Code: result.ErrorCode, Message: "Planner telemetry export failed", Retryable: &retryable,
		}
	}
	report.Metrics.Tools["telemetry.export"] = reporting.ToolMetrics{
		Calls: &calls, Succeeded: &succeeded, Failed: &failed,
	}
	if len(report.ToolCalls) >= 1000 {
		report.Truncated = true
		return
	}
	report.ToolCalls = append(report.ToolCalls, reporting.ToolCallRecord{
		CallID: "planner-telemetry-export", Tool: "telemetry.export",
		Arguments: map[string]any{}, Outcome: outcome, Error: executionError,
	})
}

func (s *Scheduler) rebuildStageMetrics(ctx context.Context, stageExecutionID string) {
	operationContext, cancel := s.terminalOperationContext(ctx)
	err := s.store.RebuildStageMetrics(operationContext, stageExecutionID, contracts.APIVersion)
	cancel()
	if err != nil {
		s.options.Logger.Warn(
			"StageMetrics persistence failed", "stage_execution_id", stageExecutionID,
		)
	}
}

func (s *Scheduler) telemetrySecrets() []string {
	return append([]string(nil), s.options.TelemetrySecrets...)
}

func (s *Scheduler) allocationReportSecrets(
	ctx context.Context,
	run runstore.WorkflowRun,
	workflow executableWorkflow,
	reservation controlplane.Reservation,
) []string {
	secrets := s.telemetrySecrets()
	if reservation.ResolvedRuntimeConfig != nil {
		settings, err := s.materializeRuntimeSettings(ctx, reservation.ResolvedRuntimeConfig.Clone())
		if err != nil {
			s.options.Logger.Warn("allocation report credential resolution was incomplete", "allocation_id", reservation.Grant.AllocationID)
		} else {
			secrets = append(secrets, settings.SecretValues()...)
		}
	}
	if binding, ok := workflow.stage.Agents[reservation.Grant.LogicalAgentName]; ok && run.ProjectHTTPTarget != nil && agentUsesHTTPRequest(binding.Template) {
		target, err := s.materializeHTTPOriginTarget(ctx, *run.ProjectHTTPTarget)
		if err != nil {
			s.options.Logger.Warn("allocation report HTTP origin credential resolution was incomplete", "allocation_id", reservation.Grant.AllocationID)
		} else {
			secrets = append(secrets, (contracts.RuntimeSettings{HTTPOriginTarget: target}).SecretValues()...)
		}
	}
	return secrets
}
