package evalservice

import (
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// The Python producer checks the same zero-call fixture. Real recorded
// Planner/Worker/summary reports count once; the executor states its zero.
// Removing those counters preserves the historical partial outcome.
func TestSharedEvalObservedHandoffUsageWithExplicitZeroExecutor(t *testing.T) {
	pool := serviceTestPool(t)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	repository := telemetry.NewRepository(pool)
	value := func(n int64) *int64 { return &n }
	for _, missing := range []bool{false, true} {
		t.Run(fmt.Sprint(missing), func(t *testing.T) {
			run := fmt.Sprintf("zero-executor-%t", missing)
			stage := run + "-stage"
			createRunningUsageRun(t, runs, run)
			recordSummarizedUsageStage(t, pool, run, stage, false)
			allocations, err := runs.ListStageAllocations(t.Context(), stage)
			if err != nil || len(allocations) != 1 {
				t.Fatalf("builder allocation: %v %v", allocations, err)
			}
			executor := allocations[0]
			executor.AllocationID = stage + "-executor"
			executor.LogicalAgentName, executor.Namespace = "planner_executor", "planner_executor"
			executor.WorkerRuntimeRef = contracts.WorkerRuntimeRef{RuntimeID: "planner", Version: "1"}
			if err := runs.RecordStageAllocation(t.Context(), executor); err != nil {
				t.Fatal(err)
			}
			var metrics reporting.ExecutionMetrics
			if err := strictjson.DecodeNonNull(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "zero-direct-model-usage.json"), &metrics); err != nil {
				t.Fatal(err)
			}
			if missing {
				metrics.InputTokens, metrics.OutputTokens, metrics.TotalTokens = nil, nil, nil
			}
			now := time.Now().UTC()
			if err := repository.RecordAllocationReport(t.Context(), telemetry.AllocationReportEnvelope{
				StageExecutionID: stage, AllocationID: executor.AllocationID,
				LogicalAgentName: executor.LogicalAgentName, ReportSchemaVersion: contracts.APIVersion,
				PerformanceCollectionPolicy: reporting.PerformanceCollectionDisabled, ReceivedAt: now,
				Report: reporting.AllocationFinalReport{
					ReportID: "final-" + executor.AllocationID, AllocationID: executor.AllocationID,
					StartedAt: now.Add(-time.Second), FinishedAt: now,
					Worker: reporting.ExecutionReport{ReportID: "worker-" + executor.AllocationID, Complete: true,
						Metrics: metrics, ToolCalls: []reporting.ToolCallRecord{}, Errors: []reporting.ExecutionError{}},
					Runtime: reporting.RuntimeReport{Complete: true, Adapters: map[runtimesettings.RuntimeAdapterRef]reporting.RuntimeAdapterMetrics{}},
				},
			}); err != nil {
				t.Fatal(err)
			}
			if err := repository.RecordPlannerReport(t.Context(), telemetry.PlannerReportEnvelope{
				StageExecutionID: stage, SessionID: stage + "-session", InvocationID: stage + "-invocation",
				StartedAt: now.Add(-time.Second), FinishedAt: now, ReportSchemaVersion: contracts.APIVersion,
				Report: reporting.ExecutionReport{ReportID: "planner-" + stage, Complete: true,
					Metrics:   reporting.ExecutionMetrics{ModelCalls: value(1), InputTokens: value(8), OutputTokens: value(2), TotalTokens: value(10), Tools: map[string]reporting.ToolMetrics{}},
					ToolCalls: []reporting.ToolCallRecord{}, Errors: []reporting.ExecutionError{}},
			}); err != nil {
				t.Fatal(err)
			}
			if _, err := repository.RebuildStageMetrics(t.Context(), stage, contracts.APIVersion); err != nil {
				t.Fatal(err)
			}
			usage, err := observedRunUsage(t, pool, run)
			coverage := "complete"
			if missing {
				coverage = "partial"
			}
			if err != nil || usage.TotalTokens.Value == nil || *usage.TotalTokens.Value != 155 || usage.TotalTokens.Completeness != coverage || usage.ModelCalls.Value == nil || *usage.ModelCalls.Value != 4 {
				t.Fatalf("handoff usage = %+v %v; want 155 tokens, 4 calls, %s", usage, err, coverage)
			}
		})
	}
}
