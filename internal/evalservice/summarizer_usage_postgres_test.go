package evalservice

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5/pgxpool"
)

func recordSummarizedUsageStage(t *testing.T, pool *pgxpool.Pool, runID, stageID string, unavailable bool) {
	t.Helper()
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	if err := runstoretest.CreateRunningNamedStage(t.Context(), runs, runID, stageID, stageID, contracts.APIVersion); err != nil {
		t.Fatal(err)
	}
	gateway := llmgateway.LLMGatewayConfigRef{GatewayID: "fixture", Version: "1", Digest: "sha256:" + strings.Repeat("b", 64)}
	configuration := &runstore.AllocationRuntimeConfiguration{ModelPolicy: contracts.ModelPolicyRef{PolicyID: "worker", Version: "1", Digest: "sha256:" + strings.Repeat("c", 64)}, Origins: runtimeconfig.ResolvedRuntimeConfigOrigins{LLMGateway: &runtimeconfig.RuntimeFieldOrigin{Layer: runtimeconfig.LayerWorkflow}}, Provenance: runtimesettings.ResolvedRuntimeConfigProvenance{Default: runtimesettings.RuntimeLabelBindingProvenance{Label: "default", BindingRevision: 1, Config: runtimesettings.RuntimeConfigRef{Name: runtimeconfig.BuiltInName, Version: runtimeconfig.BuiltInVersion, Digest: runtimeconfig.BuiltInDigest}}, RunLabels: []runtimesettings.RuntimeLabelBindingProvenance{}, AgentLabels: []runtimesettings.RuntimeLabelBindingProvenance{}, RuntimeAdapters: []runtimesettings.RuntimeAdapterRef{}, LLMGatewayConfig: &gateway, RuntimeCredentialRefs: []runtimesettings.RuntimeCredentialRef{}}}
	allocation := stageID + "-allocation"
	if err := runs.RecordStageAllocation(t.Context(), runstore.StageAllocation{AllocationID: allocation, StageExecutionID: stageID, LogicalAgentName: "builder", Namespace: "builder", AgentTemplateRef: contracts.AgentTemplateRef{TemplateID: "fixture", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64)}, WorkerRuntimeRef: contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"}, RuntimeAgentID: strings.Repeat("1", 64), RuntimeAgentInstanceID: "fixture-instance", RuntimeAgentLabelRevision: 1, RuntimeConfigurationSchemaVersion: runstore.AllocationRuntimeConfigurationSchemaVersion, RuntimeConfiguration: configuration, PerformanceCollectionPolicy: reporting.PerformanceCollectionDisabled}); err != nil {
		t.Fatal(err)
	}
	value := func(v int64) *int64 { return &v }
	summary := &reporting.WorkerSummarizerMetrics{Attempts: 1, Succeeded: 1, ModelCalls: 1, InputTokens: 20, OutputTokens: 5, TotalTokens: 25, FailureCodes: map[string]uint64{}}
	if unavailable {
		summary.InputTokens = 0
		summary.OutputTokens = 0
		summary.TotalTokens = 0
		summary.TokenUsageUnavailable = 1
	}
	repository := telemetry.NewRepository(pool)
	now := time.Now().UTC()
	envelope := telemetry.AllocationReportEnvelope{StageExecutionID: stageID, AllocationID: allocation, LogicalAgentName: "builder", ReportSchemaVersion: contracts.APIVersion, PerformanceCollectionPolicy: reporting.PerformanceCollectionDisabled, ReceivedAt: now, Report: reporting.AllocationFinalReport{AllocationID: allocation, ReportID: "final-" + allocation, StartedAt: now.Add(-time.Second), FinishedAt: now, Worker: reporting.ExecutionReport{ReportID: "worker-" + allocation, Complete: true, Metrics: reporting.ExecutionMetrics{ModelCalls: value(2), InputTokens: value(100), OutputTokens: value(20), TotalTokens: value(120), Summarizer: summary, LLMCost: &reporting.LLMCost{Currency: "USD", Calls: 3, Groups: []reporting.LLMCostGroup{{Model: "fixture", Phase: "main", Source: "provider", Calls: 2, Amount: "0.2"}, {Model: "fixture", Phase: "summarization", Source: "provider", Calls: 1, Amount: "0.1"}}}, Tools: map[string]reporting.ToolMetrics{}}, ToolCalls: []reporting.ToolCallRecord{}, Errors: []reporting.ExecutionError{}}, Runtime: reporting.RuntimeReport{Complete: true, Adapters: map[runtimesettings.RuntimeAdapterRef]reporting.RuntimeAdapterMetrics{}}}}
	if err := repository.RecordAllocationReport(t.Context(), envelope); err != nil {
		t.Fatal(err)
	}
	if _, err := repository.RebuildStageMetrics(t.Context(), stageID, contracts.APIVersion); err != nil {
		t.Fatal(err)
	}
}

func TestEvalSummarizerUsageAndKnownBudgetShareStoredReportAuthority(t *testing.T) {
	pool := serviceTestPool(t)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	store := evalstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	for _, unavailable := range []bool{false, true} {
		t.Run(fmt.Sprint(unavailable), func(t *testing.T) {
			run := fmt.Sprintf("summary-run-%t", unavailable)
			createRunningUsageRun(t, runs, run)
			recordSummarizedUsageStage(t, pool, run, run+"-stage", unavailable)
			wantTokens := float64(145)
			coverage := "complete"
			if unavailable {
				wantTokens = 120
				coverage = "partial"
			}
			for range 2 {
				usage, err := observedRunUsage(t, pool, run)
				if err != nil || usage.ModelCalls.Value == nil || *usage.ModelCalls.Value != 3 || usage.TotalTokens.Value == nil || *usage.TotalTokens.Value != wantTokens || usage.TotalTokens.Completeness != coverage {
					t.Fatalf("whole-task usage: %+v %v", usage, err)
				}
				known, err := store.KnownTokens(t.Context(), "owner", "run", &run)
				if err != nil || known != int64(wantTokens) {
					t.Fatalf("known budget %d %v", known, err)
				}
				foreign, err := store.KnownTokens(t.Context(), "foreign", "run", &run)
				if err != nil || foreign != 0 {
					t.Fatalf("foreign budget %d %v", foreign, err)
				}
			}
		})
	}
}
