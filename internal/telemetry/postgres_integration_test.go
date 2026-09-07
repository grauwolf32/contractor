package telemetry_test

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const postgresSecret = "postgres-telemetry-secret"

func TestMetricsPostgresIdempotencyRedactionAggregationAndRetention(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool)
	reports := telemetry.NewRepository(pool)
	createTelemetryRun(t, ctx, runs)

	old := time.Now().UTC().Add(-31 * 24 * time.Hour)
	oldStage := createTelemetryStage(t, ctx, runs, "stage-old", "allocation-old", "session-old")
	allocation := allocationEnvelope(oldStage, "allocation-old", old, true)
	if err := reports.RecordAllocationReport(ctx, allocation); err != nil {
		t.Fatal(err)
	}
	if err := reports.RecordAllocationReport(ctx, allocation); err != nil {
		t.Fatalf("idempotent allocation report: %v", err)
	}
	planner := plannerEnvelope(oldStage, "session-old", old)
	if err := reports.RecordPlannerReport(ctx, planner); err != nil {
		t.Fatal(err)
	}
	if err := reports.RecordPlannerReport(ctx, planner); err != nil {
		t.Fatalf("idempotent Planner report: %v", err)
	}
	metrics, err := reports.RebuildStageMetrics(ctx, oldStage, contracts.APIVersion)
	if err != nil {
		t.Fatal(err)
	}
	if metrics.Summary.ModelCalls != 5 || metrics.Summary.ToolCalls != 1 ||
		metrics.Summary.ErrorCount != 1 {
		t.Fatalf("aggregate doubled or lost counters: %+v", metrics.Summary)
	}
	var raw string
	if err := pool.QueryRow(ctx, `
SELECT report::text FROM allocation_execution_reports WHERE allocation_id = 'allocation-old'`,
	).Scan(&raw); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(raw, postgresSecret) {
		t.Fatalf("persisted report contains configured secret: %s", raw)
	}
	different := allocation
	different.Report.Worker.Complete = false
	if err := reports.RecordAllocationReport(ctx, different); !errors.Is(err, telemetry.ErrConflict) {
		t.Fatalf("different duplicate report error = %v, want conflict", err)
	}
	finishTelemetryStage(t, ctx, runs, oldStage)
	if _, err := pool.Exec(ctx, `
UPDATE stage_metrics SET expires_at = clock_timestamp() - interval '1 day'
WHERE stage_execution_id = $1`, oldStage); err != nil {
		t.Fatal(err)
	}

	activeStage := createTelemetryStage(t, ctx, runs, "stage-active", "allocation-active", "session-active")
	placeholder := allocationEnvelope(activeStage, "allocation-active", old, false)
	placeholder.Report.ReportID = "allocation-final-missing-allocation-active"
	placeholder.Report.Worker.ReportID = "worker-missing-allocation-active"
	if err := reports.RecordAllocationReport(ctx, placeholder); err != nil {
		t.Fatal(err)
	}
	late := allocationEnvelope(activeStage, "allocation-active", old.Add(time.Minute), true)
	if err := reports.RecordAllocationReport(ctx, late); err != nil {
		t.Fatalf("late complete report after placeholder: %v", err)
	}
	effective, err := reports.ListAllocationReports(ctx, activeStage)
	if err != nil || len(effective) != 1 || !effective[0].Report.Worker.Complete {
		t.Fatalf("effective late report = (%+v, %v)", effective, err)
	}
	activeMetrics, err := reports.RebuildStageMetrics(ctx, activeStage, contracts.APIVersion)
	if err != nil {
		t.Fatal(err)
	}
	if !activeMetrics.Summary.ReportsComplete {
		t.Fatalf("late complete report did not supersede placeholder: %+v", activeMetrics.Summary)
	}
	if _, err := pool.Exec(ctx, `
UPDATE stage_metrics SET expires_at = clock_timestamp() - interval '1 day'
WHERE stage_execution_id = $1`, activeStage); err != nil {
		t.Fatal(err)
	}

	newStage := createTelemetryStage(t, ctx, runs, "stage-new", "allocation-new", "session-new")
	if err := reports.RecordAllocationReport(
		ctx, allocationEnvelope(newStage, "allocation-new", time.Now().UTC(), true),
	); err != nil {
		t.Fatal(err)
	}
	if _, err := reports.RebuildStageMetrics(ctx, newStage, contracts.APIVersion); err != nil {
		t.Fatal(err)
	}
	finishTelemetryStage(t, ctx, runs, newStage)

	deleted, err := reports.CleanupExpired(ctx, time.Now().UTC(), 1)
	if err != nil {
		t.Fatal(err)
	}
	if deleted != 1 {
		t.Fatalf("first bounded cleanup deleted %d rows, want 1", deleted)
	}
	remainingDeleted, err := reports.CleanupExpired(ctx, time.Now().UTC(), 10)
	if err != nil {
		t.Fatal(err)
	}
	if remainingDeleted != 2 { // old allocation report and Planner report remain after StageMetrics.
		t.Fatalf("second cleanup deleted %d rows, want 2", remainingDeleted)
	}
	assertTelemetryCount(t, ctx, pool, "allocation_execution_reports", oldStage, 0)
	assertTelemetryCount(t, ctx, pool, "planner_execution_reports", oldStage, 0)
	assertTelemetryCount(t, ctx, pool, "stage_metrics", oldStage, 0)
	assertTelemetryCount(t, ctx, pool, "allocation_execution_reports", activeStage, 2)
	assertTelemetryCount(t, ctx, pool, "stage_metrics", activeStage, 1)
	assertTelemetryCount(t, ctx, pool, "allocation_execution_reports", newStage, 1)
	assertTelemetryCount(t, ctx, pool, "stage_metrics", newStage, 1)
}

func TestAllocationResourceHistoryUsesTerminalIdentityAndPinnedPolicy(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool)
	reports := telemetry.NewRepository(pool)
	createTelemetryRun(t, ctx, runs)

	requestedStage := createTelemetryStageWithPolicy(
		t, ctx, runs, "stage-requested", "allocation-requested", "session-requested",
		contracts.PerformanceCollectionRequested,
	)
	finishTelemetryStage(t, ctx, runs, requestedStage)
	duration, cpuUser, cpuSystem, gap := 31.0, 2.0, 1.0, 15.0
	rssStart, rssEnd, rssPeak, samples := uint64(100), uint64(125), uint64(150), uint64(4)
	envelope := allocationEnvelope(requestedStage, "allocation-requested", time.Now().UTC(), true)
	envelope.PerformanceCollectionPolicy = contracts.PerformanceCollectionRequested
	envelope.Report.Runtime.Resources = &contracts.RuntimeResources{
		Version: 1, Scope: "runtime_process", Status: contracts.ResourceComplete,
		DurationSeconds: &duration, CPUUserSeconds: &cpuUser, CPUSystemSeconds: &cpuSystem,
		RSSStartBytes: &rssStart, RSSEndBytes: &rssEnd, RSSPeakObservedBytes: &rssPeak,
		RSSSampleCount: &samples, MaxSampleGapSeconds: &gap,
	}
	if err := reports.RecordAllocationReport(ctx, envelope); err != nil {
		t.Fatal(err)
	}
	if err := runs.MarkStageAllocationReleased(ctx, "allocation-requested"); err != nil {
		t.Fatal(err)
	}

	unsupportedStage := createTelemetryStageWithPolicy(
		t, ctx, runs, "stage-unsupported", "allocation-unsupported", "session-unsupported",
		contracts.PerformanceCollectionUnsupported,
	)
	finishTelemetryStage(t, ctx, runs, unsupportedStage)
	if err := runs.MarkStageAllocationReleased(ctx, "allocation-unsupported"); err != nil {
		t.Fatal(err)
	}

	items, err := reports.ListAllocationResourceHistory(ctx, telemetry.AllocationResourceHistoryParams{
		OwnerID: "user", Limit: 10,
	})
	if err != nil || len(items) != 2 {
		t.Fatalf("allocation history = (%+v, %v)", items, err)
	}
	byID := map[string]telemetry.AllocationResourceSummary{}
	for _, item := range items {
		byID[item.AllocationID] = item
	}
	requested := byID["allocation-requested"]
	if requested.Status != telemetry.AllocationResourceAvailable || requested.Resources == nil ||
		requested.Resources.RSSPeakObservedBytes == nil || *requested.Resources.RSSPeakObservedBytes != rssPeak ||
		requested.FinishedAt.IsZero() || requested.Outcome != "succeeded" {
		t.Fatalf("requested resource summary = %+v", requested)
	}
	unsupported := byID["allocation-unsupported"]
	if unsupported.Status != telemetry.AllocationResourceUnsupported || unsupported.Resources != nil {
		t.Fatalf("unsupported resource summary = %+v", unsupported)
	}
	foreign, err := reports.ListAllocationResourceHistory(ctx, telemetry.AllocationResourceHistoryParams{
		OwnerID: "another-owner", RunID: "run-telemetry", Limit: 10,
	})
	if err != nil || len(foreign) != 0 {
		t.Fatalf("foreign history = (%+v, %v)", foreign, err)
	}
	batch, err := reports.ListStageAllocationResources(ctx, "user", []string{requestedStage, unsupportedStage})
	if err != nil || len(batch[requestedStage]) != 1 || len(batch[unsupportedStage]) != 1 {
		t.Fatalf("stage resource batch = (%+v, %v)", batch, err)
	}
}

func allocationEnvelope(
	stageExecutionID string,
	allocationID string,
	receivedAt time.Time,
	complete bool,
) telemetry.AllocationReportEnvelope {
	modelCalls, toolCalls, succeeded := int64(3), int64(1), int64(1)
	startedAt := receivedAt.Add(-time.Second)
	stopReason := "provider returned " + postgresSecret
	return telemetry.AllocationReportEnvelope{
		StageExecutionID: stageExecutionID, AllocationID: allocationID,
		LogicalAgentName: "builder", ReportSchemaVersion: contracts.APIVersion,
		Secrets: []string{postgresSecret}, ReceivedAt: receivedAt,
		Report: contracts.AllocationFinalReport{
			ReportID: "allocation-final-" + allocationID, AllocationID: allocationID,
			StartedAt: startedAt, FinishedAt: receivedAt,
			Worker: contracts.ExecutionReport{
				ReportID: "worker-" + allocationID, Complete: complete,
				Metrics: contracts.ExecutionMetrics{
					ModelCalls: &modelCalls,
					Tools: map[string]contracts.ToolMetrics{
						"probe": {Calls: &toolCalls, Succeeded: &succeeded},
					},
				},
				ToolCalls: []contracts.ToolCallRecord{{
					CallID: "call-" + allocationID, Tool: "probe",
					Arguments: map[string]any{
						"authorization": "Bearer " + postgresSecret,
					},
					Outcome: contracts.ToolCallSucceeded,
				}},
				Errors: []contracts.ExecutionError{{
					Code: "provider_error", Message: "provider returned " + postgresSecret,
				}},
			},
			Runtime: contracts.RuntimeReport{
				Complete: complete, StopReason: &stopReason,
			},
		},
	}
}

func plannerEnvelope(
	stageExecutionID string,
	sessionID string,
	receivedAt time.Time,
) telemetry.PlannerReportEnvelope {
	modelCalls := int64(2)
	return telemetry.PlannerReportEnvelope{
		StageExecutionID: stageExecutionID, SessionID: sessionID,
		InvocationID:        "invocation-" + stageExecutionID,
		StartedAt:           receivedAt.Add(-time.Second),
		FinishedAt:          receivedAt,
		ReportSchemaVersion: contracts.APIVersion,
		ReceivedAt:          receivedAt, Secrets: []string{postgresSecret},
		Report: contracts.ExecutionReport{
			ReportID: "planner-" + sessionID, Complete: true,
			Metrics: contracts.ExecutionMetrics{
				ModelCalls: &modelCalls, Tools: map[string]contracts.ToolMetrics{},
			},
			ToolCalls: []contracts.ToolCallRecord{}, Errors: []contracts.ExecutionError{},
		},
	}
}

func createTelemetryRun(t *testing.T, ctx context.Context, store *runstore.PostgresStore) {
	t.Helper()
	_, err := store.CreateRun(ctx, runstore.CreateRunParams{
		RunID: "run-telemetry", OwnerID: "user", WorkflowName: "workflow", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot:      json.RawMessage(`{"name":"workflow"}`),
		Parameters:            map[string]string{},
		RuntimeConfig:         runtimeconfig.BuiltInRunSnapshot(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(
		ctx, "run-telemetry", runstore.RunInitializing, runstore.RunRunning,
		runstore.Reason{Code: "initialized"},
	); err != nil {
		t.Fatal(err)
	}
}

func createTelemetryStage(
	t *testing.T,
	ctx context.Context,
	store *runstore.PostgresStore,
	stageExecutionID string,
	allocationID string,
	sessionID string,
) string {
	return createTelemetryStageWithPolicy(
		t, ctx, store, stageExecutionID, allocationID, sessionID,
		contracts.PerformanceCollectionDisabled,
	)
}

func createTelemetryStageWithPolicy(
	t *testing.T,
	ctx context.Context,
	store *runstore.PostgresStore,
	stageExecutionID string,
	allocationID string,
	sessionID string,
	collectionPolicy contracts.PerformanceCollectionPolicy,
) string {
	t.Helper()
	_, err := store.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: stageExecutionID, RunID: "run-telemetry", StageName: stageExecutionID,
		Attempt: 1, StageSpecSchemaVersion: contracts.APIVersion,
		StageSpecSnapshot:         json.RawMessage(`{"objective":"test"}`),
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext:              runstore.StageContextSnapshot{},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.RecordStageAllocation(ctx, runstore.StageAllocation{
		AllocationID: allocationID, StageExecutionID: stageExecutionID,
		LogicalAgentName: "builder", Namespace: "builder",
		AgentTemplateRef: contracts.AgentTemplateRef{
			TemplateID: "builder", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
		},
		WorkerRuntimeRef:                  contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"},
		RuntimeAgentID:                    strings.Repeat("1", 64),
		RuntimeAgentInstanceID:            "runtime-1",
		RuntimeAgentLabelRevision:         1,
		RuntimeConfigurationSchemaVersion: runstore.AllocationRuntimeConfigurationSchemaVersion,
		RuntimeConfiguration:              telemetryAllocationRuntimeConfiguration(),
		PerformanceCollectionPolicy:       collectionPolicy,
	}); err != nil {
		t.Fatal(err)
	}
	invocationID := "invocation-" + stageExecutionID
	startedData, err := json.Marshal(map[string]string{
		"stageExecutionId": stageExecutionID,
		"sessionId":        sessionID, "invocationId": invocationID,
	})
	if err != nil {
		t.Fatal(err)
	}
	eventID := "planner-started-" + stageExecutionID
	if err := store.StartPlanner(ctx, runstore.StartPlannerParams{
		StageExecutionID: stageExecutionID, SessionID: sessionID,
		InvocationID:       invocationID,
		StateSchemaVersion: contracts.APIVersion,
		InitialState:       json.RawMessage(`{"step":0}`),
		EventID:            eventID, EventSchemaVersion: contracts.APIVersion,
		Event: json.RawMessage(`{"kind":"planner_started"}`),
		RunEvent: runstore.RunEventAppend{
			EventID: eventID, EventSchemaVersion: contracts.APIVersion,
			Kind: runstore.RunEventPlannerStarted, Data: startedData,
		},
		Reason: runstore.Reason{Code: "planner_started"},
	}); err != nil {
		t.Fatal(err)
	}
	return stageExecutionID
}

func telemetryAllocationRuntimeConfiguration() *runstore.AllocationRuntimeConfiguration {
	gateway := contracts.LLMGatewayConfigRef{
		GatewayID: "local-litellm", Version: "1", Digest: "sha256:" + strings.Repeat("b", 64),
	}
	return &runstore.AllocationRuntimeConfiguration{
		ModelPolicy: contracts.ModelPolicyRef{
			PolicyID: "worker", Version: "1", Digest: "sha256:" + strings.Repeat("c", 64),
		},
		Origins: runtimeconfig.ResolvedRuntimeConfigOrigins{
			LLMGateway: &runtimeconfig.RuntimeFieldOrigin{Layer: runtimeconfig.LayerWorkflow},
		},
		Provenance: contracts.ResolvedRuntimeConfigProvenanceV2{
			Default: contracts.RuntimeLabelBindingProvenanceV2{
				Label: "default", BindingRevision: 1,
				Config: contracts.RuntimeConfigRefV2{
					Name: runtimeconfig.BuiltInName, Version: runtimeconfig.BuiltInVersion,
					Digest: runtimeconfig.BuiltInDigest,
				},
			},
			RunLabels:       []contracts.RuntimeLabelBindingProvenanceV2{},
			AgentLabels:     []contracts.RuntimeLabelBindingProvenanceV2{},
			RuntimeAdapters: []contracts.RuntimeAdapterRef{}, LLMGatewayConfig: &gateway,
			RuntimeCredentialRefs: []contracts.RuntimeCredentialRefV2{},
		},
	}
}

func finishTelemetryStage(
	t *testing.T,
	ctx context.Context,
	store *runstore.PostgresStore,
	stageExecutionID string,
) {
	t.Helper()
	result := contracts.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded,
		Summary: "done", Artifacts: map[string]contracts.ArtifactRef{},
	}
	if err := store.EnterFinalizing(ctx, runstore.EnterFinalizingParams{
		StageExecutionID: stageExecutionID, ResultSchemaVersion: contracts.APIVersion,
		Candidate: result, FinalizationID: "finalize-" + stageExecutionID,
		Deadline: time.Now().Add(time.Minute), Reason: runstore.Reason{Code: "planner_completed"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.CompleteStageResult(
		ctx, stageExecutionID, contracts.APIVersion, result,
	); err != nil {
		t.Fatal(err)
	}
}

func assertTelemetryCount(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	table string,
	stageExecutionID string,
	want int,
) {
	t.Helper()
	var got int
	query := `SELECT count(*) FROM ` + pgx.Identifier{table}.Sanitize() +
		` WHERE stage_execution_id = $1`
	if err := pool.QueryRow(ctx, query, stageExecutionID).Scan(&got); err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("%s rows for %s = %d, want %d", table, stageExecutionID, got, want)
	}
}

func isolatedTelemetryPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	admin, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := cryptorand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_telemetry_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop telemetry test schema: %v", err)
		}
		admin.Close()
	})
	return pool
}
