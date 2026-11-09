package runservice

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditimport"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runcost"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5"
)

type costParticipantFixture struct {
	taskRunFixture
	runs         *runstore.PostgresStore
	participants *runcost.Participants
	reports      *CostReports
	runID        string
	allocation   runstore.StageAllocation
	start        runstore.StartPlannerParams
	worker       telemetry.AllocationReportEnvelope
	planner      telemetry.PlannerReportEnvelope
}

func newCostParticipantFixture(t *testing.T) costParticipantFixture {
	t.Helper()
	f := newCommonTaskRunFixture(t, false)
	ctx := t.Context()
	created, err := f.service.CreateAudit(ctx, f.params)
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(f.pool, artifactpgtest.NewStorage())
	if _, err := runs.TransitionRun(ctx, created.Run.RunID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{LaneClass: resolved.LaneClassModel,
		StageExecutionID: "cost-stage", RunID: created.Run.RunID, StageName: "cost-stage", Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{"objective":"test"}`),
		StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{},
	}); err != nil {
		t.Fatal(err)
	}
	started, err := json.Marshal(map[string]string{"stageExecutionId": "cost-stage", "sessionId": "cost-session", "invocationId": "cost-invocation"})
	if err != nil {
		t.Fatal(err)
	}
	start := runstore.StartPlannerParams{StageExecutionID: "cost-stage", SessionID: "cost-session", InvocationID: "cost-invocation",
		StateSchemaVersion: contracts.APIVersion, InitialState: json.RawMessage(`{"step":0}`), Reason: runstore.Reason{Code: "planner_started"},
		RunEvent: runstore.RunEventAppend{EventID: "cost-started", EventSchemaVersion: contracts.APIVersion, Kind: runstore.RunEventPlannerStarted, Data: started}}
	allocation := runstore.StageAllocation{AllocationID: "cost-allocation", StageExecutionID: "cost-stage", LogicalAgentName: "builder", Namespace: "builder",
		AgentTemplateRef: contracts.AgentTemplateRef{TemplateID: "builder", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64)},
		WorkerRuntimeRef: contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"}, RuntimeAgentID: strings.Repeat("a", 64),
		RuntimeAgentInstanceID: "cost-runtime", RuntimeAgentLabelRevision: 1, RuntimeConfigurationSchemaVersion: runstore.AllocationRuntimeConfigurationSchemaVersion,
		PerformanceCollectionPolicy: reporting.PerformanceCollectionDisabled,
		RuntimeConfiguration: &runstore.AllocationRuntimeConfiguration{
			ModelPolicy: contracts.ModelPolicyRef{PolicyID: "worker", Version: "1", Digest: "sha256:" + strings.Repeat("b", 64)},
			Origins:     runtimeconfig.ResolvedRuntimeConfigOrigins{LLMGateway: &runtimeconfig.RuntimeFieldOrigin{Layer: runtimeconfig.LayerWorkflow}},
			Provenance: runtimesettings.ResolvedRuntimeConfigProvenance{
				LLMGatewayConfig: &llmgateway.LLMGatewayConfigRef{GatewayID: "test-gateway", Version: "1", Digest: "sha256:" + strings.Repeat("c", 64)},
				Default:          runtimesettings.RuntimeLabelBindingProvenance{Label: "default", BindingRevision: 1, Config: runtimesettings.RuntimeConfigRef{Name: runtimeconfig.BuiltInName, Version: runtimeconfig.BuiltInVersion, Digest: runtimeconfig.BuiltInDigest}},
				RunLabels:        []runtimesettings.RuntimeLabelBindingProvenance{}, AgentLabels: []runtimesettings.RuntimeLabelBindingProvenance{}, RuntimeAdapters: []runtimesettings.RuntimeAdapterRef{}, RuntimeCredentialRefs: []runtimesettings.RuntimeCredentialRef{},
			}}}
	// Supplied report timestamps predate registration: late delivery must still
	// fill the slot, including an incomplete first report.
	when := time.Date(2026, 11, 8, 0, 0, 0, 0, time.UTC)
	execution := func(id, amount string) reporting.ExecutionReport {
		calls := int64(1)
		return reporting.ExecutionReport{ReportID: id, Complete: true,
			Metrics:   reporting.ExecutionMetrics{ModelCalls: &calls, Tools: map[string]reporting.ToolMetrics{}, LLMCost: &reporting.LLMCost{Currency: "USD", Calls: 1, Groups: []reporting.LLMCostGroup{{Model: "model", Phase: "main", Source: "provider", Calls: 1, Amount: amount}}}},
			ToolCalls: []reporting.ToolCallRecord{}, Errors: []reporting.ExecutionError{}}
	}
	worker := telemetry.AllocationReportEnvelope{StageExecutionID: "cost-stage", AllocationID: allocation.AllocationID, LogicalAgentName: "builder",
		ReportSchemaVersion: contracts.APIVersion, ReceivedAt: when, PerformanceCollectionPolicy: reporting.PerformanceCollectionDisabled,
		Report: reporting.AllocationFinalReport{ReportID: "cost-worker-outer", AllocationID: allocation.AllocationID, StartedAt: when.Add(-time.Second), FinishedAt: when,
			Worker: execution("cost-worker", "0.1"), Runtime: reporting.RuntimeReport{Complete: true}}}
	planner := telemetry.PlannerReportEnvelope{StageExecutionID: "cost-stage", SessionID: start.SessionID, InvocationID: start.InvocationID,
		ReportSchemaVersion: contracts.APIVersion, ReceivedAt: when, StartedAt: when.Add(-time.Second), FinishedAt: when, Report: execution("cost-planner", "0.2")}
	return costParticipantFixture{taskRunFixture: f, runs: runs, participants: runcost.NewParticipants(f.pool, artifactpgtest.NewStorage()), reports: NewCostReports(f.pool), runID: created.Run.RunID,
		allocation: allocation, start: start, worker: worker, planner: planner}
}

func (f costParticipantFixture) registerWorker(t *testing.T) error {
	t.Helper()
	return persistencepostgres.Within(t.Context(), f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		// Match placement's Run-before-allocation lock order.
		if _, err := runstore.NewPostgresTxStore(tx, nil).LockRunState(t.Context(), f.runID, runstore.RunRunning); err != nil {
			return err
		}
		return f.participants.RecordStageAllocation(t.Context(), tx, f.allocation)
	})
}

func (f costParticipantFixture) requireCosts(t *testing.T, amount, unknown, reports, coverage string, deleted bool) {
	t.Helper()
	scopes := map[string]string{"audit": f.audit.AuditID, "project": f.audit.ProjectID}
	if !deleted {
		scopes["run"] = f.runID
	}
	for scope, id := range scopes {
		v, err := f.reports.GetLLMCost(t.Context(), "owner", scope, id)
		if err != nil || v.UnknownReports != unknown || v.Reports != reports || v.Coverage != coverage ||
			(amount == "" && v.Amount != nil) || (amount != "" && (v.Amount == nil || *v.Amount != amount)) {
			t.Fatalf("%s cost=%+v err=%v; want amount=%s unknown=%s reports=%s coverage=%s", scope, v, err, amount, unknown, reports, coverage)
		}
	}
	items, err := f.reports.GetAuditExecutionCosts(t.Context(), "owner", f.audit.AuditID, []string{f.params.ExecutionID})
	if err != nil || len(items) != 1 || items[0].Coverage != coverage || (amount == "" && items[0].Amount != nil) || (amount != "" && (items[0].Amount == nil || *items[0].Amount != amount)) {
		t.Fatalf("retained execution cost=%+v err=%v", items, err)
	}
}

func TestSharedPostgresMissingCostReportSurvivesTerminalRunAndDeletion(t *testing.T) {
	for _, savePlanner := range []bool{true, false} {
		t.Run(map[bool]string{true: "lost-worker", false: "all-reports-lost"}[savePlanner], func(t *testing.T) {
			f := newCostParticipantFixture(t)
			ctx := t.Context()
			if err := f.registerWorker(t); err != nil {
				t.Fatal(err)
			}
			if err := f.participants.StartPlanner(ctx, f.start); err != nil {
				t.Fatal(err)
			}
			f.requireCosts(t, "", "2", "2", "unavailable", false)
			// A definite diagnostic write failure must not prevent ordinary
			// execution completion or erase expected coverage.
			if _, err := f.pool.Exec(ctx, `ALTER TABLE allocation_execution_reports ADD CONSTRAINT reject_cost_report CHECK (false)`); err != nil {
				t.Fatal(err)
			}
			if err := f.reports.RecordAllocationReport(ctx, f.worker); err == nil {
				t.Fatal("Worker report unexpectedly saved")
			}
			if _, err := f.pool.Exec(ctx, `ALTER TABLE allocation_execution_reports DROP CONSTRAINT reject_cost_report`); err != nil {
				t.Fatal(err)
			}
			amount, unknown, coverage := "", "2", "unavailable"
			if savePlanner {
				if err := f.reports.RecordPlannerReport(ctx, f.planner); err != nil {
					t.Fatal(err)
				}
				amount, unknown, coverage = "0.2", "1", "partial"
			}
			f.requireCosts(t, amount, unknown, "2", coverage, false)
			f.finishAndDeleteCostRun(t, amount, unknown, coverage)
		})
	}
}

func (f costParticipantFixture) finishAndDeleteCostRun(t *testing.T, amount, unknown, coverage string) {
	t.Helper()
	ctx := t.Context()
	result := stageresult.StageContentResult{APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded, Summary: "done", Artifacts: map[string]contracts.ArtifactRef{}}
	if err := f.runs.EnterFinalizing(ctx, runstore.EnterFinalizingParams{StageExecutionID: "cost-stage", ResultSchemaVersion: contracts.APIVersion, Candidate: result,
		FinalizationID: "cost-finalize", Deadline: time.Now().Add(time.Minute), Reason: runstore.Reason{Code: "completed"}}); err != nil {
		t.Fatal(err)
	}
	if err := f.runs.CompleteStageResult(ctx, "cost-stage", contracts.APIVersion, result); err != nil {
		t.Fatal(err)
	}
	art := artifacts.NewService(artifactpgtest.NewStorage().Repository(f.pool))
	run, _ := art.Run(f.runID)
	written, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "builder", Name: "result"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("ordinary output")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifactstest.BindOutput(ctx, art, f.runID, "result", written.Ref); err != nil {
		t.Fatal(err)
	}
	if err := artifactstest.FreezeOutputs(ctx, art, f.runID); err != nil {
		t.Fatal(err)
	}
	if _, err := f.runs.TransitionRun(ctx, f.runID, runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "completed"}); err != nil {
		t.Fatal(err)
	}
	if err := f.runs.MarkStageAllocationReleased(ctx, f.allocation.AllocationID); err != nil {
		t.Fatal(err)
	}
	f.requireCosts(t, amount, unknown, "2", coverage, false)
	if _, err := f.reports.CleanupExpired(ctx, f.planner.ReceivedAt.Add(60*24*time.Hour), 100); err != nil {
		t.Fatal(err)
	}
	before, err := f.audits.Get(ctx, "owner", f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := f.audits.Transition(ctx, auditstore.TransitionParams{OwnerID: "owner", AuditID: before.AuditID, ExpectedRevision: before.Revision, ExpectedState: before.State,
		TargetState: auditstore.AuditPaused, IdempotencyKey: "cost-pause", RequestDigest: contentdigest.Bytes([]byte("cost-pause"))}); err != nil {
		t.Fatal(err)
	}
	cursor, err := f.runs.GetRunEventCursor(ctx, f.runID)
	if err != nil {
		t.Fatal(err)
	}
	execution, err := f.audits.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{Claim: f.params.Claim, ExecutionID: f.params.ExecutionID, RunID: f.runID, Generation: cursor.Generation, Sequence: uint64(cursor.Sequence)})
	if err != nil {
		t.Fatal(err)
	}
	access, err := auditimport.NewArtifactAccess(art, hookstore.NewClaimedArtifacts(f.pool, artifactpgtest.NewStorage()))
	if err != nil {
		t.Fatal(err)
	}
	importer, err := auditimport.New(f.audits, f.audits, f.runs, access)
	if err != nil {
		t.Fatal(err)
	}
	audit, err := f.audits.Get(ctx, "owner", f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if changed, err := importer.Collect(ctx, f.params.Claim, auditstore.ReconcileSnapshot{Audit: audit}, execution); err != nil || !changed {
		t.Fatalf("collect: %v %v", changed, err)
	}
	if err := rundeletion.New(f.pool, artifactpgtest.NewStorage()).DeleteReleasedTerminalRun(ctx, "owner", f.runID); err != nil {
		t.Fatal(err)
	}
	f.requireCosts(t, amount, unknown, "2", coverage, true)
}

func TestSharedPostgresExpectedCostSlotsReplayRollbackAndLateDelivery(t *testing.T) {
	f := newCostParticipantFixture(t)
	ctx := t.Context()
	// Invalid parent data forces registration failure after the actor write.
	if _, err := f.pool.Exec(ctx, `UPDATE projects SET llm_cost='{"currency":"EUR"}'::jsonb WHERE project_id=$1`, f.audit.ProjectID); err != nil {
		t.Fatal(err)
	}
	if err := f.registerWorker(t); err == nil {
		t.Fatal("allocation admitted without its parent coverage")
	}
	if err := f.participants.StartPlanner(ctx, f.start); err == nil {
		t.Fatal("Planner started without its parent coverage")
	}
	allocations, err := f.runs.ListStageAllocations(ctx, "cost-stage")
	if err != nil || len(allocations) != 0 {
		t.Fatalf("failed registration retained allocation: %+v %v", allocations, err)
	}
	stage, err := f.runs.GetStageExecution(ctx, "cost-stage")
	if err != nil || stage.State != runstore.StagePreparing {
		t.Fatalf("failed registration started Planner: %+v %v", stage, err)
	}
	if _, err := f.pool.Exec(ctx, `UPDATE projects SET llm_cost='{}'::jsonb WHERE project_id=$1`, f.audit.ProjectID); err != nil {
		t.Fatal(err)
	}
	f.requireCosts(t, "", "0", "0", "unavailable", false)
	var wg sync.WaitGroup
	errors := make(chan error, 8)
	for range 8 {
		wg.Go(func() { errors <- f.registerWorker(t) })
	}
	wg.Wait()
	close(errors)
	for err := range errors {
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := f.participants.StartPlanner(ctx, f.start); err != nil {
		t.Fatal(err)
	}
	f.requireCosts(t, "", "2", "2", "unavailable", false)
	// An incomplete report with an earlier timestamp still fills its slot.
	partial := f.worker
	partial.Report.ReportID = "cost-partial-outer"
	partial.Report.Worker.ReportID = "cost-partial"
	partial.Report.Worker.Complete = false
	if err := f.reports.RecordAllocationReport(ctx, partial); err != nil {
		t.Fatal(err)
	}
	if err := f.registerWorker(t); err != nil {
		t.Fatal(err)
	}
	f.requireCosts(t, "0.1", "2", "2", "partial", false)
	concurrentErrors := make(chan error, 16)
	for range 8 {
		wg.Go(func() { concurrentErrors <- f.registerWorker(t) })
		wg.Go(func() { concurrentErrors <- f.reports.RecordAllocationReport(ctx, f.worker) })
	}
	wg.Wait()
	close(concurrentErrors)
	for err := range concurrentErrors {
		if err != nil {
			t.Fatal(err)
		}
	}
	for range 2 {
		if err := f.reports.RecordAllocationReport(ctx, f.worker); err != nil {
			t.Fatal(err)
		}
		if err := f.reports.RecordPlannerReport(ctx, f.planner); err != nil {
			t.Fatal(err)
		}
		if err := f.registerWorker(t); err != nil {
			t.Fatal(err)
		}
	}
	// Active work with fully reported costs is pending, never final complete.
	f.requireCosts(t, "0.3", "0", "2", "pending", false)
}
