package runstore

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresIntegrationStageLifecycleAndSessions(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)

	run := createTestRun(t, ctx, store, "run-lifecycle")
	if run.State != RunInitializing || run.WorkflowSnapshot == nil {
		t.Fatalf("created Run = %+v", run)
	}
	run, err := store.TransitionRun(ctx, run.RunID, RunInitializing, RunRunning, Reason{Code: "initialized"})
	if err != nil {
		t.Fatalf("start Run: %v", err)
	}
	if run.State != RunRunning || run.StartedAt == nil {
		t.Fatalf("started Run = %+v", run)
	}

	revision := "revision-1"
	execution, err := store.CreateStageExecution(ctx, CreateStageExecutionParams{
		StageExecutionID: "stage-lifecycle", RunID: run.RunID, StageName: "copy", Attempt: 1,
		StageSpecSchemaVersion:    "contractor/v1alpha1",
		StageSpecSnapshot:         json.RawMessage(`{"objective":"copy"}`),
		StageContextSchemaVersion: "contractor/v1alpha1",
		StageContext: StageContextSnapshot{
			Parameters: map[string]string{"mode": "strict"},
			Artifacts: map[string]PinnedContextArtifact{
				"source": {
					Required: true,
					Artifact: &contracts.ArtifactRef{Namespace: "inputs", Name: "source", Revision: &revision},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("create StageExecution: %v", err)
	}
	if execution.State != StagePreparing || execution.PlannerSessionID != nil {
		t.Fatalf("preparing StageExecution = %+v", execution)
	}

	allocation := StageAllocation{
		AllocationID: "allocation-1", StageExecutionID: execution.StageExecutionID,
		LogicalAgentName: "builder", Namespace: "builder",
		AgentTemplateRef: contracts.AgentTemplateRef{
			TemplateID: "artifact_builder", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
		},
		WorkerRuntimeRef:                  contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"},
		RuntimeAgentID:                    strings.Repeat("1", 64),
		RuntimeAgentInstanceID:            "runtime-agent-1",
		RuntimeAgentLabelRevision:         1,
		RuntimeConfigurationSchemaVersion: AllocationRuntimeConfigurationSchemaVersion,
		RuntimeConfiguration:              testAllocationRuntimeConfiguration(),
	}
	err = store.RecordStageAllocation(ctx, allocation)
	if err != nil {
		t.Fatalf("record allocation: %v", err)
	}
	if err := store.RecordStageAllocation(ctx, allocation); err != nil {
		t.Fatalf("idempotent allocation record: %v", err)
	}
	different := allocation
	different.RuntimeAgentInstanceID = "another-runtime-agent"
	if err := store.RecordStageAllocation(ctx, different); !errors.Is(err, ErrConflict) {
		t.Fatalf("mismatched allocation record error = %v, want conflict", err)
	}
	allocations, err := store.ListStageAllocations(ctx, execution.StageExecutionID)
	if err != nil || len(allocations) != 1 || allocations[0].AllocationID != "allocation-1" {
		t.Fatalf("allocations = (%+v, %v)", allocations, err)
	}
	_, err = pool.Exec(ctx, `
UPDATE stage_allocations
SET runtime_agent_label_revision = runtime_agent_label_revision + 1
WHERE allocation_id = 'allocation-1'`)
	if persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("allocation Runtime provenance rewrite SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}
	reportFinished := time.Now().UTC()
	modelCalls := int64(3)
	reportParams := RecordStageExecutionReportParams{
		StageExecutionID: execution.StageExecutionID, AllocationID: allocation.AllocationID,
		LogicalAgentName: allocation.LogicalAgentName, ReportSchemaVersion: contracts.APIVersion,
		Report: contracts.AllocationFinalReport{
			ReportID: "allocation-report-1", AllocationID: allocation.AllocationID,
			StartedAt: reportFinished.Add(-time.Second), FinishedAt: reportFinished,
			Worker: contracts.ExecutionReport{
				ReportID: "worker-report-1", Complete: true,
				Metrics: contracts.ExecutionMetrics{
					ModelCalls: &modelCalls, Tools: map[string]contracts.ToolMetrics{},
				},
				ToolCalls: []contracts.ToolCallRecord{}, Errors: []contracts.ExecutionError{},
			},
			Runtime: contracts.RuntimeReport{Complete: true},
		},
	}
	if err := store.RecordStageExecutionReport(ctx, reportParams); err != nil {
		t.Fatalf("record execution report: %v", err)
	}
	if err := store.RecordStageExecutionReport(ctx, reportParams); err != nil {
		t.Fatalf("idempotent execution report: %v", err)
	}
	reports, err := store.ListStageExecutionReports(ctx, execution.StageExecutionID)
	if err != nil || len(reports) != 1 || reports[0].Report.Worker.Metrics.ModelCalls == nil ||
		*reports[0].Report.Worker.Metrics.ModelCalls != 3 {
		t.Fatalf("execution reports = (%+v, %v)", reports, err)
	}
	differentReport := reportParams
	differentCalls := int64(4)
	differentReport.Report.Worker.Metrics.ModelCalls = &differentCalls
	if err := store.RecordStageExecutionReport(ctx, differentReport); !errors.Is(err, ErrConflict) {
		t.Fatalf("mismatched execution report error = %v, want conflict", err)
	}
	if _, err := pool.Exec(ctx, `
UPDATE allocation_execution_reports
SET report = jsonb_set(report, '{worker,complete}', 'false')
WHERE allocation_id = 'allocation-1'`); persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("execution report rewrite SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}

	startedRunEvent := RunEventAppend{
		EventID: "event-started", EventSchemaVersion: contracts.APIVersion,
		Kind: RunEventPlannerStarted,
		Data: mustRunEventJSON(t, plannerRunEventData{
			StageExecutionID: execution.StageExecutionID,
			SessionID:        "session-1", InvocationID: "invocation-1",
		}),
	}
	err = store.StartPlanner(ctx, StartPlannerParams{
		StageExecutionID: execution.StageExecutionID,
		SessionID:        "session-1", InvocationID: "invocation-1",
		StateSchemaVersion: contracts.APIVersion, InitialState: json.RawMessage(`{"step":0}`),
		EventID: "event-started", EventSchemaVersion: contracts.APIVersion,
		Event: json.RawMessage(`{"kind":"planner_started"}`), RunEvent: startedRunEvent,
		Reason: Reason{Code: "planner_started"},
	})
	if err != nil {
		t.Fatalf("start Planner: %v", err)
	}
	execution, err = store.GetStageExecution(ctx, execution.StageExecutionID)
	if err != nil || execution.State != StageRunning || execution.PlannerSessionID == nil || *execution.PlannerSessionID != "session-1" {
		t.Fatalf("running StageExecution = (%+v, %v)", execution, err)
	}

	requestRunEvent := RunEventAppend{
		EventID: "event-1", EventSchemaVersion: contracts.APIVersion,
		Kind: RunEventPlannerRequestRecorded,
		Data: mustRunEventJSON(t, plannerRunEventData{
			StageExecutionID: execution.StageExecutionID,
			SessionID:        "session-1", InvocationID: "invocation-1",
		}),
	}
	err = store.AppendPlannerEvent(ctx, AppendPlannerEventParams{
		EventID: "event-1", SessionID: "session-1",
		StageExecutionID: execution.StageExecutionID, InvocationID: "invocation-1",
		SequenceNumber:     2,
		EventSchemaVersion: contracts.APIVersion, Event: json.RawMessage(`{"kind":"worker_request"}`),
		NewStateSchemaVersion: contracts.APIVersion, NewState: json.RawMessage(`{"step":1}`),
		RunEvent: requestRunEvent,
	})
	if err != nil {
		t.Fatalf("append Planner event: %v", err)
	}
	session, err := store.GetPlannerSession(ctx, "session-1")
	if err != nil || session.NextEventSequence != 3 ||
		string(session.State) != `{"step": 1}` && string(session.State) != `{"step":1}` {
		t.Fatalf("Planner session = (%+v, %v)", session, err)
	}
	events, err := store.ListPlannerEvents(ctx, "session-1", 0)
	if err != nil || len(events) != 2 || events[0].EventID != "event-started" ||
		events[1].EventID != "event-1" || events[0].RunEventSequence == nil ||
		*events[0].RunEventSequence != 5 || events[1].RunEventSequence == nil ||
		*events[1].RunEventSequence != 6 {
		t.Fatalf("Planner events = (%+v, %v)", events, err)
	}
	gap := requestRunEvent
	gap.EventID = "event-gap"
	err = store.AppendPlannerEvent(ctx, AppendPlannerEventParams{
		EventID: "event-gap", SessionID: "session-1",
		StageExecutionID: execution.StageExecutionID, InvocationID: "invocation-1",
		SequenceNumber: 4, EventSchemaVersion: contracts.APIVersion,
		Event:                 json.RawMessage(`{"kind":"worker_request"}`),
		NewStateSchemaVersion: contracts.APIVersion, NewState: json.RawMessage(`{"step":2}`),
		RunEvent: gap,
	})
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("Planner sequence gap error = %v, want conflict", err)
	}
	cursor, err := store.GetRunEventCursor(ctx, run.RunID)
	if err != nil || cursor.Sequence != 6 || cursor.Generation == "" {
		t.Fatalf("Run event cursor = (%+v, %v)", cursor, err)
	}
	runEvents, err := store.ListRunEvents(ctx, run.RunID, 0, 10)
	if err != nil || len(runEvents) != 6 || runEvents[0].Kind != RunEventLifecycleChanged ||
		runEvents[1].Kind != RunEventLifecycleChanged || runEvents[2].Kind != RunEventLifecycleChanged ||
		runEvents[3].Kind != RunEventLifecycleChanged || runEvents[4].Kind != RunEventPlannerStarted ||
		runEvents[5].Kind != RunEventPlannerRequestRecorded {
		t.Fatalf("Run events = (%+v, %v)", runEvents, err)
	}
	wantLifecycle := []lifecycleRunEventData{
		{RunID: run.RunID, Resource: "run", State: string(RunInitializing)},
		{RunID: run.RunID, Resource: "run", State: string(RunRunning)},
		{RunID: run.RunID, Resource: "stageExecution", StageExecutionID: execution.StageExecutionID, State: string(StagePreparing)},
		{RunID: run.RunID, Resource: "stageExecution", StageExecutionID: execution.StageExecutionID, State: string(StageRunning)},
	}
	for index, want := range wantLifecycle {
		got, decodeErr := decodeLifecycleRunEventData(runEvents[index].Data)
		if decodeErr != nil || got != want || runEvents[index].SequenceNumber != int64(index+1) {
			t.Fatalf("lifecycle event %d = (%+v, %v), want %+v", index, got, decodeErr, want)
		}
	}

	resultRevision := "revision-result-1"
	candidate := contracts.StageContentResult{
		APIVersion: contracts.APIVersion,
		Outcome:    contracts.StageSucceeded,
		Summary:    "copied",
		Artifacts: map[string]contracts.ArtifactRef{
			"copied": {Namespace: "builder", Name: "result", Revision: &resultRevision},
		},
	}
	err = store.EnterFinalizing(ctx, EnterFinalizingParams{
		StageExecutionID:    execution.StageExecutionID,
		ResultSchemaVersion: "contractor/v1alpha1", Candidate: candidate,
		FinalizationID: "finalization-1", Deadline: time.Now().Add(time.Minute),
		Reason: Reason{Code: "planner_completed"},
	})
	if err != nil {
		t.Fatalf("enter finalizing: %v", err)
	}
	err = store.EnterAborting(ctx, EnterAbortingParams{
		StageExecutionID: execution.StageExecutionID, ExpectedState: StageRunning,
		TerminationSchemaVersion: "contractor/v1alpha1",
		Termination: StageTermination{
			Outcome: TerminationInterrupted, Code: "late_abort", Message: "late abort",
			Retryable: true, Phase: TerminationRunning, OccurredAt: time.Now(),
		},
		AbortID: "late-abort", Deadline: time.Now().Add(time.Minute), Reason: Reason{Code: "late_abort"},
	})
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("late abort error = %v, want conflict", err)
	}
	if err := store.CompleteStageResult(ctx, execution.StageExecutionID, "contractor/v1alpha1", candidate); err != nil {
		t.Fatalf("complete StageResult: %v", err)
	}
	execution, err = store.GetStageExecution(ctx, execution.StageExecutionID)
	if err != nil || execution.State != StageSucceeded || execution.CandidateResult == nil || execution.AcceptedResult == nil {
		t.Fatalf("terminal StageExecution = (%+v, %v)", execution, err)
	}
	if got := *execution.AcceptedResult.Artifacts["copied"].Revision; got != resultRevision {
		t.Fatalf("accepted artifact revision = %q, want %q", got, resultRevision)
	}
	if err := store.CompleteStageResult(ctx, execution.StageExecutionID, "contractor/v1alpha1", candidate); !errors.Is(err, ErrConflict) {
		t.Fatalf("second completion error = %v, want conflict", err)
	}
	pendingRelease, err := store.ListTerminalStageExecutionsWithAllocations(ctx)
	if err != nil || len(pendingRelease) != 1 || pendingRelease[0].StageExecutionID != execution.StageExecutionID {
		t.Fatalf("pending terminal allocation release = (%+v, %v)", pendingRelease, err)
	}
	if err := store.MarkStageAllocationReleaseAttempt(ctx, allocation.AllocationID); err != nil {
		t.Fatalf("mark allocation release attempt: %v", err)
	}
	if err := store.MarkStageAllocationReleased(ctx, allocation.AllocationID); err != nil {
		t.Fatalf("mark allocation released: %v", err)
	}
	if err := store.MarkStageAllocationReleased(ctx, allocation.AllocationID); err != nil {
		t.Fatalf("repeat allocation released marker: %v", err)
	}
	allocations, err = store.ListStageAllocations(ctx, execution.StageExecutionID)
	if err != nil || len(allocations) != 1 || allocations[0].ReleaseAttemptedAt == nil ||
		allocations[0].ReleaseCompletedAt == nil {
		t.Fatalf("allocation release state = (%+v, %v)", allocations, err)
	}
	pendingRelease, err = store.ListTerminalStageExecutionsWithAllocations(ctx)
	if err != nil || len(pendingRelease) != 0 {
		t.Fatalf("completed release remained pending = (%+v, %v)", pendingRelease, err)
	}

	_, err = store.CreateStageExecution(ctx, CreateStageExecutionParams{
		StageExecutionID: "stage-aborted", RunID: run.RunID, StageName: "precheck", Attempt: 1,
		StageSpecSchemaVersion: "contractor/v1alpha1", StageSpecSnapshot: json.RawMessage(`{"objective":"check"}`),
		StageContextSchemaVersion: "contractor/v1alpha1", StageContext: StageContextSnapshot{},
	})
	if err != nil {
		t.Fatalf("create aborting StageExecution: %v", err)
	}
	termination := StageTermination{
		Outcome: TerminationInterrupted, Code: "context_artifact_missing", Message: "source is missing",
		Retryable: false, Phase: TerminationPreparing, OccurredAt: time.Now(),
	}
	err = store.EnterAborting(ctx, EnterAbortingParams{
		StageExecutionID: "stage-aborted", ExpectedState: StagePreparing,
		TerminationSchemaVersion: "contractor/v1alpha1", Termination: termination,
		AbortID: "abort-1", Deadline: time.Now().Add(time.Minute), Reason: Reason{Code: termination.Code},
	})
	if err != nil {
		t.Fatalf("enter aborting during preparation: %v", err)
	}
	if err := store.CompleteStageTermination(ctx, "stage-aborted"); err != nil {
		t.Fatalf("complete StageTermination: %v", err)
	}
	aborted, err := store.GetStageExecution(ctx, "stage-aborted")
	if err != nil || aborted.State != StageInterrupted || aborted.Termination == nil || aborted.PlannerSessionID != nil {
		t.Fatalf("interrupted StageExecution = (%+v, %v)", aborted, err)
	}

	executions, err := store.ListStageExecutions(ctx, run.RunID)
	if err != nil || len(executions) != 2 {
		t.Fatalf("StageExecution list = (%+v, %v)", executions, err)
	}

	_, err = pool.Exec(ctx, `
UPDATE stage_executions
SET candidate_stage_result = jsonb_set(candidate_stage_result, '{summary}', '"rewritten"')
WHERE stage_execution_id = 'stage-lifecycle'`)
	if persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("candidate rewrite SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}
}

func TestPostgresWorkflowRunMetadataLabelsAreAtomicImmutableAndProjected(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)

	params := testRunParams("run-metadata-labels")
	params.MetadataLabels = RunMetadataLabels{
		"purpose": "eval", "eval.id": "eval_01", "eval.leg": "a",
	}
	created, err := store.CreateRun(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(created.MetadataLabels, params.MetadataLabels) {
		t.Fatalf("created metadata labels = %v, want %v", created.MetadataLabels, params.MetadataLabels)
	}
	params.MetadataLabels["purpose"] = "mutated-by-caller"
	loaded, err := store.GetRun(ctx, created.RunID)
	if err != nil || loaded.MetadataLabels["purpose"] != "eval" {
		t.Fatalf("stored metadata labels = (%v, %v)", loaded.MetadataLabels, err)
	}
	page, err := store.ListRuns(ctx, ListRunsParams{OwnerID: params.OwnerID, Limit: 10})
	if err != nil || len(page) != 1 || page[0].MetadataLabels["eval.id"] != "eval_01" {
		t.Fatalf("listed metadata labels = (%+v, %v)", page, err)
	}

	_, err = pool.Exec(ctx, `
UPDATE workflow_run_metadata_labels SET label_value = 'b'
WHERE run_id = $1 AND label_key = 'eval.leg'`, created.RunID)
	if persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("direct metadata-label update SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}
	_, err = pool.Exec(ctx, `
DELETE FROM workflow_run_metadata_labels
WHERE run_id = $1 AND label_key = 'eval.leg'`, created.RunID)
	if persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("direct metadata-label delete SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}
	_, err = pool.Exec(ctx, `
INSERT INTO workflow_run_metadata_labels (run_id, ordinal, label_key, label_value)
VALUES ($1, 4, 'eval.id', 'different')`, created.RunID)
	if persistencepostgres.SQLState(err) != "23505" {
		t.Fatalf("duplicate metadata-label SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}
	_, err = pool.Exec(ctx, `
INSERT INTO workflow_run_metadata_labels (run_id, ordinal, label_key, label_value)
VALUES ($1, 33, 'extra', 'value')`, created.RunID)
	if persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("metadata-label count bound SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}

	if _, err := pool.Exec(ctx, `DELETE FROM workflow_runs WHERE run_id = $1`, created.RunID); err != nil {
		t.Fatalf("parent retention delete: %v", err)
	}
	var remaining int
	if err := pool.QueryRow(ctx, `
SELECT count(*) FROM workflow_run_metadata_labels WHERE run_id = $1`, created.RunID).Scan(&remaining); err != nil || remaining != 0 {
		t.Fatalf("retained metadata-label rows = %d, error = %v", remaining, err)
	}
}

func TestPostgresWorkflowRunProjectMembershipIsOwnedImmutableAndFilterable(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	projects := projectstore.NewPostgresStore(pool)
	for index, projectID := range []string{"project-one", "project-two"} {
		_, _, err := projects.Create(ctx, projectstore.CreateParams{
			ProjectID: projectID, OwnerID: "user-1", Kind: projectstore.KindProject,
			Name: projectID, IdempotencyKey: projectID,
			RequestDigest: fmt.Sprintf("sha256:%064x", index+1),
		})
		if err != nil {
			t.Fatalf("create Project %q: %v", projectID, err)
		}
	}
	store := NewPostgresStore(pool)
	projectID := "project-one"
	projectParams := testRunParams("run-project-member")
	projectParams.ProjectID = &projectID
	projectRun, err := store.CreateRun(ctx, projectParams)
	if err != nil || projectRun.ProjectID == nil || *projectRun.ProjectID != projectID {
		t.Fatalf("create Project Run = (%+v, %v)", projectRun, err)
	}
	if _, err := store.CreateRun(ctx, testRunParams("run-standalone")); err != nil {
		t.Fatal(err)
	}
	page, err := store.ListRuns(ctx, ListRunsParams{
		OwnerID: "user-1", ProjectID: &projectID, Limit: 10,
	})
	if err != nil || len(page) != 1 || page[0].RunID != projectRun.RunID ||
		page[0].ProjectID == nil || *page[0].ProjectID != projectID {
		t.Fatalf("Project Run page = (%+v, %v)", page, err)
	}

	_, err = pool.Exec(ctx, `UPDATE workflow_runs SET project_id = 'project-two' WHERE run_id = $1`, projectRun.RunID)
	if persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("Project membership mutation SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}
	foreignParams := testRunParams("run-foreign-project-member")
	foreignParams.OwnerID = "user-2"
	foreignParams.ProjectID = &projectID
	_, err = store.CreateRun(ctx, foreignParams)
	if persistencepostgres.SQLState(err) != "23503" {
		t.Fatalf("foreign Project membership SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}
}

func TestPostgresWorkflowRunQueueIsOwnedOldestFirstAndTerminalAware(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	projects := projectstore.NewPostgresStore(pool)
	for index, item := range []struct {
		id   string
		kind projectstore.Kind
		name string
	}{
		{id: "project-queue", kind: projectstore.KindProject, name: "Payment service"},
		{id: "evaluation-queue", kind: projectstore.KindEvaluation, name: "OpenAPI eval"},
	} {
		if _, _, err := projects.Create(ctx, projectstore.CreateParams{
			ProjectID: item.id, OwnerID: "user-1", Kind: item.kind, Name: item.name,
			IdempotencyKey: "create-" + item.id,
			RequestDigest:  fmt.Sprintf("sha256:%064x", index+1),
		}); err != nil {
			t.Fatal(err)
		}
	}
	store := NewPostgresStore(pool)
	create := func(runID string, projectID *string, labels RunMetadataLabels) WorkflowRun {
		t.Helper()
		params := testRunParams(runID)
		params.ProjectID = projectID
		params.MetadataLabels = labels
		run, err := store.CreateRun(ctx, params)
		if err != nil {
			t.Fatal(err)
		}
		return run
	}
	projectID, evaluationID := "project-queue", "evaluation-queue"
	standalone := create("run-queue-standalone", nil, nil)
	projectRun := create(
		"run-queue-project", &projectID, RunMetadataLabels{"purpose": "manual"},
	)
	evaluationRun := create(
		"run-queue-evaluation", &evaluationID,
		RunMetadataLabels{"purpose": "eval", "eval.id": "eval-queue-1"},
	)
	terminal := create("run-queue-terminal", nil, nil)
	foreignParams := testRunParams("run-queue-foreign")
	foreignParams.OwnerID = "user-2"
	if _, err := store.CreateRun(ctx, foreignParams); err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(
		ctx, standalone.RunID, RunInitializing, RunRunning, Reason{Code: "initialized"},
	); err != nil {
		t.Fatal(err)
	}
	if _, err := store.RequestRunCancellation(ctx, evaluationRun.RunID, WorkflowRunCancellation{
		Code: CancellationUserRequested, RequestedAt: time.Now(),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(
		ctx, terminal.RunID, RunInitializing, RunFailed, Reason{Code: "invalid_input"},
	); err != nil {
		t.Fatal(err)
	}

	first, err := store.ListRunQueue(ctx, ListRunQueueParams{OwnerID: "user-1", Limit: 2})
	if err != nil || len(first) != 2 || first[0].RunID != standalone.RunID ||
		first[1].RunID != projectRun.RunID || first[0].ProjectID != nil ||
		first[1].ProjectID == nil || first[1].ProjectName != "Payment service" ||
		first[1].ProjectKind != string(projectstore.KindProject) ||
		first[1].MetadataLabels["purpose"] != "manual" ||
		first[1].EventCursor.Generation == "" || first[1].EventCursor.Sequence < 0 {
		t.Fatalf("first Queue page = (%+v, %v)", first, err)
	}
	second, err := store.ListRunQueue(ctx, ListRunQueueParams{
		OwnerID: "user-1", AfterCreatedAt: &first[1].CreatedAt,
		AfterRunID: first[1].RunID, Limit: 2,
	})
	if err != nil || len(second) != 1 || second[0].RunID != evaluationRun.RunID ||
		second[0].ProjectKind != string(projectstore.KindEvaluation) ||
		second[0].MetadataLabels["eval.id"] != "eval-queue-1" {
		t.Fatalf("second Queue page = (%+v, %v)", second, err)
	}
	projectMembership := RunQueueProject
	projectOnly, err := store.ListRunQueue(ctx, ListRunQueueParams{
		OwnerID: "user-1", Membership: &projectMembership, Limit: 10,
	})
	if err != nil || len(projectOnly) != 1 || projectOnly[0].RunID != projectRun.RunID {
		t.Fatalf("Project Queue filter = (%+v, %v)", projectOnly, err)
	}
	cancelling := RunCancelling
	cancellingOnly, err := store.ListRunQueue(ctx, ListRunQueueParams{
		OwnerID: "user-1", State: &cancelling, Limit: 10,
	})
	if err != nil || len(cancellingOnly) != 1 || cancellingOnly[0].RunID != evaluationRun.RunID {
		t.Fatalf("cancelling Queue filter = (%+v, %v)", cancellingOnly, err)
	}
	foreign, err := store.ListRunQueue(ctx, ListRunQueueParams{OwnerID: "user-2", Limit: 10})
	if err != nil || len(foreign) != 1 || foreign[0].RunID != "run-queue-foreign" {
		t.Fatalf("foreign owner Queue = (%+v, %v)", foreign, err)
	}
	if _, err := store.TransitionRun(
		ctx, projectRun.RunID, RunInitializing, RunFailed, Reason{Code: "failed"},
	); err != nil {
		t.Fatal(err)
	}
	projectOnly, err = store.ListRunQueue(ctx, ListRunQueueParams{
		OwnerID: "user-1", Membership: &projectMembership, Limit: 10,
	})
	if err != nil || len(projectOnly) != 0 {
		t.Fatalf("terminal Project Queue = (%+v, %v)", projectOnly, err)
	}
}

func TestPostgresWorkflowRunMetadataLabelFilteringIsConjunctiveOwnedAndStable(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	create := func(runID, ownerID, leg string) WorkflowRun {
		t.Helper()
		params := testRunParams(runID)
		params.OwnerID = ownerID
		params.MetadataLabels = RunMetadataLabels{
			"purpose": "eval", "eval.id": "eval_filter", "eval.leg": leg,
		}
		run, err := store.CreateRun(ctx, params)
		if err != nil {
			t.Fatal(err)
		}
		return run
	}
	old := create("run-filter-old", "user-1", "a")
	unrelatedParams := testRunParams("run-filter-unrelated")
	unrelatedParams.MetadataLabels = RunMetadataLabels{"purpose": "manual"}
	if _, err := store.CreateRun(ctx, unrelatedParams); err != nil {
		t.Fatal(err)
	}
	newer := create("run-filter-newer", "user-1", "b")
	_ = create("run-filter-foreign", "user-2", "a")
	if _, err := store.TransitionRun(
		ctx, newer.RunID, RunInitializing, RunRunning, Reason{Code: "initialized"},
	); err != nil {
		t.Fatal(err)
	}

	selectors := []RunMetadataLabelSelector{
		{Key: "purpose", Value: "eval"}, {Key: "eval.id", Value: "eval_filter"},
	}
	first, err := store.ListRuns(ctx, ListRunsParams{
		OwnerID: "user-1", MetadataLabelSelectors: selectors, Limit: 1,
	})
	if err != nil || len(first) != 1 || first[0].RunID != newer.RunID {
		t.Fatalf("first filtered page = (%+v, %v)", first, err)
	}
	concurrent := create("run-filter-concurrent", "user-1", "a")
	second, err := store.ListRuns(ctx, ListRunsParams{
		OwnerID: "user-1", MetadataLabelSelectors: append(selectors, selectors[0]), Limit: 10,
		BeforeCreatedAt: &first[0].CreatedAt, BeforeRunID: first[0].RunID,
	})
	if err != nil || len(second) != 1 || second[0].RunID != old.RunID {
		t.Fatalf("second stable filtered page = (%+v, %v)", second, err)
	}
	all, err := store.ListRuns(ctx, ListRunsParams{
		OwnerID: "user-1", MetadataLabelSelectors: selectors, Limit: 10,
	})
	if err != nil || len(all) != 3 || all[0].RunID != concurrent.RunID ||
		all[1].RunID != newer.RunID || all[2].RunID != old.RunID {
		t.Fatalf("complete filtered group = (%+v, %v)", all, err)
	}
	running := RunRunning
	byState, err := store.ListRuns(ctx, ListRunsParams{
		OwnerID: "user-1", State: &running, MetadataLabelSelectors: selectors, Limit: 10,
	})
	if err != nil || len(byState) != 1 || byState[0].RunID != newer.RunID {
		t.Fatalf("state plus metadata-label filter = (%+v, %v)", byState, err)
	}
	legA, err := store.ListRuns(ctx, ListRunsParams{
		OwnerID: "user-1", MetadataLabelSelectors: append(selectors,
			RunMetadataLabelSelector{Key: "eval.leg", Value: "a"}), Limit: 10,
	})
	if err != nil || len(legA) != 2 || legA[0].RunID != concurrent.RunID || legA[1].RunID != old.RunID {
		t.Fatalf("filtered leg = (%+v, %v)", legA, err)
	}
	contradiction, err := store.ListRuns(ctx, ListRunsParams{
		OwnerID: "user-1", MetadataLabelSelectors: []RunMetadataLabelSelector{
			{Key: "eval.leg", Value: "a"}, {Key: "eval.leg", Value: "b"},
		}, Limit: 10,
	})
	if err != nil || len(contradiction) != 0 {
		t.Fatalf("contradictory filtered group = (%+v, %v)", contradiction, err)
	}
	foreign, err := store.ListRuns(ctx, ListRunsParams{
		OwnerID: "user-2", MetadataLabelSelectors: selectors, Limit: 10,
	})
	if err != nil || len(foreign) != 1 || foreign[0].RunID != "run-filter-foreign" {
		t.Fatalf("foreign owner filtered group = (%+v, %v)", foreign, err)
	}
	if _, err := store.ListRuns(ctx, ListRunsParams{
		OwnerID: "user-1", MetadataLabelSelectors: []RunMetadataLabelSelector{{Key: "Upper", Value: "bad"}}, Limit: 10,
	}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("invalid repository selector error = %v", err)
	}
}

func testAllocationRuntimeConfiguration() *AllocationRuntimeConfiguration {
	gateway := contracts.LLMGatewayConfigRef{
		GatewayID: "local-litellm", Version: "1", Digest: "sha256:" + strings.Repeat("b", 64),
	}
	return &AllocationRuntimeConfiguration{
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

func TestPostgresIntegrationClaimsConflictsAndExplicitTransactions(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	createTestRun(t, ctx, store, "run-race")

	var transitions int32
	var conflicts int32
	var wait sync.WaitGroup
	for range 2 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			_, err := store.TransitionRun(ctx, "run-race", RunInitializing, RunRunning, Reason{Code: "ready"})
			switch {
			case err == nil:
				atomic.AddInt32(&transitions, 1)
			case errors.Is(err, ErrConflict):
				atomic.AddInt32(&conflicts, 1)
			default:
				t.Errorf("transition race: %v", err)
			}
		}()
	}
	wait.Wait()
	if transitions != 1 || conflicts != 1 {
		t.Fatalf("transition race successes=%d conflicts=%d", transitions, conflicts)
	}

	type claimResult struct {
		run WorkflowRun
		err error
	}
	claimResults := make(chan claimResult, 2)
	for _, claimID := range []string{"claim-a", "claim-b"} {
		wait.Add(1)
		go func(claimID string) {
			defer wait.Done()
			run, err := store.ClaimRunnableRun(ctx, claimID, time.Minute)
			claimResults <- claimResult{run: run, err: err}
		}(claimID)
	}
	wait.Wait()
	close(claimResults)
	var claimed WorkflowRun
	var claims, noWork int
	for result := range claimResults {
		if result.err == nil {
			claims++
			claimed = result.run
		} else if errors.Is(result.err, ErrNoWork) {
			noWork++
		} else {
			t.Fatalf("claim race: %v", result.err)
		}
	}
	if claims != 1 || noWork != 1 || claimed.SchedulerClaim == nil {
		t.Fatalf("claim race claims=%d noWork=%d claimed=%+v", claims, noWork, claimed)
	}
	if err := store.RenewRunClaim(ctx, claimed.RunID, claimed.SchedulerClaim.ClaimID, 2*time.Minute); err != nil {
		t.Fatalf("renew claim: %v", err)
	}
	if err := store.ReleaseRunClaim(ctx, claimed.RunID, "wrong-claim"); !errors.Is(err, ErrConflict) {
		t.Fatalf("wrong release error = %v", err)
	}
	if err := store.ReleaseRunClaim(ctx, claimed.RunID, claimed.SchedulerClaim.ClaimID); err != nil {
		t.Fatalf("release claim: %v", err)
	}

	rollback := errors.New("force rollback")
	err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		txStore := NewPostgresStore(tx)
		if _, err := txStore.CreateRun(ctx, testRunParams("run-rolled-back")); err != nil {
			return err
		}
		return rollback
	})
	if !errors.Is(err, rollback) {
		t.Fatalf("transaction error = %v, want rollback sentinel", err)
	}
	if _, err := store.GetRun(ctx, "run-rolled-back"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("rolled-back Run lookup error = %v", err)
	}
	err = persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := NewPostgresStore(tx).CreateRun(ctx, testRunParams("run-committed"))
		return err
	})
	if err != nil {
		t.Fatalf("commit explicit transaction: %v", err)
	}
	if _, err := store.GetRun(ctx, "run-committed"); err != nil {
		t.Fatalf("get committed Run: %v", err)
	}
}

func TestPostgresClaimRunnableRunRotatesAfterDeferredRelease(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)

	first := createTestRun(t, ctx, store, "run-capacity-waiting")
	if _, err := store.TransitionRun(
		ctx, first.RunID, RunInitializing, RunRunning, Reason{Code: "initialized"},
	); err != nil {
		t.Fatal(err)
	}
	claimed, err := store.ClaimRunnableRun(ctx, "claim-capacity-first", time.Minute)
	if err != nil || claimed.RunID != first.RunID {
		t.Fatalf("initial claim = (%+v, %v), want first Run", claimed, err)
	}
	if err := store.ReleaseRunClaim(ctx, first.RunID, "claim-capacity-first"); err != nil {
		t.Fatal(err)
	}

	second := createTestRun(t, ctx, store, "run-compatible-newer")
	if _, err := store.TransitionRun(
		ctx, second.RunID, RunInitializing, RunRunning, Reason{Code: "initialized"},
	); err != nil {
		t.Fatal(err)
	}
	// The older Run receives one more scheduling attempt. Releasing that claim
	// after a deferred capacity result must move it behind the compatible Run.
	claimed, err = store.ClaimRunnableRun(ctx, "claim-capacity-retry", time.Minute)
	if err != nil || claimed.RunID != first.RunID {
		t.Fatalf("capacity retry claim = (%+v, %v), want older Run", claimed, err)
	}
	if err := store.ReleaseRunClaim(ctx, first.RunID, "claim-capacity-retry"); err != nil {
		t.Fatal(err)
	}
	claimed, err = store.ClaimRunnableRun(ctx, "claim-compatible", time.Minute)
	if err != nil || claimed.RunID != second.RunID {
		t.Fatalf("post-defer claim = (%+v, %v), want compatible newer Run", claimed, err)
	}
}

func TestPostgresIntegrationRunSkillSnapshotGatesRunnableClaim(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	run := createTestRun(t, ctx, store, "run-skill-pending")
	revision := "owner-revision-1"
	selection := []contracts.RunSkillSnapshot{{
		Name: "review",
		Source: &contracts.ArtifactRef{
			Namespace: contracts.AgentSkillNamespace, Name: "review", Revision: &revision,
		},
		SourceDigest: "sha256:" + strings.Repeat("a", 64), SourceSize: 1024,
	}}
	if err := store.SetRunSkillSelections(ctx, run.RunID, selection); err != nil {
		t.Fatalf("record Skill selection: %v", err)
	}
	pending, err := store.GetRun(ctx, run.RunID)
	if err != nil || pending.State != RunInitializing ||
		pending.StateReason.Code != SkillInitializationPendingReason ||
		len(pending.SkillSnapshot) != 1 {
		t.Fatalf("pending Run = (%+v, %v)", pending, err)
	}
	claimed, err := store.ClaimRunnableRun(ctx, "skill-claim", time.Minute)
	if err != nil || claimed.RunID != run.RunID || claimed.State != RunInitializing {
		t.Fatalf("claim pending Skill Run = (%+v, %v)", claimed, err)
	}

	runRevision := "run-revision-1"
	initialized := append([]contracts.RunSkillSnapshot(nil), selection...)
	initialized[0].Artifact = &contracts.ArtifactRef{
		Namespace: contracts.AgentSkillNamespace, Name: "review", Revision: &runRevision,
	}
	initialized[0].PackageDigest = initialized[0].SourceDigest
	initialized[0].ExpandedBytes = 2048
	if err := store.CompleteRunSkillInitialization(ctx, run.RunID, initialized); err != nil {
		t.Fatalf("complete Skill snapshot: %v", err)
	}
	started, err := store.TransitionRun(
		ctx, run.RunID, RunInitializing, RunRunning, Reason{Code: "initialized"},
	)
	if err != nil || started.State != RunRunning || started.SkillSnapshot[0].Artifact == nil ||
		*started.SkillSnapshot[0].Artifact.Revision != runRevision {
		t.Fatalf("started Skill Run = (%+v, %v)", started, err)
	}
	_, err = pool.Exec(ctx, `UPDATE workflow_runs SET skill_snapshot = '[]'::jsonb WHERE run_id = $1`, run.RunID)
	if persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("terminal Skill snapshot rewrite SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}

	mutated := append([]contracts.RunSkillSnapshot(nil), initialized...)
	otherRevision := "owner-revision-2"
	mutated[0].Source = &contracts.ArtifactRef{
		Namespace: contracts.AgentSkillNamespace, Name: "review", Revision: &otherRevision,
	}
	if err := store.CompleteRunSkillInitialization(ctx, run.RunID, mutated); !errors.Is(err, ErrConflict) {
		t.Fatalf("source rewrite error = %v, want conflict", err)
	}
}

func TestWorkflowRunEventNotificationIsVisibleOnlyAfterCommit(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	listener, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Release()
	if _, err := listener.Exec(ctx, "LISTEN "+runEventNotificationChannel); err != nil {
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		t.Fatal(err)
	}
	runID := "run-notify-" + hex.EncodeToString(random)
	notified := make(chan struct{})
	waitErr := make(chan error, 1)
	go func() {
		for {
			notification, err := listener.Conn().WaitForNotification(ctx)
			if err != nil {
				waitErr <- err
				return
			}
			if notification.Channel == runEventNotificationChannel && notification.Payload == runID {
				close(notified)
				return
			}
		}
	}()
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	_, err = NewPostgresStore(tx).CreateRun(ctx, CreateRunParams{
		RunID: runID, OwnerID: "user", WorkflowName: "workflow", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot:      json.RawMessage(`{"name":"workflow"}`),
		Parameters:            map[string]string{},
		RuntimeConfig:         runtimeconfig.BuiltInRunSnapshot(),
	})
	if err != nil {
		_ = tx.Rollback(ctx)
		t.Fatal(err)
	}
	select {
	case <-notified:
		_ = tx.Rollback(ctx)
		t.Fatal("WorkflowRun event notification escaped an uncommitted transaction")
	case err := <-waitErr:
		_ = tx.Rollback(ctx)
		t.Fatalf("wait before commit: %v", err)
	case <-time.After(100 * time.Millisecond):
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	select {
	case <-notified:
	case err := <-waitErr:
		t.Fatalf("wait after commit: %v", err)
	case <-time.After(2 * time.Second):
		t.Fatal("committed WorkflowRun event did not publish a wake-up notification")
	}
	events, err := NewPostgresStore(pool).ListRunEvents(ctx, runID, 0, 10)
	if err != nil || len(events) != 1 || events[0].Kind != RunEventLifecycleChanged ||
		events[0].SequenceNumber != 1 || !strings.Contains(string(events[0].Data), `"state": "initializing"`) {
		t.Fatalf("committed lifecycle events = (%+v, %v)", events, err)
	}
}

func TestPostgresIntegrationRunCancellationIsDurableAndIdempotent(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)

	requestedBy := "user-1"
	reason := "stop this work"
	requestedAt := time.Date(2026, 8, 29, 12, 0, 0, 123, time.FixedZone("request-zone", 3*60*60))
	first, err := store.RequestRunCancellation(ctx, createTestRun(t, ctx, store, "run-cancel").RunID, WorkflowRunCancellation{
		Code: CancellationUserRequested, RequestedAt: requestedAt,
		RequestedBy: &requestedBy, Reason: &reason,
	})
	if err != nil {
		t.Fatalf("request cancellation: %v", err)
	}
	if first.State != RunCancelling || first.Cancellation == nil ||
		first.Cancellation.RequestedAt.Location() != time.UTC ||
		!first.Cancellation.RequestedAt.Equal(requestedAt) || first.Cancellation.Reason == nil ||
		*first.Cancellation.Reason != reason || first.CancellationSchemaVersion == nil {
		t.Fatalf("cancelling Run = %+v", first)
	}

	differentReason := "must not replace the winner"
	repeated, err := store.RequestRunCancellation(ctx, first.RunID, WorkflowRunCancellation{
		Code: CancellationUserRequested, RequestedAt: requestedAt.Add(time.Hour), Reason: &differentReason,
	})
	if err != nil {
		t.Fatalf("repeat cancellation: %v", err)
	}
	if repeated.Cancellation == nil || repeated.Cancellation.Reason == nil ||
		*repeated.Cancellation.Reason != reason || !repeated.Cancellation.RequestedAt.Equal(requestedAt) {
		t.Fatalf("repeated cancellation replaced the first payload: %+v", repeated.Cancellation)
	}

	claimed, err := store.ClaimRunnableRun(ctx, "cancel-claim", time.Minute)
	if err != nil || claimed.RunID != first.RunID || claimed.State != RunCancelling {
		t.Fatalf("claim cancelling Run = (%+v, %v)", claimed, err)
	}
	if err := store.RenewRunClaim(ctx, first.RunID, "cancel-claim", time.Minute); err != nil {
		t.Fatalf("renew cancelling Run claim: %v", err)
	}
	cancelled, err := store.TransitionRun(
		ctx, first.RunID, RunCancelling, RunCancelled, Reason{Code: CancellationUserRequested},
	)
	if err != nil || cancelled.State != RunCancelled || cancelled.FinishedAt == nil || cancelled.Cancellation == nil {
		t.Fatalf("finish cancelled Run = (%+v, %v)", cancelled, err)
	}

	_, err = pool.Exec(ctx, `
UPDATE workflow_runs
SET run_cancellation = jsonb_set(run_cancellation, '{reason}', '"rewritten"')
WHERE run_id = $1`, first.RunID)
	if persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("cancellation rewrite SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}

	succeeded := createTestRun(t, ctx, store, "run-succeeded-before-cancel")
	succeeded, err = store.TransitionRun(ctx, succeeded.RunID, RunInitializing, RunRunning, Reason{Code: "ready"})
	if err == nil {
		succeeded, err = store.TransitionRun(ctx, succeeded.RunID, RunRunning, RunSucceeded, Reason{Code: "completed"})
	}
	if err != nil {
		t.Fatalf("finish success before cancellation: %v", err)
	}
	afterCancel, err := store.RequestRunCancellation(ctx, succeeded.RunID, WorkflowRunCancellation{
		Code: CancellationUserRequested, RequestedAt: time.Now(),
	})
	if err != nil || afterCancel.State != RunSucceeded || afterCancel.Cancellation != nil {
		t.Fatalf("cancel terminal success = (%+v, %v)", afterCancel, err)
	}
}

func TestPostgresRunRuntimeLabelsPinExactBindingsAcrossConcurrentRebind(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	credentials := pinTestRuntimeCredentials{}
	publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
		Pool: pool, RuntimeCredentials: credentials,
		PlannerTelemetryAdapters: runtimeconfig.PlannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
		Now:                      func() time.Time { return time.Date(2026, 9, 1, 4, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}
	publish := func(name, block string) runtimeconfig.Ref {
		t.Helper()
		document := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"` +
			name + `","version":"1"},"spec":` + block + `}`)
		result, publishErr := publisher.Publish(ctx, document, "publish-"+name, "operator")
		if publishErr != nil {
			t.Fatal(publishErr)
		}
		return result.Version.Ref
	}
	defaultA := publish("default-a", `{"planner":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/a"}}}`)
	defaultB := publish("default-b", `{"planner":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/b"}}}`)
	debugA := publish("debug-a", `{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/debug-a"}}}`)
	debugB := publish("debug-b", `{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/debug-b"}}}`)
	bindings := runtimeconfig.NewRepository(pool)
	if _, err := bindings.Rebind(ctx, runtimeconfig.DefaultLabel, 1, defaultA, "operator", time.Now()); err != nil {
		t.Fatal(err)
	}
	if _, err := bindings.CreateBinding(ctx, "debug", debugA, "operator", time.Now()); err != nil {
		t.Fatal(err)
	}

	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	txStore := NewPostgresStore(tx)
	pinned, err := txStore.PinRuntimeLabels(ctx, []string{"debug"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pinned.Default.BindingRevision != 2 || pinned.Default.Config != defaultA ||
		len(pinned.Labels) != 1 || pinned.Labels[0].BindingRevision != 1 || pinned.Labels[0].Config != debugA {
		t.Fatalf("old pinned bindings = %+v", pinned)
	}

	rebindDone := make(chan error, 1)
	go func() {
		_, rebindErr := bindings.Rebind(
			ctx, "debug", 1, debugB, "operator", time.Now().Add(time.Minute),
		)
		rebindDone <- rebindErr
	}()
	select {
	case err := <-rebindDone:
		t.Fatalf("binding rebind bypassed Run pin lock: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	if _, err := txStore.CreateRun(ctx, CreateRunParams{
		RunID: "run-runtime-label-old", OwnerID: "user", WorkflowName: "workflow", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot:      json.RawMessage(`{"name":"workflow"}`),
		Parameters:            map[string]string{}, RuntimeConfig: pinned,
	}); err != nil {
		t.Fatal(err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if err := <-rebindDone; err != nil {
		t.Fatal(err)
	}
	if _, err := bindings.Rebind(ctx, runtimeconfig.DefaultLabel, 2, defaultB, "operator", time.Now().Add(time.Minute)); err != nil {
		t.Fatal(err)
	}

	stored, err := NewPostgresStore(pool).GetRun(ctx, "run-runtime-label-old")
	if err != nil || stored.RuntimeConfig.Default.Config != defaultA || stored.RuntimeConfig.Labels[0].Config != debugA {
		t.Fatalf("stored immutable RuntimeConfig snapshot = (%+v, %v)", stored.RuntimeConfig, err)
	}
	if _, err := pool.Exec(ctx, `
UPDATE workflow_runs
SET runtime_config_snapshot = jsonb_set(runtime_config_snapshot, '{default,bindingRevision}', '99')
WHERE run_id = 'run-runtime-label-old'`); persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("RuntimeConfig snapshot rewrite SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}
	newTx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	newPinned, err := NewPostgresStore(newTx).PinRuntimeLabels(ctx, []string{"debug"}, nil)
	_ = newTx.Rollback(ctx)
	if err != nil || newPinned.Default.Config != defaultB || newPinned.Labels[0].Config != debugB ||
		newPinned.Default.BindingRevision != 3 || newPinned.Labels[0].BindingRevision != 2 {
		t.Fatalf("new pinned bindings = (%+v, %v)", newPinned, err)
	}
}

func TestPostgresNonTerminalCredentialUsageIncludesRuntimeConfigSnapshot(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	snapshot := runtimeconfig.BuiltInRunSnapshot()
	snapshot.LLMCredentialIDs = []string{"runtime-route"}
	if _, err := store.CreateRun(ctx, CreateRunParams{
		RunID: "run-runtime-route", OwnerID: "user", WorkflowName: "workflow", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`),
		Parameters: map[string]string{}, RuntimeConfig: snapshot,
	}); err != nil {
		t.Fatal(err)
	}
	runs, err := store.ListNonTerminalRunIDsByCredential(ctx, "runtime-route", 10)
	if err != nil || len(runs) != 1 || runs[0] != "run-runtime-route" {
		t.Fatalf("RuntimeConfig LLM credential usage = (%v, %v)", runs, err)
	}
	if _, err := store.TransitionRun(
		ctx, "run-runtime-route", RunInitializing, RunFailed, Reason{Code: "test_complete"},
	); err != nil {
		t.Fatal(err)
	}
	runs, err = store.ListNonTerminalRunIDsByCredential(ctx, "runtime-route", 10)
	if err != nil || len(runs) != 0 {
		t.Fatalf("terminal RuntimeConfig LLM credential usage = (%v, %v)", runs, err)
	}
}

func TestPostgresCredentialUsageIncludesLiveAllocationProvenance(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	run := createTestRun(t, ctx, store, "run-live-allocation-credential")
	if _, err := store.TransitionRun(
		ctx, run.RunID, RunInitializing, RunRunning, Reason{Code: "ready"},
	); err != nil {
		t.Fatal(err)
	}
	execution, err := store.CreateStageExecution(ctx, CreateStageExecutionParams{
		StageExecutionID: "stage-live-allocation-credential", RunID: run.RunID,
		StageName: "copy", Attempt: 1, StageSpecSchemaVersion: contracts.APIVersion,
		StageSpecSnapshot: json.RawMessage(`{}`), StageContextSchemaVersion: contracts.APIVersion,
		StageContext: StageContextSnapshot{},
	})
	if err != nil {
		t.Fatal(err)
	}
	configuration := testAllocationRuntimeConfiguration()
	credential := contracts.LLMCredentialRef{CredentialID: "allocation-only-key"}
	configuration.Provenance.LLMCredential = &credential
	if err := store.RecordStageAllocation(ctx, StageAllocation{
		AllocationID: "allocation-live-credential", StageExecutionID: execution.StageExecutionID,
		LogicalAgentName: "builder", Namespace: "builder",
		AgentTemplateRef: contracts.AgentTemplateRef{
			TemplateID: "builder", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64),
		},
		WorkerRuntimeRef: contracts.WorkerRuntimeRef{RuntimeID: "adk", Version: "1"},
		RuntimeAgentID:   strings.Repeat("2", 64), RuntimeAgentInstanceID: "runtime-live",
		RuntimeAgentLabelRevision:         1,
		RuntimeConfigurationSchemaVersion: AllocationRuntimeConfigurationSchemaVersion,
		RuntimeConfiguration:              configuration,
	}); err != nil {
		t.Fatal(err)
	}
	runIDs, err := store.ListNonTerminalRunIDsByCredential(ctx, credential.CredentialID, 10)
	if err != nil || len(runIDs) != 1 || runIDs[0] != run.RunID {
		t.Fatalf("live allocation credential Runs = (%v, %v)", runIDs, err)
	}
	if err := store.MarkStageAllocationReleased(ctx, "allocation-live-credential"); err != nil {
		t.Fatal(err)
	}
	runIDs, err = store.ListNonTerminalRunIDsByCredential(ctx, credential.CredentialID, 10)
	if err != nil || len(runIDs) != 0 {
		t.Fatalf("released allocation credential Runs = (%v, %v)", runIDs, err)
	}
}

type pinTestRuntimeCredentials struct{}

func (pinTestRuntimeCredentials) ValidateRuntimeCredential(context.Context, string, ...string) error {
	return nil
}

func (pinTestRuntimeCredentials) WithCredentialReferences(_ context.Context, fn func() error) error {
	return fn()
}

func createTestRun(t *testing.T, ctx context.Context, store *PostgresStore, runID string) WorkflowRun {
	t.Helper()
	run, err := store.CreateRun(ctx, testRunParams(runID))
	if err != nil {
		t.Fatalf("create test Run %q: %v", runID, err)
	}
	return run
}

func testRunParams(runID string) CreateRunParams {
	return CreateRunParams{
		RunID: runID, OwnerID: "user-1", WorkflowName: "artifact-copy", WorkflowVersion: "1",
		WorkflowSchemaVersion: "contractor/v1alpha1",
		WorkflowSnapshot:      json.RawMessage(`{"ref":{"name":"artifact-copy","version":"1"}}`),
		Parameters:            map[string]string{"mode": "strict"},
		RuntimeConfig:         runtimeconfig.BuiltInRunSnapshot(),
	}
}

func isolatedRunStorePool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	adminConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatalf("parse test database URL: %v", err)
	}
	admin, err := pgxpool.NewWithConfig(ctx, adminConfig)
	if err != nil {
		t.Fatalf("open test admin pool: %v", err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatalf("ping test database: %v", err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_test_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatalf("create isolated schema: %v", err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatalf("open isolated pool: %v", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		t.Fatalf("ping isolated pool: %v", err)
	}
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		t.Fatalf("apply migrations: %v", err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanupCtx, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop isolated schema: %v", err)
		}
		admin.Close()
	})
	return pool
}
