package public

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type detailQueryTrace struct {
	mu      sync.Mutex
	queries []string
}

func (q *detailQueryTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.queries = append(q.queries, data.SQL)
	return ctx
}
func (*detailQueryTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}
func (q *detailQueryTrace) take() []string {
	q.mu.Lock()
	defer q.mu.Unlock()
	result := q.queries
	q.queries = nil
	return result
}

// Hide optional batch interfaces to compare the identical production data
// through the pre-batch composition contract, not a hand-built JSON fixture.
type singleRunReader struct{ RunReader }
type singleMetricsReader struct{ MetricsReader }
type singlePlanReader struct{ PlannerPlanReader }

func TestPostgresRunDetailFixedBatchQueries(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	trace := &detailQueryTrace{}
	tracedConfig := pool.Config()
	tracedConfig.ConnConfig.Tracer = trace
	traced, err := pgxpool.NewWithConfig(ctx, tracedConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer traced.Close()
	runs := runstore.NewPostgresStore(pool)
	metrics := telemetry.NewRepository(pool)
	plans, err := plannersession.New(runs, plannersession.Options{})
	if err != nil {
		t.Fatal(err)
	}
	catalog, err := config.Load("../../config/testdata/valid", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := catalog.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stageJSON, err := json.Marshal(workflow.Stages[workflow.EntryStage])
	if err != nil {
		t.Fatal(err)
	}
	tracedRuns := runstore.NewPostgresStore(traced)
	tracedPlans, err := plannersession.New(tracedRuns, plannersession.Options{})
	if err != nil {
		t.Fatal(err)
	}
	batch := &handler{dependencies: Dependencies{
		Runs: tracedRuns, Metrics: telemetry.NewRepository(traced), PlannerPlans: tracedPlans,
		AllocationResources: telemetry.NewRepository(traced),
		Artifacts:           artifacts.NewService(artifacts.NewPostgresRepository(traced)),
	}}
	single := &handler{dependencies: batch.dependencies}
	single.dependencies.Runs = singleRunReader{batch.dependencies.Runs}
	single.dependencies.Metrics = singleMetricsReader{batch.dependencies.Metrics}
	single.dependencies.PlannerPlans = singlePlanReader{batch.dependencies.PlannerPlans}
	get := func(h *handler, runID, owner string) *httptest.ResponseRecorder {
		r := httptest.NewRequest(http.MethodGet, "/v1/runs/"+runID, nil)
		r.SetPathValue("runID", runID)
		r = r.WithContext(auth.WithPrincipal(ctx, auth.Principal{UserID: owner}))
		w := httptest.NewRecorder()
		h.getRun(w, r)
		return w
	}
	wantTotal := 0
	for _, n := range []int{1, 5, 30} {
		t.Run(fmt.Sprintf("stages-%d", n), func(t *testing.T) {
			runID := fmt.Sprintf("batch-run-%d", n)
			if _, err := runs.CreateRun(ctx, runstore.CreateRunParams{RunID: runID, OwnerID: "user-1", WorkflowName: "artifact-copy", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`), RuntimeConfig: runtimeconfig.BuiltInRunSnapshot()}); err != nil {
				t.Fatal(err)
			}
			if _, err := runs.TransitionRun(ctx, runID, runstore.RunInitializing, runstore.RunRunning, runstore.Reason{Code: "initialized"}); err != nil {
				t.Fatal(err)
			}
			service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
			runArtifacts, _ := service.Run(runID)
			source, err := runArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "scratch", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("exact output")}, nil)
			if err != nil {
				t.Fatal(err)
			}
			output, err := service.BindOutputExact(ctx, runID, "result", source.Ref, nil)
			if err != nil {
				t.Fatal(err)
			}
			if err := service.FreezeRunOutputs(ctx, runID); err != nil {
				t.Fatal(err)
			}
			for i := 0; i < n; i++ {
				id := fmt.Sprintf("%s-stage-%02d", runID, i)
				if _, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{StageExecutionID: id, RunID: runID, StageName: id, Attempt: 1, StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageJSON, StageContextSchemaVersion: contracts.APIVersion}); err != nil {
					t.Fatal(err)
				}
				start, err := plans.Begin(ctx, id)
				if err != nil {
					t.Fatal(err)
				}
				if i%2 == 0 {
					if err := plans.RecordPlan(ctx, start.Identity, planner.PlannerPlanTransition{Kind: planner.PlannerEventPlanChanged, Plan: planner.PlannerPlanProjection{Revision: 1, Subtasks: []planner.PlannerSubtask{{ID: "0", Objective: "Read exact source", Instructions: "Inspect", Status: planner.PlannerSubtaskPending}}, CurrentSubtaskID: "0"}}); err != nil {
						t.Fatal(err)
					}
					if _, err := metrics.RebuildStageMetrics(ctx, id, contracts.APIVersion); err != nil {
						t.Fatal(err)
					}
					// Legacy persisted allocations still have to be decoded and
					// ordered correctly; odd stages deliberately have none.
					if _, err := pool.Exec(ctx, `INSERT INTO stage_allocations (allocation_id,stage_execution_id,logical_agent_name,namespace,agent_template_ref,worker_runtime_ref,runtime_agent_instance_id)
VALUES ($1,$2,'builder','builder','{}','{}','private-physical-instance')`, "allocation-"+id, id); err != nil {
						t.Fatal(err)
					}
				}
				if i < n-1 {
					result := contracts.StageContentResult{APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded, Summary: "done", Artifacts: map[string]contracts.ArtifactRef{}}
					if err := runs.EnterFinalizing(ctx, runstore.EnterFinalizingParams{StageExecutionID: id, ResultSchemaVersion: contracts.APIVersion, Candidate: result, FinalizationID: "finalize-" + id, Deadline: time.Now().Add(time.Minute), Reason: runstore.Reason{Code: "done"}}); err != nil {
						t.Fatal(err)
					}
					if err := runs.CompleteStageResult(ctx, id, contracts.APIVersion, result); err != nil {
						t.Fatal(err)
					}
				}
			}
			if n > 1 {
				// A malformed optional summary must not suppress healthy peers.
				id := fmt.Sprintf("%s-stage-01", runID)
				if _, err := metrics.RebuildStageMetrics(ctx, id, contracts.APIVersion); err != nil {
					t.Fatal(err)
				}
				if _, err := pool.Exec(ctx, `UPDATE stage_metrics SET summary='{"modelCalls":"invalid"}' WHERE stage_execution_id=$1`, id); err != nil {
					t.Fatal(err)
				}
			}
			trace.take()
			response := get(batch, runID, "user-1")
			queries := trace.take()
			if response.Code != http.StatusOK {
				t.Fatalf("batch GET: %d %s", response.Code, response.Body.String())
			}
			if wantTotal == 0 {
				wantTotal = len(queries)
			} else if len(queries) != wantTotal {
				t.Errorf("queries grew: %d vs %d", len(queries), wantTotal)
			}
			for _, table := range []string{"stage_allocations", "stage_metrics", "planner_sessions"} {
				count := 0
				for _, query := range queries {
					if strings.Contains(query, "FROM "+table) {
						count++
						if !strings.Contains(query, "ANY($1::text[])") {
							t.Errorf("non-batch %s query: %s", table, query)
						}
					}
				}
				if count != 1 {
					t.Errorf("%s queries = %d, want 1", table, count)
				}
			}
			resourceQueries := 0
			for _, query := range queries {
				if strings.Contains(query, "LEFT JOIN LATERAL") && strings.Contains(query, "allocation_execution_reports") {
					resourceQueries++
				}
			}
			if resourceQueries != 1 {
				t.Errorf("allocation resource queries = %d, want 1", resourceQueries)
			}
			baseline := get(single, runID, "user-1")
			baselineQueries := trace.take()
			t.Logf("stages=%d total queries: per-stage=%d batched=%d; related batches=4", n, len(baselineQueries), len(queries))
			if baseline.Code != http.StatusOK || baseline.Body.String() != response.Body.String() {
				t.Fatalf("batch changed response\nbatch: %s\nsingle: %s", response.Body.String(), baseline.Body.String())
			}
			var detail runStatusResponse
			if err := json.Unmarshal(response.Body.Bytes(), &detail); err != nil {
				t.Fatal(err)
			}
			if len(detail.Attempts) != n || detail.ActiveStageExecutionID == nil || detail.Attempts[0].Plan == nil || detail.Attempts[0].Metrics == nil || detail.Outputs["result"].Revision == nil || *detail.Outputs["result"].Revision != *output.TargetRef.Revision {
				t.Fatalf("incomplete detail: %+v", detail)
			}
			if strings.Contains(response.Body.String(), "private-physical-instance") {
				t.Fatal("physical allocation identity leaked")
			}
			denied := get(batch, runID, "other-user")
			deniedQueries := trace.take()
			// GetRun includes the metadata-label read; related data must never
			// be queried before the owner guard succeeds.
			if denied.Code != http.StatusNotFound || len(deniedQueries) != 2 {
				t.Fatalf("owner isolation: status=%d queries=%d", denied.Code, len(deniedQueries))
			}
			for _, query := range deniedQueries {
				if strings.Contains(query, "FROM stage_") || strings.Contains(query, "FROM planner_sessions") {
					t.Fatal("owner denial loaded related data")
				}
			}
		})
	}
}
