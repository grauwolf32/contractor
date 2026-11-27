package streamline

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

// Ordinary Server contexts must reach the same durable plan transitions and
// Worker budget boundary as model callbacks, without creating an ADK session.
func TestPostgresPlannerOperationsWithoutADKContext(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		for _, exhausted := range []bool{false, true} {
			name := "complete"
			if exhausted {
				name = "worker-budget"
			}
			t.Run(profile+"/"+name, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
				defer cancel()
				pool := isolatedStreamlinePool(t, ctx)
				store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
				createPreparingStreamlineStage(t, ctx, store)
				sessions, err := plannersession.New(store, plannersession.Options{})
				if err != nil {
					t.Fatal(err)
				}
				workers := appendProbeWorkers()
				factory := runtimeTestFactory(t, profile, sessions, workers)
				instance, err := factory.Create(appendProbeInvocation(profile))
				if err != nil {
					t.Fatal(err)
				}
				p := instance.(*streamlinePlanner)
				defer p.stateViews.Close()
				p.deadline, _ = ctx.Deadline()
				state := newExecutionState(p.limits)
				if exhausted {
					state.workerCalls = int64(state.limits.MaxWorkerCalls)
				}
				started, err := sessions.Begin(ctx, p.invocation.StageExecutionID)
				if err != nil {
					t.Fatal(err)
				}
				identity := started.Identity
				if err := sessions.RecordRequest(ctx, identity, planner.RequestFactsFor([]string{"builder"}, p.request)); err != nil {
					t.Fatal(err)
				}
				added := p.addSubtask(ctx, state, identity, addSubtaskArgs{
					Objective: "Inspect bounded input", Instructions: "Return bounded evidence",
				})
				if !added.OK || added.Plan.CurrentSubtaskID != "0" {
					t.Fatalf("add subtask = %+v", added)
				}
				finish := finishArgs{Outcome: stageresult.StageSucceeded, Summary: "Complete", Artifacts: map[string]materialSelection{}}
				if rejected := p.finish(ctx, state, identity, finish); rejected.Accepted {
					t.Fatal("unfinished subtask authorized finish")
				}
				var result workerCallOutput
				var stop bool
				if profile == "router" {
					result, stop = p.routeWorker(ctx, state, identity, routerWorkerCallArgs{SubtaskID: "0", WorkerName: "builder"})
				} else {
					result, stop = p.callWorker(ctx, state, identity, p.workers[0], "0")
				}
				wantStatus := planner.PlannerSubtaskSucceeded
				if exhausted {
					wantStatus = planner.PlannerSubtaskFailed
					if !stop || result.Error == nil || result.Error.Code != "planner_worker_call_limit" || len(workers.bindingCalls()) != 0 {
						t.Fatalf("budget refusal: result=%+v stop=%v calls=%v", result, stop, workers.bindingCalls())
					}
				} else {
					if !result.OK || stop || len(workers.bindingCalls()) != 1 {
						t.Fatalf("dispatch: result=%+v stop=%v calls=%v", result, stop, workers.bindingCalls())
					}
					if accepted := p.finish(ctx, state, identity, finish); !accepted.Accepted {
						t.Fatalf("finish = %+v", accepted)
					}
				}
				stored, err := store.GetPlannerSession(ctx, identity.SessionID)
				if err != nil {
					t.Fatal(err)
				}
				var durable struct {
					Plan          planner.PlannerPlanProjection `json:"plan"`
					ADKEventCount int                           `json:"adkEventCount"`
				}
				if err := json.Unmarshal(stored.State, &durable); err != nil {
					t.Fatal(err)
				}
				if len(durable.Plan.Subtasks) != 1 || durable.Plan.Subtasks[0].Status != wantStatus || durable.Plan.ActiveDispatch != nil || durable.ADKEventCount != 0 {
					t.Fatalf("durable operation result = %+v", durable)
				}
			})
		}
	}
}
