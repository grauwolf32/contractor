package scheduler

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

type deferredPlacementHarness struct {
	pool      *pgxpool.Pool
	store     *runstore.PostgresStore
	scheduler *Scheduler
	workflow  workflowconfig.ResolvedWorkflow
	invoker   *postgresTestWorkerInvoker
	workers   *memoryWorkers
}

// newDeferredPlacementHarness pins allocation rows during placement, as
// production placement does, so a deferral cannot hide lost durable state.
func newDeferredPlacementHarness(t *testing.T, ctx context.Context, wrap func(AtomicPersistence) AtomicPersistence) deferredPlacementHarness {
	t.Helper()
	pool := isolatedSchedulerPool(t, ctx)
	workflow := loadSchedulerWorkflow(t)
	store := runstore.NewPostgresStore(pool)
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	runStore := createSchedulerRun(t, ctx, store, artifactService, workflow)
	workerWrite, err := runStore.Write(ctx, contracts.ArtifactRef{Namespace: "builder", Name: "copied"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("copied\n")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	sessions, err := plannersession.New(store, plannersession.Options{})
	if err != nil {
		t.Fatal(err)
	}
	inspector, err := planner.NewRunArtifactInspector("run-1", runStore)
	if err != nil {
		t.Fatal(err)
	}
	invoker := &postgresTestWorkerInvoker{result: contracts.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded, Summary: "copied",
		Artifacts: map[string]contracts.ArtifactRef{"copied": workerWrite.Ref},
	}}
	passthrough, err := planner.NewPassthroughFactory(sessions, invoker, inspector)
	if err != nil {
		t.Fatal(err)
	}
	planners, err := planner.NewRegistry(passthrough)
	if err != nil {
		t.Fatal(err)
	}
	recovery, err := gatewayrecovery.New(pool, gatewayrecovery.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	clock := staticClock{now: time.Now().UTC()}
	events := &eventRecorder{}
	allocator := &memoryAllocator{workflow: workflow, clock: clock, events: events, grants: map[string]controlplane.AllocationGrant{}}
	workers := &memoryWorkers{allocator: allocator, workflow: workflow, clock: clock, events: events}
	resolver, _ := NewArtifactServiceResolver(artifactService)
	transactions, _ := NewPostgresPersistence(pool)
	var persistence AtomicPersistence = transactions
	if wrap != nil {
		persistence = wrap(persistence)
	}
	workerSelection := workflow.Stages[workflow.EntryStage].ExecutionConfig.Agents["builder"]
	credentialProvider, err := credentials.NewStaticProvider([]credentials.StaticEntry{{
		Metadata: workflowconfig.CredentialMetadata{
			Ref: *workerSelection.Credential, LLMGateway: workerSelection.LLMGateway.Ref, Unrestricted: true,
		},
		Token: contracts.NewSecretString("scheduler-test-token"),
	}})
	if err != nil {
		t.Fatal(err)
	}
	sequence := 0
	scheduler, err := New(store, persistence, resolver, allocator, workers, planners, Options{
		PollInterval: time.Second, ClaimDuration: time.Minute, OperationTimeout: 5 * time.Second,
		PlannerTimeout: 20 * time.Second, FinalizationTimeout: 5 * time.Second, AbortTimeout: 5 * time.Second,
		RuntimeSettings: testSchedulerRuntimeSettings(), Credentials: credentialProvider, Clock: clock,
		Settings: settingsstore.NewPostgresStore(pool), GatewayRecovery: recovery,
		NewID: func(prefix string) (string, error) {
			sequence++
			return prefix + strings.Repeat("x", sequence), nil
		},
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	allocator.record = scheduler.recordReservations
	return deferredPlacementHarness{pool, store, scheduler, workflow, invoker, workers}
}

func (h deferredPlacementHarness) requireSingleSucceededStage(t *testing.T, ctx context.Context) {
	t.Helper()
	run, err := h.store.GetRun(ctx, "run-1")
	if err != nil || run.State != runstore.RunSucceeded {
		t.Fatalf("Run = (%s %+v, %v)", run.State, run.StateReason, err)
	}
	executions, err := h.store.ListStageExecutions(ctx, "run-1")
	if err != nil || len(executions) != 1 || executions[0].State != runstore.StageSucceeded {
		t.Fatalf("StageExecutions after admission = (%+v, %v)", executions, err)
	}
	allocations, err := h.store.ListStageAllocations(ctx, executions[0].StageExecutionID)
	if err != nil || len(allocations) != 1 || allocations[0].ReleaseCompletedAt == nil {
		t.Fatalf("Stage allocations = (%+v, %v)", allocations, err)
	}
}

// A gateway admission denial must not leave pinned rows behind without live
// grants, or the next claim aborts the Stage with control_plane_state_lost.
func TestPostgresGatewayDeferralDoesNotLoseControlPlaneState(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	h := newDeferredPlacementHarness(t, ctx, nil)
	pool, store, scheduler, workflow := h.pool, h.store, h.scheduler, h.workflow
	for _, reservation := range schedulerTestReservations(t, workflow.Stages[workflow.EntryStage]) {
		for _, route := range reservationModelRoutes("user-1", reservation) {
			if _, err := pool.Exec(ctx, `
INSERT INTO gateway_recovery_routes(route_key,owner_id,blocked,failure_code,next_probe_at,automatic_until)
VALUES($1,$2,true,'model_unavailable',clock_timestamp()+interval '1 hour',clock_timestamp()+interval '1 hour')`,
				route.Key(), route.OwnerID); err != nil {
				t.Fatal(err)
			}
		}
	}
	if worked, err := scheduler.RunOnce(ctx); !errors.Is(err, ErrDeferred) || !worked {
		t.Fatalf("blocked RunOnce = (%v, %v)", worked, err)
	}
	executions, err := store.ListStageExecutions(ctx, "run-1")
	if err != nil || len(executions) != 1 || executions[0].State != runstore.StagePreparing || executions[0].AdmittedAt != nil {
		t.Fatalf("deferred StageExecutions = (%+v, %v)", executions, err)
	}
	if h.invoker.calls != 0 || h.workers.prepareCalls != 0 {
		t.Fatalf("blocked route reached Workers: invoker=%d prepare=%d", h.invoker.calls, h.workers.prepareCalls)
	}
	allocations, err := store.ListStageAllocations(ctx, executions[0].StageExecutionID)
	if err != nil || len(allocations) != 0 {
		t.Fatalf("deferred placement left allocations = (%+v, %v)", allocations, err)
	}

	if _, err := pool.Exec(ctx, `UPDATE gateway_recovery_routes SET blocked=false,next_probe_at=NULL,automatic_until=NULL`); err != nil {
		t.Fatal(err)
	}
	if worked, err := scheduler.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("unblocked RunOnce = (%v, %v)", worked, err)
	}
	h.requireSingleSucceededStage(t, ctx)
}

// A paused owner's pending Run must not pin a durable placement that
// AdmitStage then rejects: that would hold Runtime slots for the whole pause.
func TestPostgresQueuePauseDoesNotPlacePendingStage(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	h := newDeferredPlacementHarness(t, ctx, nil)
	if _, err := h.pool.Exec(ctx, `
UPDATE workflow_runs SET state = 'pending', started_at = NULL, state_reason_code = 'awaiting_admission'
WHERE run_id = 'run-1'`); err != nil {
		t.Fatal(err)
	}
	allocator := h.workers.allocator
	allocator.reserveError = controlplane.ErrInsufficientCapacity
	if worked, err := h.scheduler.RunOnce(ctx); !errors.Is(err, ErrDeferred) || !worked {
		t.Fatalf("capacity-deferred RunOnce = (%v, %v)", worked, err)
	}
	setPaused := func(paused bool) {
		t.Helper()
		control, err := h.store.GetOwnerQueueControl(ctx, "user-1")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := h.store.UpdateOwnerQueueControl(ctx, runstore.UpdateOwnerQueueControlParams{
			OwnerID: "user-1", ExpectedRevision: control.Revision, Paused: paused,
		}); err != nil {
			t.Fatal(err)
		}
	}
	setPaused(true)
	allocator.reserveError = nil
	reserveCalls := allocator.reserveCalls
	if worked, err := h.scheduler.RunOnce(ctx); !errors.Is(err, ErrDeferred) || !worked {
		t.Fatalf("paused RunOnce = (%v, %v)", worked, err)
	}
	executions, err := h.store.ListStageExecutions(ctx, "run-1")
	if err != nil || len(executions) != 1 || executions[0].State != runstore.StagePreparing ||
		executions[0].AdmittedAt != nil {
		t.Fatalf("paused StageExecutions = (%v, %v)", stageExecutionStates(executions), err)
	}
	allocations, err := h.store.ListStageAllocations(ctx, executions[0].StageExecutionID)
	if err != nil || len(allocations) != 0 || allocator.reserveCalls != reserveCalls || len(allocator.grants) != 0 {
		t.Fatalf("paused placement = allocations:%d reserves:%d grants:%d err:%v",
			len(allocations), allocator.reserveCalls-reserveCalls, len(allocator.grants), err)
	}

	setPaused(false)
	if worked, err := h.scheduler.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("resumed RunOnce = (%v, %v)", worked, err)
	}
	h.requireSingleSucceededStage(t, ctx)
}

type failOnceAdmitPersistence struct {
	AtomicPersistence
	failed bool
}

func (p *failOnceAdmitPersistence) AdmitStage(ctx context.Context, runID, stageID string) (runstore.StageExecution, error) {
	if !p.failed {
		p.failed = true
		return runstore.StageExecution{}, errors.New("transient admission failure")
	}
	return p.AtomicPersistence.AdmitStage(ctx, runID, stageID)
}

// A deferral after durable placement keeps the committed reservation live, so
// the next claim reuses it instead of aborting with control_plane_state_lost.
func TestPostgresAdmitStageFailureReusesDurablePlacement(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	failing := &failOnceAdmitPersistence{}
	h := newDeferredPlacementHarness(t, ctx, func(persistence AtomicPersistence) AtomicPersistence {
		failing.AtomicPersistence = persistence
		return failing
	})
	if worked, err := h.scheduler.RunOnce(ctx); err == nil || !worked || !failing.failed {
		t.Fatalf("failed admission RunOnce = (%v, %v)", worked, err)
	}
	executions, err := h.store.ListStageExecutions(ctx, "run-1")
	if err != nil || len(executions) != 1 || executions[0].State != runstore.StagePreparing {
		t.Fatalf("StageExecutions after failed admission = (%+v, %v)", executions, err)
	}
	if worked, err := h.scheduler.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("retried RunOnce = (%v, %v)", worked, err)
	}
	h.requireSingleSucceededStage(t, ctx)
	var routes int
	if err := h.pool.QueryRow(ctx, `SELECT count(*) FROM gateway_allocation_routes`).Scan(&routes); err != nil || routes == 0 {
		t.Fatalf("reused placement did not bind model routes: %d %v", routes, err)
	}
}

// Failing a Run the Scheduler cannot progress must also end its active Stage,
// so terminal allocation recovery releases the allocation it still holds.
func TestPostgresFailActiveRunTerminatesStageAndReleasesAllocations(t *testing.T) {
	for _, test := range []struct {
		name       string
		cancel     bool
		runState   runstore.WorkflowRunState
		stageState runstore.StageExecutionState
	}{
		{"failed", false, runstore.RunFailed, runstore.StageInterrupted},
		{"cancelling", true, runstore.RunCancelled, runstore.StageCancelled},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			failing := &failOnceAdmitPersistence{}
			h := newDeferredPlacementHarness(t, ctx, func(persistence AtomicPersistence) AtomicPersistence {
				failing.AtomicPersistence = persistence
				return failing
			})
			if worked, err := h.scheduler.RunOnce(ctx); err == nil || !worked {
				t.Fatalf("placement RunOnce = (%v, %v)", worked, err)
			}
			if test.cancel {
				if _, err := h.store.RequestRunCancellation(ctx, "run-1", runstore.WorkflowRunCancellation{
					Code: runstore.CancellationUserRequested, RequestedAt: time.Now(),
				}); err != nil {
					t.Fatal(err)
				}
			}
			if err := h.scheduler.failInvalidRunState(ctx, "run-1", errors.New("invalid state")); err != nil {
				t.Fatal(err)
			}
			run, err := h.store.GetRun(ctx, "run-1")
			if err != nil || run.State != test.runState {
				t.Fatalf("Run = (%s %+v, %v)", run.State, run.StateReason, err)
			}
			executions, err := h.store.ListStageExecutions(ctx, "run-1")
			if err != nil || len(executions) != 1 || executions[0].State != test.stageState ||
				executions[0].Termination == nil || executions[0].Termination.Phase != runstore.TerminationPreparing {
				t.Fatalf("StageExecutions = (%+v, %v)", executions, err)
			}
			if _, err := h.scheduler.RunOnce(ctx); err != nil {
				t.Fatal(err)
			}
			allocations, err := h.store.ListStageAllocations(ctx, executions[0].StageExecutionID)
			if err != nil || len(allocations) != 1 || allocations[0].ReleaseCompletedAt == nil {
				t.Fatalf("Stage allocations after recovery = (%+v, %v)", allocations, err)
			}
			if _, err := h.workers.allocator.GetGrant(allocations[0].AllocationID); !errors.Is(err, controlplane.ErrAllocationNotFound) {
				t.Fatalf("allocation grant after recovery = %v", err)
			}
		})
	}
}

func stageExecutionStates(executions []runstore.StageExecution) []runstore.StageExecutionState {
	states := make([]runstore.StageExecutionState, 0, len(executions))
	for _, execution := range executions {
		states = append(states, execution.State)
	}
	return states
}

type failOnceResultCommitPersistence struct {
	AtomicPersistence
	failed bool
}

func (p *failOnceResultCommitPersistence) CommitResultProgression(ctx context.Context, value ResultProgression) error {
	if !p.failed {
		p.failed = true
		return errors.New("transient result commit failure")
	}
	return p.AtomicPersistence.CommitResultProgression(ctx, value)
}

// A finalizing Stage already won its result race, so failing the Run accepts
// that immutable candidate instead of leaving a non-terminal Stage whose
// allocations terminal recovery would never release.
func TestPostgresFailActiveRunAcceptsFinalizingCandidateAndReleasesAllocations(t *testing.T) {
	for _, test := range []struct {
		name     string
		cancel   bool
		runState runstore.WorkflowRunState
	}{
		{"failed", false, runstore.RunFailed},
		{"cancelling", true, runstore.RunCancelled},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			failing := &failOnceResultCommitPersistence{}
			h := newDeferredPlacementHarness(t, ctx, func(persistence AtomicPersistence) AtomicPersistence {
				failing.AtomicPersistence = persistence
				return failing
			})
			if worked, err := h.scheduler.RunOnce(ctx); err == nil || !worked || !failing.failed {
				t.Fatalf("interrupted result commit RunOnce = (%v, %v)", worked, err)
			}
			executions, err := h.store.ListStageExecutions(ctx, "run-1")
			if err != nil || len(executions) != 1 || executions[0].State != runstore.StageFinalizing {
				t.Fatalf("StageExecutions before Run failure = (%v, %v)", stageExecutionStates(executions), err)
			}
			if test.cancel {
				if _, err := h.store.RequestRunCancellation(ctx, "run-1", runstore.WorkflowRunCancellation{
					Code: runstore.CancellationUserRequested, RequestedAt: time.Now(),
				}); err != nil {
					t.Fatal(err)
				}
			}
			if err := h.scheduler.failInvalidRunState(ctx, "run-1", errors.New("invalid state")); err != nil {
				t.Fatal(err)
			}
			run, err := h.store.GetRun(ctx, "run-1")
			if err != nil || run.State != test.runState {
				t.Fatalf("Run = (%s %+v, %v)", run.State, run.StateReason, err)
			}
			executions, err = h.store.ListStageExecutions(ctx, "run-1")
			if err != nil || len(executions) != 1 || executions[0].State != runstore.StageSucceeded ||
				executions[0].AcceptedResult == nil || executions[0].Termination != nil {
				t.Fatalf("StageExecutions after Run failure = (%v, %v)", stageExecutionStates(executions), err)
			}
			if _, err := h.scheduler.RunOnce(ctx); err != nil {
				t.Fatal(err)
			}
			allocations, err := h.store.ListStageAllocations(ctx, executions[0].StageExecutionID)
			if err != nil || len(allocations) != 1 || allocations[0].ReleaseCompletedAt == nil {
				t.Fatalf("Stage allocations after recovery = (%+v, %v)", allocations, err)
			}
			if _, err := h.workers.allocator.GetGrant(allocations[0].AllocationID); !errors.Is(err, controlplane.ErrAllocationNotFound) {
				t.Fatalf("allocation grant after recovery = %v", err)
			}
			if err := h.store.DeleteReleasedTerminalRun(ctx, "user-1", "run-1"); err != nil {
				t.Fatalf("delete released terminal Run: %v", err)
			}
		})
	}
}
