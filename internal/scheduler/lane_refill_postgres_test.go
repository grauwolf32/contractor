package scheduler

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/settingsstore"
)

// postgresClaimAttempts records every PostgreSQL claim attempt, with or
// without work, by lane class.
type postgresClaimAttempts struct {
	*postgresLaneBlockingStore
	mu        sync.Mutex
	attempts  map[resolved.LaneClass]int
	attempted chan resolved.LaneClass
}

func (s *postgresClaimAttempts) ClaimRunnableRun(
	ctx context.Context, claimID string, lease time.Duration, excluded []string, class resolved.LaneClass,
) (runstore.WorkflowRun, error) {
	run, err := s.postgresLaneBlockingStore.ClaimRunnableRun(ctx, claimID, lease, excluded, class)
	s.mu.Lock()
	s.attempts[class]++
	s.mu.Unlock()
	s.attempted <- class
	return run, err
}

func (s *postgresClaimAttempts) counts() map[resolved.LaneClass]int {
	s.mu.Lock()
	defer s.mu.Unlock()
	counts := make(map[resolved.LaneClass]int, len(s.attempts))
	for class, count := range s.attempts {
		counts[class] = count
	}
	return counts
}

// With the migration's one model and two tool lanes and no runnable Run, each
// fill round makes one real PostgreSQL claim attempt per pool; once one model
// and two tool Runs are runnable, the next start claims all three before any
// completes and makes no empty claim (SCH-R181, LIM-SCH-21).
func TestPostgresSchedulerSupervisorClaimsOncePerIdleRoundAndFillsLanes(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	postgresStore := runstore.NewPostgresStore(pool)
	settings := settingsstore.NewPostgresStore(pool)
	if current, err := settings.GetSchedulerSettings(ctx); err != nil ||
		current.MaxConcurrentRuns != 1 || current.MaxConcurrentToolRuns != 2 {
		t.Fatalf("migrated Scheduler settings = (%+v, %v), want 1 model and 2 tool lanes", current, err)
	}
	store := &postgresClaimAttempts{
		postgresLaneBlockingStore: newPostgresLaneBlockingStore(postgresStore),
		attempts:                  make(map[resolved.LaneClass]int), attempted: make(chan resolved.LaneClass, 64),
	}
	clock := newManualPollClock()
	scheduler := newPostgresSupervisor(t, pool, store, settings, clock)
	started := countLaneStarts(scheduler)
	run := func() (context.CancelFunc, chan error) {
		runContext, stop := context.WithCancel(ctx)
		done := make(chan error, 1)
		go func() { done <- scheduler.Run(runContext) }()
		return stop, done
	}

	stop, done := run()
	for round := 1; round <= 2; round++ {
		if round > 1 {
			clock.fireWhenArmedBefore(t, ctx, postgresSupervisorPoll, 2)
		}
		got := map[resolved.LaneClass]int{}
		for range len(resolved.LaneClasses) {
			got[awaitValue(t, ctx, store.attempted, "idle PostgreSQL claim")]++
		}
		if got[resolved.LaneClassModel] != 1 || got[resolved.LaneClassTool] != 1 {
			t.Fatalf("idle round %d claims = %v, want one per pool", round, got)
		}
	}
	stop()
	if err := awaitValue(t, ctx, done, "idle PostgreSQL Scheduler shutdown"); err != nil {
		t.Fatal(err)
	}
	if attempts := store.counts(); attempts[resolved.LaneClassModel] != 2 || attempts[resolved.LaneClassTool] != 2 ||
		started.Load() != 4 {
		t.Fatalf("idle claims = %v with %d lanes, want two per pool and four lanes", attempts, started.Load())
	}

	snapshot, err := configload.Load("../../testdata/configs-scan")
	if err != nil {
		t.Fatal(err)
	}
	toolWorkflow, err := snapshot.Workflow("nuclei-target@1")
	if err != nil {
		t.Fatal(err)
	}
	modelWorkflow := loadSchedulerWorkflow(t)
	runIDs := []string{"run-model-1", "run-tool-1", "run-tool-2", "run-model-2", "run-tool-3"}
	for _, runID := range runIDs {
		workflow, parameters := modelWorkflow, map[string]string{"objective": "hold lane"}
		if strings.HasPrefix(runID, "run-tool") {
			workflow, parameters = toolWorkflow, map[string]string{"target": "https://target.test"}
		}
		createRunningPoolRun(t, ctx, postgresStore, runID, workflow, parameters)
	}
	stop, done = run()
	claimed := map[string]bool{}
	for range 3 {
		claimed[awaitValue(t, ctx, store.claimed, "PostgreSQL claim")] = true
	}
	for range 3 {
		awaitValue(t, ctx, store.attempted, "PostgreSQL claim attempt")
	}
	if !claimed["run-model-1"] || !claimed["run-tool-1"] || !claimed["run-tool-2"] {
		t.Fatalf("claimed %v, want the first model and the first two tool Runs", claimed)
	}
	if active, maximum := store.postgresLaneBlockingStore.counts(); active != 3 || maximum != 3 {
		t.Fatalf("lanes holding Runs = %d (maximum %d), want all three before any completes", active, maximum)
	}
	stop()
	if err := awaitValue(t, ctx, done, "busy PostgreSQL Scheduler shutdown"); err != nil {
		t.Fatal(err)
	}
	if attempts := store.counts(); attempts[resolved.LaneClassModel] != 3 || attempts[resolved.LaneClassTool] != 4 ||
		started.Load() != 7 {
		t.Fatalf("claims = %v with %d lanes, want 3 model and 4 tool claims and seven lanes", attempts, started.Load())
	}
	for _, runID := range runIDs {
		current, err := postgresStore.GetRun(ctx, runID)
		if err != nil || current.State != runstore.RunRunning || current.SchedulerClaim != nil {
			t.Fatalf("Run %s after shutdown = (%s, claim %v, %v), want running without a claim",
				runID, current.State, current.SchedulerClaim, err)
		}
	}
}

// createRunningPoolRun creates a running Run of workflow whose entry Stage
// sets its lane class.
func createRunningPoolRun(
	t *testing.T, ctx context.Context, store *runstore.PostgresStore, runID string,
	workflow resolved.ResolvedWorkflow, parameters map[string]string,
) {
	t.Helper()
	workflowJSON, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runstoretest.CreateRun(ctx, store, runstore.CreateRunParams{
		LaneClass: resolved.EntryLaneClass(workflow),
		RunID:     runID, OwnerID: "user-refill", WorkflowName: workflow.Ref.Name,
		WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot: workflowJSON, Parameters: parameters,
		RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(
		ctx, runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"},
	); err != nil {
		t.Fatal(err)
	}
	if _, err := store.TransitionRun(
		ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"},
	); err != nil {
		t.Fatal(err)
	}
}
