package scheduler

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// cancellationCleanupReads counts the Stage-history and allocation reads of a
// cancelling Run's cleanup (SCH-R179). The next unreleased-history and batch
// reads fail with the injected errors, once each.
type cancellationCleanupReads struct {
	Store
	histories, unreleased, allocationReads int
	batchIDs                               [][]string
	unreleasedErr, batchErr                error
}

func (s *cancellationCleanupReads) HasStageExecutions(ctx context.Context, runID string) (bool, error) {
	s.histories++
	return s.Store.HasStageExecutions(ctx, runID)
}

func (s *cancellationCleanupReads) ListStageLineage(
	ctx context.Context, runID, stageName string,
) ([]runstore.StageLineage, error) {
	s.histories++
	return s.Store.ListStageLineage(ctx, runID, stageName)
}

func (s *cancellationCleanupReads) ListStageExecutionsWithUnreleasedAllocations(
	ctx context.Context, runID string,
) ([]runstore.StageExecution, error) {
	s.unreleased++
	if err := s.unreleasedErr; err != nil {
		s.unreleasedErr = nil
		return nil, err
	}
	return s.Store.ListStageExecutionsWithUnreleasedAllocations(ctx, runID)
}

func (s *cancellationCleanupReads) ListStageAllocations(ctx context.Context, id string) ([]runstore.StageAllocation, error) {
	s.allocationReads++
	return s.Store.ListStageAllocations(ctx, id)
}

func (s *cancellationCleanupReads) ListStageAllocationsBatch(
	ctx context.Context, ids []string,
) (map[string][]runstore.StageAllocation, error) {
	s.batchIDs = append(s.batchIDs, slices.Clone(ids))
	if err := s.batchErr; err != nil {
		s.batchErr = nil
		return nil, err
	}
	return s.Store.ListStageAllocationsBatch(ctx, ids)
}

const (
	historicalAttempts        = 22
	releasedHistoricalAttempt = 20
)

// historicalCancellation is run-1 on PostgreSQL with 22 terminal attempts of
// its Stage and no active one: attempts 1-20 completed their release, the
// grant of attempt 21 is gone without a completed release (the process ended
// mid-release), and attempt 22 still holds its live grant. Its Run is then
// cancelled.
type historicalCancellation struct {
	deferredPlacementHarness
	reads       *cancellationCleanupReads
	allocations map[string]string // StageExecution ID -> allocation ID
}

func historicalAttemptID(attempt int) string { return fmt.Sprintf("stage-history-%02d", attempt) }

func newHistoricalCancellation(t *testing.T, ctx context.Context) historicalCancellation {
	t.Helper()
	h := newDeferredPlacementHarness(t, ctx, nil)
	allocator := h.workers.allocator
	allocations := createTerminalAttempts(t, ctx, h, historicalAttempts,
		func(attempt int, id string, reservations []controlplane.Reservation) {
			switch {
			case attempt <= releasedHistoricalAttempt:
				if err := h.scheduler.releaseTerminal(id, reservations); err != nil {
					t.Fatalf("release attempt %d: %v", attempt, err)
				}
			case attempt == releasedHistoricalAttempt+1:
				delete(allocator.grants, reservations[0].Grant.AllocationID)
				allocator.cached = nil
			}
		})
	requireUnreleasedAllocations(t, ctx, h, allocations,
		historicalAttemptID(historicalAttempts-1), historicalAttemptID(historicalAttempts))
	requestTestCancellation(t, ctx, h.store)
	reads := &cancellationCleanupReads{Store: h.store}
	h.scheduler.store = reads
	return historicalCancellation{deferredPlacementHarness: h, reads: reads, allocations: allocations}
}

// createTerminalAttempts creates count interrupted attempts of run-1's Stage
// on PostgreSQL, named by historicalAttemptID. Each is placed through the
// harness allocator, which pins its durable allocation row, and ended with
// its grant still live; ended then sees the attempt before the next one is
// created. It returns each attempt's allocation ID by StageExecution ID.
func createTerminalAttempts(
	t *testing.T, ctx context.Context, h deferredPlacementHarness, count int,
	ended func(attempt int, id string, reservations []controlplane.Reservation),
) map[string]string {
	t.Helper()
	persistence, err := NewPostgresPersistence(h.pool)
	if err != nil {
		t.Fatal(err)
	}
	runArtifacts, err := artifacts.NewService(artifactpostgres.NewRepository(h.pool)).Run("run-1")
	if err != nil {
		t.Fatal(err)
	}
	input, err := runArtifacts.Metadata(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
	if err != nil {
		t.Fatal(err)
	}
	run, err := h.store.GetRun(ctx, "run-1")
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := decodeExecutableWorkflow(run)
	if err != nil {
		t.Fatal(err)
	}
	spec, err := json.Marshal(workflow.stage)
	if err != nil {
		t.Fatal(err)
	}
	pins := []ContextPin{{Name: "source", Ref: input.Ref}}
	stageContext := runstore.StageContextSnapshot{
		Parameters: map[string]string{"objective": "copy exactly"},
		Artifacts:  map[string]runstore.PinnedContextArtifact{"source": {Required: true, Artifact: &input.Ref}},
	}
	allocations := make(map[string]string, count)
	var previous *string
	for attempt := 1; attempt <= count; attempt++ {
		id := historicalAttemptID(attempt)
		if _, err := persistence.CreateStageWithContext(ctx, runstore.CreateStageExecutionParams{
			LaneClass:        resolved.LaneClassModel,
			StageExecutionID: id, RunID: "run-1", StageName: workflow.stageName, Attempt: attempt,
			PreviousExecutionID: previous, ExecutionConfigVariant: runstore.StageExecutionConfigBase,
			StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: spec,
			StageContextSchemaVersion: contracts.APIVersion, StageContext: stageContext,
		}, pins); err != nil {
			t.Fatalf("create attempt %d: %v", attempt, err)
		}
		previous = &id
		execution, err := h.store.GetStageExecution(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		requirements, err := auditBindingRequirements(run, workflow, execution.StageContext)
		if err != nil {
			t.Fatal(err)
		}
		reservations, err := h.workers.allocator.ReserveAllContext(ctx, stagePlacementRequest(run, workflow, execution, requirements))
		if err != nil || len(reservations) != 1 {
			t.Fatalf("place attempt %d = (%d, %v)", attempt, len(reservations), err)
		}
		allocations[id] = reservations[0].Grant.AllocationID
		if err := persistence.EnterAbortingWithTermination(ctx, "run-1", runstore.EnterAbortingParams{
			StageExecutionID: id, ExpectedState: runstore.StagePreparing,
			TerminationSchemaVersion: contracts.APIVersion,
			Termination: runstore.StageTermination{
				Outcome: runstore.TerminationInterrupted, Code: "runtime_unavailable",
				Message: "Runtime Agent became unavailable", Retryable: true,
				Phase: runstore.TerminationPreparing, OccurredAt: time.Now().UTC(),
			},
			AbortID: "abort-" + id, Deadline: time.Now().Add(time.Minute),
			Reason: runstore.Reason{Code: "stage_interrupted"},
		}); err != nil {
			t.Fatalf("abort attempt %d: %v", attempt, err)
		}
		if err := h.store.CompleteStageTermination(ctx, id); err != nil {
			t.Fatalf("end attempt %d: %v", attempt, err)
		}
		ended(attempt, id, reservations)
	}
	return allocations
}

// requireUnreleasedAllocations checks that exactly the allocations of the
// given StageExecutions have no completed release.
func requireUnreleasedAllocations(
	t *testing.T, ctx context.Context, h deferredPlacementHarness, allocations map[string]string, want ...string,
) {
	t.Helper()
	ids := make([]string, 0, len(allocations))
	for id := range allocations {
		ids = append(ids, id)
	}
	rows, err := h.store.ListStageAllocationsBatch(ctx, ids)
	if err != nil {
		t.Fatal(err)
	}
	var unreleased []string
	for id, recorded := range rows {
		if len(recorded) != 1 || recorded[0].AllocationID != allocations[id] {
			t.Fatalf("allocations of %s = %+v", id, recorded)
		}
		if recorded[0].ReleaseCompletedAt == nil {
			unreleased = append(unreleased, id)
		}
	}
	slices.Sort(unreleased)
	if len(rows) != len(allocations) || !slices.Equal(unreleased, want) {
		t.Fatalf("StageExecutions with an incomplete release = %v, want %v", unreleased, want)
	}
}

func requireRunState(t *testing.T, ctx context.Context, h deferredPlacementHarness, want runstore.WorkflowRunState) {
	t.Helper()
	run, err := h.store.GetRun(ctx, "run-1")
	if err != nil || run.State != want {
		t.Fatalf("Run = (%s %+v, %v), want %s", run.State, run.StateReason, err, want)
	}
	if want == runstore.RunCancelled && run.StateReason.Code != runstore.CancellationUserRequested {
		t.Fatalf("cancelled Run reason = %+v", run.StateReason)
	}
}

// A cancelling Run without an active Stage reads only the attempts whose
// release is incomplete and their allocations in one statement, never the
// history of completed releases. It fences and releases the live grant, ends
// the Run cancelled, and leaves the release whose grant is already gone to
// terminal release recovery, which completes it (SCH-R179, SCH-R71).
func TestPostgresCancellationReadsOnlyIncompleteReleases(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	f := newHistoricalCancellation(t, ctx)
	live := f.allocations[historicalAttemptID(historicalAttempts)]
	releaseCalls := f.workers.releaseCalls

	worked, err := f.scheduler.progressOneClaim(ctx, resolved.LaneClassModel, nil)
	if err != nil || !worked {
		t.Fatalf("cancellation claim = (%v, %v)", worked, err)
	}

	requireRunState(t, ctx, f.deferredPlacementHarness, runstore.RunCancelled)
	pending := []string{historicalAttemptID(historicalAttempts - 1), historicalAttemptID(historicalAttempts)}
	if f.reads.histories != 0 || f.reads.allocationReads != 0 || f.reads.unreleased != 1 ||
		len(f.reads.batchIDs) != 1 || !slices.Equal(f.reads.batchIDs[0], pending) {
		t.Fatalf("cleanup reads = other history reads %d, per-Stage allocation reads %d, unreleased histories %d, batches %v; "+
			"want one unreleased history and one batch of %v",
			f.reads.histories, f.reads.allocationReads, f.reads.unreleased, f.reads.batchIDs, pending)
	}
	if !f.workers.allocator.fenced[live] || f.workers.releaseCalls != releaseCalls+1 {
		t.Fatalf("live allocation fenced=%t, release batches=%d", f.workers.allocator.fenced[live],
			f.workers.releaseCalls-releaseCalls)
	}
	if _, err := f.workers.allocator.GetGrant(live); !errors.Is(err, controlplane.ErrAllocationNotFound) {
		t.Fatalf("live grant after cancellation = %v", err)
	}
	requireUnreleasedAllocations(t, ctx, f.deferredPlacementHarness, f.allocations,
		historicalAttemptID(historicalAttempts-1))

	if worked, err := f.scheduler.recoverTerminalRelease(ctx); err != nil || !worked {
		t.Fatalf("terminal release recovery = (%v, %v)", worked, err)
	}
	requireUnreleasedAllocations(t, ctx, f.deferredPlacementHarness, f.allocations)
}

// Each failure of the cleanup keeps its outcome: a failed history read keeps
// the Run cancelling for the next claim; a failed allocation read and a failed
// Runtime release end the Run cancelled and leave the release to terminal
// release recovery, which completes it once the Runtime answers.
func TestPostgresCancellationCleanupFailuresKeepRetries(t *testing.T) {
	for _, test := range []struct {
		name   string
		inject func(historicalCancellation)
		// unreleased are the attempts whose release recovery completes.
		unreleased []int
	}{
		{"history read fails", func(f historicalCancellation) {
			f.reads.unreleasedErr = errors.New("injected history read failure")
		}, []int{historicalAttempts - 1}},
		{"allocation read fails", func(f historicalCancellation) {
			f.reads.batchErr = errors.New("injected allocation read failure")
		}, []int{historicalAttempts - 1, historicalAttempts}},
		{"Runtime release fails", func(f historicalCancellation) {
			f.workers.releaseErrors = map[string]error{
				f.allocations[historicalAttemptID(historicalAttempts)]: errors.New("injected release failure"),
			}
		}, []int{historicalAttempts - 1, historicalAttempts}},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			f := newHistoricalCancellation(t, ctx)
			test.inject(f)

			worked, err := f.scheduler.progressOneClaim(ctx, resolved.LaneClassModel, nil)
			if f.reads.histories != 0 || f.reads.allocationReads != 0 {
				t.Fatalf("cleanup read the other Stage history %d times and per-Stage allocations %d times",
					f.reads.histories, f.reads.allocationReads)
			}
			if test.name == "history read fails" {
				if err == nil {
					t.Fatal("failed history read ended the claim without an error")
				}
				requireRunState(t, ctx, f.deferredPlacementHarness, runstore.RunCancelling)
				worked, err = f.scheduler.progressOneClaim(ctx, resolved.LaneClassModel, nil)
			}
			if err != nil || !worked {
				t.Fatalf("cancellation claim = (%v, %v)", worked, err)
			}
			requireRunState(t, ctx, f.deferredPlacementHarness, runstore.RunCancelled)
			want := make([]string, len(test.unreleased))
			for index, attempt := range test.unreleased {
				want[index] = historicalAttemptID(attempt)
			}
			requireUnreleasedAllocations(t, ctx, f.deferredPlacementHarness, f.allocations, want...)

			// One recovery listing takes every pending attempt: there are
			// fewer than its operation bound (SCH-R180).
			f.workers.releaseErrors = nil
			if worked, err := f.scheduler.recoverTerminalRelease(ctx); err != nil || !worked {
				t.Fatalf("terminal release recovery = (%v, %v)", worked, err)
			}
			requireUnreleasedAllocations(t, ctx, f.deferredPlacementHarness, f.allocations)
			if worked, err := f.scheduler.recoverTerminalRelease(ctx); err != nil || worked {
				t.Fatalf("terminal release recovery after completion = (%v, %v), want no work", worked, err)
			}
			if _, err := f.workers.allocator.GetGrant(f.allocations[historicalAttemptID(historicalAttempts)]); err == nil {
				t.Fatal("live grant outlived terminal release recovery")
			}
		})
	}
}

// countingCredentials counts LLM credential resolutions in front of another
// resolver.
type countingCredentials struct {
	CredentialResolver
	calls int
}

func (c *countingCredentials) ResolveLLMCredential(
	ctx context.Context, ref llmgateway.LLMCredentialRef, gateway llmgateway.LLMGatewayConfigRef,
) (contracts.SecretString, error) {
	c.calls++
	return c.CredentialResolver.ResolveLLMCredential(ctx, ref, gateway)
}

// On PostgreSQL, a Stage's metrics are visible once its Planner completes,
// its final report persistence rebuilds them once, and the persisted Worker
// report is redacted exactly, whether the Stage kept its preparation secrets
// or lost them and resolved its credential again (SCH-R178, SCH-R51).
func TestPostgresStageEndReportPersistence(t *testing.T) {
	const workerToken = "scheduler-test-token" // the harness's static Worker credential
	for _, test := range []struct {
		name        string
		loseSecrets bool
		resolutions int
	}{
		{"preparation secrets", false, 1},
		{"preparation secrets lost", true, 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			h := newDeferredPlacementHarness(t, ctx, nil)
			repository := telemetry.NewRepository(h.pool)
			reports := &stageEndReports{inner: repository}
			credentials := &countingCredentials{CredentialResolver: h.scheduler.options.Credentials}
			options := h.scheduler.options
			options.Credentials, options.Reports = credentials, reports
			var (
				scheduler    *Scheduler
				stageID      string
				atFinalizing map[string]telemetry.StageMetricsRecord
				observeErr   error
			)
			workers := stageEndWorkers{memoryWorkers: h.workers, secrets: []string{workerToken}, beforeFinalize: func() {
				execution, err := h.store.GetActiveStageExecution(ctx, "run-1")
				if err != nil {
					observeErr = err
					return
				}
				stageID = execution.StageExecutionID
				atFinalizing, observeErr = repository.GetStageMetricsBatch(ctx, []string{stageID})
				if test.loseSecrets {
					scheduler.preparedSecrets.forget(stageID)
				}
			}}
			var err error
			scheduler, err = New(h.store, h.scheduler.persistence, h.scheduler.artifacts, h.scheduler.allocator,
				workers, h.scheduler.planners, options)
			if err != nil {
				t.Fatal(err)
			}

			if worked, err := scheduler.RunOnce(ctx); err != nil || !worked {
				t.Fatalf("RunOnce = (%v, %v)", worked, err)
			}

			requireRunState(t, ctx, h, runstore.RunSucceeded)
			if observeErr != nil || atFinalizing[stageID].Metrics.Planner == nil {
				t.Fatalf("metrics when finalization began = (%+v, %v), want the completed Planner's", atFinalizing, observeErr)
			}
			if credentials.calls != test.resolutions {
				t.Fatalf("Worker credential resolutions = %d, want %d", credentials.calls, test.resolutions)
			}
			want := []string{"planner_report", "rebuild", "planner_report", "allocation_report", "rebuild"}
			if !slices.Equal(reports.ops, want) {
				t.Fatalf("report operations = %v, want %v", reports.ops, want)
			}
			stored, err := repository.ListAllocationReports(ctx, stageID)
			if err != nil || len(stored) != 1 {
				t.Fatalf("persisted allocation reports = (%d, %v)", len(stored), err)
			}
			requireStageEndRedaction(t, stored[0].Report, []string{workerToken})
			metrics, err := repository.GetStageMetricsBatch(ctx, []string{stageID})
			if err != nil || metrics[stageID].Metrics.Planner == nil || len(metrics[stageID].Metrics.Workers) != 1 {
				t.Fatalf("final StageMetrics = (%+v, %v), want the Planner and the Worker", metrics, err)
			}
			requireNoPreparedSecrets(t, scheduler)
		})
	}
}
