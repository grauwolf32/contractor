package scheduler

import (
	"context"
	"errors"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type operationBudgetObservations struct {
	mu   sync.Mutex
	seen map[string]bool
}

func (o *operationBudgetObservations) record(name string, ctx context.Context) {
	_, bounded := ctx.Deadline()
	o.mu.Lock()
	defer o.mu.Unlock()
	if previous, ok := o.seen[name]; ok {
		bounded = bounded && previous
	}
	o.seen[name] = bounded
}
func (o *operationBudgetObservations) require(t *testing.T, names ...string) {
	t.Helper()
	o.mu.Lock()
	defer o.mu.Unlock()
	for _, name := range names {
		if !o.seen[name] {
			t.Errorf("%s had no individual operation deadline", name)
		}
	}
}

type budgetObservedStore struct {
	Store
	observed *operationBudgetObservations
}

func (s budgetObservedStore) GetRunState(ctx context.Context, id string) (runstore.WorkflowRunState, error) {
	s.observed.record("Run state", ctx)
	return s.Store.GetRunState(ctx, id)
}
func (s budgetObservedStore) GetActiveStageExecution(ctx context.Context, id string) (runstore.StageExecution, error) {
	s.observed.record("active Stage", ctx)
	return s.Store.GetActiveStageExecution(ctx, id)
}
func (s budgetObservedStore) ListStageAllocations(ctx context.Context, id string) ([]runstore.StageAllocation, error) {
	s.observed.record("allocations", ctx)
	return s.Store.ListStageAllocations(ctx, id)
}

type budgetObservedArtifacts struct {
	ArtifactResolver
	observed *operationBudgetObservations
}

func (r budgetObservedArtifacts) Resolve(ctx context.Context, id string, ref contracts.ArtifactRef) (ResolvedArtifact, error) {
	r.observed.record("artifact metadata", ctx)
	return r.ArtifactResolver.Resolve(ctx, id, ref)
}

type budgetObservedPersistence struct {
	AtomicPersistence
	observed *operationBudgetObservations
}

func (p budgetObservedPersistence) AdmitStage(ctx context.Context, runID, stageID string) (runstore.StageExecution, error) {
	p.observed.record("admission", ctx)
	return p.AtomicPersistence.AdmitStage(ctx, runID, stageID)
}
func (p budgetObservedPersistence) StartStageOperation(ctx context.Context, runID, stageID, claimID string) (runstore.StageExecution, error) {
	p.observed.record("operation start", ctx)
	return p.AtomicPersistence.(operationPersistence).StartStageOperation(ctx, runID, stageID, claimID)
}

// No outer deadline can mask an unbounded call here. Exercise the actual lane
// through its ordinary Store/ArtifactResolver and Planner admission boundaries.
func TestSchedulerBoundsIndividualStoreAndMetadataCalls(t *testing.T) {
	h := newSchedulerHarness(t)
	o := &operationBudgetObservations{seen: map[string]bool{}}
	options := h.scheduler.options
	credentials := options.Credentials
	options.Credentials = credentialResolverFunc(func(ctx context.Context, ref llmgateway.LLMCredentialRef, gateway llmgateway.LLMGatewayConfigRef) (contracts.SecretString, error) {
		o.record("credential", ctx)
		return credentials.ResolveLLMCredential(ctx, ref, gateway)
	})
	s, err := New(budgetObservedStore{h.store, o}, budgetObservedPersistence{h.persistence, o}, budgetObservedArtifacts{h.artifacts, o}, h.allocator, h.workers, h.planners, options)
	if err != nil {
		t.Fatal(err)
	}
	if worked, err := s.RunOnce(context.Background()); err != nil || !worked {
		t.Fatalf("RunOnce=%v %v", worked, err)
	}
	o.require(t, "Run state", "active Stage", "allocations", "artifact metadata", "admission", "credential")
}

func TestPostgresServerOperationAdmissionAndStartHaveIndividualBudgets(t *testing.T) {
	h := newOperationHarness(t, false)
	o := &operationBudgetObservations{seen: map[string]bool{}}
	s := h.scheduler(t, budgetObservedPersistence{h.transactions, o}, h.registry)
	if _, err := s.RunOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	o.require(t, "admission", "operation start")
}

type blockedStateStore struct{ Store }

func (blockedStateStore) GetRunState(ctx context.Context, _ string) (runstore.WorkflowRunState, error) {
	<-ctx.Done()
	return "", ctx.Err()
}

// A blocked store read ends at its own budget and releases the claim. The
// virtual clock makes this a deadline outcome check, not a wall-clock assertion.
func TestSchedulerBlockedStoreReadEndsAtOperationBudget(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := newSchedulerHarness(t)
		options := h.scheduler.options
		options.OperationTimeout = time.Millisecond
		s, err := New(blockedStateStore{h.store}, h.persistence, h.artifacts, h.allocator, h.workers, h.planners, options)
		if err != nil {
			t.Fatal(err)
		}
		if worked, err := s.RunOnce(context.Background()); !worked || !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("blocked read=%v %v", worked, err)
		}
		if h.store.claimID != "" || h.planners.runCalls != 0 {
			t.Fatalf("timed-out read kept its claim or started a Planner: claim=%q calls=%d", h.store.claimID, h.planners.runCalls)
		}
	})
}
