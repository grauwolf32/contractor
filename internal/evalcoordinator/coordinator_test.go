package evalcoordinator

import (
	"context"
	"errors"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"sync"
	"testing"
	"time"
)

type testClaims struct {
	mu       sync.Mutex
	claims   []evalstore.Claim
	released map[string]bool
}

func (s *testClaims) Claim(context.Context, string, time.Duration, int) ([]evalstore.Claim, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := s.claims
	s.claims = nil
	return out, nil
}
func (s *testClaims) ReleaseClaim(ctx context.Context, c evalstore.Claim) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.released[c.ExperimentID] = ctx.Err() == nil
	return nil
}

type blockingTicks struct {
	mu      sync.Mutex
	started map[string]bool
}

func (s *blockingTicks) Tick(ctx context.Context, c evalstore.Claim) (bool, error) {
	s.mu.Lock()
	s.started[c.ExperimentID] = true
	s.mu.Unlock()
	<-ctx.Done()
	return false, ctx.Err()
}

func TestCoordinatorReleasesEveryClaimAfterBoundedOperations(t *testing.T) {
	store := &testClaims{claims: []evalstore.Claim{{ExperimentID: "first"}, {ExperimentID: "second"}}, released: map[string]bool{}}
	service := &blockingTicks{started: map[string]bool{}}
	c, err := New(store, service, Options{Lease: time.Second, OperationTimeout: 20 * time.Millisecond, Batch: 2})
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.RunOnce(t.Context())
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	if len(service.started) != 2 || !store.released["first"] || !store.released["second"] {
		t.Fatal("one failed operation stranded another claim")
	}
}

func TestCoordinatorShutdownReleasesWithIndependentContext(t *testing.T) {
	store := &testClaims{claims: []evalstore.Claim{{ExperimentID: "first"}}, released: map[string]bool{}}
	service := &blockingTicks{started: map[string]bool{}}
	c, err := New(store, service, Options{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err = c.Run(ctx); err != nil {
		t.Fatal(err)
	}
	if !store.released["first"] {
		t.Fatal("shutdown used cancelled release context")
	}
}
