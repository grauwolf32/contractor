//go:build integration

package auditcontroller

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

// heldStepStore is the real claim store whose reconcile step is held at its
// snapshot read until the test lets it end. With untilCancelled the step also
// waits for its pass to be cancelled and ends with that cancellation;
// otherwise it ends with errHeldStep. Every release is recorded with the
// context it ran under and the store's answer.
type heldStepStore struct {
	Store
	untilCancelled bool
	entered        chan struct{}
	proceed        chan struct{}

	mu         sync.Mutex
	events     []string
	contextErr error
	result     error
}

var errHeldStep = errors.New("held reconcile step failed")

func (s *heldStepStore) GetReconcileSnapshot(
	ctx context.Context, _ auditstore.ControllerClaim,
) (auditstore.ReconcileSnapshot, error) {
	close(s.entered)
	if s.untilCancelled {
		<-ctx.Done()
	}
	<-s.proceed
	s.mu.Lock()
	s.events = append(s.events, "step ended")
	s.mu.Unlock()
	if s.untilCancelled {
		return auditstore.ReconcileSnapshot{}, ctx.Err()
	}
	return auditstore.ReconcileSnapshot{}, errHeldStep
}

func (s *heldStepStore) ReleaseClaim(ctx context.Context, claim auditstore.ControllerClaim) error {
	contextErr := ctx.Err()
	err := s.Store.ReleaseClaim(ctx, claim)
	s.mu.Lock()
	s.events = append(s.events, "released")
	s.contextErr, s.result = contextErr, err
	s.mu.Unlock()
	return err
}

func (h *postgresControllerHarness) auditClaim(t *testing.T, ctx context.Context) (holder *string, epoch int64) {
	t.Helper()
	if err := h.pool.QueryRow(ctx, `
SELECT holder_id, epoch FROM audit_controller_claims WHERE audit_id = $1`,
		h.started.Audit.AuditID).Scan(&holder, &epoch); err != nil {
		t.Fatal(err)
	}
	return holder, epoch
}

// A Controller pass cancelled during a reconcile step joins the step before
// it releases the claim, and the release commits on the real store under a
// live context, so the next holder can lease the Audit at once (PER-R26).
func TestPostgresControllerReleasesTheClaimOfACancelledStep(t *testing.T) {
	ctx, cancelTest := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancelTest()
	harness := newCommonControllerHarness(t)
	controller := harness.commonController(t)
	// Only the cancellation may end the step, never its own budget.
	controller.options.OperationTimeout = time.Minute
	held := &heldStepStore{
		Store: controller.store, untilCancelled: true, entered: make(chan struct{}), proceed: make(chan struct{}),
	}
	controller.store = held
	passContext, cancelPass := context.WithCancel(ctx)
	done := make(chan error, 1)
	go func() {
		_, err := controller.RunOnce(passContext)
		done <- err
	}()
	<-held.entered
	if holder, _ := harness.auditClaim(t, ctx); holder == nil || *holder != "postgres-audit-controller" {
		t.Fatalf("claim holder during the step = %v", holder)
	}
	cancelPass()
	close(held.proceed)
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled pass error = %v, want the step's cancellation", err)
	}
	held.mu.Lock()
	events, contextErr, result := slices.Clone(held.events), held.contextErr, held.result
	held.mu.Unlock()
	if !slices.Equal(events, []string{"step ended", "released"}) {
		t.Fatalf("events = %v, want the step joined before the release", events)
	}
	if contextErr != nil || result != nil {
		t.Fatalf("release ran with context error %v and returned %v", contextErr, result)
	}
	holder, epoch := harness.auditClaim(t, ctx)
	if holder != nil {
		t.Fatalf("claim still held by %s after the cancelled pass", *holder)
	}
	next, err := harness.audits.Claim(ctx, auditstore.ClaimParams{HolderID: "next-holder", Lease: 5 * time.Second, Limit: 1})
	if err != nil || len(next) != 1 || next[0].AuditID != harness.started.Audit.AuditID || int64(next[0].Epoch) != epoch+1 {
		t.Fatalf("next lease = %+v, %v, want the Audit at epoch %d", next, err, epoch+1)
	}
}

// A claim whose lease ended during a reconcile step and that another
// Controller took over is not released by the stale holder: the real store
// reports the claim lost, the pass reports only the step's own failure, and
// the new holder keeps its claim.
func TestPostgresControllerPassToleratesAClaimTakenOverDuringItsStep(t *testing.T) {
	ctx, cancelTest := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancelTest()
	harness := newCommonControllerHarness(t)
	controller := harness.commonController(t)
	held := &heldStepStore{Store: controller.store, entered: make(chan struct{}), proceed: make(chan struct{})}
	controller.store = held
	done := make(chan error, 1)
	go func() {
		_, err := controller.RunOnce(ctx)
		done <- err
	}()
	<-held.entered
	if _, err := harness.pool.Exec(ctx, `
UPDATE audit_controller_claims SET expires_at = clock_timestamp() WHERE audit_id = $1`,
		harness.started.Audit.AuditID); err != nil {
		t.Fatal(err)
	}
	rival, err := harness.audits.Claim(ctx, auditstore.ClaimParams{HolderID: "rival-holder", Lease: 20 * time.Second, Limit: 1})
	if err != nil || len(rival) != 1 || rival[0].AuditID != harness.started.Audit.AuditID {
		t.Fatalf("rival lease = %+v, %v", rival, err)
	}
	close(held.proceed)
	err = <-done
	if !errors.Is(err, errHeldStep) || errors.Is(err, auditstore.ErrClaimLost) {
		t.Fatalf("pass error = %v, want only the step's failure", err)
	}
	held.mu.Lock()
	result := held.result
	held.mu.Unlock()
	if !errors.Is(result, auditstore.ErrClaimLost) {
		t.Fatalf("stale release returned %v, want the store's lost claim", result)
	}
	holder, epoch := harness.auditClaim(t, ctx)
	if holder == nil || *holder != "rival-holder" || uint64(epoch) != rival[0].Epoch {
		t.Fatalf("claim after the stale release = %v at epoch %d, want rival-holder at %d", holder, epoch, rival[0].Epoch)
	}
	if err := harness.audits.ReleaseClaim(ctx, rival[0]); err != nil {
		t.Fatal(err)
	}
}
