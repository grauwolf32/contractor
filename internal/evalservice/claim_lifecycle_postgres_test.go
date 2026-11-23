package evalservice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/evalcoordinator"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// heldTicks holds the tick of a real claim until the test lets it end. With
// untilCancelled the tick also waits for its pass to be cancelled and ends
// with that cancellation; otherwise it ends with errHeldTick.
type heldTicks struct {
	untilCancelled bool
	entered        chan struct{}
	proceed        chan struct{}
	events         *eventLog
}

var errHeldTick = errors.New("held tick failed")

func (s *heldTicks) Tick(ctx context.Context, claim evalstore.Claim) (bool, error) {
	close(s.entered)
	if s.untilCancelled {
		<-ctx.Done()
	}
	<-s.proceed
	s.events.record("tick ended")
	if s.untilCancelled {
		return false, ctx.Err()
	}
	return false, errHeldTick
}

// recordedReleases is the real claim store that records every release with
// the context it ran under and the store's answer.
type recordedReleases struct {
	evalcoordinator.Store
	events     *eventLog
	mu         sync.Mutex
	contextErr error
	result     error
}

func (s *recordedReleases) ReleaseClaim(ctx context.Context, claim evalstore.Claim) error {
	contextErr := ctx.Err()
	err := s.Store.ReleaseClaim(ctx, claim)
	s.mu.Lock()
	s.contextErr, s.result = contextErr, err
	s.mu.Unlock()
	s.events.record("released")
	return err
}

type eventLog struct {
	mu     sync.Mutex
	events []string
}

func (l *eventLog) record(event string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, event)
}

func (l *eventLog) list() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return slices.Clone(l.events)
}

// claimableExperiment returns an experiment whose pending prepare command
// makes it claimable.
func (h *serviceHarness) claimableExperiment(t *testing.T) evalstore.Experiment {
	t.Helper()
	e := h.create(t, "workflow")
	h.command(t, e, "prepare")
	return h.get(t, e.ID)
}

func (h *serviceHarness) heldCoordinator(
	t *testing.T, ticks *heldTicks, releases *recordedReleases, operationTimeout time.Duration,
) *evalcoordinator.Coordinator {
	t.Helper()
	coordinator, err := evalcoordinator.New(releases, ticks, evalcoordinator.Options{
		HolderID: "held-coordinator", Lease: 5 * time.Minute, OperationTimeout: operationTimeout, Batch: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	return coordinator
}

func (h *serviceHarness) claimHolder(t *testing.T, experimentID string) (holder *string, epoch int64) {
	t.Helper()
	if err := h.pool.QueryRow(t.Context(), `
SELECT holder_id, epoch FROM eval_controller_claims WHERE experiment_id = $1`, experimentID).Scan(&holder, &epoch); err != nil {
		t.Fatal(err)
	}
	return holder, epoch
}

// A coordinator pass cancelled during a tick joins the tick before it
// releases the claim, and the release commits on the real store under a
// live context, so the next holder can lease the experiment at once
// (PER-R26).
func TestSharedPostgresCoordinatorReleasesTheClaimOfACancelledTick(t *testing.T) {
	h := newHarness(t)
	e := h.claimableExperiment(t)
	events := &eventLog{}
	ticks := &heldTicks{untilCancelled: true, entered: make(chan struct{}), proceed: make(chan struct{}), events: events}
	releases := &recordedReleases{Store: evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage()), events: events}
	// Only the cancellation may end the tick, never its own budget.
	coordinator := h.heldCoordinator(t, ticks, releases, 4*time.Minute)
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() {
		_, err := coordinator.RunOnce(ctx)
		done <- err
	}()
	<-ticks.entered
	if holder, _ := h.claimHolder(t, e.ID); holder == nil || *holder != "held-coordinator" {
		t.Fatalf("claim holder during the tick = %v", holder)
	}
	cancel()
	close(ticks.proceed)
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled pass error = %v, want the tick's cancellation", err)
	}
	if got := events.list(); !slices.Equal(got, []string{"tick ended", "released"}) {
		t.Fatalf("events = %v, want the tick joined before the release", got)
	}
	if releases.contextErr != nil || releases.result != nil {
		t.Fatalf("release ran with context error %v and returned %v", releases.contextErr, releases.result)
	}
	holder, epoch := h.claimHolder(t, e.ID)
	if holder != nil {
		t.Fatalf("claim still held by %s after the cancelled pass", *holder)
	}
	next, err := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage()).Claim(t.Context(), "next-coordinator", time.Minute, 1)
	if err != nil || len(next) != 1 || next[0].ExperimentID != e.ID || next[0].Epoch != epoch+1 {
		t.Fatalf("next lease = %+v, %v, want experiment %s at epoch %d", next, err, e.ID, epoch+1)
	}
}

// A claim whose lease ended during a tick and that another coordinator took
// over is not released by the stale holder: the real store reports the claim
// lost, the pass reports only the tick's own failure, and the new holder
// keeps its claim.
func TestSharedPostgresCoordinatorPassToleratesAClaimTakenOverDuringItsTick(t *testing.T) {
	h := newHarness(t)
	e := h.claimableExperiment(t)
	events := &eventLog{}
	ticks := &heldTicks{entered: make(chan struct{}), proceed: make(chan struct{}), events: events}
	releases := &recordedReleases{Store: evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage()), events: events}
	coordinator := h.heldCoordinator(t, ticks, releases, time.Minute)
	done := make(chan error, 1)
	go func() {
		_, err := coordinator.RunOnce(t.Context())
		done <- err
	}()
	<-ticks.entered
	if _, err := h.pool.Exec(t.Context(), `
UPDATE eval_controller_claims SET expires_at = clock_timestamp() WHERE experiment_id = $1`, e.ID); err != nil {
		t.Fatal(err)
	}
	rival, err := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage()).Claim(t.Context(), "rival-coordinator", time.Minute, 1)
	if err != nil || len(rival) != 1 || rival[0].ExperimentID != e.ID {
		t.Fatalf("rival lease = %+v, %v", rival, err)
	}
	close(ticks.proceed)
	err = <-done
	if !errors.Is(err, errHeldTick) || errors.Is(err, evalstore.ErrClaimLost) {
		t.Fatalf("pass error = %v, want only the tick's failure", err)
	}
	if !errors.Is(releases.result, evalstore.ErrClaimLost) {
		t.Fatalf("stale release returned %v, want the store's lost claim", releases.result)
	}
	holder, epoch := h.claimHolder(t, e.ID)
	if holder == nil || *holder != "rival-coordinator" || epoch != rival[0].Epoch {
		t.Fatalf("claim after the stale release = %v at epoch %d, want rival-coordinator at %d", holder, epoch, rival[0].Epoch)
	}
}

// journalClock answers its first reading with first and every later one
// with later, so a cancellation time read again instead of replayed from its
// journaled intent shows up as later.
type journalClock struct {
	mu           sync.Mutex
	readings     int
	first, later time.Time
}

func (c *journalClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.readings++
	if c.readings == 1 {
		return c.first
	}
	return c.later
}

func (c *journalClock) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.readings
}

func (h *serviceHarness) suboperation(t *testing.T, experimentID, kind string) (state string, request []byte, found bool) {
	t.Helper()
	err := h.pool.QueryRow(t.Context(), `
SELECT state, request FROM eval_suboperations WHERE experiment_id = $1 AND kind = $2`, experimentID, kind).Scan(&state, &request)
	if err != nil {
		return "", nil, false
	}
	return state, request, true
}

// tickUntil ticks until done, tolerating the failures the fixtures inject.
func (h *serviceHarness) tickUntil(t *testing.T, c *evalcoordinator.Coordinator, what string, done func() bool) {
	t.Helper()
	for range 10 {
		if done() {
			return
		}
		if _, err := c.RunOnce(t.Context()); err != nil &&
			!errors.Is(err, errResponseLost) && !errors.Is(err, errCancelUnavailable) {
			t.Fatal(err)
		}
	}
	if !done() {
		t.Fatalf("ten ticks did not reach %s", what)
	}
}

func journaledRequestTime(t *testing.T, request []byte) time.Time {
	t.Helper()
	var journaled cancellationRequest
	if err := json.Unmarshal(request, &journaled); err != nil {
		t.Fatal(err)
	}
	return journaled.RequestedAt
}

// failingFirstCancel fails the first Audit cancellation before it reaches the
// Audit service, as a lost connection would, and records every attempt. A
// non-nil move runs another mutation of the Audit before that failure.
type failingFirstCancel struct {
	AuditExecutor
	move     func(context.Context, string) error
	mu       sync.Mutex
	attempts []auditservice.MutationParams
}

var errCancelUnavailable = errors.New("fixture: Audit cancellation unavailable")

func (a *failingFirstCancel) Cancel(ctx context.Context, p auditservice.MutationParams) (auditservice.MutationResult, error) {
	a.mu.Lock()
	a.attempts = append(a.attempts, p)
	first := len(a.attempts) == 1
	a.mu.Unlock()
	if first {
		if a.move != nil {
			if err := a.move(ctx, p.AuditID); err != nil {
				return auditservice.MutationResult{}, err
			}
		}
		return auditservice.MutationResult{}, errCancelUnavailable
	}
	return a.AuditExecutor.Cancel(ctx, p)
}

// A member's cancellation reads the Driver clock once, when it journals the
// intent. The Run cancellation and every replay of an interrupted request
// carry that journaled time, never a later reading. An Audit member's retry
// repeats its idempotency key and request digest only while the Audit
// revision is unchanged; after the revision moved it sends new ones for the
// same journaled request (EVM-R115).
func TestSharedPostgresCancellationReplaysItsJournaledRequestTime(t *testing.T) {
	first := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)
	later := first.Add(24 * time.Hour)

	t.Run("workflow", func(t *testing.T) {
		h := newHarness(t)
		clock := &journalClock{first: first, later: later}
		h.driver.Now = clock.now
		e := h.prepared(t, "workflow")
		h.command(t, e, "start")
		c := h.coordinator(t, "controller")
		h.tickUntil(t, c, "an accepted Run submission", func() bool {
			var accepted int
			if err := h.pool.QueryRow(t.Context(), `
SELECT count(*) FROM eval_submissions WHERE experiment_id = $1 AND state = 'accepted'`, e.ID).Scan(&accepted); err != nil {
				t.Fatal(err)
			}
			return accepted == 1
		})
		var runID string
		if err := h.pool.QueryRow(t.Context(), `SELECT run_id FROM workflow_runs`).Scan(&runID); err != nil {
			t.Fatal(err)
		}
		h.command(t, h.get(t, e.ID), "cancel")

		// Hold the Run row so the first cancellation request cannot commit
		// after its intent was journaled.
		lock, err := h.pool.Begin(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		if _, err = lock.Exec(t.Context(), `SELECT 1 FROM workflow_runs WHERE run_id = $1 FOR UPDATE`, runID); err != nil {
			t.Fatal(err)
		}
		interrupted, err := evalcoordinator.New(evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage()), h.service, evalcoordinator.Options{
			HolderID: "interrupted", Lease: 5 * time.Second, OperationTimeout: time.Second,
		})
		if err != nil {
			t.Fatal(err)
		}
		_, _ = interrupted.RunOnce(t.Context())
		if err = lock.Rollback(t.Context()); err != nil {
			t.Fatal(err)
		}
		state, request, found := h.suboperation(t, e.ID, "cancel")
		if !found || state != "intent" || !journaledRequestTime(t, request).Equal(first) {
			t.Fatalf("interrupted cancellation journal = (%q, %s, %t), want an intent stamped %s", state, request, found, first)
		}

		h.tickUntil(t, c, "an acknowledged Run cancellation", func() bool {
			state, _, _ := h.suboperation(t, e.ID, "cancel")
			return state == "succeeded"
		})
		run, err := runstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage()).GetRun(t.Context(), runID)
		if err != nil {
			t.Fatal(err)
		}
		if run.Cancellation == nil || !run.Cancellation.RequestedAt.Equal(first) {
			t.Fatalf("Run cancellation = %+v, want the journaled time %s", run.Cancellation, first)
		}
		if _, request, _ = h.suboperation(t, e.ID, "cancel"); !journaledRequestTime(t, request).Equal(first) || clock.count() != 1 {
			t.Fatalf("journal %s after %d clock readings, want one reading stamped %s", request, clock.count(), first)
		}
	})

	for _, test := range []struct {
		name  string
		moved bool
	}{{name: "audit"}, {name: "audit after its revision moved", moved: true}} {
		t.Run(test.name, func(t *testing.T) {
			h := newHarness(t)
			clock := &journalClock{first: first, later: later}
			h.driver.Now = clock.now
			e := h.prepared(t, "audit")
			h.command(t, e, "start")
			c := h.coordinator(t, "controller")
			h.tickUntil(t, c, "a started Audit", func() bool {
				state, _, _ := h.suboperation(t, e.ID, "audit-start")
				return state == "succeeded"
			})
			cancels := &failingFirstCancel{AuditExecutor: h.driver.Audits}
			if test.moved {
				cancels.move = func(ctx context.Context, auditID string) error {
					_, err := h.pool.Exec(ctx, `UPDATE audits SET revision = revision + 1 WHERE audit_id = $1`, auditID)
					return err
				}
			}
			h.driver.Audits = cancels
			h.command(t, h.get(t, e.ID), "cancel")
			var journaled []byte
			h.tickUntil(t, c, "an acknowledged Audit cancellation", func() bool {
				state, request, _ := h.suboperation(t, e.ID, "cancel")
				if journaled == nil && request != nil {
					journaled = request
				}
				return state == "succeeded"
			})
			_, request, _ := h.suboperation(t, e.ID, "cancel")
			if !bytes.Equal(request, journaled) || !journaledRequestTime(t, request).Equal(first) || clock.count() != 1 {
				t.Fatalf("journal %s (first %s) after %d clock readings, want one unchanged request stamped %s", request, journaled, clock.count(), first)
			}
			cancels.mu.Lock()
			defer cancels.mu.Unlock()
			if len(cancels.attempts) != 2 {
				t.Fatalf("Audit cancellation attempts = %+v, want the interrupted one and its retry", cancels.attempts)
			}
			interrupted, retry := cancels.attempts[0], cancels.attempts[1]
			sameKey := interrupted.IdempotencyKey == retry.IdempotencyKey && interrupted.RequestDigest == retry.RequestDigest
			if moved := retry.ExpectedRevision == interrupted.ExpectedRevision+1; moved != test.moved || sameKey == test.moved {
				t.Fatalf("Audit cancellation attempts = %+v, want the key and digest repeated exactly while the revision is unchanged", cancels.attempts)
			}
		})
	}
}
