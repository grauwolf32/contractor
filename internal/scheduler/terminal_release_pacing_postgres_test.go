package scheduler

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type pacedBehaviour int

const (
	pacedHealthy  pacedBehaviour = iota
	pacedBlocked                 // answers only when the test finishes it
	pacedConflict                // answers HTTP 409 at once
	pacedLost                    // unreachable, and its control lease has expired
)

// pacedRelease is the Scheduler's allocator and the production Runtime batch
// controller's allocation registry for terminal release recovery. It keeps
// the live grants and paces each Stage's Runtime release as the test says,
// safely for concurrent Stage operations (SCH-R126). It also records the
// Stage operations in flight, the Stages whose release gate the Scheduler
// holds less those a test lane owns.
type pacedRelease struct {
	scheduler *Scheduler

	mu          sync.Mutex
	live        map[string]controlplane.Reservation // by allocation ID
	lost        map[string]bool                     // by allocation ID
	behaviour   map[string]pacedBehaviour           // by StageExecution ID
	gates       map[string]chan error               // a blocked Stage's Runtime answer
	calls       map[string]int                      // Runtime release calls by Stage
	laneOwned   map[string]bool                     // gates a test lane holds
	maxInFlight int

	entered  chan string // a Runtime release began, by StageExecution ID
	returned chan string // a Runtime release returned
	released chan string // a durable release completed
}

func newPacedRelease() *pacedRelease {
	return &pacedRelease{
		live: make(map[string]controlplane.Reservation), lost: make(map[string]bool),
		behaviour: make(map[string]pacedBehaviour), gates: make(map[string]chan error),
		calls: make(map[string]int), laneOwned: make(map[string]bool),
		entered: make(chan string, 64), returned: make(chan string, 64), released: make(chan string, 64),
	}
}

func (p *pacedRelease) set(stageExecutionID string, behaviour pacedBehaviour) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.behaviour[stageExecutionID] = behaviour
	if behaviour == pacedBlocked {
		p.gates[stageExecutionID] = make(chan error, 1)
	}
}

// finish answers the blocked Runtime release of a Stage with err.
func (p *pacedRelease) finish(stageExecutionID string, err error) {
	p.mu.Lock()
	gate := p.gates[stageExecutionID]
	p.mu.Unlock()
	gate <- err
}

func (p *pacedRelease) loseRuntime(allocationID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.lost[allocationID] = true
}

func (p *pacedRelease) callCount(stageExecutionID string) int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls[stageExecutionID]
}

func (p *pacedRelease) isLive(allocationID string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	_, live := p.live[allocationID]
	return live
}

// inFlight samples the recovery Stage operations in flight and returns them.
func (p *pacedRelease) inFlight() int {
	p.scheduler.releaseMu.Lock()
	gates := make([]string, 0, len(p.scheduler.releasing))
	for stageExecutionID := range p.scheduler.releasing {
		gates = append(gates, stageExecutionID)
	}
	p.scheduler.releaseMu.Unlock()
	p.mu.Lock()
	defer p.mu.Unlock()
	operations := 0
	for _, stageExecutionID := range gates {
		if !p.laneOwned[stageExecutionID] {
			operations++
		}
	}
	p.maxInFlight = max(p.maxInFlight, operations)
	return operations
}

func (p *pacedRelease) observedMaximum() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.maxInFlight
}

func (p *pacedRelease) ReserveAllContext(context.Context, controlplane.ReservationRequest) ([]controlplane.Reservation, error) {
	return nil, errors.New("unexpected placement")
}

func (p *pacedRelease) GetGrant(allocationID string) (controlplane.AllocationGrant, error) {
	reservation, err := p.GetReservation(allocationID)
	return reservation.Grant, err
}

func (p *pacedRelease) GetReservation(allocationID string) (controlplane.Reservation, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	reservation, live := p.live[allocationID]
	if !live {
		return controlplane.Reservation{}, controlplane.ErrAllocationNotFound
	}
	return reservation, nil
}

func (p *pacedRelease) SetWriteFence(allocationID string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	reservation, live := p.live[allocationID]
	if !live {
		return controlplane.ErrAllocationNotFound
	}
	reservation.Grant.WriteFenced = true
	p.live[allocationID] = reservation
	return nil
}

func (p *pacedRelease) DiscardUnprepared(string) error {
	return errors.New("unexpected placement discard")
}

func (p *pacedRelease) PollAllocationLosses() []controlplane.AllocationLoss { return nil }

func (p *pacedRelease) PrepareReservation(controlplane.Reservation) (controlplane.Reservation, error) {
	return controlplane.Reservation{}, errors.New("unexpected preparation")
}

func (p *pacedRelease) SetAllocationPhase(allocationID string, _ controlplane.AllocationAuthoritativePhase, _ *controlplane.SafeReason) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, live := p.live[allocationID]; !live {
		return controlplane.ErrAllocationNotFound
	}
	return nil
}

func (p *pacedRelease) RecordAllocationReport(string, reporting.AllocationFinalReport) error {
	return errors.New("unexpected allocation report")
}

func (p *pacedRelease) Release(allocationID string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, live := p.live[allocationID]; !live {
		return controlplane.ErrAllocationNotFound
	}
	delete(p.live, allocationID)
	return nil
}

// ReleaseLost removes a grant only when its Runtime's control lease expired,
// as the Control Plane registry does.
func (p *pacedRelease) ReleaseLost(allocationID string) (bool, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, live := p.live[allocationID]; !live {
		return false, controlplane.ErrAllocationNotFound
	}
	if !p.lost[allocationID] {
		return false, nil
	}
	delete(p.live, allocationID)
	return true, nil
}

// pacedRuntime is the Runtime the production batch controller releases
// through.
type pacedRuntime struct{ paced *pacedRelease }

func (r pacedRuntime) Prepare(context.Context, controlplane.Reservation, runtimesettings.WorkerExecutionSettings) (contracts.WorkerHandle, error) {
	return contracts.WorkerHandle{}, errors.New("unexpected Runtime prepare")
}

func (r pacedRuntime) Finalize(context.Context, controlplane.Reservation, string, time.Time) (reporting.AllocationFinalReport, error) {
	return reporting.AllocationFinalReport{}, errors.New("unexpected Runtime finalize")
}

func (r pacedRuntime) Abort(context.Context, controlplane.Reservation, string, contracts.TerminationError, time.Time) (reporting.AllocationFinalReport, error) {
	return reporting.AllocationFinalReport{}, errors.New("unexpected Runtime abort")
}

func (r pacedRuntime) Release(ctx context.Context, reservation controlplane.Reservation) error {
	stageExecutionID := reservation.Grant.StageExecutionID
	p := r.paced
	p.mu.Lock()
	p.calls[stageExecutionID]++
	behaviour, gate := p.behaviour[stageExecutionID], p.gates[stageExecutionID]
	p.mu.Unlock()
	p.inFlight()
	p.entered <- stageExecutionID
	defer func() { p.returned <- stageExecutionID }()
	switch behaviour {
	case pacedBlocked:
		select {
		case err := <-gate:
			return err
		case <-ctx.Done():
			return ctx.Err()
		}
	case pacedConflict:
		// A Runtime conflict answers at once; it says nothing about the
		// network and costs no cleanup deadline.
		return &controlplane.RuntimeAPIError{StatusCode: http.StatusConflict, Code: "allocation_state_conflict"}
	case pacedLost:
		return errors.New("Runtime Agent is unreachable")
	}
	return nil
}

// pacedReleaseStore is the real PostgreSQL store with a listing it can fail
// once and counts, and with the operations' store calls sampled.
type pacedReleaseStore struct {
	Store
	paced   *pacedRelease
	stageOf map[string]string // allocation ID -> StageExecution ID

	mu      sync.Mutex
	lists   int
	listErr error
	listed  chan struct{}
}

func (s *pacedReleaseStore) ListTerminalStageExecutionsWithAllocations(ctx context.Context) ([]runstore.StageExecution, error) {
	s.mu.Lock()
	s.lists++
	err := s.listErr
	s.listErr = nil
	s.mu.Unlock()
	defer func() { s.listed <- struct{}{} }()
	if err != nil {
		return nil, err
	}
	return s.Store.ListTerminalStageExecutionsWithAllocations(ctx)
}

func (s *pacedReleaseStore) ListStageAllocations(ctx context.Context, stageExecutionID string) ([]runstore.StageAllocation, error) {
	s.paced.inFlight()
	return s.Store.ListStageAllocations(ctx, stageExecutionID)
}

func (s *pacedReleaseStore) MarkStageAllocationReleaseAttempt(ctx context.Context, allocationIDs ...string) error {
	s.paced.inFlight()
	return s.Store.MarkStageAllocationReleaseAttempt(ctx, allocationIDs...)
}

func (s *pacedReleaseStore) MarkStageAllocationReleased(ctx context.Context, allocationIDs ...string) error {
	s.paced.inFlight()
	err := s.Store.MarkStageAllocationReleased(ctx, allocationIDs...)
	if err == nil {
		for _, allocationID := range allocationIDs {
			s.paced.released <- s.stageOf[allocationID]
		}
	}
	return err
}

func (s *pacedReleaseStore) failNextList(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.listErr = err
}

func (s *pacedReleaseStore) listCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lists
}

// waitLists waits until recovery listed count times in total.
func (s *pacedReleaseStore) waitLists(t *testing.T, count int) {
	t.Helper()
	for s.listCount() < count {
		awaitValue(t, t.Context(), s.listed, "recovery listing")
	}
}

// pacedTerminalRelease is run-1 on PostgreSQL with terminal attempts whose
// grants are all live, recovered by the production monitor through the
// production Runtime batch controller with test-paced Runtime answers.
type pacedTerminalRelease struct {
	h           deferredPlacementHarness
	paced       *pacedRelease
	store       *pacedReleaseStore
	clock       *manualPollClock
	allocations map[string]string // StageExecution ID -> allocation ID
	failed      chan struct{}     // monitor observed an ended failed operation
}

func newPacedTerminalRelease(t *testing.T, ctx context.Context, attempts int) pacedTerminalRelease {
	t.Helper()
	h := newDeferredPlacementHarness(t, ctx, nil)
	paced := newPacedRelease()
	allocations := createTerminalAttempts(t, ctx, h, attempts,
		func(_ int, _ string, reservations []controlplane.Reservation) {
			for _, reservation := range reservations {
				paced.live[reservation.Grant.AllocationID] = reservation
			}
		})
	stageOf := make(map[string]string, len(allocations))
	for stageExecutionID, allocationID := range allocations {
		stageOf[allocationID] = stageExecutionID
	}
	batch, err := controlplane.NewRuntimeBatchController(pacedRuntime{paced: paced}, paced, controlplane.RuntimeBatchOptions{
		PrepareTimeout: time.Minute, CleanupTimeout: time.Hour,
	})
	if err != nil {
		t.Fatal(err)
	}
	store := &pacedReleaseStore{Store: h.store, paced: paced, stageOf: stageOf, listed: make(chan struct{}, 64)}
	clock := newManualPollClock()
	h.scheduler.store = store
	h.scheduler.allocator = paced
	h.scheduler.workers = batch
	h.scheduler.options.Clock = clock
	paced.scheduler = h.scheduler
	failed := make(chan struct{}, 64)
	h.scheduler.options.Logger = slog.New(releaseFailureHandler{failed: failed})
	return pacedTerminalRelease{h: h, paced: paced, store: store, clock: clock, allocations: allocations, failed: failed}
}

// startMonitor runs the production terminal release monitor until stop; done
// closes when it has returned.
func (f pacedTerminalRelease) startMonitor(ctx context.Context) (stop context.CancelFunc, done chan struct{}) {
	monitorContext, stop := context.WithCancel(ctx)
	done = make(chan struct{})
	go func() {
		defer close(done)
		f.h.scheduler.monitorTerminalRelease(monitorContext)
	}()
	return stop, done
}

// stopMonitor stops the monitor and answers every Runtime release still
// blocked, so a failed test ends instead of waiting for the monitor forever.
func (f pacedTerminalRelease) stopMonitor(stop context.CancelFunc, done <-chan struct{}) {
	stop()
	f.paced.mu.Lock()
	for _, gate := range f.paced.gates {
		select {
		case gate <- errors.New("test ended"):
		default:
		}
	}
	f.paced.mu.Unlock()
	<-done
}

// attempted reports the StageExecutions whose release recovery has marked an
// attempt on PostgreSQL.
func (f pacedTerminalRelease) attempted(t *testing.T, ctx context.Context) map[string]bool {
	t.Helper()
	ids := make([]string, 0, len(f.allocations))
	for stageExecutionID := range f.allocations {
		ids = append(ids, stageExecutionID)
	}
	rows, err := f.h.store.ListStageAllocationsBatch(ctx, ids)
	if err != nil {
		t.Fatal(err)
	}
	attempted := make(map[string]bool, len(rows))
	for stageExecutionID, allocations := range rows {
		attempted[stageExecutionID] = len(allocations) == 1 && allocations[0].ReleaseAttemptedAt != nil
	}
	return attempted
}

// requireOperationsEnded checks gates only after the monitor acknowledged
// the corresponding outcomes, so a retained gate fails without polling.
func (f pacedTerminalRelease) requireOperationsEnded(t *testing.T, ids ...string) {
	t.Helper()
	f.h.scheduler.releaseMu.Lock()
	defer f.h.scheduler.releaseMu.Unlock()
	for _, id := range ids {
		if _, held := f.h.scheduler.releasing[id]; held {
			t.Fatalf("completed recovery retained gate %s", id)
		}
	}
}

// The monitor reports a failed recovery only after recoverTerminalStage has
// returned and released its gate. These events replace polling releaseMu.
type releaseFailureHandler struct{ failed chan struct{} }

func (h releaseFailureHandler) Enabled(context.Context, slog.Level) bool { return true }
func (h releaseFailureHandler) Handle(_ context.Context, record slog.Record) error {
	if record.Message == "terminal allocation release recovery failed" {
		h.failed <- struct{}{}
	}
	return nil
}
func (h releaseFailureHandler) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h releaseFailureHandler) WithGroup(string) slog.Handler      { return h }

func receiveStage(t *testing.T, channel <-chan string, operation string) string {
	t.Helper()
	return awaitValue(t, t.Context(), channel, operation)
}

func waitClosed(t *testing.T, done <-chan struct{}, operation string) {
	t.Helper()
	awaitValue(t, t.Context(), done, operation)
}

// Terminal release recovery runs at most four Stage operations at once
// (SCH-R180, LIM-SCH-20). The first listing starts exactly four; the other
// Stages are neither called nor marked attempted. While the first Stage's
// Runtime does not answer, every other Stage starts and completes through
// the remaining slots, each with one Runtime call, and the blocked Stage
// completes once its Runtime answers. Recovery lists again only after an
// operation completed useful work.
func TestPostgresTerminalReleaseRecoveryBoundsOperationsAndPassesBlockedPeer(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	const attempts = 2*terminalReleaseRecoveryConcurrency + 2
	f := newPacedTerminalRelease(t, ctx, attempts)
	blocked := historicalAttemptID(1)
	var first []string
	for attempt := 1; attempt <= terminalReleaseRecoveryConcurrency; attempt++ {
		first = append(first, historicalAttemptID(attempt))
		f.paced.set(historicalAttemptID(attempt), pacedBlocked)
	}
	stop, done := f.startMonitor(ctx)
	defer f.stopMonitor(stop, done)

	var entered []string
	for range terminalReleaseRecoveryConcurrency {
		entered = append(entered, receiveStage(t, f.paced.entered, "Runtime release"))
	}
	slices.Sort(entered)
	if !slices.Equal(entered, first) || f.paced.inFlight() != terminalReleaseRecoveryConcurrency {
		t.Fatalf("first operations = %v with %d in flight, want %v", entered, f.paced.inFlight(), first)
	}
	attempted := f.attempted(t, ctx)
	for attempt := terminalReleaseRecoveryConcurrency + 1; attempt <= attempts; attempt++ {
		id := historicalAttemptID(attempt)
		if attempted[id] || f.paced.callCount(id) != 0 {
			t.Fatalf("Stage %s beyond the bound started: attempted %t, Runtime calls %d", id, attempted[id], f.paced.callCount(id))
		}
	}

	for _, id := range first[1:] {
		f.paced.finish(id, nil)
	}
	released := map[string]bool{}
	for range attempts - 1 {
		released[receiveStage(t, f.paced.released, "durable release")] = true
	}
	if released[blocked] || len(released) != attempts-1 {
		t.Fatalf("released while %s was blocked: %v", blocked, released)
	}
	requireUnreleasedAllocations(t, ctx, f.h, f.allocations, blocked)
	if !f.paced.isLive(f.allocations[blocked]) {
		t.Fatal("blocked Stage lost its grant before its Runtime answered")
	}
	for id := range f.allocations {
		if calls := f.paced.callCount(id); calls != 1 {
			t.Fatalf("Stage %s had %d Runtime release calls, want 1", id, calls)
		}
	}

	f.paced.finish(blocked, nil)
	if id := receiveStage(t, f.paced.released, "blocked durable release"); id != blocked {
		t.Fatalf("released %s, want %s", id, blocked)
	}
	// One initial listing and one after each of the ten completed releases.
	f.store.waitLists(t, 1+attempts)
	stop()
	waitClosed(t, done, "recovery stop")
	requireUnreleasedAllocations(t, ctx, f.h, f.allocations)
	if lists := f.store.listCount(); lists != 1+attempts {
		t.Fatalf("recovery listed %d times, want %d", lists, 1+attempts)
	}
	if maximum := f.paced.observedMaximum(); maximum != terminalReleaseRecoveryConcurrency {
		t.Fatalf("Stage operations in flight reached %d, want at most and exactly %d",
			maximum, terminalReleaseRecoveryConcurrency)
	}
}

// A failed operation frees its slot without listing again, and its Stage is
// retried behind the Stages never attempted (SCH-R71, SCH-R180). A Runtime
// that answers HTTP 409 ends its operation at once without waiting for any
// cleanup deadline; a Runtime whose control lease expired is released as
// lost; a failed listing waits for the next poll; a release that reached its
// cleanup deadline keeps its grant until a later attempt succeeds.
func TestPostgresTerminalReleaseRecoveryRotatesFailuresBehindUnattemptedStages(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	f := newPacedTerminalRelease(t, ctx, 6)
	conflict, timedOut, slowA, slowB := historicalAttemptID(1), historicalAttemptID(2), historicalAttemptID(3), historicalAttemptID(4)
	healthy, lost := historicalAttemptID(5), historicalAttemptID(6)
	f.paced.set(conflict, pacedConflict)
	for _, id := range []string{timedOut, slowA, slowB} {
		f.paced.set(id, pacedBlocked)
	}
	f.paced.set(lost, pacedLost)
	f.paced.loseRuntime(f.allocations[lost])
	f.store.failNextList(errors.New("injected listing failure"))
	poll := f.h.scheduler.options.PollInterval
	stop, done := f.startMonitor(ctx)
	defer f.stopMonitor(stop, done)

	f.store.waitLists(t, 1)
	for id := range f.allocations {
		if f.paced.callCount(id) != 0 {
			t.Fatalf("failed listing started Stage %s", id)
		}
	}
	f.clock.fireWhenArmedBefore(t, t.Context(), poll, 1)
	var entered []string
	for range terminalReleaseRecoveryConcurrency {
		entered = append(entered, receiveStage(t, f.paced.entered, "Runtime release"))
	}
	slices.Sort(entered)
	if want := []string{conflict, timedOut, slowA, slowB}; !slices.Equal(entered, want) {
		t.Fatalf("first operations = %v, want %v", entered, want)
	}
	if id := receiveStage(t, f.paced.returned, "Runtime conflict"); id != conflict {
		t.Fatalf("first Runtime answer from %s, want the conflict of %s", id, conflict)
	}
	awaitValue(t, t.Context(), f.failed, "first failed recovery outcome")
	f.requireOperationsEnded(t, conflict)
	// Nothing lists again before the poll: the failure is not useful work.
	attempted := f.attempted(t, ctx)
	if f.paced.callCount(healthy) != 0 || f.paced.callCount(lost) != 0 || attempted[healthy] || attempted[lost] {
		t.Fatalf("failure started more work: healthy calls %d, lost calls %d, attempted %v",
			f.paced.callCount(healthy), f.paced.callCount(lost), attempted)
	}
	if !attempted[conflict] || !f.paced.isLive(f.allocations[conflict]) {
		t.Fatalf("conflicting release: attempted %t, live %t", attempted[conflict], f.paced.isLive(f.allocations[conflict]))
	}

	f.clock.fireWhenArmedBefore(t, t.Context(), poll, 1)
	var rotated []string
	for range 3 {
		rotated = append(rotated, receiveStage(t, f.paced.entered, "rotated Runtime release"))
	}
	if want := []string{healthy, lost, conflict}; !slices.Equal(rotated, want) {
		t.Fatalf("operations after the poll = %v, want %v: never attempted Stages before the failed one", rotated, want)
	}
	for _, want := range []string{healthy, lost} {
		if id := receiveStage(t, f.paced.released, "durable release"); id != want {
			t.Fatalf("released %s, want %s", id, want)
		}
	}
	requireUnreleasedAllocations(t, ctx, f.h, f.allocations, conflict, timedOut, slowA, slowB)
	if f.paced.isLive(f.allocations[lost]) {
		t.Fatal("lost Runtime kept its grant")
	}

	f.paced.finish(timedOut, context.DeadlineExceeded)
	// The conflict retry and injected cleanup timeout have ended. The
	// initial conflict was acknowledged before its retry above.
	for range 2 {
		awaitValue(t, t.Context(), f.failed, "failed recovery outcome")
	}
	f.requireOperationsEnded(t, timedOut, conflict)
	requireUnreleasedAllocations(t, ctx, f.h, f.allocations, conflict, timedOut, slowA, slowB)
	if !f.paced.isLive(f.allocations[timedOut]) {
		t.Fatal("release past its cleanup deadline lost its grant")
	}
	f.paced.set(conflict, pacedHealthy)
	f.paced.set(timedOut, pacedHealthy)
	f.paced.finish(slowA, nil)
	f.paced.finish(slowB, nil)
	released := map[string]bool{}
	for len(released) < 4 {
		released[receiveStage(t, f.paced.released, "durable release")] = true
	}
	for _, id := range []string{conflict, timedOut, slowA, slowB} {
		if !released[id] {
			t.Fatalf("released %v, want %s", released, id)
		}
	}
	// Listings: the failed one, one per poll and one after each of the six
	// completed releases; none after a failure.
	f.store.waitLists(t, 9)
	stop()
	waitClosed(t, done, "recovery stop")
	requireUnreleasedAllocations(t, ctx, f.h, f.allocations)
	if lists := f.store.listCount(); lists != 9 {
		t.Fatalf("recovery listed %d times, want 9", lists)
	}
	for id, want := range map[string]int{conflict: 3, timedOut: 2, slowA: 1, slowB: 1, healthy: 1, lost: 1} {
		if calls := f.paced.callCount(id); calls != want {
			t.Fatalf("Stage %s had %d Runtime release calls, want %d", id, calls, want)
		}
	}
	if maximum := f.paced.observedMaximum(); maximum > terminalReleaseRecoveryConcurrency {
		t.Fatalf("Stage operations in flight reached %d, above %d", maximum, terminalReleaseRecoveryConcurrency)
	}
}

// Recovery never starts a Stage whose release a lane owns, and stopping it
// waits for its operations: an operation in flight at the stop completes its
// durable release afterwards, and nothing outlives the monitor (SCH-R180,
// FAIL-SCH-9).
func TestPostgresTerminalReleaseRecoveryStopJoinsOperations(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	f := newPacedTerminalRelease(t, ctx, 3)
	blocked, laneOwned, healthy := historicalAttemptID(1), historicalAttemptID(2), historicalAttemptID(3)
	f.paced.set(blocked, pacedBlocked)
	if !f.h.scheduler.beginTerminalRelease(laneOwned) {
		t.Fatal("failed to take the lane's release gate")
	}
	defer f.h.scheduler.endTerminalRelease(laneOwned)
	f.paced.mu.Lock()
	f.paced.laneOwned[laneOwned] = true
	f.paced.mu.Unlock()
	stop, done := f.startMonitor(ctx)
	defer f.stopMonitor(stop, done)

	if id := receiveStage(t, f.paced.released, "healthy durable release"); id != healthy {
		t.Fatalf("released %s, want %s", id, healthy)
	}
	for f.paced.callCount(blocked) == 0 {
		receiveStage(t, f.paced.entered, "blocked Runtime release")
	}
	stop()
	select {
	case <-done:
		t.Fatal("recovery stopped while its operation was in flight")
	default:
	}
	f.paced.finish(blocked, nil)
	waitClosed(t, done, "recovery stop after its operation")
	requireUnreleasedAllocations(t, ctx, f.h, f.allocations, laneOwned)
	if calls, attempted := f.paced.callCount(laneOwned), f.attempted(t, ctx)[laneOwned]; calls != 0 || attempted {
		t.Fatalf("recovery released the lane's Stage: Runtime calls %d, attempted %t", calls, attempted)
	}
	if inFlight := f.paced.inFlight(); inFlight != 0 {
		t.Fatalf("%d operations outlived the monitor", inFlight)
	}
}
