package scheduler

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/settingsstore"
)

// lanePool is owned by the supervisor goroutine. A round admits at most the
// lanes missing when it opened, with one claim in flight; lanes already doing
// useful work may complete and be replaced independently of that round.
type lanePool struct {
	class                             resolved.LaneClass
	desired, active, wanted, claiming int
	round                             uint64
}

type laneEventKind uint8

const (
	claimAcquired laneEventKind = iota
	laneFinished
)

type laneEvent struct {
	kind   laneEventKind
	class  resolved.LaneClass
	round  uint64
	worked bool
	err    error
}

// Run supervises the independently sized model and tool pools. Only this
// goroutine changes pool state; lanes send claim-acquired and lane-finished
// events. Maintenance remains outside the semantic execution limits.
func (s *Scheduler) Run(ctx context.Context) error {
	if !s.beginProductionRun() {
		return errors.New("Workflow Scheduler is already running")
	}
	defer s.endProductionRun()
	desired, err := s.readLaneLimits(ctx)
	if ctx.Err() != nil {
		return nil
	}
	if err != nil {
		return fmt.Errorf("load Workflow Scheduler concurrency setting: %w", err)
	}
	monitorContext, cancelMonitor := context.WithCancel(context.WithoutCancel(ctx))
	var monitor sync.WaitGroup
	monitor.Add(2)
	go func() { defer monitor.Done(); s.monitorAllocationLosses(monitorContext) }()
	go func() { defer monitor.Done(); s.monitorTerminalRelease(monitorContext) }()

	// At most two events per possible lane, so lanes can finish after the
	// supervisor stops receiving, including during a pool decrease's drain.
	events := make(chan laneEvent, 2*settingsstore.MaximumLanes)
	var lanes sync.WaitGroup
	pools := make(map[resolved.LaneClass]*lanePool, len(resolved.LaneClasses))
	for _, class := range resolved.LaneClasses {
		pools[class] = &lanePool{class: class}
	}
	admissionHealthy := true
	canStart := func(pool *lanePool) bool { return admissionHealthy && ctx.Err() == nil && pool.active < pool.desired }
	startLane := func(pool *lanePool) {
		pool.active++
		pool.claiming++
		lanes.Add(1)
		class, started := pool.class, pool.round
		go func() {
			defer lanes.Done()
			worked, laneErr := s.progressOneClaim(ctx, class, func() { events <- laneEvent{kind: claimAcquired, class: class, round: started} })
			events <- laneEvent{kind: laneFinished, class: class, round: started, worked: worked, err: laneErr}
		}()
	}
	// Idle pools attempt one claim per round. Busy pools fill the next lane
	// as soon as the previous claim succeeds (SCH-R181), without waiting for
	// that Run's work to finish.
	continueFill := func(pool *lanePool) {
		if pool.claiming == 0 && pool.wanted > 0 && canStart(pool) {
			pool.wanted--
			startLane(pool)
		}
	}
	fillLanes := func() {
		for _, class := range resolved.LaneClasses {
			pool := pools[class]
			pool.desired = desired[class]
			pool.round++
			pool.wanted = max(0, pool.desired-pool.active)
			continueFill(pool)
		}
	}
	defer func() { lanes.Wait(); cancelMonitor(); monitor.Wait() }()
	fillLanes()
	refresh := s.after(s.options.PollInterval)
	for {
		select {
		case <-ctx.Done():
			return nil
		case event := <-events:
			pool := pools[event.class]
			if event.kind == claimAcquired {
				pool.claiming--
				continueFill(pool)
				continue
			}
			pool.active--
			if !event.worked {
				pool.claiming--
				if event.round == pool.round {
					pool.wanted = 0 // No work or a failed claim ends its own round.
				} else {
					// An old claim occupied a slot when the current round opened;
					// its completion must not consume that round's first attempt.
					pool.wanted = min(pool.wanted+1, max(0, pool.desired-pool.active))
					continueFill(pool)
				}
			}
			if event.err != nil && !errors.Is(event.err, ErrDeferred) && ctx.Err() == nil {
				s.options.Logger.Error("Workflow Scheduler lane failed", "lane_class", string(event.class), "error", event.err)
			}
			// Replace successful work eagerly. Deferred or failed work waits for
			// the next poll/wake, so blocked Runs cannot cause a hot loop.
			if event.worked && event.err == nil && canStart(pool) {
				startLane(pool)
			}
			continue
		case <-s.wake:
		case <-refresh:
			refresh = s.after(s.options.PollInterval)
		}
		next, refreshErr := s.readLaneLimits(ctx)
		if refreshErr != nil {
			admissionHealthy = false
			if ctx.Err() == nil {
				s.options.Logger.Error("refresh Workflow Scheduler concurrency setting failed", "error", refreshErr)
			}
			continue
		}
		desired, admissionHealthy = next, true
		fillLanes()
	}
}

// progressOneClaim is the production lane primitive of the pool of class. It
// owns exactly one claim/progression/release cycle of a Run of that class and
// performs no global maintenance. It calls admitted, when not nil, once its
// claim succeeded and the Run is registered, before progressing the Run; it
// reports work (true) exactly when it claimed a Run.
func (s *Scheduler) progressOneClaim(
	ctx context.Context, class resolved.LaneClass, admitted func(),
) (bool, error) {
	claimID, err := s.newID("claim_")
	if err != nil {
		return false, fmt.Errorf("generate Scheduler claim ID: %w", err)
	}
	ownershipContext, cancelOwnership := context.WithCancelCause(ctx)
	defer cancelOwnership(nil)
	ownershipContext = context.WithValue(ownershipContext, runOwnershipContextKey{}, ownershipContext)
	executionContext, cancelExecution := context.WithCancelCause(ownershipContext)
	defer cancelExecution(nil)
	claimContext, cancelClaim := context.WithTimeout(ctx, s.options.OperationTimeout)
	leaseExpiresAt := s.now().Add(s.options.ClaimDuration)
	run, err := s.claimRun(claimContext, claimID, cancelExecution, class)
	cancelClaim()
	if errors.Is(err, runstore.ErrNoWork) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	ownershipContext = WithRunClaim(ownershipContext, run.RunID, claimID)
	executionContext = WithRunClaim(executionContext, run.RunID, claimID)
	if admitted != nil {
		admitted()
	}

	// Cancelling Planner work does not release ownership of the Run. Keep its
	// claim renewable through cancellation/lease-loss cleanup, which can exceed
	// the original lease when remote teardown or report persistence is slow.
	defer s.unregisterActiveRun(run.RunID, claimID)
	stopRenewal := make(chan struct{})
	renewalContext, cancelRenewal := context.WithCancel(ownershipContext)
	defer cancelRenewal()
	var renewal sync.WaitGroup
	renewal.Add(1)
	go func() {
		defer renewal.Done()
		s.renewClaim(renewalContext, cancelOwnership, cancelExecution, stopRenewal, run.RunID, claimID, run.State, leaseExpiresAt)
	}()

	err = s.executeRun(executionContext, run, claimID, class)
	if errors.Is(err, runstore.ErrQueuePaused) {
		err = ErrDeferred
	}
	if cause := context.Cause(executionContext); ownershipContext.Err() == nil &&
		(errors.Is(cause, ErrRunCancellationRequested) || errors.Is(cause, ErrAllocationLeaseLost)) {
		// The interrupted context intentionally cannot be reused for cleanup.
		// Retain the same durable claim while entering the authoritative bounded path.
		loadContext, cancelLoad := context.WithTimeout(ownershipContext, s.options.OperationTimeout)
		current, loadErr := s.store.GetRun(loadContext, run.RunID)
		cancelLoad()
		if loadErr != nil {
			err = loadErr
		} else if current.SchedulerLaneClass != class {
			// The interrupted attempt handed the Run to the other pool, which
			// claims it in its own claim order (SCH-R121).
			err = nil
		} else if current.State == runstore.RunCancelling ||
			(errors.Is(cause, ErrAllocationLeaseLost) && (current.State == runstore.RunRunning || current.State == runstore.RunWaiting)) {
			err = s.executeRun(ownershipContext, current, claimID, class)
		} else {
			err = nil
		}
	}
	close(stopRenewal)
	cancelRenewal()
	cancelExecution(nil)
	renewal.Wait()
	if cause := context.Cause(ownershipContext); errors.Is(cause, ErrClaimLost) {
		err = errors.Join(cause, err)
	}

	// Once released, another lane of this process may claim the Run and
	// register its own cancellation; this lane must not shadow or remove it.
	s.unregisterActiveRun(run.RunID, claimID)
	releaseContext, cancelRelease := context.WithTimeout(context.Background(), s.options.OperationTimeout)
	var releaseErr error
	if errors.Is(err, ErrDeferred) {
		releaseErr = s.store.DeferRunClaim(releaseContext, run.RunID, claimID)
	} else {
		releaseErr = s.store.ReleaseRunClaim(releaseContext, run.RunID, claimID)
	}
	cancelRelease()
	if releaseErr != nil && !errors.Is(releaseErr, runstore.ErrConflict) {
		err = errors.Join(err, releaseErr)
	}
	if cause := context.Cause(executionContext); cause != nil &&
		!errors.Is(cause, context.Canceled) && !errors.Is(cause, ErrRunCancellationRequested) &&
		!errors.Is(cause, ErrAllocationLeaseLost) && err == nil {
		err = cause
	}
	return true, err
}

// claimRun claims a Run of class and excludes every lane of either pool that
// has not joined its work yet. A host pause
// may expire SQL leases while those lanes still have live Planner invocations.
// Serialize selection through registration so a second claim cannot miss a
// returned-but-not-yet-registered owner. Cancellation and renewal never take
// claimMu, and activeMu is held only while copying the bounded lane registry.
func (s *Scheduler) claimRun(
	ctx context.Context, claimID string, cancel context.CancelCauseFunc, class resolved.LaneClass,
) (runstore.WorkflowRun, error) {
	s.claimMu.Lock()
	defer s.claimMu.Unlock()
	if err := ctx.Err(); err != nil {
		return runstore.WorkflowRun{}, err
	}
	s.activeMu.Lock()
	excluded := make([]string, 0, len(s.active))
	for runID := range s.active {
		excluded = append(excluded, runID)
	}
	s.activeMu.Unlock()
	run, err := s.store.ClaimRunnableRun(ctx, claimID, s.options.ClaimDuration, excluded, class)
	if err == nil {
		s.registerActiveRun(run.RunID, claimID, cancel)
	}
	return run, err
}

// readLaneLimits reads the size of each lane pool from the one durable
// settings row; an invalid row is never replaced by a guessed default.
func (s *Scheduler) readLaneLimits(ctx context.Context) (map[resolved.LaneClass]int, error) {
	readContext, cancel := context.WithTimeout(ctx, s.options.OperationTimeout)
	defer cancel()
	settings, err := s.options.Settings.GetSchedulerSettings(readContext)
	if err != nil {
		return nil, err
	}
	if settings.MaxConcurrentRuns < settingsstore.MinimumConcurrentRuns ||
		settings.MaxConcurrentRuns > settingsstore.MaximumConcurrentRuns ||
		settings.MaxConcurrentToolRuns < settingsstore.MinimumConcurrentToolRuns ||
		settings.MaxConcurrentToolRuns > settingsstore.MaximumConcurrentToolRuns ||
		settings.Revision == 0 || settings.UpdatedAt.IsZero() {
		return nil, settingsstore.ErrInvariant
	}
	return map[resolved.LaneClass]int{
		resolved.LaneClassModel: settings.MaxConcurrentRuns,
		resolved.LaneClassTool:  settings.MaxConcurrentToolRuns,
	}, nil
}

func (s *Scheduler) beginProductionRun() bool {
	s.runMu.Lock()
	defer s.runMu.Unlock()
	if s.running {
		return false
	}
	s.running = true
	return true
}

func (s *Scheduler) endProductionRun() {
	s.runMu.Lock()
	s.running = false
	s.runMu.Unlock()
}

func (s *Scheduler) monitorAllocationLosses(ctx context.Context) {
	for {
		s.pollAllocationLosses()
		select {
		case <-ctx.Done():
			return
		case <-s.after(s.options.LeaseScanInterval):
		}
	}
}

// monitorTerminalRelease runs terminal release recovery outside the lanes
// (SCH-R159) with at most terminalReleaseRecoveryConcurrency Stage operations
// in flight (SCH-R180, LIM-SCH-20), so a Runtime Agent that holds one Stage's
// release until its cleanup deadline delays only that Stage. It lists again
// once an operation completed useful work and otherwise each poll interval; a
// failed operation frees its slot without listing, so a failing Runtime or
// store cannot create a hot loop. It returns only after every operation it
// started has ended.
func (s *Scheduler) monitorTerminalRelease(ctx context.Context) {
	type outcome struct {
		worked bool
		err    error
	}
	// Each operation sends one outcome and at most the bound are in flight,
	// so no operation blocks once the monitor stops receiving.
	outcomes := make(chan outcome, terminalReleaseRecoveryConcurrency)
	var operations sync.WaitGroup
	defer operations.Wait()
	start := func(execution runstore.StageExecution) {
		operations.Add(1)
		go func() {
			defer operations.Done()
			worked, err := s.recoverTerminalStage(ctx, execution)
			outcomes <- outcome{worked: worked, err: err}
		}()
	}
	inFlight, list := 0, true
	var poll <-chan time.Time
	for {
		if list && inFlight < terminalReleaseRecoveryConcurrency {
			list = false
			started, err := s.startTerminalReleaseRecovery(ctx, terminalReleaseRecoveryConcurrency-inFlight, start)
			inFlight += started
			if err != nil && ctx.Err() == nil {
				s.options.Logger.Warn("list terminal allocation releases failed", "error", err)
			}
		}
		if poll == nil {
			poll = s.after(s.options.PollInterval)
		}
		select {
		case <-ctx.Done():
			return
		case result := <-outcomes:
			inFlight--
			if result.err != nil && ctx.Err() == nil {
				s.options.Logger.Warn("terminal allocation release recovery failed", "error", result.err)
			}
			list = list || (result.worked && result.err == nil)
		case <-poll:
			poll = nil
			list = true
		}
	}
}

func (s *Scheduler) pollAllocationLosses() {
	for _, loss := range s.allocator.PollAllocationLosses() {
		interrupted := s.interruptStageAllocationLoss(loss)
		s.options.Logger.Warn(
			"Runtime Agent allocation was lost",
			"run_id", loss.RunID,
			"stage_execution_id", loss.StageExecutionID,
			"allocation_id", loss.AllocationID,
			"reason", loss.Reason,
			"interrupted_current_stage", interrupted,
		)
		// A loss for a Stage that is no longer current interrupts nothing; its
		// fenced grant is reclaimed by terminal release recovery. Waking lets
		// recovery and any waiting lane run promptly in either case.
		s.Wake()
	}
}

func (s *Scheduler) renewClaim(
	ctx context.Context,
	cancelOwnership context.CancelCauseFunc,
	cancelExecution context.CancelCauseFunc,
	stop <-chan struct{},
	runID string,
	claimID string,
	claimedState runstore.WorkflowRunState,
	leaseExpiresAt time.Time,
) {
	// A lane renews every third of the lease (SCH-R166, LIM-SCH-15). The
	// renewal also reports the Run state, so a cancellation the in-process
	// Cancel did not reach still interrupts the lane.
	interval := s.options.ClaimDuration / 3
	if interval <= 0 {
		interval = time.Millisecond
	}
	// Transient store failures are retried every poll interval while the
	// last known lease still has a safety margin; past that point another
	// Scheduler may reclaim it.
	retry := min(s.options.PollInterval, interval)
	if retry <= 0 {
		retry = time.Millisecond
	}
	safetyMargin := s.options.ClaimDuration / 3
	keepRenewing := func(err error) bool {
		if !claimOwnershipLost(err) && s.now().Before(leaseExpiresAt.Add(-safetyMargin)) {
			s.options.Logger.Warn(
				"WorkflowRun claim renewal failed; retrying",
				"run_id", runID, "lease_expires_at", leaseExpiresAt, "error", err,
			)
			return true
		}
		cancelOwnership(errors.Join(ErrClaimLost, err))
		return false
	}
	wait := interval
	for {
		select {
		case <-ctx.Done():
			return
		case <-stop:
			return
		case <-s.after(wait):
		}
		renewContext, renewCancel := context.WithTimeout(ctx, s.options.OperationTimeout)
		renewStarted := s.now()
		state, err := s.store.RenewRunClaim(renewContext, runID, claimID, s.options.ClaimDuration)
		if err != nil {
			if ctx.Err() != nil {
				renewCancel()
				return
			}
			if errors.Is(err, runstore.ErrConflict) {
				// A terminal transition atomically clears its own claim: a
				// terminal Run holds none. Renewal racing that commit is
				// normal completion, not stolen ownership.
				current, loadErr := s.store.GetRunState(renewContext, runID)
				if loadErr == nil && runstore.RunLifecycleTerminal.Includes(current) {
					renewCancel()
					return
				}
			}
			renewCancel()
			if ctx.Err() != nil {
				return
			}
			if !keepRenewing(err) {
				return
			}
			wait = retry
			continue
		}
		renewCancel()
		wait = interval
		leaseExpiresAt = renewStarted.Add(s.options.ClaimDuration)
		// A Run claimed while already cancelling is using executionContext
		// for its bounded abort or finalization; do not interrupt that cleanup.
		if state == runstore.RunCancelling && claimedState != runstore.RunCancelling {
			cancelExecution(ErrRunCancellationRequested)
		}
	}
}

// claimOwnershipLost reports store answers that prove the claim can no longer
// be renewed, as opposed to transient failures worth retrying.
func claimOwnershipLost(err error) bool {
	return errors.Is(err, runstore.ErrConflict) || errors.Is(err, runstore.ErrNotFound) ||
		errors.Is(err, runstore.ErrInvalid)
}
