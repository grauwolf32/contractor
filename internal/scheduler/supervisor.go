package scheduler

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/settingsstore"
)

// Run supervises the PostgreSQL-authoritative number of concurrent Run lanes.
// Each lane owns at most one durable claim, while maintenance remains outside
// the semantic execution limit so cancellation and release can always drain.
func (s *Scheduler) Run(ctx context.Context) error {
	if !s.beginProductionRun() {
		return errors.New("Workflow Scheduler is already running")
	}
	defer s.endProductionRun()

	desired, err := s.readConcurrentRunLimit(ctx)
	if ctx.Err() != nil {
		return nil
	}
	if err != nil {
		return fmt.Errorf("load Workflow Scheduler concurrency setting: %w", err)
	}
	monitorContext, cancelMonitor := context.WithCancel(context.WithoutCancel(ctx))
	var monitor sync.WaitGroup
	monitor.Add(3)
	go func() {
		defer monitor.Done()
		s.monitorAllocationLosses(monitorContext)
	}()
	go func() {
		defer monitor.Done()
		s.monitorTerminalRelease(monitorContext)
	}()
	go func() {
		defer monitor.Done()
		s.monitorMetricsRetention(monitorContext)
	}()

	type laneResult struct {
		worked bool
		err    error
	}
	results := make(chan laneResult, settingsstore.MaximumConcurrentRuns)
	var lanes sync.WaitGroup
	activeLanes := 0
	admissionHealthy := true
	startLane := func() {
		activeLanes++
		lanes.Add(1)
		go func() {
			defer lanes.Done()
			worked, laneErr := s.progressOneClaim(ctx)
			results <- laneResult{worked: worked, err: laneErr}
		}()
	}
	fillLanes := func() {
		for admissionHealthy && ctx.Err() == nil && activeLanes < desired {
			startLane()
		}
	}
	defer func() {
		lanes.Wait()
		cancelMonitor()
		monitor.Wait()
	}()
	fillLanes()
	refresh := s.after(s.options.PollInterval)
	for {
		select {
		case <-ctx.Done():
			return nil
		case result := <-results:
			activeLanes--
			if result.err != nil && !errors.Is(result.err, ErrDeferred) && ctx.Err() == nil {
				s.options.Logger.Error("Workflow Scheduler lane failed", "error", result.err)
			}
			// Preserve the serial loop's eager progression only after useful,
			// successful work. No-work and deferred attempts wait for a bounded
			// poll/wake so incompatible Runs cannot create a hot loop.
			if result.worked && result.err == nil && admissionHealthy &&
				ctx.Err() == nil && activeLanes < desired {
				startLane()
			}
		case <-s.wake:
			next, refreshErr := s.readConcurrentRunLimit(ctx)
			if refreshErr != nil {
				admissionHealthy = false
				if ctx.Err() == nil {
					s.options.Logger.Error("refresh Workflow Scheduler concurrency setting failed", "error", refreshErr)
				}
				continue
			}
			desired, admissionHealthy = next, true
			fillLanes()
		case <-refresh:
			refresh = s.after(s.options.PollInterval)
			next, refreshErr := s.readConcurrentRunLimit(ctx)
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
}

// RunOnce claims and advances at most one WorkflowRun.
func (s *Scheduler) RunOnce(ctx context.Context) (bool, error) {
	s.pollAllocationLosses()
	released, terminalReleaseErr := s.recoverTerminalRelease(ctx)
	if terminalReleaseErr != nil {
		s.options.Logger.Warn("terminal allocation release recovery failed", "error", terminalReleaseErr)
	}
	worked, err := s.progressOneClaim(ctx)
	if !worked && err == nil {
		return released && terminalReleaseErr == nil, nil
	}
	return worked, err
}

// progressOneClaim is the production lane primitive. It owns exactly one
// claim/progression/release cycle and performs no global maintenance.
func (s *Scheduler) progressOneClaim(ctx context.Context) (bool, error) {
	claimID, err := s.newID("claim_")
	if err != nil {
		return false, fmt.Errorf("generate Scheduler claim ID: %w", err)
	}
	claimContext, cancelClaim := context.WithTimeout(ctx, s.options.OperationTimeout)
	run, err := s.store.ClaimRunnableRun(claimContext, claimID, s.options.ClaimDuration)
	cancelClaim()
	if errors.Is(err, runstore.ErrNoWork) {
		return false, nil
	}
	if err != nil {
		return false, err
	}

	// Cancelling Planner work does not release ownership of the Run. Keep its
	// claim renewable through cancellation/lease-loss cleanup, which can exceed
	// the original lease when remote teardown or report persistence is slow.
	ownershipContext, cancelOwnership := context.WithCancelCause(ctx)
	defer cancelOwnership(nil)
	ownershipContext = context.WithValue(ownershipContext, runOwnershipContextKey{}, ownershipContext)
	executionContext, cancelExecution := context.WithCancelCause(ownershipContext)
	s.activeMu.Lock()
	s.active[run.RunID] = cancelExecution
	s.activeMu.Unlock()
	defer func() {
		s.activeMu.Lock()
		if current := s.active[run.RunID]; current != nil {
			delete(s.active, run.RunID)
		}
		s.activeMu.Unlock()
	}()
	stopRenewal := make(chan struct{})
	renewalContext, cancelRenewal := context.WithCancel(ownershipContext)
	defer cancelRenewal()
	var renewal sync.WaitGroup
	renewal.Add(1)
	go func() {
		defer renewal.Done()
		s.renewClaim(renewalContext, cancelOwnership, cancelExecution, stopRenewal, run.RunID, claimID)
	}()

	err = s.executeRun(executionContext, run)
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
		} else if current.State == runstore.RunCancelling ||
			(errors.Is(cause, ErrAllocationLeaseLost) && (current.State == runstore.RunRunning || current.State == runstore.RunWaiting)) {
			err = s.executeRun(ownershipContext, current)
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

	releaseContext, cancelRelease := context.WithTimeout(context.Background(), s.options.OperationTimeout)
	releaseErr := s.store.ReleaseRunClaim(releaseContext, run.RunID, claimID)
	cancelRelease()
	if releaseErr != nil && !errors.Is(releaseErr, runstore.ErrConflict) && err == nil {
		err = releaseErr
	}
	if cause := context.Cause(executionContext); cause != nil &&
		!errors.Is(cause, context.Canceled) && !errors.Is(cause, ErrRunCancellationRequested) &&
		!errors.Is(cause, ErrAllocationLeaseLost) && err == nil {
		err = cause
	}
	return true, err
}

func (s *Scheduler) readConcurrentRunLimit(ctx context.Context) (int, error) {
	readContext, cancel := context.WithTimeout(ctx, s.options.OperationTimeout)
	defer cancel()
	settings, err := s.options.Settings.GetSchedulerSettings(readContext)
	if err != nil {
		return 0, err
	}
	if settings.MaxConcurrentRuns < settingsstore.MinimumConcurrentRuns ||
		settings.MaxConcurrentRuns > settingsstore.MaximumConcurrentRuns ||
		settings.Revision == 0 || settings.UpdatedAt.IsZero() {
		return 0, settingsstore.ErrInvariant
	}
	return settings.MaxConcurrentRuns, nil
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

func (s *Scheduler) monitorTerminalRelease(ctx context.Context) {
	for {
		worked, err := s.recoverTerminalRelease(ctx)
		if err != nil && ctx.Err() == nil {
			s.options.Logger.Warn("terminal allocation release recovery failed", "error", err)
		}
		if worked && err == nil {
			continue
		}
		select {
		case <-ctx.Done():
			return
		case <-s.after(s.options.PollInterval):
		}
	}
}

func (s *Scheduler) monitorMetricsRetention(ctx context.Context) {
	for {
		cleanupContext, cancel := context.WithTimeout(ctx, s.options.OperationTimeout)
		deleted, err := s.store.CleanupExpiredTelemetry(
			cleanupContext, s.now(), s.options.MetricsCleanupBatch,
		)
		cancel()
		if err != nil && ctx.Err() == nil {
			s.options.Logger.Warn("telemetry retention cleanup failed", "error", err)
		} else if deleted > 0 {
			s.options.Logger.Info("expired telemetry removed", "rows", deleted)
		}
		select {
		case <-ctx.Done():
			return
		case <-s.after(s.options.MetricsCleanupInterval):
		}
	}
}

func (s *Scheduler) pollAllocationLosses() {
	for _, loss := range s.allocator.PollAllocationLosses() {
		s.options.Logger.Warn(
			"Runtime Agent allocation was lost",
			"run_id", loss.RunID,
			"stage_execution_id", loss.StageExecutionID,
			"allocation_id", loss.AllocationID,
			"reason", loss.Reason,
		)
		s.interruptRun(loss.RunID, &AllocationLeaseLossError{Loss: loss})
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
) {
	interval := s.options.ClaimDuration / 3
	if s.options.PollInterval < interval {
		interval = s.options.PollInterval
	}
	if interval <= 0 {
		interval = time.Millisecond
	}
	for {
		select {
		case <-ctx.Done():
			return
		case <-stop:
			return
		case <-s.after(interval):
		}
		renewContext, renewCancel := context.WithTimeout(ctx, s.options.OperationTimeout)
		err := s.store.RenewRunClaim(renewContext, runID, claimID, s.options.ClaimDuration)
		if err != nil {
			if ctx.Err() != nil {
				renewCancel()
				return
			}
			if errors.Is(err, runstore.ErrConflict) {
				// A terminal transition atomically clears its own claim. Renewal
				// racing that commit is normal completion, not stolen ownership.
				current, loadErr := s.store.GetRun(renewContext, runID)
				if loadErr == nil && terminalRunReleasedClaim(current) {
					renewCancel()
					return
				}
			}
			renewCancel()
			if ctx.Err() != nil {
				return
			}
			cancelOwnership(errors.Join(ErrClaimLost, err))
			return
		}
		current, err := s.store.GetRun(renewContext, runID)
		renewCancel()
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			cancelOwnership(errors.Join(ErrClaimLost, err))
			return
		}
		if terminalRunReleasedClaim(current) {
			return
		}
		if current.State == runstore.RunCancelling {
			cancelExecution(ErrRunCancellationRequested)
		}
	}
}

func terminalRunReleasedClaim(run runstore.WorkflowRun) bool {
	return run.SchedulerClaim == nil &&
		(run.State == runstore.RunSucceeded || run.State == runstore.RunFailed || run.State == runstore.RunCancelled)
}
