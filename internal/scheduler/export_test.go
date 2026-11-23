package scheduler

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// RunOnce claims and advances at most one WorkflowRun after one maintenance
// tick (SCH-R125): it tries the pools in resolved.LaneClasses order and stops
// at the first claim. Production lanes call progressOneClaim and leave
// maintenance to the supervisor.
func (s *Scheduler) RunOnce(ctx context.Context) (bool, error) {
	s.pollAllocationLosses()
	released, terminalReleaseErr := s.recoverTerminalRelease(ctx)
	if terminalReleaseErr != nil {
		s.options.Logger.Warn("terminal allocation release recovery failed", "error", terminalReleaseErr)
	}
	for _, class := range resolved.LaneClasses {
		worked, err := s.progressOneClaim(ctx, class, nil)
		if worked || err != nil {
			return worked, err
		}
	}
	return released && terminalReleaseErr == nil, nil
}

// recoverTerminalRelease is RunOnce's terminal release recovery tick: one
// listing that starts the Stage operations the monitor would start with every
// slot free (SCH-R180), run one after another so the tick stays
// deterministic. It reports whether any operation had work, and every
// failure.
func (s *Scheduler) recoverTerminalRelease(ctx context.Context) (bool, error) {
	worked := false
	var failures []error
	_, err := s.startTerminalReleaseRecovery(ctx, terminalReleaseRecoveryConcurrency,
		func(execution runstore.StageExecution) {
			stageWorked, stageErr := s.recoverTerminalStage(ctx, execution)
			worked = worked || stageWorked
			failures = append(failures, stageErr)
		})
	if err != nil {
		return false, err
	}
	return worked, errors.Join(failures...)
}
