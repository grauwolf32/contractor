package scheduler

import (
	"context"

	"github.com/grauwolf32/contractor/internal/config/resolved"
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
		worked, err := s.progressOneClaim(ctx, class)
		if worked || err != nil {
			return worked, err
		}
	}
	return released && terminalReleaseErr == nil, nil
}
