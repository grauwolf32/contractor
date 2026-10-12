package scheduler

import "context"

// RunOnce claims and advances at most one WorkflowRun after one maintenance
// tick (SCH-R125). Production lanes call progressOneClaim and leave
// maintenance to the supervisor.
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
