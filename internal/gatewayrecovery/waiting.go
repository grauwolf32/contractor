package gatewayrecovery

import (
	"context"
	"time"
)

// EndStage unregisters exactly this Stage. Late updates cannot recreate it.
func (s *Service) EndStage(runID, stageID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if run := s.runs[runID]; run != nil && run.stageID == stageID {
		s.endStage(run)
	}
}
func (s *Service) endStage(run *runState) {
	s.failureReceipts -= len(run.failures)
	for participant := range run.waits {
		s.clearWaiting(run, participant)
	}
	for binding := range run.bindings {
		delete(s.bindings, binding)
	}
	for key := range run.routes {
		state := s.routes[key]
		if state.probeRunID == run.id {
			state.release()
		}
		delete(state.members, run.id)
		if !state.blocked && len(state.members) == 0 {
			delete(s.routes, key)
		}
	}
	delete(s.runs, run.id)
	close(run.changed)
}

func (s *Service) ReleaseAdmission(runID, stageID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	run := s.runs[runID]
	if run == nil || run.stageID != stageID {
		return
	}
	for key := range run.routes {
		state := s.routes[key]
		if state.probeRunID == runID && state.probeID == "" {
			state.release()
		}
	}
}

// Waiting returns a stable wait reason and a broadcast channel for its next
// change. Scheduler reconciles this projection with the durable Run separately.
func (s *Service) Waiting(runID, stageID string) (string, <-chan struct{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	run := s.runs[runID]
	if run == nil || run.stageID != stageID {
		return "", nil
	}
	key := ""
	for _, route := range run.waits {
		if s.routes[route].blocked && (key == "" || route < key) {
			key = route
		}
	}
	if key == "" {
		return "", run.changed
	}
	return s.routes[key].code, run.changed
}

type Status struct {
	Code           string     `json:"code"`
	Since          time.Time  `json:"since"`
	NextRetryAt    *time.Time `json:"nextRetryAt,omitempty"`
	AutomaticUntil time.Time  `json:"automaticUntil"`
	RequiresRetry  bool       `json:"requiresRetry"`
}

// ReadStatus is a live process snapshot. The reader supplies the exact active
// Stage from its durable Run snapshot, preventing status from a later Stage.
func (s *Service) ReadStatus(ctx context.Context, runID, stageID string) (*Status, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	run := s.runs[runID]
	if run == nil || run.stageID != stageID {
		return nil, nil
	}
	now := s.now()
	var chosen *routeState
	var chosenKey string
	for key := range run.routes {
		state := s.routes[key]
		if !state.blocked {
			continue
		}
		if chosen == nil || state.automaticExpired(now) && !chosen.automaticExpired(now) ||
			state.automaticExpired(now) == chosen.automaticExpired(now) && (state.next.After(chosen.next) || state.next.Equal(chosen.next) && key < chosenKey) {
			chosen, chosenKey = state, key
		}
	}
	if chosen == nil {
		return nil, nil
	}
	status := &Status{Code: chosen.code, Since: chosen.blockedAt, AutomaticUntil: chosen.automaticUntil, RequiresRetry: chosen.automaticExpired(now)}
	if !status.RequiresRetry {
		next := chosen.next
		if chosen.probeUntil.After(next) {
			next = chosen.probeUntil
		}
		status.NextRetryAt = &next
	}
	return status, nil
}

func (s *Service) Retry(ctx context.Context, ownerID, runID string) error {
	if err := checkContext(ctx); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	run := s.runs[runID]
	if run == nil || run.ownerID != ownerID {
		return ErrUnavailable
	}
	changed := false
	now := s.now()
	for key := range run.routes {
		state := s.routes[key]
		if state.blocked {
			state.automaticUntil, state.next = now.Add(s.policy.AutomaticWindow), now
			changed = true
		}
	}
	if !changed {
		return ErrUnavailable
	}
	return nil
}

// EndRun removes any admission registration after Scheduler durably ends a Run
// without progressing its Stage, including an unreadable Workflow snapshot.
func (s *Service) EndRun(runID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if run := s.runs[runID]; run != nil {
		s.endStage(run)
	}
}
