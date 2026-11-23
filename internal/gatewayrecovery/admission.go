package gatewayrecovery

import "context"

// Admit checks the candidate Stage routes before placement. Live invocations
// have priority over queued Runs. Re-admission relinquishes earlier reserved
// probes and replaces route membership, including after a denied admission.
func (s *Service) Admit(ctx context.Context, runID, stageID string, routes []Route) (bool, error) {
	if err := checkContext(ctx); err != nil {
		return false, err
	}
	if runID == "" || stageID == "" || len(routes) > maximumBindingsPerRun {
		return false, ErrInvalid
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	keys := make(map[string]Route, len(routes))
	owner := ""
	missing := 0
	for _, route := range routes {
		if route.OwnerID == "" || route.Model == "" || (owner != "" && owner != route.OwnerID) {
			return false, ErrInvalid
		}
		owner = route.OwnerID
		key := route.Key()
		if _, exists := keys[key]; exists {
			continue
		}
		keys[key] = route
		if s.routes[key] == nil {
			missing++
		}
	}
	if len(s.routes)+missing > maximumRoutes || (s.runs[runID] == nil && len(s.runs) >= maximumRuns) {
		return false, errCapacity
	}
	run := s.runs[runID]
	if run != nil && run.stageID != stageID {
		s.endStage(run)
		run = nil
	}
	if run == nil {
		run = &runState{id: runID, stageID: stageID, ownerID: owner, routes: make(map[string]struct{}),
			bindings: make(map[bindingKey]string), waits: make(map[bindingKey]string),
			failures: make(map[string]struct{}), changed: make(chan struct{})}
		s.runs[runID] = run
	}
	if run.active {
		return false, ErrInvalid
	}
	for key := range run.routes {
		state := s.routes[key]
		if state.probeRunID == runID && state.probeID == "" {
			state.release()
		}
		if _, keep := keys[key]; !keep {
			delete(state.members, runID)
			delete(run.routes, key)
			if len(state.members) == 0 && !state.blocked {
				delete(s.routes, key)
			}
		}
	}
	for key := range keys {
		state := s.routes[key]
		if state == nil {
			state = &routeState{members: make(map[string]*runState)}
			s.routes[key] = state
		}
		state.members[runID] = run
		run.routes[key] = struct{}{}
	}
	now := s.now()
	allowed := true
	for key := range keys {
		state := s.routes[key]
		if !state.blocked {
			continue
		}

		if state.waiters != 0 || state.automaticExpired(now) || now.Before(state.next) || state.activeProbe(now) {
			allowed = false
		}
	}
	if allowed {
		for key := range keys {
			state := s.routes[key]
			if state.blocked {
				state.probeRunID = runID
				state.probeUntil = now.Add(max(s.workerRequestTimeout, s.plannerRequestTimeout) + s.policy.MaxDelay)
			}
		}
	}
	return allowed, nil
}

// Bind records a trusted Scheduler allocation binding before Runtime preparation.
func (s *Service) Bind(ctx context.Context, allocationID, runID, stageID string, route Route) error {
	if err := checkContext(ctx); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	run := s.runs[runID]
	if run == nil || run.stageID != stageID {
		return ErrUnavailable
	}
	key := route.Key()
	if _, ok := run.routes[key]; !ok {
		return ErrUnavailable
	}
	binding := bindingKey{"allocation:" + allocationID, route.Model}
	if old, exists := run.bindings[binding]; exists {
		if old != key {
			return ErrInvalid
		}
		return nil
	}
	if len(run.bindings) >= maximumBindingsPerRun || len(s.bindings) >= maximumBindings {
		return errCapacity
	}
	run.bindings[binding] = key
	s.bindings[binding] = run
	return nil
}

// Activate opens the registered Stage's participants only while its Planner is
// running. The Scheduler closes them before beginning finalization or abort.
func (s *Service) Activate(runID, stageID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if run := s.runs[runID]; run != nil && run.stageID == stageID {
		run.active = true
	}
}
