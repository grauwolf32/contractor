package gatewayrecovery

import (
	"context"
	"math"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
)

// Request contains only safe control metadata. Provider messages and secrets
// must be classified and discarded before reaching the recovery authority.
type Request struct {
	Model             string  `json:"model"`
	RequestID         string  `json:"requestId"`
	Action            string  `json:"action"`
	Code              string  `json:"code,omitempty"`
	RetryAfterSeconds float64 `json:"retryAfterSeconds,omitempty"`
}

func (r Request) Validate() error {
	// Same opaque-identifier bound used by public resource identifiers.
	const maximumIdentifierLength = 256
	if r.RequestID == "" || len(r.RequestID) > maximumIdentifierLength || r.Model == "" || len(r.Model) > maximumIdentifierLength {
		return ErrInvalid
	}
	if math.IsNaN(r.RetryAfterSeconds) || math.IsInf(r.RetryAfterSeconds, 0) || r.RetryAfterSeconds < 0 {
		return ErrInvalid
	}
	switch r.Action {
	case "acquire", "succeeded", "finished", "released":
		return nil
	case "failed":
		if ValidFailure(r.Code) {
			return nil
		}
	}
	return ErrInvalid
}

// Terminal reports the terminal update of a granted attempt: an observed
// response (succeeded, finished) or an abandoned call (released). Each one
// relinquishes the sender's probe and wait, so it stays deliverable after the
// sender's Stage began to end (SCH-R108): from a write-fenced allocation and
// while the Run is cancelling. acquire and failed are not terminal: an ending
// Stage must neither start a model call nor make its Run wait.
func (r Request) Terminal() bool {
	switch r.Action {
	case "succeeded", "finished", "released":
		return true
	default:
		return false
	}
}

// Update resolves an authenticated allocation from the Scheduler's memory binding.
func (s *Service) Update(ctx context.Context, allocationID string, request Request) (Decision, error) {
	if err := request.Validate(); err != nil {
		return Decision{}, err
	}
	if err := checkContext(ctx); err != nil {
		return Decision{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	binding := bindingKey{"allocation:" + allocationID, request.Model}
	run := s.bindings[binding]
	if run == nil {
		if request.Terminal() {
			return Decision{}, nil
		}
		return Decision{}, ErrUnavailable
	}
	return s.update(run, binding, run.bindings[binding], request, s.workerRequestTimeout)
}

type Participant struct {
	service                    *Service
	runID, stageID, key, model string
	signatures                 llmgateway.GatewayFailureSignatures
}

// Planner references an already admitted Stage; it never registers a missing
// participant. An old Participant cannot recreate state after cleanup or restart.
func (s *Service) Planner(runID, stageID string, route Route, signatures llmgateway.GatewayFailureSignatures) *Participant {
	return &Participant{s, runID, stageID, route.Key(), route.Model, signatures}
}
func (p *Participant) Update(ctx context.Context, request Request) (Decision, error) {
	request.Model = p.model
	if err := request.Validate(); err != nil {
		return Decision{}, err
	}
	if err := checkContext(ctx); err != nil {
		return Decision{}, err
	}
	s := p.service
	s.mu.Lock()
	defer s.mu.Unlock()
	run := s.runs[p.runID]
	if run == nil || run.stageID != p.stageID {
		if request.Terminal() {
			return Decision{}, nil
		}
		return Decision{}, ErrUnavailable
	}
	return s.update(run, bindingKey{"planner:" + p.stageID, p.model}, p.key, request, s.plannerRequestTimeout)
}

// update holds only the process mutex; no database or network work runs here.
func (s *Service) update(run *runState, participant bindingKey, key string, request Request, timeout time.Duration) (Decision, error) {
	result := Decision{RetryAfterSeconds: s.policy.MaxDelay.Seconds()}
	state := s.routes[key]
	if state == nil {
		return result, ErrUnavailable
	}
	if _, ok := run.routes[key]; !ok {
		return result, ErrUnavailable
	}
	if !run.active && !request.Terminal() {
		return result, ErrUnavailable
	}
	now := s.now()
	switch request.Action {
	case "failed":
		// Request IDs are opaque: frame the route and ID rather than concatenate.
		receipt := key + ":" + request.RequestID
		if _, seen := run.failures[receipt]; !seen {
			if s.failureReceipts >= maximumFailures {
				return result, errCapacity
			}
			run.failures[receipt] = struct{}{}
			s.failureReceipts++
			delay := max(s.policy.failureDelay(state.failures), time.Duration(min(request.RetryAfterSeconds, s.policy.AutomaticWindow.Seconds())*float64(time.Second)))
			if !state.blocked {
				state.blockedAt = now
				state.automaticUntil = now.Add(s.policy.AutomaticWindow)
			}
			state.blocked, state.code = true, request.Code
			state.failures++
			if next := now.Add(delay); next.After(state.next) {
				state.next = next
			}
			if state.probeID == request.RequestID {
				state.release()
			}
		} else if !state.blocked {
			return result, nil
		}
		s.setWaiting(run, participant, key)
		run.notify()
	case "acquire":
		result.Code = state.code
		if !state.blocked {
			result.Allowed = true
			return result, nil
		}
		if run.waits[participant] != key {
			s.setWaiting(run, participant, key)
		}
		// Another participant may have changed the shared route's cause.
		// Wake this Run's projection even when its membership already exists.
		run.notify()
		result.RequiresRetry = state.automaticExpired(now)
		if result.RequiresRetry {
			return result, nil
		}
		ownProbe := state.activeProbe(now) && state.probeRunID == run.id && (state.probeID == "" || state.probeID == request.RequestID)
		if !ownProbe && (state.activeProbe(now) || now.Before(state.next)) {
			until := state.next
			if state.activeProbe(now) {
				until = state.probeUntil
			}
			result.RetryAfterSeconds = max(s.policy.InitialDelay.Seconds(), min(s.policy.MaxDelay.Seconds(), until.Sub(now).Seconds()))
			return result, nil
		}
		state.probeID, state.probeRunID, state.probeUntil = request.RequestID, run.id, now.Add(timeout+s.policy.InitialDelay)
		result.Allowed = true
	case "succeeded", "finished":
		if state.probeID == request.RequestID && state.probeRunID == run.id {
			state.blocked, state.code, state.failures = false, "", 0
			state.blockedAt, state.next, state.automaticUntil = time.Time{}, time.Time{}, time.Time{}
			state.release()
		}
		if _, waits := run.waits[participant]; waits {
			s.clearWaiting(run, participant)
			run.notify()
		}
	case "released":
		if state.probeID == request.RequestID && state.probeRunID == run.id {
			state.release()
			if _, waits := run.waits[participant]; waits {
				s.clearWaiting(run, participant)
				run.notify()
			}
		}
	}
	return result, nil
}

func (p Policy) failureDelay(failures int64) time.Duration {
	delay := p.InitialDelay
	for failures > 0 && delay < p.MaxDelay {
		if delay > p.MaxDelay/2 {
			return p.MaxDelay
		}
		delay *= 2
		failures--
	}
	return min(delay, p.MaxDelay)
}

func (s *Service) setWaiting(run *runState, participant bindingKey, key string) {
	if old, exists := run.waits[participant]; exists {
		if old == key {
			return
		}
		s.routes[old].waiters--
	}
	run.waits[participant] = key
	s.routes[key].waiters++
}
func (s *Service) clearWaiting(run *runState, participant bindingKey) {
	if old, exists := run.waits[participant]; exists {
		s.routes[old].waiters--
		delete(run.waits, participant)
	}
}
