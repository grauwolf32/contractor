package gatewayrecovery

import (
	"context"
	"errors"
	"math"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
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

// Update resolves the route from an authenticated allocation's durable binding.
func (s *Service) Update(ctx context.Context, allocationID string, request Request) (Decision, error) {
	if err := request.Validate(); err != nil {
		return Decision{}, err
	}
	var runID, key string
	if err := s.pool.QueryRow(ctx, `
SELECT run_id,route_key FROM gateway_allocation_routes
WHERE allocation_id=$1 AND model=$2`, allocationID, request.Model).Scan(&runID, &key); err != nil {
		return Decision{}, err
	}
	return s.update(ctx, runID, "allocation:"+allocationID, key, request, s.workerRequestTimeout)
}

// Participant is a Server-side planner's invocation, bound to the same authority
// used by Runtime model calls. The route must first pass Admit.
type Participant struct {
	service                          *Service
	runID, participantID, key, model string
	signatures                       llmgateway.GatewayFailureSignatures
}

// Planner binds a planner invocation to its route; signatures are the selected
// Gateway's effective failure signatures, applied to every response it reads.
func (s *Service) Planner(runID, stageID string, route Route, signatures llmgateway.GatewayFailureSignatures) *Participant {
	return &Participant{s, runID, "planner:" + stageID, route.Key(), route.Model, signatures}
}
func (p *Participant) Update(ctx context.Context, request Request) (Decision, error) {
	request.Model = p.model
	if err := request.Validate(); err != nil {
		return Decision{}, err
	}
	return p.service.update(ctx, p.runID, p.participantID, p.key, request, p.service.plannerRequestTimeout)
}

// update applies one participant request. requestTimeout bounds that
// participant's own model request and so sizes a probe lease it acquires.
func (s *Service) update(
	ctx context.Context, runID, participantID, key string, request Request, requestTimeout time.Duration,
) (Decision, error) {
	if decision, unchanged, err := s.unchanged(ctx, runID, participantID, key, request); err != nil || unchanged {
		return decision, err
	}
	result := Decision{RetryAfterSeconds: s.policy.MaxDelay.Seconds()}
	err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := lockRun(ctx, tx, s.runs, runID, request.Terminal()); err != nil {
			return err
		}
		state, err := readRoute(ctx, tx, key)
		if err != nil {
			return err
		}
		now, err := transactionNow(ctx, tx)
		if err != nil {
			return err
		}
		switch request.Action {
		case "failed":
			receipt, err := tx.Exec(ctx, `INSERT INTO gateway_recovery_failures(route_key,request_id,run_id)
VALUES($1,$2,$3) ON CONFLICT DO NOTHING`, key, request.RequestID, runID)
			if err != nil {
				return err
			}
			if receipt.RowsAffected() == 0 && !state.blocked {
				return nil
			}
			if receipt.RowsAffected() != 0 {
				delay := s.policy.failureDelay(state.failures)
				// A hint beyond the automatic window causes manual waiting; it must not
				// force an early retry or overflow time.Duration.
				delay = max(delay, time.Duration(min(request.RetryAfterSeconds, s.policy.AutomaticWindow.Seconds())*float64(time.Second)))
				_, err = tx.Exec(ctx, `
UPDATE gateway_recovery_routes SET blocked=true,failure_code=$2,
 failure_count=failure_count+1,
 blocked_at=CASE WHEN blocked THEN blocked_at ELSE $3 END,
 automatic_until=CASE WHEN blocked THEN automatic_until ELSE $4 END,
 next_probe_at=GREATEST(next_probe_at,$5),
 probe_id=CASE WHEN probe_id=$6 THEN NULL ELSE probe_id END,
 probe_run_id=CASE WHEN probe_id=$6 THEN NULL ELSE probe_run_id END,
 probe_until=CASE WHEN probe_id=$6 THEN NULL ELSE probe_until END
WHERE route_key=$1`, key, request.Code, now, now.Add(s.policy.AutomaticWindow), now.Add(delay), request.RequestID)
				if err != nil {
					return err
				}
			}
			return setWaiting(ctx, tx, s.runs, participantID, request.Model, runID, key, request.Code)
		case "acquire":
			result.Code = state.code
			if !state.blocked {
				result.Allowed = true
				return nil
			}
			if err := setWaiting(ctx, tx, s.runs, participantID, request.Model, runID, key, state.code); err != nil {
				return err
			}
			result.RequiresRetry = state.automaticExpired(now)
			if result.RequiresRetry {
				return nil
			}
			ownProbe := state.activeProbe(now) && state.probeRunID != nil && *state.probeRunID == runID && (state.probeID == nil || *state.probeID == request.RequestID)
			if !ownProbe && (state.activeProbe(now) || state.next == nil || now.Before(*state.next)) {
				until := state.next
				if state.activeProbe(now) {
					until = state.probeUntil
				}
				if until != nil {
					result.RetryAfterSeconds = max(s.policy.InitialDelay.Seconds(), min(s.policy.MaxDelay.Seconds(), until.Sub(now).Seconds()))
				}
				return nil
			}
			_, err = tx.Exec(ctx, `
UPDATE gateway_recovery_routes SET probe_id=$2,probe_run_id=$3,probe_until=$4
WHERE route_key=$1`, key, request.RequestID, runID, now.Add(requestTimeout+s.policy.InitialDelay))
			result.Allowed = err == nil
			return err
		case "succeeded", "finished":
			// Only a response to the current probe may reopen the route. A permanent
			// input error proves the model transport answered but fails this invocation.
			if state.probeID != nil && *state.probeID == request.RequestID {
				_, err = tx.Exec(ctx, `
UPDATE gateway_recovery_routes SET blocked=false,failure_code='',failure_count=0,
 blocked_at=NULL,next_probe_at=NULL,automatic_until=NULL,
 probe_id=NULL,probe_run_id=NULL,probe_until=NULL
WHERE route_key=$1`, key)
				if err != nil {
					return err
				}
			}
			return clearWaiting(ctx, tx, s.runs, participantID, request.Model, runID)
		case "released":
			// The Runtime abandoned its granted call without observing a model
			// result. Relinquish only its current probe; a stale release must not
			// clear a newer probe or its participant's wait.
			if state.probeID == nil || *state.probeID != request.RequestID {
				return nil
			}
			_, err = tx.Exec(ctx, `
UPDATE gateway_recovery_routes SET probe_id=NULL,probe_run_id=NULL,probe_until=NULL
WHERE route_key=$1 AND probe_id=$2`, key, request.RequestID)
			if err != nil {
				return err
			}
			return clearWaiting(ctx, tx, s.runs, participantID, request.Model, runID)
		}
		return ErrInvalid
	})
	return result, err
}

// unchanged decides, from one read without row locks or writes, a request
// that changes nothing (SCH-R165): an acquire on a route that is not blocked,
// or a terminal update whose participant holds neither the route's probe nor,
// for succeeded and finished, a wait while its Run is not waiting. Every
// model call of a healthy route is such a pair, so it no longer serializes on
// the Run row and the route row shared by the owner's Runs. The decision is
// the one the locking path would make at the moment of the read: a
// concurrent failure that blocks the route commits after it. Any other
// request, and one whose Run or route the read does not find, takes the
// locking path, which also reports those errors.
func (s *Service) unchanged(
	ctx context.Context, runID, participantID, key string, request Request,
) (Decision, bool, error) {
	var runState, code string
	var blocked, waits bool
	var probeID *string
	err := s.pool.QueryRow(ctx, `
SELECT run.state, route.blocked, route.failure_code, route.probe_id,
       EXISTS (SELECT 1 FROM gateway_recovery_waits WHERE participant_id = $3 AND model = $4)
FROM workflow_runs AS run, gateway_recovery_routes AS route
WHERE run.run_id = $1 AND route.route_key = $2`,
		runID, key, participantID, request.Model).Scan(&runState, &blocked, &code, &probeID, &waits)
	if errors.Is(err, pgx.ErrNoRows) {
		return Decision{}, false, nil
	}
	if err != nil {
		return Decision{}, false, err
	}
	decision := Decision{RetryAfterSeconds: s.policy.MaxDelay.Seconds()}
	switch {
	case request.Action == "acquire":
		if blocked || !acceptsUpdate(runState, false) {
			return Decision{}, false, nil
		}
		decision.Allowed, decision.Code = true, code
		return decision, true, nil
	case request.Terminal():
		ownProbe := probeID != nil && *probeID == request.RequestID
		// released clears only its own probe; succeeded and finished also
		// clear the participant's wait and may return a waiting Run to running.
		clearsWait := request.Action != "released" && (waits || runState == "waiting")
		if ownProbe || clearsWait || !acceptsUpdate(runState, true) {
			return Decision{}, false, nil
		}
		return decision, true, nil
	default:
		return Decision{}, false, nil
	}
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
