package gatewayrecovery

import (
	"context"
	"math"
	"time"

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
	case "acquire", "succeeded", "finished":
		return nil
	case "failed":
		if ValidFailure(r.Code) {
			return nil
		}
	}
	return ErrInvalid
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
	return s.update(ctx, runID, "allocation:"+allocationID, key, request)
}

// Participant is a Server-side planner's invocation, bound to the same authority
// used by Runtime model calls. The route must first pass Admit.
type Participant struct {
	service                          *Service
	runID, participantID, key, model string
}

func (s *Service) Planner(runID, stageID string, route Route) *Participant {
	return &Participant{s, runID, "planner:" + stageID, route.Key(), route.Model}
}
func (p *Participant) Update(ctx context.Context, request Request) (Decision, error) {
	request.Model = p.model
	if err := request.Validate(); err != nil {
		return Decision{}, err
	}
	return p.service.update(ctx, p.runID, p.participantID, p.key, request)
}

func (s *Service) update(ctx context.Context, runID, participantID, key string, request Request) (Decision, error) {
	result := Decision{RequestTimeoutSeconds: s.policy.RequestTimeout.Seconds(), RetryAfterSeconds: s.policy.MaxDelay.Seconds()}
	err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := lockRun(ctx, tx, runID); err != nil {
			return err
		}
		state, err := readRoute(ctx, tx, key)
		if err != nil {
			return err
		}
		now := time.Now()
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
			return setWaiting(ctx, tx, participantID, request.Model, runID, key, request.Code)
		case "acquire":
			result.Code = state.code
			if !state.blocked {
				result.Allowed = true
				return nil
			}
			if err := setWaiting(ctx, tx, participantID, request.Model, runID, key, state.code); err != nil {
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
WHERE route_key=$1`, key, request.RequestID, runID, now.Add(s.policy.RequestTimeout+s.policy.InitialDelay))
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
			return clearWaiting(ctx, tx, participantID, request.Model, runID)
		}
		return ErrInvalid
	})
	return result, err
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
