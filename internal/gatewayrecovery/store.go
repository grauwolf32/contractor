package gatewayrecovery

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Service struct {
	pool   *pgxpool.Pool
	policy Policy
}

func New(pool *pgxpool.Pool, policy Policy) (*Service, error) {
	if pool == nil {
		return nil, ErrInvalid
	}
	if err := policy.Validate(); err != nil {
		return nil, err
	}
	return &Service{pool: pool, policy: policy}, nil
}

type routeState struct {
	blocked                          bool
	code                             string
	failures                         int64
	next, automaticUntil, probeUntil *time.Time
	probeID, probeRunID              *string
}

func readRoute(ctx context.Context, tx pgx.Tx, key string) (routeState, error) {
	var state routeState
	err := tx.QueryRow(ctx, `
SELECT blocked, failure_code, failure_count, next_probe_at, automatic_until,
       probe_until, probe_id, probe_run_id
FROM gateway_recovery_routes WHERE route_key=$1 FOR UPDATE`, key).Scan(
		&state.blocked, &state.code, &state.failures, &state.next, &state.automaticUntil,
		&state.probeUntil, &state.probeID, &state.probeRunID)
	return state, err
}

// transactionNow reads the database clock so route deadlines written here
// compare consistently with the clock_timestamp() predicates in Status and
// Admit, whatever skew exists between Server and PostgreSQL hosts.
func transactionNow(ctx context.Context, tx pgx.Tx) (time.Time, error) {
	var now time.Time
	if err := tx.QueryRow(ctx, `SELECT clock_timestamp()`).Scan(&now); err != nil {
		return time.Time{}, err
	}
	return now, nil
}

func lockRun(ctx context.Context, tx pgx.Tx, runID string) error {
	var state string
	if err := tx.QueryRow(ctx, `SELECT state FROM workflow_runs WHERE run_id=$1 FOR UPDATE`, runID).Scan(&state); err != nil {
		return err
	}
	if state != "pending" && state != "running" && state != "waiting" {
		return ErrUnavailable
	}
	return nil
}

func (s routeState) automaticExpired(now time.Time) bool {
	return s.automaticUntil == nil || !now.Before(*s.automaticUntil)
}
func (s routeState) activeProbe(now time.Time) bool {
	return s.probeUntil != nil && now.Before(*s.probeUntil)
}

func IsUnavailable(err error) bool {
	return errors.Is(err, ErrUnavailable) || errors.Is(err, pgx.ErrNoRows)
}
