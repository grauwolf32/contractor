package gatewayrecovery

import (
	"context"
	"sort"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// Admit checks all routes before reserving any probe. Existing invocations get
// first opportunity to recover; new Runs do not drain through an unavailable model.
func (s *Service) Admit(ctx context.Context, runID string, routes []Route) (bool, error) {
	allowed := true
	ordered := append([]Route(nil), routes...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Key() < ordered[j].Key() })
	err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := lockRun(ctx, tx, runID); err != nil {
			return err
		}
		now, err := transactionNow(ctx, tx)
		if err != nil {
			return err
		}
		probes := map[string]bool{}
		for _, route := range ordered {
			key := route.Key()
			if _, err := tx.Exec(ctx, `
INSERT INTO gateway_recovery_routes(route_key,owner_id) VALUES($1,$2)
ON CONFLICT DO NOTHING`, key, route.OwnerID); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `
INSERT INTO gateway_run_routes(run_id,route_key) VALUES($1,$2)
ON CONFLICT DO NOTHING`, runID, key); err != nil {
				return err
			}
			state, err := readRoute(ctx, tx, key)
			if err != nil {
				return err
			}
			if !state.blocked {
				continue
			}
			var waiter bool
			if err := tx.QueryRow(ctx, `
SELECT EXISTS(SELECT 1 FROM gateway_recovery_waits w
JOIN workflow_runs r USING(run_id)
WHERE w.route_key=$1 AND r.state IN ('running','waiting'))`, key).Scan(&waiter); err != nil {
				return err
			}
			if waiter || state.automaticExpired(now) || state.next == nil || now.Before(*state.next) || state.activeProbe(now) {
				allowed = false
				if _, err := tx.Exec(ctx, `
UPDATE workflow_runs SET state_reason_code=$2,
 state_reason_message='Waiting for the model gateway', updated_at=clock_timestamp()
WHERE run_id=$1 AND state='pending' AND state_reason_code IS DISTINCT FROM $2`, runID, state.code); err != nil {
					return err
				}
			} else {
				probes[key] = true
			}
		}
		if !allowed {
			return nil
		}
		for key := range probes {
			// Include preparation time in this reservation. A crashed scheduler cannot
			// hold the gate indefinitely; the first request takes a request-sized lease.
			if _, err := tx.Exec(ctx, `
UPDATE gateway_recovery_routes SET probe_id=NULL,probe_run_id=$2,probe_until=$3
WHERE route_key=$1`, key, runID, now.Add(s.policy.RequestTimeout+s.policy.MaxDelay)); err != nil {
				return err
			}
		}
		return nil
	})
	return allowed, err
}

func (s *Service) Bind(ctx context.Context, allocationID, runID string, route Route) error {
	_, err := s.pool.Exec(ctx, `
INSERT INTO gateway_allocation_routes(allocation_id,model,run_id,route_key)
VALUES($1,$2,$3,$4) ON CONFLICT(allocation_id,model) DO NOTHING`, allocationID, route.Model, runID, route.Key())
	return err
}
