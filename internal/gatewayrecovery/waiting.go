package gatewayrecovery

import (
	"context"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func setWaiting(ctx context.Context, tx pgx.Tx, participantID, model, runID, key, code string) error {
	if _, err := tx.Exec(ctx, `
INSERT INTO gateway_recovery_waits(participant_id,model,run_id,route_key)
VALUES($1,$2,$3,$4) ON CONFLICT DO NOTHING`, participantID, model, runID, key); err != nil {
		return err
	}
	_, err := tx.Exec(ctx, `
UPDATE workflow_runs SET state='waiting',state_reason_code=$2,
 state_reason_message='Waiting for the model gateway',updated_at=clock_timestamp()
WHERE run_id=$1 AND state IN ('running','waiting')
 AND (state<>'waiting' OR state_reason_code IS DISTINCT FROM $2)`, runID, code)
	return err
}

func clearWaiting(ctx context.Context, tx pgx.Tx, participantID, model, runID string) error {
	if _, err := tx.Exec(ctx, `DELETE FROM gateway_recovery_waits WHERE participant_id=$1 AND model=$2`, participantID, model); err != nil {
		return err
	}
	return resumeRun(ctx, tx, runID)
}

// resumeRun returns a waiting Run to running once no participant of it waits.
func resumeRun(ctx context.Context, tx pgx.Tx, runID string) error {
	_, err := tx.Exec(ctx, `
UPDATE workflow_runs SET state='running',state_reason_code='gateway_recovered',
 state_reason_message='',updated_at=clock_timestamp()
WHERE run_id=$1 AND state='waiting'
 AND NOT EXISTS(SELECT 1 FROM gateway_recovery_waits WHERE run_id=$1)`, runID)
	return err
}

// EndStage drops the model recovery state of a Run's ending Stage inside the
// transaction that ends it; the caller already holds the Run row lock. The
// Stage's waits and route membership go, and any probe the Run holds or
// reserved is relinquished without reopening the route or resetting its
// retry window, so another Run can probe at once instead of after the lease
// (SCH-R107, SCH-R108). A waiting Run returns to running.
//
// Route rows are locked in key order, like Admit and Retry, so a Run holding
// probes on several routes cannot deadlock with them. gateway_recovery_routes
// holds one row per distinct owner model route, so the probe scan is bounded
// by that catalog, not by Runs or Stages.
func EndStage(ctx context.Context, tx pgx.Tx, runID string) error {
	if _, err := tx.Exec(ctx, `DELETE FROM gateway_recovery_waits WHERE run_id=$1`, runID); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx, `DELETE FROM gateway_run_routes WHERE run_id=$1`, runID); err != nil {
		return err
	}
	if err := relinquishProbes(ctx, tx, runID, false); err != nil {
		return err
	}
	return resumeRun(ctx, tx, runID)
}

// ReleaseAdmission relinquishes the probes that admission reserved for a Run
// whose placement was discarded before its Stage was admitted, inside the
// caller's transaction, which holds the Run row lock. The Run may wait for a
// paused queue for long; its reservation must not refuse other Runs' probes
// meanwhile. Its route membership stays, so its status still names the route.
func ReleaseAdmission(ctx context.Context, tx pgx.Tx, runID string) error {
	return relinquishProbes(ctx, tx, runID, true)
}

// relinquishProbes clears the probes a Run holds, or only those admission
// reserved, locking route rows in key order like Admit and Retry.
func relinquishProbes(ctx context.Context, tx pgx.Tx, runID string, admissionOnly bool) error {
	_, err := tx.Exec(ctx, `
UPDATE gateway_recovery_routes SET probe_id=NULL,probe_run_id=NULL,probe_until=NULL
WHERE route_key IN (
 SELECT route_key FROM gateway_recovery_routes
 WHERE probe_run_id=$1 AND (NOT $2 OR probe_id IS NULL)
 ORDER BY route_key FOR UPDATE)`, runID, admissionOnly)
	return err
}

// Status deliberately excludes provider data, route identities and credentials.
type Status struct {
	Code           string     `json:"code"`
	Since          time.Time  `json:"since"`
	NextRetryAt    *time.Time `json:"nextRetryAt,omitempty"`
	AutomaticUntil time.Time  `json:"automaticUntil"`
	RequiresRetry  bool       `json:"requiresRetry"`
}

// ReadStatus reads the recovery status of an active Run through db, a pool or
// the caller's transaction, so a Run detail can read it in the same snapshot
// as the Run state. It reports nil when no route of the Run is blocked.
func ReadStatus(ctx context.Context, db persistencepostgres.DBTX, runID string) (*Status, error) {
	var status Status
	err := db.QueryRow(ctx, `
SELECT g.failure_code,g.blocked_at,
 CASE WHEN g.automatic_until>clock_timestamp()
      THEN GREATEST(g.next_probe_at,g.probe_until) ELSE NULL END,
 g.automatic_until,g.automatic_until<=clock_timestamp()
FROM gateway_recovery_routes g JOIN gateway_run_routes b USING(route_key)
JOIN workflow_runs r USING(run_id)
WHERE b.run_id=$1 AND g.blocked AND r.state IN ('pending','running','waiting')
ORDER BY (g.automatic_until<=clock_timestamp()) DESC,g.next_probe_at DESC
LIMIT 1`, runID).Scan(&status.Code, &status.Since, &status.NextRetryAt, &status.AutomaticUntil, &status.RequiresRetry)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &status, nil
}

func (s *Service) Retry(ctx context.Context, ownerID, runID string) error {
	return persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := lockRun(ctx, tx, runID, false); err != nil {
			return err
		}
		var owner string
		if err := tx.QueryRow(ctx, `SELECT owner_id FROM workflow_runs WHERE run_id=$1`, runID).Scan(&owner); err != nil {
			return err
		}
		if owner != ownerID {
			return ErrUnavailable
		}
		// Take route locks in the same order as admission, avoiding multi-route
		// deadlocks. An active probe retains its lease when the user clicks Retry.
		rows, err := tx.Query(ctx, `
SELECT g.route_key FROM gateway_recovery_routes g
JOIN gateway_run_routes b USING(route_key)
WHERE b.run_id=$1 AND g.owner_id=$2 AND g.blocked
ORDER BY g.route_key FOR UPDATE OF g`, runID, ownerID)
		if err != nil {
			return err
		}
		keys, err := pgx.CollectRows(rows, pgx.RowTo[string])
		if err != nil {
			return err
		}
		if len(keys) == 0 {
			return ErrUnavailable
		}
		_, err = tx.Exec(ctx, `
UPDATE gateway_recovery_routes SET automatic_until=clock_timestamp()+$2,next_probe_at=clock_timestamp()
WHERE route_key=ANY($1)`, keys, s.policy.AutomaticWindow)
		return err
	})
}
