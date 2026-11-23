package scheduler

import (
	"context"
	"github.com/jackc/pgx/v5/pgxpool"
	"testing"
)

// fixtureClaim supplies real ownership to direct lifecycle fixture mutations.
// Its lease is deliberately expired so the next production lane may claim it;
// unchanged ownership remains valid for fixture writes (SCH-R120).
func fixtureClaim(t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID string) context.Context {
	t.Helper()
	if _, err := pool.Exec(ctx, `UPDATE workflow_runs
 SET scheduler_claim_id='fixture-claim',scheduler_claimed_at=clock_timestamp()-interval '2 minutes',scheduler_claim_expires_at=clock_timestamp()-interval '1 minute'
 WHERE run_id=$1 AND scheduler_claim_id IS NULL AND state IN ('initializing','pending','running','waiting','cancelling')`, runID); err != nil {
		t.Fatal(err)
	}
	var id string
	if err := pool.QueryRow(ctx, `SELECT COALESCE(scheduler_claim_id,'fixture-claim') FROM workflow_runs WHERE run_id=$1`, runID).Scan(&id); err != nil {
		t.Fatal(err)
	}
	return WithRunClaim(ctx, runID, id)
}

func memoryFixtureClaim(ctx context.Context, h *schedulerHarness) context.Context {
	h.store.claimID = "fixture-claim"
	return WithRunClaim(ctx, h.store.run.RunID, "fixture-claim")
}
