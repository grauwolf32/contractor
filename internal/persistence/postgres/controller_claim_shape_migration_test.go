package postgres

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

// Migration 000014 (controller_claim_shape) gives eval_controller_claims the
// shape of every controller claim table (PER-R25). No lease survives it:
// each held Eval claim, live or expired, is released with its epoch kept, so
// the Eval coordinator leases it again at a higher epoch and a write under a
// pre-upgrade lease finds no live claim. The upgraded schema starts at
// version 13 with free claims at epoch 0 and 5, held claims at nonzero epochs
// (one live, one expired, one whose holder the new holder bound rejects) and
// a held Audit claim. The test fails if the release or the epoch
// preservation is removed, or if the migration touches other rows.
func TestPostgresControllerClaimShapeMigrationReleasesEvalLeases(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, url)
	previous, _ := embeddedMigration(t, "controller_claim_shape")
	installMigrationPrefix(t, ctx, pool, previous)

	longHolder := strings.Repeat("h", 300)
	claims := []struct {
		experiment string
		epoch      int64
		holder     string // empty for a free claim
		expiresIn  string // the lease end relative to the upgrade
	}{
		{experiment: "never-leased", epoch: 0},
		{experiment: "released", epoch: 5},
		{experiment: "held-live", epoch: 4, holder: "old-holder", expiresIn: "1 minute"},
		{experiment: "held-expired", epoch: 7, holder: "old-holder", expiresIn: "-1 minute"},
		{experiment: "held-long-holder", epoch: 2, holder: longHolder, expiresIn: "1 minute"},
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO projects (project_id, owner_id, kind, name, request_idempotency_key, request_digest)
VALUES ('claims-project', 'owner', 'project', 'Claims', 'claims-project', 'sha256:' || repeat('a', 64)),
       ('claims-evaluation', 'owner', 'evaluation', 'Claims evaluation', 'claims-evaluation', 'sha256:' || repeat('b', 64));
INSERT INTO audits (
    audit_id, owner_id, project_id, profile_name, profile_version, profile_digest,
    profile_snapshot, input_selection, max_rounds, batch_size, max_items_per_round,
    max_items_total, max_submitted_runs, max_item_run_attempts, max_evidence_bytes
) VALUES ('claims-audit', 'owner', 'claims-project', 'profile', '1', 'sha256:' || repeat('a', 64),
          '{}', '{}', 1, 1, 10, 10, 10, 1, 1024);
INSERT INTO audit_controller_claims (audit_id, epoch, holder_id, claimed_at, expires_at)
VALUES ('claims-audit', 3, 'audit-holder', clock_timestamp(), clock_timestamp() + interval '1 minute')`); err != nil {
		t.Fatal(err)
	}
	for _, claim := range claims {
		if _, err := pool.Exec(ctx, `
INSERT INTO eval_experiments (
    experiment_id, owner_id, project_id, portable_id, control_mode, name, state, max_in_flight, wall_ms
) VALUES ($1, 'owner', 'claims-evaluation', $1, 'external', 'Claims', 'running', 1, 1000)`, claim.experiment); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `
INSERT INTO eval_controller_claims (experiment_id, epoch, holder_id, expires_at)
VALUES ($1, $2, NULLIF($3, ''), clock_timestamp() + NULLIF($4, '')::interval)`,
			claim.experiment, claim.epoch, claim.holder, claim.expiresIn); err != nil {
			t.Fatal(err)
		}
	}
	const unrelated = `
SELECT (SELECT string_agg(concat_ws('|', experiment_id, state, revision, updated_at), ',' ORDER BY experiment_id)
          FROM eval_experiments),
       (SELECT string_agg(to_jsonb(claim)::text, ',' ORDER BY audit_id) FROM audit_controller_claims AS claim)`
	var experimentsBefore, auditClaimsBefore string
	if err := pool.QueryRow(ctx, unrelated).Scan(&experimentsBefore, &auditClaimsBefore); err != nil {
		t.Fatal(err)
	}

	result, err := ApplyMigrations(ctx, pool, MigrationBudgets{})
	if err != nil {
		t.Fatalf("upgrade the schema with held Eval claims: %v", err)
	}
	if len(result.AppliedVersions) == 0 || result.AppliedVersions[0] != previous+1 {
		t.Fatalf("applied versions %v, want the upgrade to start at %d", result.AppliedVersions, previous+1)
	}
	for _, claim := range claims {
		var epoch int64
		var free bool
		if err := pool.QueryRow(ctx, `
SELECT epoch, holder_id IS NULL AND claimed_at IS NULL AND expires_at IS NULL
  FROM eval_controller_claims WHERE experiment_id = $1`, claim.experiment).Scan(&epoch, &free); err != nil {
			t.Fatal(err)
		}
		if epoch != claim.epoch || !free {
			t.Fatalf("Eval claim %s after the upgrade = epoch %d, free %t; want free at epoch %d",
				claim.experiment, epoch, free, claim.epoch)
		}
	}
	var experimentsAfter, auditClaimsAfter string
	if err := pool.QueryRow(ctx, unrelated).Scan(&experimentsAfter, &auditClaimsAfter); err != nil {
		t.Fatal(err)
	}
	if experimentsAfter != experimentsBefore || auditClaimsAfter != auditClaimsBefore {
		t.Fatalf("the upgrade changed unrelated rows:\nexperiments %s\nbefore      %s\nAudit claims %s\nbefore       %s",
			experimentsAfter, experimentsBefore, auditClaimsAfter, auditClaimsBefore)
	}

	// The Eval store proves and releases claims through this helper.
	evalClaims := NewControllerClaims("eval_controller_claims", "experiment_id")
	lock := func(experiment, holder string, epoch int64) error {
		return InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			return evalClaims.Lock(ctx, tx, experiment, holder, epoch)
		})
	}
	// No pre-upgrade lease authorizes anything, not even the live one.
	for _, claim := range claims {
		if claim.holder == "" {
			continue
		}
		if live, err := evalClaims.Live(ctx, pool, claim.experiment, claim.holder, claim.epoch); err != nil || live {
			t.Fatalf("pre-upgrade lease of %s live = %t, %v", claim.experiment, live, err)
		}
		if err := lock(claim.experiment, claim.holder, claim.epoch); !errors.Is(err, ErrControllerClaimLost) {
			t.Fatalf("write under the pre-upgrade lease of %s = %v, want the claim lost", claim.experiment, err)
		}
		if err := evalClaims.Renew(ctx, pool, claim.experiment, claim.holder, claim.epoch, time.Minute); !errors.Is(err, ErrControllerClaimLost) {
			t.Fatalf("renewal of the pre-upgrade lease of %s = %v, want the claim lost", claim.experiment, err)
		}
		if err := evalClaims.Release(ctx, pool, claim.experiment, claim.holder, claim.epoch); !errors.Is(err, ErrControllerClaimLost) {
			t.Fatalf("release of the pre-upgrade lease of %s = %v, want the claim lost", claim.experiment, err)
		}
	}

	// Leasing again advances every epoch; the same holder name at its old
	// epoch stays fenced off while the new lease authorizes writes.
	rows, err := pool.Query(ctx, `
WITH candidates AS (
    SELECT claim.ctid AS claim_tid
      FROM eval_controller_claims AS claim
     WHERE `+evalClaims.AvailableCondition("claim")+`
     FOR UPDATE SKIP LOCKED
)`+evalClaims.LeaseSQL(), "old-holder", time.Minute.Milliseconds())
	if err != nil {
		t.Fatal(err)
	}
	leased := map[string]int64{}
	for rows.Next() {
		claim, err := ScanControllerClaim(rows)
		if err != nil {
			t.Fatal(err)
		}
		leased[claim.SubjectID] = claim.Epoch
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if len(leased) != len(claims) {
		t.Fatalf("leased %v after the upgrade, want every Eval claim", leased)
	}
	for _, claim := range claims {
		if leased[claim.experiment] != claim.epoch+1 {
			t.Fatalf("lease of %s after the upgrade at epoch %d, want %d", claim.experiment, leased[claim.experiment], claim.epoch+1)
		}
		if err := lock(claim.experiment, "old-holder", claim.epoch); !errors.Is(err, ErrControllerClaimLost) {
			t.Fatalf("write of %s at the epoch before the new lease = %v, want the claim lost", claim.experiment, err)
		}
		if err := lock(claim.experiment, "old-holder", claim.epoch+1); err != nil {
			t.Fatalf("write of %s under the new lease: %v", claim.experiment, err)
		}
	}
	if _, err := pool.Exec(ctx, `
UPDATE eval_controller_claims SET claimed_at = NULL WHERE experiment_id = 'held-live'`); Classify(err) != ClassInvalid ||
		ConstraintName(err) != "eval_controller_claim_shape" {
		t.Fatalf("held Eval claim without its lease start = %v, want the claim shape rejection", err)
	}
}
