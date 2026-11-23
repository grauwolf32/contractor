package projectstore

import (
	"context"
	"fmt"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// deletionClaimFixtureSQL loads 12,000 Projects of 30 owners; every
// twentieth is being deleted. Four consecutive deletions share deletion_requested_at, and
// the Project IDs order a tie differently from the load order. The 16 oldest
// deletions hold an unexpired claim; of the others every third holds an
// expired claim and the rest none.
const deletionClaimFixtureSQL = `
INSERT INTO projects (project_id, owner_id, kind, name, request_idempotency_key, request_digest,
    lifecycle_state, deletion_phase, deletion_requested_at, deletion_claim_id, deletion_claimed_at, deletion_claim_expires_at)
SELECT 'project-' || md5(n::text), 'owner-' || (n % 30), 'project', 'Fixture', 'create-' || n, 'sha256:' || repeat('a', 64),
       CASE WHEN deleting THEN 'deleting' ELSE 'active' END,
       CASE WHEN deleting THEN 'draining' END,
       CASE WHEN deleting THEN timestamptz '2026-09-01 00:00:00+00' + (rank / 4) * interval '1 second' END,
       CASE WHEN held OR expired THEN 'claim-' || n END,
       CASE WHEN held OR expired THEN timestamptz '2026-09-01 00:00:00+00' END,
       CASE WHEN held THEN timestamptz '2999-01-01 00:00:00+00'
            WHEN expired THEN timestamptz '2026-09-01 00:01:00+00' END
  FROM generate_series(1, 12000) AS n
 CROSS JOIN LATERAL (SELECT n % 20 = 0 AS deleting, n / 20 AS rank) AS d
 CROSS JOIN LATERAL (SELECT deleting AND rank <= 16 AS held,
                            deleting AND rank > 16 AND rank % 3 = 0 AS expired) AS c`

// TestPostgresProjectDeletionClaimFollowsRequestOrder claims Project
// deletions among 600 deletions and 11,400 active Projects. The claim reads
// projects_deletion_claim_idx in its own order, (deletion_requested_at,
// project_id) over the deleting Projects, so in the custom and the generic
// plan it reads the held claims ahead of the first claimable deletion, that
// deletion and its row again for the update: 20 rows here, where the index
// led by the claim expiry read and sorted all 600 deletions (603 rows).
// Forty claims, half under each plan, take the claimable deletions in exactly
// the order of an independent read, across ties, and skip every held claim.
func TestPostgresProjectDeletionClaimFollowsRequestOrder(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	pool := isolatedProjectPool(t, ctx, databaseURL)
	if _, err := pool.Exec(ctx, deletionClaimFixtureSQL); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `VACUUM ANALYZE projects`); err != nil {
		t.Fatal(err)
	}
	var deleting, held int
	if err := pool.QueryRow(ctx, `SELECT count(*), count(*) FILTER (WHERE deletion_claim_expires_at > clock_timestamp())
 FROM projects WHERE lifecycle_state = 'deleting'`).Scan(&deleting, &held); err != nil || deleting != 600 || held != 16 {
		t.Fatalf("fixture holds %d deletions, %d held: %v", deleting, held, err)
	}

	const claims = 40
	rows, err := pool.Query(ctx, `SELECT project_id FROM projects
 WHERE lifecycle_state = 'deleting' AND (deletion_claim_id IS NULL OR deletion_claim_expires_at <= clock_timestamp())
 ORDER BY deletion_requested_at, project_id LIMIT $1`, claims)
	if err != nil {
		t.Fatal(err)
	}
	want, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil || len(want) != claims {
		t.Fatalf("independent read = %d, %v", len(want), err)
	}

	const budget = 16 + 2 + 8 // held claims ahead, the claimed row read twice, superseded entries
	for _, mode := range explaintest.Modes {
		analysis, err := explaintest.ExplainPrepared(ctx, pool, mode, claimProjectDeletionSQL,
			"claim-measured", time.Minute.Microseconds(), time.Hour.Microseconds(), DeletionPurgingRuns, DeletionPurgingArtifacts)
		if err != nil {
			t.Fatal(err)
		}
		if read := analysis.Reads.Of("projects"); read > budget {
			t.Fatalf("%s: the claim read %d Projects (budget %d):\n%s", mode, read, budget, analysis.Plan)
		}
		t.Logf("%s: the claim read %d Projects", mode, analysis.Reads.Of("projects"))
	}

	var got []string
	for index, mode := range explaintest.Modes {
		config := pool.Config()
		config.ConnConfig.RuntimeParams["plan_cache_mode"] = string(mode)
		claimer, err := pgxpool.NewWithConfig(ctx, config)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(claimer.Close)
		store := NewPostgresStore(claimer)
		for n := range claims / len(explaintest.Modes) {
			claim, found, err := store.ClaimDeletion(ctx, ClaimDeletionParams{
				ClaimID: fmt.Sprintf("claim-%d-%d", index, n), Duration: time.Minute, PurgeDuration: time.Hour,
			})
			if err != nil || !found {
				t.Fatalf("%s claim %d = (%t, %v)", mode, n, found, err)
			}
			got = append(got, claim.ProjectID)
		}
	}
	if !slices.Equal(got, want) {
		t.Fatalf("claims took %v, want %v", got, want)
	}
}
