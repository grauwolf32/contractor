package scanstore

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/jackc/pgx/v5/pgxpool"
)

// The sweep-claim fixture holds sweepsHeld active sweeps that another
// controller holds and that are due first, sweepsDue due active sweeps and
// sweepsCancelling due cancelling sweeps with free claims after them,
// sweepsLater active sweeps not due for an hour, and sweepsSettled settled
// sweeps; the second measurement adds sweepsRetained more settled sweeps.
const (
	sweepsHeld       = 100
	sweepsDue        = 50
	sweepsCancelling = 20
	sweepsLater      = 50
	sweepsSettled    = 2_000
	sweepsRetained   = 20_000
)

// sweepFixtureSQL adds $2 sweeps named $1-<n> in state $3, due $4 from now,
// held by holder $5 for an hour when it is set. Settled sweeps admitted their
// one member.
const sweepFixtureSQL = `
INSERT INTO scan_sweeps (sweep_id, owner_id, workflow, snapshot, snapshot_digest, state, total, pending,
       admitted, skipped, unsettled, settled_at, next_tick_at, holder_id, claimed_at, expires_at)
SELECT $1 || '-' || lpad(n::text, 5, '0'), 'owner', 'scan', '\x7b7d'::bytea, 'sha256:' || repeat('a', 64), $3, 1,
       CASE WHEN $3 IN ('completed', 'cancelled') THEN 0 ELSE 1 END,
       CASE WHEN $3 IN ('completed', 'cancelled') THEN 1 ELSE 0 END, 0,
       CASE WHEN $3 IN ('completed', 'cancelled') THEN 0 ELSE 1 END,
       CASE WHEN $3 IN ('completed', 'cancelled') THEN clock_timestamp() END,
       clock_timestamp() + $4::interval + n * interval '1 millisecond',
       $5, CASE WHEN $5::text IS NOT NULL THEN clock_timestamp() END,
       CASE WHEN $5::text IS NOT NULL THEN clock_timestamp() + interval '1 hour' END
  FROM generate_series(1, $2::integer) AS n`

// TestPostgresSweepClaimReadsOnlyActiveSweeps measures the durable sweep
// controller's claim (V414-028, sibling of the Audit claim): in custom and
// generic plans it reads only the active and cancelling sweeps of
// scan_sweeps_ready_idx, here the held sweeps due before the first free one,
// leases that one, and adding retained settled sweeps changes neither.
func TestPostgresSweepClaimReadsOnlyActiveSweeps(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	pool := isolatedSweepPool(t, ctx)
	insert := func(db *pgxpool.Pool, groups []sweepGroup) {
		t.Helper()
		for _, group := range groups {
			if _, err := db.Exec(ctx, sweepFixtureSQL, group.prefix, group.count, group.state, group.due, group.holder); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := db.Exec(ctx, `VACUUM (INDEX_CLEANUP ON, ANALYZE) scan_sweeps`); err != nil {
			t.Fatal(err)
		}
	}
	insert(pool, []sweepGroup{
		{"held", "active", "-2 hours", sweepsHeld, "sweep-other"},
		{"due", "active", "-1 hour", sweepsDue, nil},
		{"cancelling", "cancelling", "-1 hour", sweepsCancelling, nil},
		{"later", "active", "1 hour", sweepsLater, nil},
		{"settled", "completed", "-3 hours", sweepsSettled, nil},
	})
	analyze := func() {
		if _, err := pool.Exec(ctx, `VACUUM (INDEX_CLEANUP ON, ANALYZE) scan_sweeps`); err != nil {
			t.Fatal(err)
		}
	}
	analyze()
	// The ready index holds every active and cancelling sweep; the claim
	// passes over the held ones due first and leases the first free one.
	ready := int64(sweepsHeld + sweepsDue + sweepsCancelling + sweepsLater)
	type measurement struct{ reads, blocks int64 }
	explain := func(db explaintest.Beginner, phase string, mode explaintest.Mode, leased float64, want int64) measurement {
		t.Helper()
		analysis, err := explaintest.ExplainPrepared(ctx, db, mode, claimSQL, "sweep-controller", int64(30_000))
		if err != nil {
			t.Fatal(err)
		}
		read := analysis.Reads.Of("scan_sweeps")
		if read > ready || read != want || analysis.Plan.Rows != leased {
			t.Errorf("%s, %s plan: the sweep claim leased %.0f sweeps and read %d, want %.0f and %d of the %d ready sweeps:\n%s",
				phase, mode, analysis.Plan.Rows, read, leased, want, ready, analysis.Plan)
		}
		// The candidates are the one limited InitPlan; the lease written
		// after them touches whatever pages hold free space.
		measured := measurement{reads: read, blocks: -1}
		for _, node := range analysis.Plan.Nodes() {
			if node.Relationship == "InitPlan" && node.Type == "Limit" {
				measured.blocks = node.Blocks()
			}
		}
		if measured.blocks < 0 {
			t.Fatalf("the sweep claim plan has no limited candidates InitPlan:\n%s", analysis.Plan)
		}
		t.Logf("%s, %s plan: %d sweeps read, %d blocks for the candidates", phase, mode, read, measured.blocks)
		return measured
	}
	measure := func(phase string) map[explaintest.Mode]measurement {
		t.Helper()
		measured := map[explaintest.Mode]measurement{}
		for _, mode := range explaintest.Modes {
			analyze()
			measured[mode] = explain(pool, phase, mode, 1, sweepsHeld+1)
		}
		return measured
	}
	before := measure(fmt.Sprintf("%d settled sweeps", sweepsSettled))
	if _, err := pool.Exec(ctx, sweepFixtureSQL, "retained", sweepsRetained, "cancelled", "-3 hours", nil); err != nil {
		t.Fatal(err)
	}
	after := measure(fmt.Sprintf("%d settled sweeps", sweepsSettled+sweepsRetained))
	for mode, measured := range after {
		// Each sweep read and the one locked may lie one B-tree level or
		// one page further away in the larger table.
		if measured.reads != before[mode].reads || measured.blocks > before[mode].blocks+measured.reads+1 {
			t.Errorf("%s plan: the sweep claim read %+v with the retained sweeps, %+v without", mode, measured, before[mode])
		}
	}
	claim, ok, err := New(pool).Claim(ctx, "sweep-controller", 30*time.Second)
	if err != nil || !ok || claim.SubjectID != "due-00001" || claim.HolderID != "sweep-controller" || claim.Epoch != 1 {
		t.Fatalf("claim = (%+v, %t, %v), want due-00001 at epoch 1", claim, ok, err)
	}
	// With every due sweep held, the claim reads the whole ready index and
	// leases nothing. The held state is inserted in a schema of its own: the
	// dead versions an UPDATE leaves stay in the index while any other
	// transaction of the database is older than it.
	heldPool := isolatedSweepPool(t, ctx)
	insert(heldPool, []sweepGroup{
		{"held", "active", "-2 hours", sweepsHeld + sweepsDue, "sweep-other"},
		{"cancelling", "cancelling", "-1 hour", sweepsCancelling, "sweep-other"},
		{"later", "active", "1 hour", sweepsLater, nil},
		{"settled", "completed", "-3 hours", sweepsSettled + sweepsRetained, nil},
	})
	for _, mode := range explaintest.Modes {
		explain(heldPool, "every due sweep held", mode, 0, ready)
	}
}

// sweepGroup is one group of the sweep-claim fixture: count sweeps named
// prefix-<n> in state, due due from now and held by holder when it is set.
type sweepGroup struct {
	prefix, state, due string
	count              int
	holder             any
}

// isolatedSweepPool returns a pool on a fresh migrated schema of
// CONTRACTOR_TEST_DATABASE_URL, dropped when the test ends.
func isolatedSweepPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{})
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	return pool
}
