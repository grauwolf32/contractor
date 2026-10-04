package evalstore

import (
	"context"
	"testing"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/jackc/pgx/v5/pgxpool"
)

type viewRows struct{ generations, members, pairs, charts, progress int }

func countViewRows(t *testing.T, pool *pgxpool.Pool, id string) viewRows {
	t.Helper()
	var r viewRows
	err := pool.QueryRow(context.Background(), `
SELECT (SELECT count(*) FROM eval_view_generations WHERE experiment_id = $1),
    (SELECT 2 * count(*) FROM eval_view_pairs WHERE experiment_id = $1),
    (SELECT count(*) FROM eval_view_pairs WHERE experiment_id = $1),
    (SELECT count(*) FROM eval_view_charts WHERE experiment_id = $1),
    (SELECT count(*) FROM eval_progress_observations WHERE experiment_id = $1)
`, id).Scan(&r.generations, &r.members, &r.pairs, &r.charts, &r.progress)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestPostgresEvalViewPublicationSkipsUnchangedAndPrunesSuperseded(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	scope := setupProject(t, pool, "owner", "eval")
	e := createExperiment(t, pool, scope, "experiment", "trace-1", "external-workflow")
	ms := members(t, pool, e)
	// A submission starts the clock, so every new generation records progress.
	if _, err := admit(t, pool, e, ms[0].MemberID, nil, "submit-first"); err != nil {
		t.Fatal(err)
	}
	claim := oneClaim(t, pool)
	reader := NewPostgresStore(pool)
	comparison := evaldomain.Comparison{Baseline: "a", Candidate: "b", Gates: evaldomain.Gates{MinCandidateEndToEndPass: 1}}
	project := func(state string) {
		t.Helper()
		dirty, err := reader.DirtyMembers(ctx, e.OwnerID, e.ID, 100)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range dirty {
			for _, m := range ms {
				if m.MemberID != d.MemberID {
					continue
				}
				v := evaldomain.MemberView{Member: evaldomain.MemberIdentity{ID: m.MemberID, SuiteID: m.SuiteID, CaseID: m.CaseID, Sample: m.Sample, VariantID: m.VariantID, CaseSHA256: m.CaseSHA256, BindingSHA256: m.BindingSHA256, Eligibility: m.Eligibility}, Execution: &evaldomain.ExecutionView{State: state}, Assessment: "unscored"}
				if state == "unknown" {
					reason := "Fixture execution outcome is being reconciled."
					v.Execution.Reason = &reason
				}
				mustTx(t, pool, func(st *Store) error {
					return st.ProjectMember(ctx, scope, e.ID, m.MemberID, claim, d.Revision, v, false, nil)
				})
			}
		}
	}
	publish := func() *View {
		t.Helper()
		var v *View
		mustTx(t, pool, func(st *Store) error {
			var err error
			v, err = st.PublishView(ctx, scope, e.ID, claim, comparison, true)
			return err
		})
		if v == nil || v.Freshness != "current" {
			t.Fatalf("published view = %+v", v)
		}
		return v
	}
	dirty := func() {
		t.Helper()
		if _, err := pool.Exec(ctx, `SELECT contractor_eval_dirty($1,$2)`, e.ID, ms[1].MemberID); err != nil {
			t.Fatal(err)
		}
	}

	project("not_submitted")
	first := publish()
	one := countViewRows(t, pool, e.ID)
	if one.generations != 1 || one.members != len(ms) || one.pairs != len(ms)/2 || one.charts == 0 || one.progress != 1 {
		t.Fatalf("first generation rows = %+v", one)
	}

	// A dirty event that leaves every selected document unchanged.
	dirty()
	if stale, err := reader.LatestView(ctx, e.OwnerID, e.ID); err != nil || stale.Freshness != "stale" {
		t.Fatalf("dirty view = %+v, %v", stale, err)
	}
	project("not_submitted")
	same := publish()
	if same.Snapshot != first.Snapshot || same.Generation != first.Generation || !same.CreatedAt.Equal(first.CreatedAt) {
		t.Fatalf("unchanged documents created generation %d (%s), want %d (%s)", same.Generation, same.Snapshot, first.Generation, first.Snapshot)
	}
	if rows := countViewRows(t, pool, e.ID); rows != one {
		t.Fatalf("unchanged publication rows = %+v, want %+v", rows, one)
	}

	// A visible change replaces the generation and prunes the superseded one.
	dirty()
	project("unknown")
	changed := publish()
	if changed.Snapshot == first.Snapshot || changed.Generation <= first.Generation {
		t.Fatalf("changed documents kept snapshot %s generation %d", changed.Snapshot, changed.Generation)
	}
	if rows := countViewRows(t, pool, e.ID); rows.generations != 1 || rows.members != one.members || rows.pairs != one.pairs || rows.charts != one.charts || rows.progress != 2 {
		t.Fatalf("replacement rows = %+v, want one generation like %+v and two progress observations", rows, one)
	}
	page := func(generation int64) int {
		t.Helper()
		members, err := reader.SelectedMemberPage(ctx, SelectedPageParams{OwnerID: e.OwnerID, ExperimentID: e.ID, Generation: generation, AfterOrdinal: -1, Limit: 100})
		if err != nil {
			t.Fatal(err)
		}
		return len(members.Items)
	}
	if page(first.Generation) != 0 || page(changed.Generation) != len(ms) {
		t.Fatal("superseded generation retained or current generation unreadable")
	}
	pairs, err := reader.PairPage(ctx, SelectedPageParams{OwnerID: e.OwnerID, ExperimentID: e.ID, Generation: changed.Generation, AfterOrdinal: -1, Limit: 100})
	if err != nil || len(pairs.Items) != len(ms)/2 {
		t.Fatalf("current pairs = %d, %v", len(pairs.Items), err)
	}
	for _, statement := range []string{
		`DELETE FROM eval_view_generations WHERE experiment_id = $1`,
		`DELETE FROM eval_view_pairs WHERE experiment_id = $1`,
		`DELETE FROM eval_view_charts WHERE experiment_id = $1`,
		`UPDATE eval_view_generations SET pins_verified = NOT pins_verified WHERE experiment_id = $1`,
	} {
		if _, err := pool.Exec(ctx, statement, e.ID); err == nil {
			t.Fatalf("current generation changed by %s", statement)
		}
	}

	// Recovered documents equal to the pruned generation still get a new snapshot.
	dirty()
	project("not_submitted")
	recovered := publish()
	if recovered.Snapshot == first.Snapshot || recovered.Snapshot == changed.Snapshot || recovered.Generation <= changed.Generation {
		t.Fatalf("recovery reused snapshot %s generation %d", recovered.Snapshot, recovered.Generation)
	}
	if rows := countViewRows(t, pool, e.ID); rows.generations != 1 || rows.members != one.members || rows.progress != 3 {
		t.Fatalf("recovery rows = %+v", rows)
	}
}
