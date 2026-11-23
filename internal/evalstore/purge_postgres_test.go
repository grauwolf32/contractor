package evalstore

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// purgePass is one committed Eval purge pass and the row mutations
// PostgreSQL counted in its transaction: every insert, update and delete of
// every table, cascades and trigger writes included.
type purgePass struct {
	PurgeProgress
	Mutations int64
	tables    string
}

// xactMutationsSQL counts the current transaction's row mutations in every
// table of the test schema.
const xactMutationsSQL = `
SELECT COALESCE(sum(pg_stat_get_xact_tuples_inserted(c.oid) + pg_stat_get_xact_tuples_updated(c.oid)
    + pg_stat_get_xact_tuples_deleted(c.oid)), 0)::bigint
FROM pg_class c
WHERE c.relnamespace = current_schema()::regnamespace AND c.relkind = 'r'`

// xactMutationsByTableSQL names the tables the counters of xactMutationsSQL
// cover, for a failure message.
const xactMutationsByTableSQL = `
SELECT COALESCE(string_agg(c.relname || '=' || (pg_stat_get_xact_tuples_inserted(c.oid) + pg_stat_get_xact_tuples_updated(c.oid)
    + pg_stat_get_xact_tuples_deleted(c.oid)), ' ' ORDER BY c.relname), '')
FROM pg_class c
WHERE c.relnamespace = current_schema()::regnamespace AND c.relkind = 'r'
    AND pg_stat_get_xact_tuples_inserted(c.oid) + pg_stat_get_xact_tuples_updated(c.oid) + pg_stat_get_xact_tuples_deleted(c.oid) > 0`

// runPass runs one purge pass in its own transaction and counts its row
// mutations before the commit.
func runPass(ctx context.Context, pool *pgxpool.Pool, fn func(*Store) (PurgeProgress, error)) (purgePass, error) {
	var pass purgePass
	err := pg.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		// The counters also hold earlier transactions of the connection whose
		// statistics are not flushed yet, so the pass is their difference.
		var earlier int64
		if err := tx.QueryRow(ctx, xactMutationsSQL).Scan(&earlier); err != nil {
			return err
		}
		progress, err := fn(NewTxStore(tx, artifactpgtest.NewStorage()))
		if err != nil {
			return err
		}
		pass.PurgeProgress = progress
		if err = tx.QueryRow(ctx, xactMutationsSQL).Scan(&pass.Mutations); err != nil {
			return err
		}
		pass.Mutations -= earlier
		return tx.QueryRow(ctx, xactMutationsByTableSQL).Scan(&pass.tables)
	})
	return pass, err
}

// checkPass holds a committed pass to its budget: the rows it accounted are
// exactly the rows PostgreSQL changed, at most CleanupBatchRows, their stored
// size is within LIM-EVM-20, and a pass that is not done made progress.
func checkPass(t *testing.T, label string, pass purgePass) {
	t.Helper()
	if pass.Mutations != pass.Rows {
		t.Fatalf("%s accounted %d row mutations, PostgreSQL counted %d: %s", label, pass.Rows, pass.Mutations, pass.tables)
	}
	if pass.Rows > pg.CleanupBatchRows || pass.Bytes > purgeBatchBytes {
		t.Fatalf("%s spent %d rows and %d bytes, budget %d rows and %d bytes", label, pass.Rows, pass.Bytes, pg.CleanupBatchRows, purgeBatchBytes)
	}
	if !pass.Done && pass.Rows == 0 {
		t.Fatalf("%s made no progress and is not done", label)
	}
}

var claimHolders atomic.Int64

// claimExperiment takes the Eval controller claim of one experiment under a
// new holder, as a new controller would, and releases any other it got.
func claimExperiment(t *testing.T, pool *pgxpool.Pool, id string) Claim {
	t.Helper()
	ctx := context.Background()
	store := NewPostgresStore(pool, artifactpgtest.NewStorage())
	claims, err := store.Claim(ctx, fmt.Sprintf("controller-%d", claimHolders.Add(1)), time.Minute, 100)
	if err != nil {
		t.Fatal(err)
	}
	var out *Claim
	for _, claim := range claims {
		if claim.ExperimentID == id {
			out = &claim
			continue
		}
		if err := store.ReleaseClaim(ctx, claim); err != nil {
			t.Fatal(err)
		}
	}
	if out == nil {
		t.Fatalf("experiment %s is not claimable", id)
	}
	return *out
}

func releaseClaim(t *testing.T, pool *pgxpool.Pool, claim Claim) {
	t.Helper()
	if err := NewPostgresStore(pool, artifactpgtest.NewStorage()).ReleaseClaim(context.Background(), claim); err != nil && !errors.Is(err, ErrClaimLost) {
		t.Fatal(err)
	}
}

// purgeExperimentUntilDone runs experiment purge passes, each under a fresh
// claim in its own transaction, until one reports Done.
func purgeExperimentUntilDone(t *testing.T, pool *pgxpool.Pool, scope Scope, id string) []purgePass {
	t.Helper()
	var passes []purgePass
	for len(passes) < 100 {
		claim := claimExperiment(t, pool, id)
		pass, err := runPass(context.Background(), pool, func(s *Store) (PurgeProgress, error) {
			return s.Purge(context.Background(), scope, id, claim)
		})
		if err != nil {
			t.Fatalf("experiment purge pass %d: %v", len(passes)+1, err)
		}
		releaseClaim(t, pool, claim)
		checkPass(t, fmt.Sprintf("experiment purge pass %d", len(passes)+1), pass)
		passes = append(passes, pass)
		if pass.Done {
			return passes
		}
	}
	t.Fatalf("experiment purge did not finish: %+v", passes)
	return nil
}

// purgeProjectUntilDone runs Project Eval purge passes, each in its own
// transaction, until one reports Done.
func purgeProjectUntilDone(t *testing.T, pool *pgxpool.Pool, scope Scope) []purgePass {
	t.Helper()
	var passes []purgePass
	for len(passes) < 100 {
		pass, err := runPass(context.Background(), pool, func(s *Store) (PurgeProgress, error) {
			return s.PurgeProject(context.Background(), scope)
		})
		if err != nil {
			t.Fatalf("Project Eval purge pass %d: %v", len(passes)+1, err)
		}
		checkPass(t, fmt.Sprintf("Project Eval purge pass %d", len(passes)+1), pass)
		passes = append(passes, pass)
		if pass.Done {
			return passes
		}
	}
	t.Fatalf("Project Eval purge did not finish: %+v", passes)
	return nil
}

// evalTables lists the tables of the test schema whose rows an Eval purge
// removes, with every column the schema gives them, independent of the
// purge's own lists.
func evalTables(t *testing.T, pool *pgxpool.Pool) map[string][]string {
	t.Helper()
	rows, err := pool.Query(context.Background(), `
SELECT table_name, array_agg(column_name::text ORDER BY ordinal_position)
FROM information_schema.columns
WHERE table_schema = current_schema() AND (table_name LIKE 'eval\_%' OR table_name = 'mutation_receipts')
GROUP BY table_name`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	out := map[string][]string{}
	for rows.Next() {
		var table string
		var columns []string
		if err := rows.Scan(&table, &columns); err != nil {
			t.Fatal(err)
		}
		out[table] = columns
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}

// experimentFootprint is the rows and stored bytes an experiment holds in
// every Eval table with an experiment_id column.
type experimentFootprint struct {
	rows  map[string]int64
	bytes int64
}

func footprint(t *testing.T, pool *pgxpool.Pool, id string) experimentFootprint {
	t.Helper()
	out := experimentFootprint{rows: map[string]int64{}}
	for table, columns := range evalTables(t, pool) {
		hasExperiment := false
		terms := make([]string, len(columns))
		for i, column := range columns {
			hasExperiment = hasExperiment || column == "experiment_id"
			terms[i] = "COALESCE(pg_column_size(" + column + "), 0)"
		}
		if !hasExperiment {
			continue
		}
		var rows, bytes int64
		if err := pool.QueryRow(context.Background(), `SELECT count(*), COALESCE(sum(`+strings.Join(terms, " + ")+`), 0)::bigint FROM `+table+` WHERE experiment_id = $1`, id).Scan(&rows, &bytes); err != nil {
			t.Fatal(err)
		}
		if rows > 0 {
			out.rows[table] = rows
		}
		out.bytes += bytes
	}
	return out
}

func countWhere(t *testing.T, pool *pgxpool.Pool, query string, args ...any) int64 {
	t.Helper()
	var n int64
	if err := pool.QueryRow(context.Background(), query, args...).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// incompressibleGaps returns count collection gaps of size random base64
// characters each, which neither pglz nor lz4 shrinks.
func incompressibleGaps(t *testing.T, count, size int) []string {
	t.Helper()
	gaps := make([]string, count)
	raw := make([]byte, size*3/4)
	for i := range gaps {
		if _, err := rand.Read(raw); err != nil {
			t.Fatal(err)
		}
		gaps[i] = base64.StdEncoding.EncodeToString(raw)
	}
	return gaps
}

// populatedExperiment builds an external experiment of eight members through
// the store's own writes, with no outstanding work:
// an admitted member whose run creation was rejected, so its submission and
// resolved suboperation are drained; evidenceResults results of 1,024
// evidence references and largeResults results of about 880 KB of
// incompressible collection gaps, all on one other member; a human assessment
// selected with the first result; a published view with its pairs, charts and
// progress observation; and the mutation receipt of every keyed write. The
// one execution tombstone, which an ordinary Run deletion writes through its
// trigger, is seeded directly.
func populatedExperiment(t *testing.T, pool *pgxpool.Pool, scope Scope, id, portable string, evidenceResults, largeResults int) Experiment {
	t.Helper()
	if evidenceResults+largeResults < 1 {
		t.Fatal("a populated experiment needs a result")
	}
	ctx := context.Background()
	e := createExperiment(t, pool, scope, id, portable, "external-workflow")
	ms := members(t, pool, e)
	var base evaldomain.ResultInput
	if err := json.Unmarshal(fixture(t, "result", "ResultInput").Bytes(), &base); err != nil {
		t.Fatal(err)
	}
	base.PlanSHA256 = planDigest(t, pool, e)
	reviewed := base.MemberID
	var admitted Member
	for _, m := range ms {
		if m.MemberID != reviewed {
			admitted = m
			break
		}
	}
	if _, err := admit(t, pool, e, admitted.MemberID, nil, id+"-submit"); err != nil {
		t.Fatal(err)
	}
	claim := claimExperiment(t, pool, e.ID)
	op := Suboperation{Kind: "run-create", Key: admitted.SubmissionKey, Request: json.RawMessage(`{"workflow":"trace-a@1"}`)}
	mustTx(t, pool, func(s *Store) error {
		return s.RecoverMember(scope, e.ID, admitted.MemberID, claim).PutSuboperation(ctx, op)
	})
	mustTx(t, pool, func(s *Store) error {
		return s.RecoverMember(scope, e.ID, admitted.MemberID, claim).ResolveSuboperation(ctx, op.Kind, []byte(`{"rejected":true}`), true)
	})
	mustTx(t, pool, func(s *Store) error { return s.RecoverMember(scope, e.ID, admitted.MemberID, claim).Settle(ctx) })
	if _, err := pool.Exec(ctx, `
INSERT INTO eval_execution_tombstones(experiment_id, member_id, execution_kind, execution_id, terminal_state, never_started)
VALUES ($1, $2, 'run', $3, 'failed', false)`, e.ID, admitted.MemberID, id+"-deleted-run"); err != nil {
		t.Fatal(err)
	}

	put := func(key, operation string, doc evaldomain.Frozen) {
		t.Helper()
		mustTx(t, pool, func(s *Store) error {
			_, err := s.PutRecord(ctx, RecordParams{
				Scope: scope, ExperimentID: e.ID, MemberID: reviewed, ActorID: scope.OwnerID, Operation: operation,
				Mutation: mutation(t, key, 0, doc), Build: func(Experiment) (evaldomain.Frozen, error) { return doc, nil },
			})
			return err
		})
	}
	var results []evaldomain.Frozen
	for i := range evidenceResults {
		result := base
		result.Source.ID = fmt.Sprintf("trace-evidence-%d", i)
		result.Evidence = make([]evaldomain.Evidence, 1024)
		for j := range result.Evidence {
			artifact := base.Evidence[0].Artifact
			artifact.Name = fmt.Sprintf("trace-%d-%04d", i, j)
			result.Evidence[j] = evaldomain.Evidence{ID: fmt.Sprintf("evidence-%04d", j), Artifact: artifact}
		}
		results = append(results, freeze(t, "ResultInput", result))
	}
	for i := range largeResults {
		result := base
		result.Source.ID = fmt.Sprintf("trace-large-%d", i)
		result.Collection = evaldomain.Collection{Status: "partial", Gaps: incompressibleGaps(t, 110, 8000)}
		results = append(results, freeze(t, "ResultInput", result))
	}
	for i, doc := range results {
		put(fmt.Sprintf("%s-result-%d", id, i), "result", doc)
	}
	var assessment evaldomain.AssessmentInput
	if err := json.Unmarshal(fixture(t, "human-assessment", "AssessmentInput").Bytes(), &assessment); err != nil {
		t.Fatal(err)
	}
	assessment.ResultSHA256 = results[0].Digest()
	review := freeze(t, "AssessmentInput", assessment)
	put(id+"-review", "assessment", review)
	reviewSHA := review.Digest()
	current, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).Get(ctx, scope.OwnerID, e.ID)
	if err != nil {
		t.Fatal(err)
	}
	selection := evaldomain.SelectionInput{PlanSHA256: base.PlanSHA256, Selections: []evaldomain.SelectionEntry{{
		MemberID: reviewed, ResultSHA256: results[0].Digest(), AssessmentSHA256: &reviewSHA,
	}}}
	selectionDoc := freeze(t, "SelectionInput", selection)
	mustTx(t, pool, func(s *Store) error {
		_, err := s.SelectRecords(ctx, scope, e.ID, scope.OwnerID, selection, mutation(t, id+"-select", current.Revision, selectionDoc))
		return err
	})

	dirty, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).DirtyMembers(ctx, scope.OwnerID, e.ID, 100)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range dirty {
		for _, m := range ms {
			if m.MemberID != d.MemberID {
				continue
			}
			view := evaldomain.MemberView{Member: evaldomain.MemberIdentity{
				ID: m.MemberID, SuiteID: m.SuiteID, CaseID: m.CaseID, Sample: m.Sample, VariantID: m.VariantID,
				CaseSHA256: m.CaseSHA256, BindingSHA256: m.BindingSHA256, Eligibility: m.Eligibility,
			}, Execution: &evaldomain.ExecutionView{State: "not_submitted"}, Assessment: "unscored"}
			mustTx(t, pool, func(s *Store) error {
				return s.ProjectMember(ctx, scope, e.ID, m.MemberID, claim, d.Revision, view, false, nil)
			})
		}
	}
	comparison := evaldomain.Comparison{Baseline: "a", Candidate: "b", Gates: evaldomain.Gates{MinCandidateEndToEndPass: 1}}
	mustTx(t, pool, func(s *Store) error {
		_, err := s.PublishView(ctx, scope, e.ID, claim, comparison, true)
		return err
	})
	releaseClaim(t, pool, claim)
	if current, err = NewPostgresStore(pool, artifactpgtest.NewStorage()).Get(ctx, scope.OwnerID, e.ID); err != nil {
		t.Fatal(err)
	}
	if current.Outstanding != 0 {
		t.Fatalf("populated experiment is not drained: %+v", current)
	}
	view, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).LatestView(ctx, scope.OwnerID, e.ID)
	if err != nil || view == nil {
		t.Fatalf("populated experiment has no selected view: %+v, %v", view, err)
	}
	return current
}

// fenceExperiment requests the experiment's deletion under the key
// id-delete and returns it fenced.
func fenceExperiment(t *testing.T, pool *pgxpool.Pool, scope Scope, e Experiment) Experiment {
	t.Helper()
	ctx := context.Background()
	deletion := fixture(t, "delete", "Delete")
	mustTx(t, pool, func(s *Store) error {
		_, err := s.BeginDeletion(ctx, scope, e.ID, mutation(t, e.ID+"-delete", e.Revision, deletion))
		return err
	})
	fenced, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).Get(ctx, scope.OwnerID, e.ID)
	if err != nil || fenced.DeletionRequestedAt == nil || fenced.Revision != e.Revision+1 {
		t.Fatalf("fenced experiment = %+v, %v", fenced, err)
	}
	return fenced
}

// An admitted experiment whose eight members hold more than three batches of
// dependent rows on one member, and more stored bytes than one pass may
// remove, is purged in passes that each stay within CleanupBatchRows and
// LIM-EVM-20, exactly as PostgreSQL counts them. A failed or cancelled pass
// removes nothing and keeps the committed passes, a new controller resumes
// from what is left, the selected view goes before any of its rows, the
// experiment row goes last, and its receipts still replay after the purge
// (EVM-R243, EVM-R226).
func TestSharedPostgresEvalExperimentPurgeIsBoundedAndRestartable(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	scope := setupProject(t, pool, "owner", "eval")
	other := setupProject(t, pool, "other", "other-eval")
	e := fenceExperiment(t, pool, scope, populatedExperiment(t, pool, scope, "exp", "trace-1", 3, 20))
	foreign := fenceExperiment(t, pool, other, populatedExperiment(t, pool, other, "foreign", "trace-1", 1, 0))
	before := footprint(t, pool, e.ID)
	foreignBefore := footprint(t, pool, foreign.ID)
	dependents := int64(0)
	for table, rows := range before.rows {
		if table != "eval_experiments" {
			dependents += rows
		}
	}
	if before.rows["eval_evidence_refs"] < 3*pg.CleanupBatchRows || dependents <= 3*pg.CleanupBatchRows || before.bytes <= purgeBatchBytes {
		t.Fatalf("fixture holds %d dependent rows (%d evidence references) and %d bytes; want more than three batches and one byte budget", dependents, before.rows["eval_evidence_refs"], before.bytes)
	}
	receipts := func(project string) int64 {
		return countWhere(t, pool, `SELECT count(*) FROM mutation_receipts WHERE key_scope[1] = 'eval' AND project_id = $1`, project)
	}
	receiptsBefore := receipts(scope.ProjectID)

	// A pass that fails, or whose caller is cancelled before the commit,
	// removes nothing, before the first pass and after a committed one.
	claim := claimExperiment(t, pool, e.ID)
	failPasses := func(label string, viewKept bool) {
		t.Helper()
		want := footprint(t, pool, e.ID)
		failed := errors.New("injected failure after the pass")
		_, err := runPass(ctx, pool, func(s *Store) (PurgeProgress, error) {
			if _, err := s.Purge(ctx, scope, e.ID, claim); err != nil {
				return PurgeProgress{}, err
			}
			return PurgeProgress{}, failed
		})
		if !errors.Is(err, failed) {
			t.Fatalf("%s: failed pass = %v", label, err)
		}
		cancelled, cancel := context.WithCancel(ctx)
		err = pg.InTx(cancelled, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			_, err := NewTxStore(tx, artifactpgtest.NewStorage()).Purge(cancelled, scope, e.ID, claim)
			cancel()
			return err
		})
		if err == nil {
			t.Fatalf("%s: a pass cancelled before its commit committed", label)
		}
		if now := footprint(t, pool, e.ID); now.bytes != want.bytes || fmt.Sprint(now.rows) != fmt.Sprint(want.rows) {
			t.Fatalf("%s: failed passes left %+v, want %+v", label, now, want)
		}
		if view, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).LatestView(ctx, scope.OwnerID, e.ID); err != nil || (view != nil) != viewKept {
			t.Fatalf("%s: view after failed passes = %+v, %v", label, view, err)
		}
	}
	failPasses("before the first pass", true)
	releaseClaim(t, pool, claim)

	var passes []purgePass
	previous := before
	for len(passes) < 100 {
		passClaim := claimExperiment(t, pool, e.ID)
		pass, err := runPass(ctx, pool, func(s *Store) (PurgeProgress, error) { return s.Purge(ctx, scope, e.ID, passClaim) })
		if err != nil {
			t.Fatalf("pass %d: %v", len(passes)+1, err)
		}
		releaseClaim(t, pool, passClaim)
		label := fmt.Sprintf("pass %d", len(passes)+1)
		checkPass(t, label, pass)
		passes = append(passes, pass)
		now := footprint(t, pool, e.ID)
		removed := previous.bytes - now.bytes
		// The pass charged every byte it removed; beyond that only the
		// rewritten queue and collection rows.
		if removed > pass.Bytes || pass.Bytes-removed > 64<<10 {
			t.Fatalf("%s charged %d bytes but removed %d", label, pass.Bytes, removed)
		}
		if len(passes) == 1 {
			// The first pass removes the selected view before its rows; the
			// experiment's identity and plan stay until the last pass.
			if view, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).LatestView(ctx, scope.OwnerID, e.ID); err != nil || view != nil {
				t.Fatalf("view after the first pass = %+v, %v", view, err)
			}
			if purging, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).Purging(ctx, scope.OwnerID, e.ID); err != nil || !purging {
				t.Fatalf("Purging after the first pass = %t, %v", purging, err)
			}
			if _, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).FrozenPlan(ctx, scope.OwnerID, e.ID); err != nil {
				t.Fatalf("frozen plan during the purge: %v", err)
			}
		}
		if !pass.Done {
			if now.rows["eval_experiments"] != 1 {
				t.Fatalf("%s removed the experiment before it was done", label)
			}
			// No parent goes while one of its children is left.
			if now.rows["eval_members"] < 8 && now.rows["eval_records"]+now.rows["eval_evidence_refs"]+now.rows["eval_submissions"]+now.rows["eval_member_projections"]+now.rows["eval_execution_tombstones"] > 0 {
				t.Fatalf("%s removed members before their children: %+v", label, now.rows)
			}
		}
		previous = now
		if len(passes) == 2 {
			claim = claimExperiment(t, pool, e.ID)
			failPasses("after two passes", false)
			releaseClaim(t, pool, claim)
		}
		if pass.Done {
			break
		}
	}
	t.Logf("fixture: %d dependent rows, %d stored bytes, rows per table %v; passes (rows, bytes, done):", dependents, before.bytes, before.rows)
	for i, pass := range passes {
		t.Logf("  %d: %d, %d, %t", i+1, pass.Rows, pass.Bytes, pass.Done)
	}
	byteBound := false
	for _, pass := range passes[:len(passes)-1] {
		byteBound = byteBound || pass.Rows < pg.CleanupBatchRows
	}
	if len(passes) < 4 || !byteBound {
		t.Fatalf("purge passes = %+v; want more than three, one of them ended by the byte budget", passes)
	}
	if left := footprint(t, pool, e.ID); len(left.rows) != 0 {
		t.Fatalf("purge left %+v", left.rows)
	}
	if now := footprint(t, pool, foreign.ID); now.bytes != foreignBefore.bytes || fmt.Sprint(now.rows) != fmt.Sprint(foreignBefore.rows) {
		t.Fatalf("another owner's experiment changed: %+v, want %+v", now, foreignBefore)
	}
	// The experiment's receipts stay and replay until its Project is purged.
	if got := receipts(scope.ProjectID); got != receiptsBefore || got == 0 {
		t.Fatalf("receipts after the experiment purge = %d, want %d", got, receiptsBefore)
	}
	deletion := fixture(t, "delete", "Delete")
	mustTx(t, pool, func(s *Store) error {
		r, err := s.BeginDeletion(ctx, scope, e.ID, mutation(t, "exp-delete", e.Revision-1, deletion))
		if err == nil && !r.Replayed {
			t.Fatal("deletion receipt lost after purge")
		}
		return err
	})
	// A controller whose acknowledgement of the last pass was lost finds
	// nothing left to purge.
	mustTx(t, pool, func(s *Store) error {
		_, err := s.Purge(ctx, scope, e.ID, claim)
		if !code(err, "eval_not_found") {
			t.Fatalf("purge after the last pass = %v", err)
		}
		return nil
	})
}

// Purge removes nothing while the experiment is not fenced, has outstanding
// work, an unresolved side-effect intent or a live execution workspace, or
// while the caller's claim or owner scope is not current; the experiment's
// view stays published (EVM-R243).
func TestSharedPostgresEvalPurgeWaitsForDrainAndAuthority(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	scope := setupProject(t, pool, "owner", "eval")
	e := populatedExperiment(t, pool, scope, "exp", "trace-1", 1, 0)
	attempt := func(label string, s Scope, claim Claim, want func(error) bool) {
		t.Helper()
		before := footprint(t, pool, e.ID)
		_, err := runPass(ctx, pool, func(st *Store) (PurgeProgress, error) { return st.Purge(ctx, s, e.ID, claim) })
		if !want(err) {
			t.Fatalf("%s: purge = %v", label, err)
		}
		if now := footprint(t, pool, e.ID); now.bytes != before.bytes || fmt.Sprint(now.rows) != fmt.Sprint(before.rows) {
			t.Fatalf("%s removed rows: %+v, want %+v", label, now.rows, before.rows)
		}
		if view, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).LatestView(ctx, scope.OwnerID, e.ID); err != nil || view == nil {
			t.Fatalf("%s removed the selected view: %+v, %v", label, view, err)
		}
	}
	drain := func(err error) bool { return errors.Is(err, ErrDrain) }
	exec := func(statement string) {
		t.Helper()
		if _, err := pool.Exec(ctx, statement, e.ID); err != nil {
			t.Fatal(err)
		}
	}
	claim := claimExperiment(t, pool, e.ID)
	attempt("not fenced", scope, claim, drain)
	releaseClaim(t, pool, claim)
	e = fenceExperiment(t, pool, scope, e)
	claim = claimExperiment(t, pool, e.ID)
	stale := claim
	stale.Epoch--
	attempt("stale claim", scope, stale, func(err error) bool { return errors.Is(err, ErrClaimLost) })
	attempt("foreign owner", Scope{"other", scope.ProjectID}, claim, func(err error) bool { return code(err, "eval_not_found") })

	exec(`UPDATE eval_experiments SET outstanding_count = 1, ` + observe + ` WHERE experiment_id = $1`)
	attempt("outstanding work", scope, claim, drain)
	exec(`UPDATE eval_experiments SET outstanding_count = 0, ` + observe + ` WHERE experiment_id = $1`)

	exec(`
INSERT INTO eval_suboperations(experiment_id, member_id, kind, operation_key, request)
SELECT experiment_id, member_id, 'cancel', 'unresolved-cancel', '{}'
FROM eval_submissions
WHERE experiment_id = $1`)
	attempt("unresolved intent", scope, claim, drain)
	exec(`DELETE FROM eval_suboperations WHERE experiment_id = $1 AND kind = 'cancel'`)

	setupProject(t, pool, scope.OwnerID, "audit-workspace")
	exec(`UPDATE eval_submissions SET execution_project_id = 'audit-workspace' WHERE experiment_id = $1`)
	attempt("live execution workspace", scope, claim, drain)
	releaseClaim(t, pool, claim)
}

// worstCaseSuites is an upper bound of the largest suites document a view can
// store (LIM-EVM-20): 5,000 suites, the most that 10,000 members in A/B pairs
// form (LIM-EVF-8), each named by a 128-character identifier, with both
// 128-character variants' counters at 10,000, the longest ratio values and a
// conclusion padded beyond the longest one.
func worstCaseSuites() map[string]evaldomain.Summary {
	value := 1.2345678901234567e-05
	counts := evaldomain.Counts{
		Expected: 10000, Eligible: 10000, Unsupported: 10000, Blocked: 10000, Submitted: 10000, Terminal: 10000,
		Missing: 10000, Conflicting: 10000, CollectionComplete: 10000, Scored: 10000, QualityPassed: 10000,
		ExecutionSucceeded: 10000, EndToEndPassed: 10000,
	}
	ratio := evaldomain.Ratio{Numerator: 10000, Denominator: 10000, Value: &value}
	quality := evaldomain.Quality{ExecutionSuccess: ratio, EndToEndPass: ratio, ConditionalQuality: ratio}
	a, b := strings.Repeat("a", 128), strings.Repeat("b", 128)
	summary := evaldomain.Summary{
		Counts: map[string]evaldomain.Counts{a: counts, b: counts}, Quality: map[string]evaldomain.Quality{a: quality, b: quality},
		TerminalPairs: 10000, CompleteQualityPairs: 10000, CompleteTokenPairs: 10000, Conclusion: "inconclusive-xxxxxxxxxxxx",
	}
	suites := make(map[string]evaldomain.Summary, 5000)
	for i := range 5000 {
		suites[fmt.Sprintf("%0128d", i)] = summary
	}
	return suites
}

// The largest row a validated write can store is a progress observation of
// the worst-case suites, about 10.7 MiB stored without compression. It fits a
// fresh pass: a pass that has already spent the queue row's worst-case 8.9
// MiB stops before it and the next pass removes it first. A row larger than
// the whole byte budget, which no validated write stores and is seeded here
// only to prove progress, goes alone as the first row of a fresh pass, so no
// pass ends without progress and no size is refused (EVM-R243).
func TestSharedPostgresEvalPurgeFitsLargestLegalRowInFreshPass(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	// Incompressible content is the worst case: store these columns as they are.
	for _, statement := range []string{
		`ALTER TABLE eval_projection_queue ALTER COLUMN suites SET STORAGE EXTERNAL`,
		`ALTER TABLE eval_progress_observations ALTER COLUMN counts SET STORAGE EXTERNAL`,
	} {
		if _, err := pool.Exec(ctx, statement); err != nil {
			t.Fatal(err)
		}
	}
	suites := worstCaseSuites()
	counts := bytesOf(map[string]any{"a": 10000, "b": 10000, "suites": suites})
	const largestQueueSuites, largestObservation = 9_295_001, 11_260_056
	if size := len(bytesOf(suites)); size != largestQueueSuites {
		t.Fatalf("worst-case suites document = %d bytes, want %d", size, largestQueueSuites)
	}
	observationBytes := func(id string) []int64 {
		t.Helper()
		rows, err := pool.Query(ctx, `SELECT (`+purgeRowBytes["eval_progress_observations"]+`)::bigint FROM eval_progress_observations WHERE experiment_id = $1 ORDER BY sequence`, id)
		if err != nil {
			t.Fatal(err)
		}
		sizes, err := pgx.CollectRows(rows, pgx.RowTo[int64])
		if err != nil {
			t.Fatal(err)
		}
		return sizes
	}

	scope := setupProject(t, pool, "owner", "eval")
	e := populatedExperiment(t, pool, scope, "exp", "trace-1", 1, 0)
	if _, err := pool.Exec(ctx, `UPDATE eval_projection_queue SET suites = $2 WHERE experiment_id = $1`, e.ID, bytesOf(suites)); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO eval_progress_observations(experiment_id, counts) VALUES ($1, $2)`, e.ID, counts); err != nil {
		t.Fatal(err)
	}
	queueBytes := countWhere(t, pool, `SELECT (`+purgeRowBytes["eval_projection_queue"]+`)::bigint FROM eval_projection_queue WHERE experiment_id = $1`, e.ID)
	sizes := observationBytes(e.ID)
	largest := sizes[len(sizes)-1]
	if queueBytes < largestQueueSuites || largest < largestObservation || largest >= purgeBatchBytes || queueBytes+largest <= purgeBatchBytes {
		t.Fatalf("queue row %d bytes and observation %d bytes; want the worst cases, each within %d bytes and not both", queueBytes, largest, purgeBatchBytes)
	}
	e = fenceExperiment(t, pool, scope, e)
	passes := purgeExperimentUntilDone(t, pool, scope, e.ID)
	if len(passes) < 2 || passes[0].Bytes < queueBytes || passes[1].Bytes < largest {
		t.Fatalf("passes = %+v; want the queue row in the first and the largest observation in the second", passes)
	}

	// A directly seeded oversized row: the first row of a fresh pass, alone.
	scope = setupProject(t, pool, "owner-2", "eval-2")
	e = populatedExperiment(t, pool, scope, "oversized", "trace-1", 1, 0)
	oversized := `{"pad":"` + strings.Repeat("x", purgeBatchBytes) + `"}`
	if _, err := pool.Exec(ctx, `INSERT INTO eval_progress_observations(experiment_id, counts) VALUES ($1, $2)`, e.ID, []byte(oversized)); err != nil {
		t.Fatal(err)
	}
	if sizes := observationBytes(e.ID); sizes[len(sizes)-1] <= purgeBatchBytes {
		t.Fatalf("seeded observation is %d bytes, want more than %d", sizes[len(sizes)-1], purgeBatchBytes)
	}
	e = fenceExperiment(t, pool, scope, e)
	var oversizedPasses []purgePass
	for len(oversizedPasses) < 20 {
		claim := claimExperiment(t, pool, e.ID)
		pass, err := runPass(ctx, pool, func(s *Store) (PurgeProgress, error) { return s.Purge(ctx, scope, e.ID, claim) })
		if err != nil {
			t.Fatal(err)
		}
		releaseClaim(t, pool, claim)
		oversizedPasses = append(oversizedPasses, pass)
		if pass.Mutations != pass.Rows || pass.Rows > pg.CleanupBatchRows || (!pass.Done && pass.Rows == 0) ||
			(pass.Bytes > purgeBatchBytes && pass.Rows != 1) {
			t.Fatalf("pass %d = %+v", len(oversizedPasses), pass)
		}
		if pass.Done {
			break
		}
	}
	alone := false
	for _, pass := range oversizedPasses {
		alone = alone || pass.Rows == 1 && pass.Bytes > purgeBatchBytes
	}
	if !alone || !oversizedPasses[len(oversizedPasses)-1].Done {
		t.Fatalf("oversized row passes = %+v; want it removed alone and the purge done", oversizedPasses)
	}
}

// preparedNative creates a native experiment of the fixture plan and
// prepares and starts it, so it holds its frozen plan, members, two plan
// resources and two commands.
func preparedNative(t *testing.T, pool *pgxpool.Pool, scope Scope, id string) Experiment {
	t.Helper()
	ctx := context.Background()
	e := createExperiment(t, pool, scope, id, "trace-1", "create-workflow")
	prepare := evaldomain.Command{Kind: "prepare"}
	mustTx(t, pool, func(s *Store) error {
		_, err := s.Command(ctx, CommandParams{Scope: scope, ExperimentID: e.ID, CommandID: id + "-prepare", Command: prepare,
			Mutation: mutation(t, id+"-prepare", e.Revision, freeze(t, "Command", prepare))})
		return err
	})
	claim := claimExperiment(t, pool, e.ID)
	var registration evaldomain.ExternalRegistration
	if err := json.Unmarshal(fixture(t, "registration", "ExternalRegistration").Bytes(), &registration); err != nil {
		t.Fatal(err)
	}
	setup := bytesOf(map[string]any{"variants": registration.Variants, "checks": registration.Checks, "comparison": registration.Comparison, "budgets": registration.Budgets})
	cases := map[string]evaldomain.Case{}
	for _, recipe := range registration.Recipes {
		cases[recipe.MemberID] = recipe.Case
	}
	plan := fixture(t, "portable-plan", "playground.plan/v1")
	resources := []PlanResource{
		{Path: "datasets/one.json", Document: fixture(t, "dataset", "DatasetInput")},
		{Path: "datasets/two.json", Document: fixture(t, "dataset", "DatasetInput")},
	}
	mustTx(t, pool, func(s *Store) error {
		return s.FreezePrepared(ctx, scope, e.ID, claim, plan, setup, cases, resources...)
	})
	releaseClaim(t, pool, claim)
	e, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).Get(ctx, scope.OwnerID, e.ID)
	if err != nil {
		t.Fatal(err)
	}
	start := evaldomain.Command{Kind: "start", PlanSHA256: plan.Digest()}
	mustTx(t, pool, func(s *Store) error {
		_, err := s.Command(ctx, CommandParams{Scope: scope, ExperimentID: e.ID, CommandID: id + "-start", Command: start,
			Mutation: mutation(t, id+"-start", e.Revision, freeze(t, "Command", start))})
		return err
	})
	if e, err = NewPostgresStore(pool, artifactpgtest.NewStorage()).Get(ctx, scope.OwnerID, e.ID); err != nil {
		t.Fatal(err)
	}
	return e
}

// A deleting Project's experiments, then its dataset revisions, collection
// row and Eval receipts share one budget per pass: experiments one after
// another within a pass, receipts only once no experiment or dataset revision
// is left. Another Project of the owner and another owner's Project keep all
// their rows, and a pass after the last finds nothing to do (EVM-R243).
func TestSharedPostgresEvalProjectPurgeSharesOneBudget(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	scope := setupProject(t, pool, "owner", "eval")
	sibling := setupProject(t, pool, "owner", "sibling")
	other := setupProject(t, pool, "other", "other-eval")
	putDataset(t, pool, scope)
	putDataset(t, pool, sibling)
	putDataset(t, pool, other)
	dataset := fixture(t, "dataset", "DatasetInput")
	for i := range 400 {
		mustTx(t, pool, func(s *Store) error {
			_, err := s.PutDataset(ctx, scope, fmt.Sprintf("r%d", i+2), dataset, mutation(t, fmt.Sprintf("dataset-%d", i), 0, dataset))
			return err
		})
	}
	experiments := []string{populatedExperiment(t, pool, scope, "heavy", "trace-1", 2, 4).ID}
	for i := range 3 {
		experiments = append(experiments, createExperiment(t, pool, scope, fmt.Sprintf("draft-%d", i), fmt.Sprintf("draft-%d", i), "create-workflow").ID)
	}
	siblingDraft := createExperiment(t, pool, sibling, "sibling-draft", "draft-0", "create-workflow")
	foreign := populatedExperiment(t, pool, other, "foreign", "trace-1", 1, 0)

	type projectRows struct{ experiments, datasets, collection, receipts int64 }
	projectFootprint := func(owner, project string) projectRows {
		t.Helper()
		var r projectRows
		if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM eval_experiments WHERE owner_id = $1 AND project_id = $2),
    (SELECT count(*) FROM eval_dataset_revisions WHERE owner_id = $1 AND project_id = $2),
    (SELECT count(*) FROM eval_collections WHERE owner_id = $1 AND project_id = $2),
    (SELECT count(*) FROM mutation_receipts WHERE key_scope[1] = 'eval' AND owner_id = $1 AND project_id = $2)`,
			owner, project).Scan(&r.experiments, &r.datasets, &r.collection, &r.receipts); err != nil {
			t.Fatal(err)
		}
		return r
	}
	siblingBefore, otherBefore := projectFootprint("owner", "sibling"), projectFootprint("other", "other-eval")
	siblingDraftBefore, foreignBefore := footprint(t, pool, siblingDraft.ID), footprint(t, pool, foreign.ID)
	if before := projectFootprint("owner", "eval"); before.experiments != 4 || before.datasets != 401 || before.receipts < 401 {
		t.Fatalf("deleting Project holds %+v", before)
	}
	if err := transaction(pool, func(s *Store) error { _, err := s.PurgeProject(ctx, scope); return err }); !code(err, "eval_not_ready") {
		t.Fatalf("purge of an active Project = %v", err)
	}
	beginProjectDeletion(t, pool, scope)

	var passes []purgePass
	shared := false
	previous := map[string]int{}
	for _, id := range experiments {
		previous[id] = len(footprint(t, pool, id).rows)
	}
	for len(passes) < 100 {
		before := projectFootprint("owner", "eval")
		rowsBefore := map[string]int64{}
		for _, id := range experiments {
			for _, n := range footprint(t, pool, id).rows {
				rowsBefore[id] += n
			}
		}
		pass, err := runPass(ctx, pool, func(s *Store) (PurgeProgress, error) { return s.PurgeProject(ctx, scope) })
		if err != nil {
			t.Fatalf("pass %d: %v", len(passes)+1, err)
		}
		label := fmt.Sprintf("Project pass %d", len(passes)+1)
		checkPass(t, label, pass)
		passes = append(passes, pass)
		after := projectFootprint("owner", "eval")
		changed := 0
		for _, id := range experiments {
			var rows int64
			for _, n := range footprint(t, pool, id).rows {
				rows += n
			}
			if rows != rowsBefore[id] {
				changed++
			}
		}
		shared = shared || changed > 1
		if after.datasets < before.datasets && after.experiments > 0 {
			t.Fatalf("%s removed dataset revisions while %d experiments remain", label, after.experiments)
		}
		if after.receipts < before.receipts && (after.experiments > 0 || after.datasets > 0 || after.collection > 0) {
			t.Fatalf("%s removed receipts before the Project's other Eval rows: %+v", label, after)
		}
		if pass.Done {
			if after != (projectRows{}) {
				t.Fatalf("%s is done but left %+v", label, after)
			}
			break
		}
	}
	for i, pass := range passes {
		t.Logf("Project pass %d: %d rows, %d bytes, done %t", i+1, pass.Rows, pass.Bytes, pass.Done)
	}
	if !shared || len(passes) < 3 {
		t.Fatalf("Project passes = %+v; want several, one of them spanning experiments", passes)
	}
	if again := purgeProjectUntilDone(t, pool, scope); len(again) != 1 || again[0].Rows != 0 {
		t.Fatalf("a pass after the last = %+v", again)
	}
	if got := projectFootprint("owner", "sibling"); got != siblingBefore {
		t.Fatalf("sibling Project changed: %+v, want %+v", got, siblingBefore)
	}
	if got := projectFootprint("other", "other-eval"); got != otherBefore {
		t.Fatalf("other owner's Project changed: %+v, want %+v", got, otherBefore)
	}
	for id, want := range map[string]experimentFootprint{siblingDraft.ID: siblingDraftBefore, foreign.ID: foreignBefore} {
		if got := footprint(t, pool, id); got.bytes != want.bytes || fmt.Sprint(got.rows) != fmt.Sprint(want.rows) {
			t.Fatalf("experiment %s changed: %+v, want %+v", id, got, want)
		}
	}
	if countWhere(t, pool, `SELECT count(*) FROM eval_collections WHERE owner_id = 'owner' AND project_id = ''`) != 1 {
		t.Fatal("owner collection row removed")
	}
}

// A started native experiment's plan resources and commands go in its purge,
// while its receipts stay until its Project's purge, which then removes them
// with the Project's dataset revision and collection row (EVM-R243,
// EVM-R226).
func TestSharedPostgresEvalPurgeRemovesNativePlanResourcesAndCommands(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	scope := setupProject(t, pool, "owner", "eval")
	putDataset(t, pool, scope)
	e := fenceExperiment(t, pool, scope, preparedNative(t, pool, scope, "native"))
	before := footprint(t, pool, e.ID)
	if before.rows["eval_plan_resources"] != 2 || before.rows["eval_commands"] != 2 || before.rows["eval_members"] != 8 {
		t.Fatalf("native experiment holds %v", before.rows)
	}
	passes := purgeExperimentUntilDone(t, pool, scope, e.ID)
	if left := footprint(t, pool, e.ID); len(left.rows) != 0 {
		t.Fatalf("purge left %v", left.rows)
	}
	if _, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).CommandRecord(ctx, scope.OwnerID, e.ID, "native-prepare"); !code(err, "eval_not_found") {
		t.Fatalf("command after the purge = %v", err)
	}
	receipts := countWhere(t, pool, `SELECT count(*) FROM mutation_receipts WHERE key_scope[1] = 'eval' AND project_id = $1`, scope.ProjectID)
	if receipts < 4 || len(passes) != 1 {
		t.Fatalf("receipts = %d, passes = %+v", receipts, passes)
	}
	beginProjectDeletion(t, pool, scope)
	project := purgeProjectUntilDone(t, pool, scope)
	// Dataset revision, collection row and receipts are left to the Project.
	if len(project) != 1 || project[0].Rows != 3+1+receipts {
		t.Fatalf("Project passes = %+v, want one of %d rows", project, 3+1+receipts)
	}
	if n := countWhere(t, pool, `SELECT count(*) FROM mutation_receipts WHERE key_scope[1] = 'eval' AND project_id = $1`, scope.ProjectID); n != 0 {
		t.Fatalf("receipts after the Project purge = %d", n)
	}
}
