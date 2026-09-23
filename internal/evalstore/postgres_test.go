package evalstore

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func testPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	admin, err := pgxpool.New(ctx, url)
	if err != nil {
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err = rand.Read(random); err != nil {
		t.Fatal(err)
	}
	schema := "eval_test_" + hex.EncodeToString(random)
	quoted := pgx.Identifier{schema}.Sanitize()
	if _, err = admin.Exec(ctx, `CREATE SCHEMA `+quoted); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	cfg, err := pgxpool.ParseConfig(url)
	if err != nil {
		t.Fatal(err)
	}
	cfg.ConnConfig.RuntimeParams["search_path"] = schema
	cfg.ConnConfig.RuntimeParams["statement_timeout"] = "10000"
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		_, err := admin.Exec(context.Background(), `DROP SCHEMA `+quoted+` CASCADE`)
		admin.Close()
		if err != nil {
			t.Error(err)
		}
	})
	if _, err = pg.ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	return pool
}
func transaction(pool *pgxpool.Pool, fn func(*Store) error) error {
	return pg.InTx(context.Background(), pool, pgx.TxOptions{}, func(tx pgx.Tx) error { return fn(NewTxStore(tx)) })
}
func mustTx(t *testing.T, pool *pgxpool.Pool, fn func(*Store) error) {
	t.Helper()
	if err := transaction(pool, fn); err != nil {
		t.Fatal(err)
	}
}
func fixture(t *testing.T, name, kind string) evaldomain.Frozen {
	t.Helper()
	b, err := os.ReadFile("../../api/testdata/evals/valid/" + name + ".json")
	if err != nil {
		t.Fatal(err)
	}
	f, err := evaldomain.Freeze(kind, b)
	if err != nil {
		t.Fatal(name, err)
	}
	return f
}
func freeze(t *testing.T, kind string, value any) evaldomain.Frozen {
	t.Helper()
	f, err := evaldomain.Freeze(kind, bytesOf(value))
	if err != nil {
		t.Fatal(err)
	}
	return f
}
func mutation(t *testing.T, key string, revision int64, doc evaldomain.Frozen) evaldomain.MutationIdentity {
	t.Helper()
	etag := ""
	if revision > 0 {
		etag = fmt.Sprintf(`"%d"`, revision)
	}
	m, err := evaldomain.IdentifyMutation(key, etag, revision > 0, doc.Kind(), doc.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	return m
}
func setupProject(t *testing.T, pool *pgxpool.Pool, owner, project string) Scope {
	t.Helper()
	_, _, err := projectstore.NewPostgresStore(pool).Create(context.Background(), projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindEvaluation, Name: project, IdempotencyKey: project, RequestDigest: evaldomain.Digest([]byte(project))})
	if err != nil {
		t.Fatal(err)
	}
	return Scope{owner, project}
}
func code(err error, want string) bool {
	var d *evaldomain.Error
	return errors.As(err, &d) && d.Code == want
}
func putDataset(t *testing.T, pool *pgxpool.Pool, scope Scope) {
	t.Helper()
	doc := fixture(t, "dataset", "DatasetInput")
	m := mutation(t, "dataset", 0, doc)
	mustTx(t, pool, func(s *Store) error { _, err := s.PutDataset(context.Background(), scope, "r1", doc, m); return err })
}
func createExperiment(t *testing.T, pool *pgxpool.Pool, scope Scope, id, portable, fixtureName string) Experiment {
	t.Helper()
	doc := fixture(t, fixtureName, "CreateExperiment")
	m := mutation(t, id, 0, doc)
	mustTx(t, pool, func(s *Store) error {
		_, err := s.Create(context.Background(), CreateParams{Scope: scope, ID: id, PortableID: portable, Document: doc, Mutation: m})
		return err
	})
	e, err := NewPostgresStore(pool).Get(context.Background(), scope.OwnerID, id)
	if err != nil {
		t.Fatal(err)
	}
	return e
}
func members(t *testing.T, pool *pgxpool.Pool, e Experiment) []Member {
	t.Helper()
	m, err := NewPostgresStore(pool).Members(context.Background(), e.OwnerID, e.ID, -1, 100)
	if err != nil {
		t.Fatal(err)
	}
	return m
}
func oneClaim(t *testing.T, pool *pgxpool.Pool) Claim {
	t.Helper()
	c, err := NewPostgresStore(pool).Claim(context.Background(), "controller", time.Minute, 1)
	if err != nil || len(c) != 1 {
		t.Fatalf("claim: %v %v", c, err)
	}
	return c[0]
}
func planDigest(t *testing.T, pool *pgxpool.Pool, e Experiment) string {
	t.Helper()
	p, err := NewPostgresStore(pool).FrozenPlan(context.Background(), e.OwnerID, e.ID)
	if err != nil {
		t.Fatal(err)
	}
	return p.SHA256
}
func admit(t *testing.T, pool *pgxpool.Pool, e Experiment, member string, claim *Claim, key string) (Receipt, error) {
	t.Helper()
	// Read the digest before opening the transaction: a second pool
	// connection taken while every connection holds a transaction deadlocks
	// the concurrent callers on hosts where the pool is small.
	digest := planDigest(t, pool, e)
	doc := freeze(t, "Submission", evaldomain.Submission{PlanSHA256: digest})
	m := mutation(t, key, 0, doc)
	var out Receipt
	err := transaction(pool, func(s *Store) error {
		var err error
		out, err = s.Admit(context.Background(), Admission{Scope{e.OwnerID, e.ProjectID}, e.ID, member, digest, claim, m})
		return err
	})
	return out, err
}
func beginProjectDeletion(t *testing.T, pool *pgxpool.Pool, scope Scope) {
	t.Helper()
	p, err := projectstore.NewPostgresStore(pool).Get(context.Background(), scope.OwnerID, scope.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = projectstore.NewPostgresStore(pool).BeginDeletion(context.Background(), projectstore.BeginDeletionParams{OwnerID: scope.OwnerID, ProjectID: scope.ProjectID, ExpectedRevision: p.Revision})
	if err != nil {
		t.Fatal(err)
	}
}

func TestPostgresEvalDraftCASOwnerReplayAndPrivateDatasets(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	a := setupProject(t, pool, "owner-a", "eval-a")
	b := setupProject(t, pool, "owner-b", "eval-b")
	putDataset(t, pool, a)
	putDataset(t, pool, b)
	reader := NewPostgresStore(pool)
	d, err := reader.Dataset(ctx, a, "trace-small", "r1")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(d.Document.Bytes(), fixture(t, "dataset", "DatasetInput").Bytes()) {
		t.Fatal("exact dataset bytes changed")
	}
	if _, err = reader.Dataset(ctx, Scope{b.OwnerID, a.ProjectID}, "trace-small", "r1"); !code(err, "eval_not_found") {
		t.Fatalf("foreign dataset: %v", err)
	}
	list, err := reader.ListDatasets(ctx, a, "", "", 1)
	if err != nil || len(list) != 1 {
		t.Fatal(list, err)
	}
	if strings.Contains(string(bytesOf(list)), "PRIVATE_") {
		t.Fatal("private data leaked in dataset list")
	}
	e := createExperiment(t, pool, a, "exp-a", "trace-1", "create-workflow")
	createExperiment(t, pool, b, "exp-b", "trace-1", "create-workflow")
	update := fixture(t, "draft-update", "DraftUpdate")
	ids := []evaldomain.MutationIdentity{mutation(t, "edit-a", e.Revision, update), mutation(t, "edit-b", e.Revision, update)}
	receipts := make([]Receipt, 2)
	errs := make([]error, 2)
	var wg sync.WaitGroup
	for i := range 2 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			errs[i] = transaction(pool, func(s *Store) error {
				var err error
				receipts[i], err = s.UpdateDraft(ctx, a, e.ID, update, ids[i])
				return err
			})
		}(i)
	}
	wg.Wait()
	winner := -1
	for i, err := range errs {
		if err == nil {
			if winner != -1 {
				t.Fatal("two CAS writes won")
			}
			winner = i
		} else if !code(err, "eval_revision_mismatch") {
			t.Fatal(err)
		}
	}
	if winner < 0 {
		t.Fatal(errs)
	}
	mustTx(t, pool, func(s *Store) error {
		r, err := s.UpdateDraft(ctx, a, e.ID, update, ids[winner])
		if err == nil && (!r.Replayed || !bytes.Equal(r.Response, receipts[winner].Response)) {
			t.Error("replay changed receipt")
		}
		return err
	})
	bad := ids[winner]
	bad.RequestSHA256 = evaldomain.Digest([]byte("different revision"))
	err = transaction(pool, func(s *Store) error { _, err := s.UpdateDraft(ctx, a, e.ID, update, bad); return err })
	if !code(err, "eval_idempotency_conflict") {
		t.Fatal(err)
	}
	err = transaction(pool, func(s *Store) error {
		_, err := s.UpdateDraft(ctx, Scope{b.OwnerID, a.ProjectID}, e.ID, update, ids[winner])
		return err
	})
	if !code(err, "eval_not_found") {
		t.Fatal(err)
	}
	if _, err = reader.Get(ctx, b.OwnerID, e.ID); !code(err, "eval_not_found") {
		t.Fatal(err)
	}
	// Pinned revisions cannot be deleted even by the trusted purge path.
	err = transaction(pool, func(s *Store) error {
		if _, err := s.db.Exec(ctx, `SELECT set_config('contractor.eval_purge','on',true)`); err != nil {
			return err
		}
		_, err := s.db.Exec(ctx, `DELETE FROM eval_dataset_revisions WHERE project_id=$1`, a.ProjectID)
		return err
	})
	if err == nil {
		t.Fatal("deleted pinned dataset")
	}
	if _, err = reader.UpdateDraft(ctx, a, e.ID, update, ids[0]); !errors.Is(err, ErrTransaction) {
		t.Fatal("mutation outside transaction", err)
	}
}

func TestPostgresEvalExternalFreezeAdmissionAndDeletionFence(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	scope := setupProject(t, pool, "owner", "eval")
	e := createExperiment(t, pool, scope, "exp", "trace-1", "external-workflow")
	ms := members(t, pool, e)
	if len(ms) != 8 {
		t.Fatal(len(ms))
	}
	doc := fixture(t, "external-workflow", "CreateExperiment")
	var input evaldomain.CreateExperiment
	json.Unmarshal(doc.Bytes(), &input)
	p, err := NewPostgresStore(pool).FrozenPlan(ctx, e.OwnerID, e.ID)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(p.Document.Bytes(), bytesOf(input.Registration)) || p.SHA256 != input.Registration.SourcePlanSHA256 {
		t.Fatal("registration identity changed")
	}
	for _, sql := range []string{`UPDATE eval_frozen_plans SET document=convert_to('{}','UTF8')`, `UPDATE eval_members SET eligibility='blocked'`, `UPDATE eval_experiments SET control_mode='server',revision=revision+1,updated_at=clock_timestamp()`} {
		if _, err = pool.Exec(ctx, sql); err == nil {
			t.Fatal("immutable data changed")
		}
	}
	errs := make([]error, len(ms))
	var wg sync.WaitGroup
	for i, m := range ms {
		wg.Add(1)
		go func(i int, m Member) {
			defer wg.Done()
			_, errs[i] = admit(t, pool, e, m.MemberID, nil, fmt.Sprintf("submit-%d", i))
		}(i, m)
	}
	wg.Wait()
	winner := -1
	for i, err := range errs {
		if err == nil {
			if winner >= 0 {
				t.Fatal("in-flight allowance exceeded")
			}
			winner = i
		} else if !code(err, "eval_not_ready") {
			t.Fatal(err)
		}
	}
	if winner < 0 {
		t.Fatal(errs)
	}
	live, err := NewPostgresStore(pool).Get(ctx, e.OwnerID, e.ID)
	if err != nil || live.Outstanding != 1 || live.StartedAt == nil || live.DeadlineAt == nil {
		t.Fatal(live, err)
	}
	r, err := admit(t, pool, e, ms[winner].MemberID, nil, fmt.Sprintf("submit-%d", winner))
	if err != nil || !r.Replayed {
		t.Fatal(r, err)
	}
	beginProjectDeletion(t, pool, scope)
	r, err = admit(t, pool, e, ms[winner].MemberID, nil, fmt.Sprintf("submit-%d", winner))
	if err != nil || !r.Replayed {
		t.Fatal("exact replay after fence", err)
	}
	_, err = admit(t, pool, e, ms[(winner+1)%len(ms)].MemberID, nil, "after-delete")
	if !code(err, "eval_project_deleting") {
		t.Fatal(err)
	}
	after, err := NewPostgresStore(pool).Get(ctx, e.OwnerID, e.ID)
	if err != nil || after.DeletionRequestedAt == nil || after.Outstanding != 1 || !after.DeadlineAt.Equal(*live.DeadlineAt) {
		t.Fatal(after, err)
	}
	if blocked, err := NewPostgresStore(pool).ProjectBlocked(ctx, scope.OwnerID, scope.ProjectID); err != nil || !blocked {
		t.Fatal(blocked, err)
	}
	if err = transaction(pool, func(s *Store) error { return s.PurgeProject(ctx, scope) }); !errors.Is(err, ErrDrain) {
		t.Fatal(err)
	}
}

func TestPostgresEvalNativeFreezeStaleClaimsAndSuboperationRecovery(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	scope := setupProject(t, pool, "owner", "eval")
	putDataset(t, pool, scope)
	e := createExperiment(t, pool, scope, "exp", "trace-1", "create-workflow")
	prepare := freeze(t, "Command", evaldomain.Command{Kind: "prepare"})
	cmd := CommandParams{Scope: scope, ExperimentID: e.ID, CommandID: "prepare-command", Command: evaldomain.Command{Kind: "prepare"}, Mutation: mutation(t, "prepare", e.Revision, prepare)}
	mustTx(t, pool, func(s *Store) error { _, err := s.Command(ctx, cmd); return err })
	claim := oneClaim(t, pool)
	reg := fixture(t, "registration", "ExternalRegistration")
	var r evaldomain.ExternalRegistration
	json.Unmarshal(reg.Bytes(), &r)
	setup := bytesOf(map[string]any{"variants": r.Variants, "checks": r.Checks, "comparison": r.Comparison, "budgets": r.Budgets})
	cases := map[string]evaldomain.Case{}
	for _, recipe := range r.Recipes {
		cases[recipe.MemberID] = recipe.Case
	}
	plan := fixture(t, "portable-plan", "playground.plan/v1")
	mustTx(t, pool, func(s *Store) error { return s.FreezePrepared(ctx, scope, e.ID, claim, plan, setup, cases) })
	saved, err := NewPostgresStore(pool).FrozenPlan(ctx, scope.OwnerID, e.ID)
	if err != nil || !bytes.Equal(saved.Document.Bytes(), plan.Bytes()) {
		t.Fatal("plan bytes changed", err)
	}
	e, err = NewPostgresStore(pool).Get(ctx, scope.OwnerID, e.ID)
	if err != nil {
		t.Fatal(err)
	}
	start := evaldomain.Command{Kind: "start", PlanSHA256: plan.Digest()}
	startDoc := freeze(t, "Command", start)
	startParams := CommandParams{Scope: scope, ExperimentID: e.ID, CommandID: "start-command", Command: start, Mutation: mutation(t, "start", e.Revision, startDoc)}
	mustTx(t, pool, func(s *Store) error { _, err := s.Command(ctx, startParams); return err })
	ms := members(t, pool, e)
	if _, err = admit(t, pool, e, ms[0].MemberID, nil, "wrong-controller"); !code(err, "eval_external_control") {
		t.Fatal(err)
	}
	if _, err = admit(t, pool, e, ms[0].MemberID, &claim, "native-submit"); err != nil {
		t.Fatal(err)
	}
	op := Suboperation{Kind: "run-create", Key: ms[0].SubmissionKey, Request: json.RawMessage(`{"workflow":"trace-a@1"}`)}
	mustTx(t, pool, func(s *Store) error { return s.PutSuboperation(ctx, scope, e.ID, ms[0].MemberID, claim, op) })
	// Force a replacement epoch while an old holder is blocked on the claim row.
	blocker, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	_, err = blocker.Exec(ctx, `UPDATE eval_controller_claims SET holder_id='replacement',epoch=epoch+1 WHERE experiment_id=$1`, e.ID)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		done <- transaction(pool, func(s *Store) error {
			return s.ResolveSuboperation(ctx, scope, e.ID, ms[0].MemberID, op.Kind, claim, []byte(`{"rejected":true}`), true)
		})
	}()
	// Confirm it reached a lock wait before publishing the new epoch.
	deadline := time.Now().Add(5 * time.Second)
	waiting := false
	for time.Now().Before(deadline) {
		if err = pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE datname=current_database() AND wait_event_type='Lock' AND query LIKE '%eval_controller_claims%' AND pid<>pg_backend_pid())`).Scan(&waiting); err != nil {
			t.Fatal(err)
		}
		if waiting {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !waiting {
		t.Fatal("old holder never waited on claim")
	}
	if err = blocker.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if err = <-done; !errors.Is(err, ErrClaimLost) {
		t.Fatal("stale holder committed", err)
	}
	if err = NewPostgresStore(pool).ReleaseClaim(ctx, claim); !errors.Is(err, ErrClaimLost) {
		t.Fatal(err)
	}
	replacement := claim
	replacement.HolderID = "replacement"
	replacement.Epoch++
	old, err := NewPostgresStore(pool).Suboperation(ctx, scope.OwnerID, e.ID, ms[0].MemberID, op.Kind)
	if err != nil || old.State != "intent" {
		t.Fatal(old, err)
	}
	mustTx(t, pool, func(s *Store) error {
		return s.ResolveSuboperation(ctx, scope, e.ID, ms[0].MemberID, op.Kind, replacement, []byte(`{"rejected":true}`), true)
	})
	mustTx(t, pool, func(s *Store) error { return s.Settle(ctx, scope, e.ID, ms[0].MemberID, replacement) })
	mustTx(t, pool, func(s *Store) error { return s.Settle(ctx, scope, e.ID, ms[0].MemberID, replacement) })
	e, err = NewPostgresStore(pool).Get(ctx, scope.OwnerID, e.ID)
	if err != nil || e.Outstanding != 0 {
		t.Fatal(e, err)
	}
}

func TestPostgresEvalBoundedPagesPinnedPurgeAndReceiptRetention(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	scope := setupProject(t, pool, "owner", "eval")
	other := setupProject(t, pool, "other", "other-eval")
	putDataset(t, pool, scope)
	putDataset(t, pool, other)
	for i := range 12 {
		createExperiment(t, pool, scope, fmt.Sprintf("exp-%02d", i), fmt.Sprintf("portable-%02d", i), "create-workflow")
	}
	reader := NewPostgresStore(pool)
	page, err := reader.List(ctx, ListParams{OwnerID: scope.OwnerID, ProjectID: scope.ProjectID, Limit: 5})
	if err != nil || len(page.Items) != 5 {
		t.Fatal(page, err)
	}
	page2, err := reader.List(ctx, ListParams{OwnerID: scope.OwnerID, ProjectID: scope.ProjectID, AfterID: page.Items[4].ID, Limit: 5, Revision: &page.Revision})
	if err != nil || len(page2.Items) != 5 || page2.Items[0].ID == page.Items[4].ID {
		t.Fatal(page2, err)
	}
	if _, err = reader.List(ctx, ListParams{OwnerID: scope.OwnerID, Limit: 101}); !code(err, "eval_invalid") {
		t.Fatal(err)
	}
	del := fixture(t, "delete", "Delete")
	m := mutation(t, "delete", 1, del)
	mustTx(t, pool, func(s *Store) error { _, err := s.BeginDeletion(ctx, scope, "exp-00", m); return err })
	if _, err = reader.List(ctx, ListParams{OwnerID: scope.OwnerID, Limit: 5, ProjectID: scope.ProjectID, Revision: &page.Revision}); !code(err, "eval_view_changed") {
		t.Fatal(err)
	}
	mustTx(t, pool, func(s *Store) error { return s.Purge(ctx, scope, "exp-00") })
	mustTx(t, pool, func(s *Store) error {
		r, err := s.BeginDeletion(ctx, scope, "exp-00", m)
		if err == nil && !r.Replayed {
			t.Fatal("delete receipt lost")
		}
		return err
	})
	if _, err = reader.Dataset(ctx, scope, "trace-small", "r1"); err != nil {
		t.Fatal("still-pinned revision removed", err)
	}
	beginProjectDeletion(t, pool, scope)
	mustTx(t, pool, func(s *Store) error { return s.PurgeProject(ctx, scope) })
	for _, table := range []string{"eval_experiments", "eval_dataset_revisions", "eval_mutation_receipts"} {
		var n int
		if err = pool.QueryRow(ctx, `SELECT count(*) FROM `+table+` WHERE project_id=$1`, scope.ProjectID).Scan(&n); err != nil || n != 0 {
			t.Fatal(table, n, err)
		}
	}
	if _, err = reader.Dataset(ctx, other, "trace-small", "r1"); err != nil {
		t.Fatal("foreign data purged", err)
	}
}

func TestPostgresEvalAdmissionAndProjectDeletionSerializeBothOrders(t *testing.T) {
	for _, deletionFirst := range []bool{false, true} {
		t.Run(fmt.Sprint(deletionFirst), func(t *testing.T) {
			pool := testPool(t)
			ctx := context.Background()
			scope := setupProject(t, pool, "owner", "eval")
			e := createExperiment(t, pool, scope, "exp", "trace-1", "external-workflow")
			m := members(t, pool, e)[0]
			submit := freeze(t, "Submission", evaldomain.Submission{PlanSHA256: planDigest(t, pool, e)})
			p := Admission{scope, e.ID, m.MemberID, planDigest(t, pool, e), nil, mutation(t, "submit", 0, submit)}
			tx, err := pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer tx.Rollback(ctx)
			done := make(chan error, 1)
			if deletionFirst {
				_, _, err = projectstore.NewPostgresStore(tx).BeginDeletion(ctx, projectstore.BeginDeletionParams{OwnerID: scope.OwnerID, ProjectID: scope.ProjectID, ExpectedRevision: 1})
				if err != nil {
					t.Fatal(err)
				}
				go func() { done <- transaction(pool, func(s *Store) error { _, err := s.Admit(ctx, p); return err }) }()
			} else {
				if _, err = NewTxStore(tx).Admit(ctx, p); err != nil {
					t.Fatal(err)
				}
				go func() {
					_, _, err := projectstore.NewPostgresStore(pool).BeginDeletion(ctx, projectstore.BeginDeletionParams{OwnerID: scope.OwnerID, ProjectID: scope.ProjectID, ExpectedRevision: 1})
					done <- err
				}()
			}
			select {
			case err := <-done:
				t.Fatalf("competing mutation passed Project lock: %v", err)
			case <-time.After(40 * time.Millisecond):
			}
			if err = tx.Commit(ctx); err != nil {
				t.Fatal(err)
			}
			err = <-done
			if deletionFirst && !code(err, "eval_project_deleting") || !deletionFirst && err != nil {
				t.Fatal(err)
			}
			e, err = NewPostgresStore(pool).Get(ctx, scope.OwnerID, e.ID)
			if err != nil {
				t.Fatal(err)
			}
			want := 1
			if deletionFirst {
				want = 0
			}
			if e.Outstanding != want || e.DeletionRequestedAt == nil {
				t.Fatal(e)
			}
		})
	}
}

func TestPostgresEvalAuditWorkspaceDependencyFencesWithoutDeletingExperiment(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	scope := setupProject(t, pool, "owner", "eval")
	e := createExperiment(t, pool, scope, "exp", "trace-1", "external-audit")
	m := members(t, pool, e)[0]
	if m.ExecutionKind != "audit" {
		t.Fatal(m.ExecutionKind)
	}
	if _, err := admit(t, pool, e, m.MemberID, nil, "submit"); err != nil {
		t.Fatal(err)
	}
	claim := oneClaim(t, pool)
	op := Suboperation{Kind: "project-create", Key: m.SubmissionKey + "-project", Request: json.RawMessage(`{"kind":"project"}`)}
	mustTx(t, pool, func(s *Store) error { return s.PutSuboperation(ctx, scope, e.ID, m.MemberID, claim, op) })
	mustTx(t, pool, func(s *Store) error {
		_, _, err := projectstore.NewPostgresStore(s.tx).Create(ctx, projectstore.CreateParams{OwnerID: scope.OwnerID, ProjectID: "audit-workspace", Kind: projectstore.KindProject, Name: "Audit member", IdempotencyKey: op.Key, RequestDigest: evaldomain.Digest(op.Request)})
		if err != nil {
			return err
		}
		return s.RegisterExecutionProject(ctx, scope, e.ID, m.MemberID, "audit-workspace", op.Key, claim)
	})
	mustTx(t, pool, func(s *Store) error {
		return s.ResolveSuboperation(ctx, scope, e.ID, m.MemberID, op.Kind, claim, []byte(`{"projectId":"audit-workspace"}`), false)
	})
	child := Scope{scope.OwnerID, "audit-workspace"}
	beginProjectDeletion(t, pool, child)
	current, err := NewPostgresStore(pool).Get(ctx, scope.OwnerID, e.ID)
	if err != nil || current.State != "cancelling" || current.DeletionRequestedAt != nil {
		t.Fatal("child deletion must not erase experiment", current, err)
	}
	if blocked, err := NewPostgresStore(pool).ProjectBlocked(ctx, child.OwnerID, child.ProjectID); err != nil || !blocked {
		t.Fatal(blocked, err)
	}
	mustTx(t, pool, func(s *Store) error { return s.Settle(ctx, scope, e.ID, m.MemberID, claim) })
	if blocked, err := NewPostgresStore(pool).ProjectBlocked(ctx, child.OwnerID, child.ProjectID); err != nil || blocked {
		t.Fatal(blocked, err)
	}
	if _, err = admit(t, pool, e, members(t, pool, e)[1].MemberID, nil, "after-child-delete"); !code(err, "eval_not_ready") {
		t.Fatal(err)
	}
}

func TestPostgresEvalFrozenDeadlineAndOptionalTokenAllowance(t *testing.T) {
	for _, tokenFence := range []bool{false, true} {
		t.Run(fmt.Sprint(tokenFence), func(t *testing.T) {
			pool := testPool(t)
			ctx := context.Background()
			scope := setupProject(t, pool, "owner", "eval")
			var input evaldomain.CreateExperiment
			if err := json.Unmarshal(fixture(t, "external-workflow", "CreateExperiment").Bytes(), &input); err != nil {
				t.Fatal(err)
			}
			input.Registration.Budgets.MaxInFlight = 8
			if tokenFence {
				zero := int64(0)
				input.Registration.Budgets.MaxObservedTotalTokens = &zero
			} else {
				input.Registration.Budgets.WallMS = 1
			}
			doc := freeze(t, "CreateExperiment", input)
			m := mutation(t, "create", 0, doc)
			mustTx(t, pool, func(s *Store) error {
				_, err := s.Create(ctx, CreateParams{Scope: scope, ID: "exp", PortableID: "trace-1", Document: doc, Mutation: m})
				return err
			})
			e, err := NewPostgresStore(pool).Get(ctx, scope.OwnerID, "exp")
			if err != nil {
				t.Fatal(err)
			}
			ms := members(t, pool, e)
			_, err = admit(t, pool, e, ms[0].MemberID, nil, "first")
			if tokenFence {
				if !code(err, "eval_budget_exhausted") {
					t.Fatal(err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			time.Sleep(5 * time.Millisecond)
			_, err = admit(t, pool, e, ms[1].MemberID, nil, "second")
			if !code(err, "eval_budget_exhausted") {
				t.Fatal(err)
			}
			if r, err := admit(t, pool, e, ms[0].MemberID, nil, "first"); err != nil || !r.Replayed {
				t.Fatal("deadline invalidated accepted receipt", r, err)
			}
		})
	}
}
