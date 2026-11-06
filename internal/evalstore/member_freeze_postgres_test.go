package evalstore

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"regexp"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// freezeWriteTracer counts client statements that write the frozen matrix,
// by table.
type freezeWriteTracer struct {
	mu     sync.Mutex
	writes map[string]int
}

var freezeWrite = regexp.MustCompile(`(?i)^\s*(?:update|insert\s+into)\s+(eval_experiments|eval_members|eval_plan_resources)\b`)

func (f *freezeWriteTracer) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	if match := freezeWrite.FindStringSubmatch(data.SQL); match != nil {
		f.mu.Lock()
		f.writes[strings.ToLower(match[1])]++
		f.mu.Unlock()
	}
	return ctx
}
func (*freezeWriteTracer) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

// tracedFreezeWrites returns a pool whose freeze writes the tracer counts.
func tracedFreezeWrites(t *testing.T, pool *pgxpool.Pool) (*pgxpool.Pool, func() map[string]int) {
	t.Helper()
	trace := &freezeWriteTracer{writes: map[string]int{}}
	config := pool.Config()
	config.ConnConfig.Tracer = trace
	traced, err := pgxpool.NewWithConfig(t.Context(), config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(traced.Close)
	return traced, func() map[string]int {
		trace.mu.Lock()
		defer trace.mu.Unlock()
		return maps.Clone(trace.writes)
	}
}

type nativeFreeze struct {
	scope Scope
	e     Experiment
	claim Claim
	plan  evaldomain.Frozen
	setup []byte
	cases map[string]evaldomain.Case
	last  string // the manifest member inserted last
}

// newNativeFreeze returns a claimed native experiment in preparing state and
// the fixture plan, setup and member cases that FreezePrepared accepts.
func newNativeFreeze(t *testing.T, pool *pgxpool.Pool) nativeFreeze {
	t.Helper()
	ctx := t.Context()
	scope := setupProject(t, pool, "owner", "eval")
	putDataset(t, pool, scope)
	e := createExperiment(t, pool, scope, "exp", "trace-1", "create-workflow")
	prepare := freeze(t, "Command", evaldomain.Command{Kind: "prepare"})
	cmd := CommandParams{Scope: scope, ExperimentID: e.ID, CommandID: "prepare-command", Command: evaldomain.Command{Kind: "prepare"}, Mutation: mutation(t, "prepare", e.Revision, prepare)}
	mustTx(t, pool, func(s *Store) error { _, err := s.Command(ctx, cmd); return err })
	var r evaldomain.ExternalRegistration
	if err := json.Unmarshal(fixture(t, "registration", "ExternalRegistration").Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	cases := map[string]evaldomain.Case{}
	for _, recipe := range r.Recipes {
		cases[recipe.MemberID] = recipe.Case
	}
	plan := fixture(t, "portable-plan", "playground.plan/v1")
	manifest, err := evaldomain.PublicPlanProjection(plan)
	if err != nil {
		t.Fatal(err)
	}
	return nativeFreeze{
		scope: scope, e: e, claim: oneClaim(t, pool), plan: plan, cases: cases,
		setup: bytesOf(map[string]any{"variants": r.Variants, "checks": r.Checks, "comparison": r.Comparison, "budgets": r.Budgets}),
		last:  manifest.Members[len(manifest.Members)-1].MemberID,
	}
}

func (f nativeFreeze) freeze(ctx context.Context, s *Store) error {
	return s.FreezePrepared(ctx, f.scope, f.e.ID, f.claim, f.plan, f.setup, f.cases)
}

// caseOfProjectionSize returns c with task parameters that make its
// execution projection exactly size bytes long.
func caseOfProjectionSize(t *testing.T, c evaldomain.Case, size int) evaldomain.Case {
	t.Helper()
	c.Source = nil
	c.Task.Parameters = map[string]string{}
	for i := 0; ; i++ {
		key := fmt.Sprintf("p%03d", i)
		c.Task.Parameters[key] = ""
		remaining := size - len(bytesOf(c))
		if remaining <= 8192 {
			c.Task.Parameters[key] = strings.Repeat("x", remaining)
			break
		}
		c.Task.Parameters[key] = strings.Repeat("x", 8192)
	}
	visible, err := evaldomain.ExecutionProjection(c)
	if err != nil {
		t.Fatal(err)
	}
	if n := len(bytesOf(visible)); n != size {
		t.Fatalf("case projection is %d bytes, want %d", n, size)
	}
	return c
}

// recipeOverflowCase returns c with an execution projection of exactly
// MaxDocumentBytes. The case alone is admissible, but with any variant its
// recipe exceeds eval_members_recipe_check.
func recipeOverflowCase(t *testing.T, c evaldomain.Case) evaldomain.Case {
	t.Helper()
	return caseOfProjectionSize(t, c, evaldomain.MaxDocumentBytes)
}

// enlargeCases gives every distinct case of the freeze an execution
// projection of size bytes.
func (f *nativeFreeze) enlargeCases(t *testing.T, size int) {
	t.Helper()
	large := map[string]evaldomain.Case{}
	for member, c := range f.cases {
		if _, ok := large[c.ID]; !ok {
			large[c.ID] = caseOfProjectionSize(t, c, size)
		}
		f.cases[member] = large[c.ID]
	}
}

func rows(t *testing.T, pool *pgxpool.Pool, query string, args ...any) int {
	t.Helper()
	var n int
	if err := pool.QueryRow(t.Context(), query, args...).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// assertNothingFrozen requires that a failed freeze left no member, member
// projection, projection queue row, plan or plan resource of experiment id.
func assertNothingFrozen(t *testing.T, pool *pgxpool.Pool, id string) {
	t.Helper()
	if n := rows(t, pool, `SELECT count(*) FROM eval_experiments WHERE experiment_id=$1 AND plan_document IS NOT NULL`, id); n != 0 {
		t.Fatal("failed freeze left its plan")
	}
	for _, table := range []string{"eval_members", "eval_member_projections", "eval_projection_queue", "eval_plan_resources"} {
		if n := rows(t, pool, `SELECT count(*) FROM `+table+` WHERE experiment_id = $1`, id); n != 0 {
			t.Fatalf("failed freeze left %d %s rows", n, table)
		}
	}
}

// assertStillPreparing requires the native experiment to be unchanged by a
// failed freeze: preparing, with no expected matrix.
func assertStillPreparing(t *testing.T, pool *pgxpool.Pool, f nativeFreeze) {
	t.Helper()
	e, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).Get(t.Context(), f.scope.OwnerID, f.e.ID)
	if err != nil {
		t.Fatal(err)
	}
	if e.State != evaldomain.StatePreparing || e.Expected != 0 {
		t.Fatalf("failed freeze changed the experiment: state=%s expected=%d", e.State, e.Expected)
	}
	assertNothingFrozen(t, pool, f.e.ID)
}

// assertNoExternalCreate requires that a failed registration left no
// experiment, frozen matrix or create receipt.
func assertNoExternalCreate(t *testing.T, pool *pgxpool.Pool, id string) {
	t.Helper()
	if n := rows(t, pool, `SELECT count(*) FROM eval_experiments WHERE experiment_id = $1`, id); n != 0 {
		t.Fatal("failed registration left its experiment")
	}
	if n := rows(t, pool, `SELECT count(*) FROM mutation_receipts WHERE key_scope[1] = 'eval' AND operation = 'experiment-create'`); n != 0 {
		t.Fatalf("failed registration left %d create receipts", n)
	}
	assertNothingFrozen(t, pool, id)
}

// A freeze is one transaction: whether a member is rejected before any write,
// the database rejects a row of the first or of a later member statement, or
// the transaction fails after the whole matrix was inserted, nothing of the
// plan, its members, their projections or the create receipt remains, and a
// retry freezes the complete matrix.
func TestSharedPostgresEvalFreezeLeavesNoPartialMatrix(t *testing.T) {
	injected := errors.New("injected transaction failure")
	t.Run("member without recipe is rejected before any write", func(t *testing.T) {
		pool := testPool(t)
		f := newNativeFreeze(t, pool)
		f.cases["0000000000000000000000000000000000000000000000000000000000000000"] = f.cases[f.last]
		delete(f.cases, f.last)
		traced, writes := tracedFreezeWrites(t, pool)
		if err := transaction(traced, func(s *Store) error { return f.freeze(t.Context(), s) }); !code(err, "eval_member_conflict") {
			t.Fatalf("freeze without a member recipe: %v", err)
		}
		if sent := writes(); len(sent) != 0 {
			t.Fatalf("rejected matrix sent freeze writes %v before validation finished", sent)
		}
		assertStillPreparing(t, pool, f)
	})
	t.Run("database rejects the last member row", func(t *testing.T) {
		pool := testPool(t)
		f := newNativeFreeze(t, pool)
		valid := f.cases[f.last]
		f.cases[f.last] = recipeOverflowCase(t, valid)
		err := transaction(pool, func(s *Store) error { return f.freeze(t.Context(), s) })
		var pgErr *pgconn.PgError
		if !errors.As(err, &pgErr) || pgErr.ConstraintName != "eval_members_recipe_check" {
			t.Fatalf("oversized last recipe: %v", err)
		}
		assertStillPreparing(t, pool, f)
		f.cases[f.last] = valid
		mustTx(t, pool, func(s *Store) error { return f.freeze(t.Context(), s) })
		if n := rows(t, pool, `SELECT count(*) FROM eval_members WHERE experiment_id = $1`, f.e.ID); n != len(f.cases) {
			t.Fatalf("retried freeze stored %d of %d members", n, len(f.cases))
		}
	})
	t.Run("database rejects a row of a later member statement", func(t *testing.T) {
		pool := testPool(t)
		f := newNativeFreeze(t, pool)
		f.enlargeCases(t, 950_000)
		f.cases[f.last] = recipeOverflowCase(t, f.cases[f.last])
		traced, writes := tracedFreezeWrites(t, pool)
		err := transaction(traced, func(s *Store) error { return f.freeze(t.Context(), s) })
		var pgErr *pgconn.PgError
		if !errors.As(err, &pgErr) || pgErr.ConstraintName != "eval_members_recipe_check" {
			t.Fatalf("oversized last recipe: %v", err)
		}
		if sent := writes()["eval_members"]; sent != 2 {
			t.Fatalf("freeze sent %d member statements, want the first batch stored and the second rejected", sent)
		}
		assertStillPreparing(t, pool, f)
	})
	t.Run("native transaction fails after the matrix insert", func(t *testing.T) {
		pool := testPool(t)
		f := newNativeFreeze(t, pool)
		err := transaction(pool, func(s *Store) error {
			if err := f.freeze(t.Context(), s); err != nil {
				return err
			}
			return injected
		})
		if !errors.Is(err, injected) {
			t.Fatal(err)
		}
		assertStillPreparing(t, pool, f)
	})
	for _, failure := range []string{"invalid resource after the matrix insert", "external transaction fails after the receipt"} {
		t.Run(failure, func(t *testing.T) {
			pool := testPool(t)
			scope := setupProject(t, pool, "owner", "eval")
			doc := fixture(t, "external-workflow", "CreateExperiment")
			params := CreateParams{Scope: scope, ID: "exp", PortableID: "trace-1", Document: doc, Mutation: mutation(t, "exp", 0, doc)}
			failed := params
			want := injected
			if failure == "invalid resource after the matrix insert" {
				// putResources runs after persistPlan has inserted the members.
				failed.Resources = []PlanResource{{Path: "/absolute", Document: doc}}
				want = nil
			}
			err := transaction(pool, func(s *Store) error {
				if _, err := s.Create(t.Context(), failed); err != nil {
					return err
				}
				return injected
			})
			if want == nil && !code(err, "eval_invalid") || want != nil && !errors.Is(err, want) {
				t.Fatalf("failed registration: %v", err)
			}
			assertNoExternalCreate(t, pool, params.ID)
			var receipt Receipt
			mustTx(t, pool, func(s *Store) error {
				var err error
				receipt, err = s.Create(t.Context(), params)
				return err
			})
			e, err := NewPostgresStore(pool, artifactpgtest.NewStorage()).Get(t.Context(), scope.OwnerID, params.ID)
			if err != nil {
				t.Fatal(err)
			}
			if receipt.Replayed || e.Expected != 8 || rows(t, pool, `SELECT count(*) FROM eval_members WHERE experiment_id = $1`, e.ID) != 8 {
				t.Fatalf("retried registration: replayed=%v expected=%d", receipt.Replayed, e.Expected)
			}
		})
	}
}

// Recipes above a fifth of the 4 MiB statement bound split the matrix: the
// fixture's eight members with ~950 KB recipes freeze with two statements of
// four, the projection queue is created by the first and advanced by the
// second, and every recipe is stored whole.
func TestSharedPostgresEvalFreezeSplitsMatrixAtRecipeByteBound(t *testing.T) {
	pool := testPool(t)
	f := newNativeFreeze(t, pool)
	f.enlargeCases(t, 950_000)
	var setup struct {
		Variants []evaldomain.Variant `json:"variants"`
	}
	if err := json.Unmarshal(f.setup, &setup); err != nil {
		t.Fatal(err)
	}
	variants := map[string]evaldomain.Variant{}
	for _, v := range setup.Variants {
		variants[v.ID] = v
	}
	manifest, err := evaldomain.PublicPlanProjection(f.plan)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string][]byte{}
	for _, m := range manifest.Members {
		visible, err := evaldomain.ExecutionProjection(f.cases[m.MemberID])
		if err != nil {
			t.Fatal(err)
		}
		recipe := bytesOf(Recipe{Case: visible, Variant: variants[m.VariantID]})
		// Four recipes fit one statement and a fifth does not.
		if len(recipe) <= freezeBatchBytes/5 || len(recipe) > freezeBatchBytes/4 {
			t.Fatalf("fixture recipe is %d bytes", len(recipe))
		}
		want[m.MemberID] = recipe
	}
	traced, writes := tracedFreezeWrites(t, pool)
	mustTx(t, traced, func(s *Store) error { return f.freeze(t.Context(), s) })
	if sent := writes()["eval_members"]; sent != 2 {
		t.Fatalf("%d members with large recipes froze with %d member statements, want 2", len(want), sent)
	}
	var revision, published int64
	if err = pool.QueryRow(t.Context(), `SELECT revision, published_revision FROM eval_projection_queue WHERE experiment_id = $1`, f.e.ID).Scan(&revision, &published); err != nil {
		t.Fatal(err)
	}
	if revision != 2 || published != 0 {
		t.Fatalf("projection queue revision %d published %d, want created once and advanced once", revision, published)
	}
	if n := rows(t, pool, `SELECT count(*) FROM eval_member_projections WHERE experiment_id = $1 AND revision = 1`, f.e.ID); n != len(want) {
		t.Fatalf("%d member projections created once, want %d", n, len(want))
	}
	stored, err := pool.Query(t.Context(), `SELECT member_id, recipe FROM eval_members WHERE experiment_id = $1`, f.e.ID)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string][]byte{}
	for stored.Next() {
		var id string
		var recipe []byte
		if err = stored.Scan(&id, &recipe); err != nil {
			t.Fatal(err)
		}
		got[id] = recipe
	}
	if err = stored.Err(); err != nil {
		t.Fatal(err)
	}
	if !maps.EqualFunc(got, want, bytes.Equal) {
		t.Fatalf("stored %d recipes that differ from the %d expected recipe encodings", len(got), len(want))
	}
}
