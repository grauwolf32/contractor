package evalstore

import (
	"bytes"
	"context"
	"reflect"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type queryCounter struct {
	mu    sync.Mutex
	count int
}

func (q *queryCounter) TraceQueryStart(ctx context.Context, _ *pgx.Conn, _ pgx.TraceQueryStartData) context.Context {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.count++
	return ctx
}
func (*queryCounter) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}
func (q *queryCounter) take() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	count := q.count
	q.count = 0
	return count
}

func TestPostgresNextMembersReadsBatchInOneQuery(t *testing.T) {
	pool := testPool(t)
	ctx := t.Context()
	scope := setupProject(t, pool, "owner", "eval")
	e := createExperiment(t, pool, scope, "exp", "trace-1", "external-workflow")
	want := members(t, pool, e)
	trace := &queryCounter{}
	config := pool.Config()
	config.ConnConfig.Tracer = trace
	traced, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer traced.Close()
	reader := NewPostgresStore(traced)
	got, err := reader.NextMembers(ctx, e.OwnerID, e.ID, len(want))
	if err != nil {
		t.Fatal(err)
	}
	if queries := trace.take(); queries != 1 {
		t.Fatalf("NextMembers issued %d queries for %d members", queries, len(got))
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("batched members differ from single reads:\n got %+v\nwant %+v", got, want)
	}
	if _, err := admit(t, pool, e, want[0].MemberID, nil, "submit-first"); err != nil {
		t.Fatal(err)
	}
	trace.take()
	got, err = reader.NextMembers(ctx, e.OwnerID, e.ID, 2)
	if err != nil || trace.take() != 1 || !reflect.DeepEqual(got, want[1:3]) {
		t.Fatalf("next unsubmitted members = %+v, %v", got, err)
	}
}

func TestPostgresFrozenPlanRestoresStoredDocumentWithoutRevalidation(t *testing.T) {
	pool := testPool(t)
	ctx := t.Context()
	scope := setupProject(t, pool, "owner", "eval")
	e := createExperiment(t, pool, scope, "exp", "trace-1", "external-workflow")
	// Only DDL can bypass eval_plans_immutable. The probe field breaks the
	// closed schema, so a read that revalidated the plan would reject it.
	for _, statement := range []string{
		`ALTER TABLE eval_frozen_plans DISABLE TRIGGER eval_plans_immutable`,
		`UPDATE eval_frozen_plans SET document=convert_to('{"probe":true,'||substr(convert_from(document,'UTF8'),2),'UTF8')`,
		`ALTER TABLE eval_frozen_plans ENABLE TRIGGER eval_plans_immutable`,
	} {
		if _, err := pool.Exec(ctx, statement); err != nil {
			t.Fatal(err)
		}
	}
	var stored []byte
	var digest, identity string
	if err := pool.QueryRow(ctx, `SELECT document, document_sha256, plan_sha256 FROM eval_frozen_plans WHERE experiment_id=$1`, e.ID).Scan(&stored, &digest, &identity); err != nil {
		t.Fatal(err)
	}
	if _, err := evaldomain.Freeze("ExternalRegistration", stored); !code(err, "eval_invalid") {
		t.Fatalf("probe kept the stored plan schema-valid: %v", err)
	}
	plan, err := NewPostgresStore(pool).FrozenPlan(ctx, e.OwnerID, e.ID)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Document.Kind() != "ExternalRegistration" || !bytes.Equal(plan.Document.Bytes(), stored) || plan.Document.Digest() != digest || plan.SHA256 != identity {
		t.Fatalf("restored plan kind=%s digest=%s identity=%s", plan.Document.Kind(), plan.Document.Digest(), plan.SHA256)
	}
	if digest != evaldomain.Digest(stored) {
		t.Fatal("generated document digest differs from the stored bytes")
	}
}
