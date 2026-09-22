package postgres

import (
	"context"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestDatabaseBudgetsAreFiniteAndOrdered(t *testing.T) {
	b, err := (Budgets{}).normalized()
	if err != nil || b != DefaultBudgets() {
		t.Fatalf("default budgets: %+v %v", b, err)
	}
	for _, b := range []Budgets{
		{AcquireTimeout: -1}, {QueryTimeout: -1}, {StatementTimeout: -1}, {LockTimeout: -1}, {IdleTransactionTimeout: -1},
		{AcquireTimeout: time.Microsecond}, {QueryTimeout: 25 * time.Hour}, {LockTimeout: 15 * time.Second},
		{StatementTimeout: 20 * time.Second}, {QueryTimeout: time.Second},
	} {
		if _, err := b.normalized(); err == nil {
			t.Errorf("invalid budgets accepted: %+v", b)
		}
	}
	if timeoutSetting(1500*time.Microsecond) != "2" {
		t.Fatal("sub-millisecond rounding can disable timeout")
	}
}

func TestMigrationBudgetsFollowConfiguredTimeouts(t *testing.T) {
	b, total, err := (MigrationBudgets{}).normalized()
	if err != nil || b.StatementTimeout != 120*time.Second || b.QueryTimeout != 125*time.Second ||
		b.LockTimeout != 10*time.Second || total != 15*time.Minute {
		t.Fatalf("default migration budgets: %+v %s %v", b, total, err)
	}
	b, total, err = (MigrationBudgets{StatementTimeout: time.Hour, LockTimeout: time.Minute}).normalized()
	if err != nil || b.StatementTimeout != time.Hour || b.QueryTimeout != time.Hour+5*time.Second ||
		b.LockTimeout != time.Minute || total != time.Hour+5*time.Second+5*time.Minute {
		t.Fatalf("configured migration budgets: %+v %s %v", b, total, err)
	}
	for _, m := range []MigrationBudgets{
		{StatementTimeout: -1}, {LockTimeout: -1}, {LockTimeout: 120 * time.Second},
		{StatementTimeout: time.Second, LockTimeout: time.Second}, {StatementTimeout: 24 * time.Hour},
	} {
		if err := m.Validate(); err == nil {
			t.Errorf("invalid migration budgets accepted: %+v", m)
		}
	}
	ctx, cancel, err := WithMigrationBudgets(t.Context(), MigrationBudgets{StatementTimeout: 20 * time.Minute, LockTimeout: 30 * time.Second})
	if err != nil {
		t.Fatal(err)
	}
	defer cancel()
	query := (&budgetTracer{budgets: DefaultBudgets()}).TraceQueryStart(ctx, nil, pgx.TraceQueryStartData{})
	deadline, _ := query.Deadline()
	if remaining := time.Until(deadline); remaining < 20*time.Minute || remaining > 20*time.Minute+5*time.Second {
		t.Fatalf("migration tracer deadline %s does not follow statement timeout", remaining)
	}
	(&budgetTracer{}).TraceQueryEnd(query, nil, pgx.TraceQueryEndData{})
}

func TestDatabaseTracerDeadlinesDoNotEscapeIndividualOperations(t *testing.T) {
	tracer := &budgetTracer{budgets: DefaultBudgets()}
	parent := t.Context()
	acquire := tracer.TraceAcquireStart(parent, nil, pgxpool.TraceAcquireStartData{})
	if _, ok := acquire.Deadline(); !ok {
		t.Fatal("acquire has no deadline")
	}
	tracer.TraceAcquireEnd(acquire, nil, pgxpool.TraceAcquireEndData{})
	if acquire.Err() == nil || parent.Err() != nil {
		t.Fatal("acquire timer was leaked or parent was cancelled")
	}
	query := tracer.TraceQueryStart(parent, nil, pgx.TraceQueryStartData{})
	if _, ok := query.Deadline(); !ok {
		t.Fatal("query has no deadline")
	}
	tracer.TraceQueryEnd(query, nil, pgx.TraceQueryEndData{})
	if query.Err() == nil || parent.Err() != nil {
		t.Fatal("query timer was leaked or parent was cancelled")
	}
	short, cancel := context.WithTimeout(parent, 50*time.Millisecond)
	defer cancel()
	query = tracer.TraceQueryStart(short, nil, pgx.TraceQueryStartData{})
	deadline, _ := query.Deadline()
	want, _ := short.Deadline()
	if deadline != want {
		t.Fatal("caller deadline was extended")
	}
	tracer.TraceQueryEnd(query, nil, pgx.TraceQueryEndData{})
}
