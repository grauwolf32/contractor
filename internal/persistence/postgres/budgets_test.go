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
