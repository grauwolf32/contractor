package postgres

import (
	"context"
	"testing"
	"time"
)

func TestOperationBudgetValidationAndCallerDeadline(t *testing.T) {
	parent, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := WithOperationBudgets(parent, Budgets{QueryTimeout: -1}); err == nil {
		t.Fatal("invalid operation policy accepted")
	}
	b := Budgets{250 * time.Millisecond, 2 * time.Second, 1500 * time.Millisecond, 100 * time.Millisecond, 2 * time.Second}
	ctx, err := WithOperationBudgets(parent, b)
	if err != nil {
		t.Fatal(err)
	}
	tracer := &budgetTracer{budgets: DefaultBudgets()}
	if tracer.policy(ctx) != b || tracer.policy(parent) != DefaultBudgets() {
		t.Fatal("policy leaked to parent")
	}
	deadline, _ := ctx.Deadline()
	want, _ := parent.Deadline()
	if deadline != want {
		t.Fatal("earlier deadline extended")
	}
}
