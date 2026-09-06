package postgres

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/jackc/pgx/v5"
)

// Budgets bounds database work, not the lifetime of an HTTP stream or LISTEN
// subscription. Zero fields select ordinary defaults; disabling a budget is
// deliberately not supported. Caller deadlines always take precedence.
type Budgets struct {
	AcquireTimeout         time.Duration
	QueryTimeout           time.Duration
	StatementTimeout       time.Duration
	LockTimeout            time.Duration
	IdleTransactionTimeout time.Duration
}

func DefaultBudgets() Budgets {
	return Budgets{2 * time.Second, 20 * time.Second, 15 * time.Second, 2 * time.Second, 30 * time.Second}
}

func (b Budgets) normalized() (Budgets, error) {
	defaults := DefaultBudgets()
	for _, pair := range []struct {
		value    *time.Duration
		fallback time.Duration
	}{
		{&b.AcquireTimeout, defaults.AcquireTimeout}, {&b.QueryTimeout, defaults.QueryTimeout},
		{&b.StatementTimeout, defaults.StatementTimeout}, {&b.LockTimeout, defaults.LockTimeout},
		{&b.IdleTransactionTimeout, defaults.IdleTransactionTimeout},
	} {
		if *pair.value == 0 {
			*pair.value = pair.fallback
		}
		if *pair.value < time.Millisecond || *pair.value > 24*time.Hour {
			return Budgets{}, fmt.Errorf("PostgreSQL budgets must be between 1ms and 24h")
		}
	}
	if b.LockTimeout >= b.StatementTimeout || b.StatementTimeout >= b.QueryTimeout {
		return Budgets{}, fmt.Errorf("PostgreSQL budgets require lock < statement < query timeout")
	}
	return b, nil
}

type maintenanceBudgetKey struct{}

// WithOperationBudgets selects one validated operation policy. Callers own an
// outer deadline and apply overrides transaction-locally, never with session SET.
func WithOperationBudgets(ctx context.Context, budgets Budgets) (context.Context, error) {
	budgets, err := budgets.normalized()
	if err != nil {
		return nil, err
	}
	return context.WithValue(ctx, maintenanceBudgetKey{}, budgets), nil
}

// WithMigrationBudget bounds the entire migrator (including advisory-lock
// acquisition) to 15 minutes. InTx installs these server settings with SET
// LOCAL, so commit/rollback cannot leak maintenance settings into the pool.
func WithMigrationBudget(ctx context.Context) (context.Context, context.CancelFunc) {
	return withMaintenanceBudget(ctx, Budgets{5 * time.Second, 125 * time.Second, 120 * time.Second, 10 * time.Second, 60 * time.Second}, 15*time.Minute)
}

// WithCleanupBudget allows bounded physical collection to do more work than
// an ordinary request. It is still cancellable, including during shutdown.
func WithCleanupBudget(ctx context.Context) (context.Context, context.CancelFunc) {
	return withMaintenanceBudget(ctx, Budgets{2 * time.Second, 65 * time.Second, 60 * time.Second, 5 * time.Second, 30 * time.Second}, 2*time.Minute)
}

func withMaintenanceBudget(ctx context.Context, b Budgets, total time.Duration) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.WithValue(ctx, maintenanceBudgetKey{}, b), total)
}

func timeoutSetting(d time.Duration) string {
	return strconv.FormatInt(int64((d+time.Millisecond-1)/time.Millisecond), 10)
}

// ApplyTransactionBudget applies only an explicitly selected maintenance
// policy to an existing transaction. Ordinary transactions inherit pool
// settings. Never issue session-level SET on a pooled connection.
func ApplyTransactionBudget(ctx context.Context, tx pgx.Tx) error {
	b, ok := ctx.Value(maintenanceBudgetKey{}).(Budgets)
	if !ok {
		return nil
	}
	_, err := tx.Exec(ctx, `SELECT set_config('statement_timeout',$1,true),
set_config('lock_timeout',$2,true), set_config('idle_in_transaction_session_timeout',$3,true)`,
		timeoutSetting(b.StatementTimeout), timeoutSetting(b.LockTimeout), timeoutSetting(b.IdleTransactionTimeout))
	if err != nil {
		return WrapError("configure PostgreSQL transaction budgets", err)
	}
	return nil
}
