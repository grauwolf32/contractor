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

// Validate checks a process policy without opening a database connection.
func (b Budgets) Validate() error {
	_, err := b.normalized()
	return err
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

// MigrationBudgets are the operator-selectable migrator settings. Zero fields
// select DefaultMigrationBudgets.
type MigrationBudgets struct {
	StatementTimeout time.Duration
	LockTimeout      time.Duration
}

func DefaultMigrationBudgets() MigrationBudgets {
	return MigrationBudgets{StatementTimeout: 120 * time.Second, LockTimeout: 10 * time.Second}
}

// migrationQueryMargin keeps the client-side tracer deadline just beyond the
// server statement timeout so PostgreSQL reports the timeout first.
const migrationQueryMargin = 5 * time.Second

func (m MigrationBudgets) normalized() (Budgets, time.Duration, error) {
	defaults := DefaultMigrationBudgets()
	if m.StatementTimeout == 0 {
		m.StatementTimeout = defaults.StatementTimeout
	}
	if m.LockTimeout == 0 {
		m.LockTimeout = defaults.LockTimeout
	}
	if m.StatementTimeout < time.Millisecond || m.StatementTimeout > 24*time.Hour-migrationQueryMargin ||
		m.LockTimeout < time.Millisecond || m.LockTimeout >= m.StatementTimeout {
		return Budgets{}, 0, fmt.Errorf("PostgreSQL migration budgets require 1ms <= lock timeout < statement timeout <= 24h-5s")
	}
	b, err := Budgets{5 * time.Second, m.StatementTimeout + migrationQueryMargin, m.StatementTimeout, m.LockTimeout, 60 * time.Second}.normalized()
	if err != nil {
		return Budgets{}, 0, err
	}
	// The whole migrator gets at least 15 minutes and always outlives one
	// statement by a margin that still allows lock waits and ledger writes.
	return b, max(15*time.Minute, b.QueryTimeout+5*time.Minute), nil
}

// Validate checks migrator overrides without opening a database connection.
func (m MigrationBudgets) Validate() error {
	_, _, err := m.normalized()
	return err
}

// WithMigrationBudget bounds the entire migrator (including advisory-lock
// acquisition) to 15 minutes. InTx installs these server settings with SET
// LOCAL, so commit/rollback cannot leak maintenance settings into the pool.
func WithMigrationBudget(ctx context.Context) (context.Context, context.CancelFunc) {
	ctx, cancel, err := WithMigrationBudgets(ctx, MigrationBudgets{})
	if err != nil {
		panic(err) // defaults are statically valid
	}
	return ctx, cancel
}

// WithMigrationBudgets is WithMigrationBudget with operator overrides. The
// tracer query deadline follows the statement timeout plus a small margin.
func WithMigrationBudgets(ctx context.Context, m MigrationBudgets) (context.Context, context.CancelFunc, error) {
	b, total, err := m.normalized()
	if err != nil {
		return nil, nil, err
	}
	ctx, cancel := withMaintenanceBudget(ctx, b, total)
	return ctx, cancel, nil
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
