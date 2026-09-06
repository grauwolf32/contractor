package postgres

import (
	"context"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// pgx explicitly uses the context returned by these hooks for the operation.
// This covers direct pool and transaction repository calls without changing
// the DBTX boundary. QueryEnd runs when Rows is closed, not when Query returns.
// Never inspect/log SQL, arguments, the connection config or raw errors here.
type budgetTracer struct {
	budgets     Budgets
	logger      *slog.Logger
	lastPoolLog atomic.Int64
}
type acquireBudgetKey struct{}
type queryBudgetKey struct{}
type budgetSpan struct {
	cancel  context.CancelFunc
	started time.Time
}

func (t *budgetTracer) policy(ctx context.Context) Budgets {
	if b, ok := ctx.Value(maintenanceBudgetKey{}).(Budgets); ok {
		return b
	}
	return t.budgets
}
func startBudget(ctx context.Context, key any, timeout time.Duration) context.Context {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	return context.WithValue(ctx, key, budgetSpan{cancel: cancel, started: time.Now()})
}
func endBudget(ctx context.Context, key any) budgetSpan {
	span := ctx.Value(key).(budgetSpan)
	span.cancel()
	return span
}
func (t *budgetTracer) TraceAcquireStart(ctx context.Context, _ *pgxpool.Pool, _ pgxpool.TraceAcquireStartData) context.Context {
	return startBudget(ctx, acquireBudgetKey{}, t.policy(ctx).AcquireTimeout)
}
func (t *budgetTracer) TraceAcquireEnd(ctx context.Context, pool *pgxpool.Pool, data pgxpool.TraceAcquireEndData) {
	span := endBudget(ctx, acquireBudgetKey{})
	elapsed := time.Since(span.started)
	if t.logger == nil || (data.Err == nil && elapsed < 100*time.Millisecond) {
		return
	}
	// At most one diagnostic per second per pool, including under saturation.
	now := time.Now().UnixNano()
	previous := t.lastPoolLog.Load()
	if now-previous < int64(time.Second) || !t.lastPoolLog.CompareAndSwap(previous, now) {
		return
	}
	stat := pool.Stat()
	result := "slow"
	if data.Err != nil {
		result = "failed"
	}
	t.logger.WarnContext(ctx, "PostgreSQL pool acquire pressure",
		"outcome", result, "wait_ms", elapsed.Milliseconds(),
		"max_connections", stat.MaxConns(), "total_connections", stat.TotalConns(),
		"acquired_connections", stat.AcquiredConns(), "idle_connections", stat.IdleConns(),
		"empty_acquire_count", stat.EmptyAcquireCount(), "canceled_acquire_count", stat.CanceledAcquireCount(),
		"acquire_duration_ms", stat.AcquireDuration().Milliseconds())
}
func (t *budgetTracer) TraceQueryStart(ctx context.Context, _ *pgx.Conn, _ pgx.TraceQueryStartData) context.Context {
	return startBudget(ctx, queryBudgetKey{}, t.policy(ctx).QueryTimeout)
}
func (*budgetTracer) TraceQueryEnd(ctx context.Context, _ *pgx.Conn, _ pgx.TraceQueryEndData) {
	endBudget(ctx, queryBudgetKey{})
}
