package performance

import (
	"context"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const DiagnosticApplicationName = "contractor-performance"

func diagnosticBudgets(maintenance bool) postgres.Budgets {
	b := postgres.Budgets{AcquireTimeout: 250 * time.Millisecond, QueryTimeout: time.Second,
		StatementTimeout: 750 * time.Millisecond, LockTimeout: 100 * time.Millisecond,
		IdleTransactionTimeout: 2 * time.Second}
	if maintenance {
		b.QueryTimeout, b.StatementTimeout = 2*time.Second, 1500*time.Millisecond
	}
	return b
}

// NewDiagnosticPool never pings and has no warm connections. Network/auth
// failures become unavailable observations, never application startup failures.
func NewDiagnosticPool(ctx context.Context, databaseURL string) (*pgxpool.Pool, error) {
	config, err := postgres.PoolConfig(databaseURL, postgres.PoolOptions{
		MaxConnections: 1, ConnectTimeout: 250 * time.Millisecond, Budgets: diagnosticBudgets(false), DisableWarmConnections: true,
	})
	if err != nil {
		return nil, err
	}
	config.MaxConns, config.MinConns, config.MinIdleConns = 1, 0, 0
	config.ConnConfig.RuntimeParams["application_name"] = DiagnosticApplicationName
	config.ConnConfig.Tracer = &diagnosticTracer{QueryTracer: config.ConnConfig.Tracer, AcquireTracer: config.ConnConfig.Tracer.(pgxpool.AcquireTracer)}
	return pgxpool.NewWithConfig(ctx, config)
}

// pgx's ConnectTimeout is per host and begins after DNS resolution; pooled
// constructors may outlive a cancelled acquire. Bound the entire connection
// attempt as well, including DNS and every configured fallback, without changing
// the working pool's tracer or its ordinary operation policies.
type diagnosticTracer struct {
	pgx.QueryTracer
	pgxpool.AcquireTracer
}
type diagnosticConnectKey struct{}

func (*diagnosticTracer) TraceConnectStart(ctx context.Context, _ pgx.TraceConnectStartData) context.Context {
	ctx, cancel := context.WithTimeout(ctx, 250*time.Millisecond)
	return context.WithValue(ctx, diagnosticConnectKey{}, cancel)
}
func (*diagnosticTracer) TraceConnectEnd(ctx context.Context, _ pgx.TraceConnectEndData) {
	ctx.Value(diagnosticConnectKey{}).(context.CancelFunc)()
}

type DatabaseStore struct{ pool *pgxpool.Pool }

func NewDatabaseStore(pool *pgxpool.Pool) *DatabaseStore { return &DatabaseStore{pool: pool} }
func (s *DatabaseStore) Close()                          { s.pool.Close() }

// Statistics snapshots and settings are transaction-local. Rollback uses the
// same deadline; pgx destroys the connection when rollback cannot be confirmed.
// No independent cleanup timeout may extend the five-second diagnostic cycle.
func (s *DatabaseStore) transaction(ctx context.Context, maintenance, readOnly bool, fn func(context.Context, pgx.Tx) error) error {
	b := diagnosticBudgets(maintenance)
	ctx, cancel := context.WithTimeout(ctx, b.QueryTimeout)
	defer cancel()
	ctx, err := postgres.WithOperationBudgets(ctx, b)
	if err != nil {
		return err
	}
	options := pgx.TxOptions{}
	if readOnly {
		options.AccessMode = pgx.ReadOnly
	}
	tx, err := s.pool.BeginTx(ctx, options)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	if err = postgres.ApplyTransactionBudget(ctx, tx); err != nil {
		return err
	}
	if err = fn(ctx, tx); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

// Statistics views only, scoped to current_database. No SQL text or per-table,
// user or application identifiers leave PostgreSQL. Hidden states are counted
// separately, not fabricated as idle/active. No exact bloat verdict is inferred.
const databaseStatisticsSQL = `
WITH activity AS (
 SELECT count(*) FILTER (WHERE backend_type = 'client backend') AS clients,
 count(*) FILTER (WHERE backend_type = 'client backend' AND state = 'active') AS active,
 count(*) FILTER (WHERE backend_type = 'client backend' AND state = 'idle') AS idle,
 count(*) FILTER (WHERE backend_type = 'client backend' AND state IN ('idle in transaction','idle in transaction (aborted)')) AS idle_tx,
 count(*) FILTER (WHERE backend_type = 'client backend' AND wait_event_type = 'Lock') AS locks,
 count(*) FILTER (WHERE backend_type IS NULL OR (backend_type = 'client backend' AND state IS NULL)) AS hidden,
 coalesce(bool_or(backend_type = 'client backend' AND state = 'disabled'),false) AS activity_disabled,
 count(*) FILTER (WHERE backend_type = 'autovacuum worker') AS autovacuum_workers,
 max(greatest(0, extract(epoch FROM clock_timestamp()-xact_start))) FILTER (WHERE backend_type = 'client backend' AND xact_start IS NOT NULL)::double precision AS longest_tx,
 max(greatest(0, extract(epoch FROM clock_timestamp()-state_change))) FILTER (WHERE backend_type = 'client backend' AND state IN ('idle in transaction','idle in transaction (aborted)'))::double precision AS longest_idle_tx
 FROM pg_stat_activity WHERE datid = (SELECT oid FROM pg_database WHERE datname=current_database()) AND pid <> pg_backend_pid()
), tables AS (
 SELECT coalesce(sum(n_live_tup),0)::bigint AS live, coalesce(sum(n_dead_tup),0)::bigint AS dead,
 coalesce(sum(vacuum_count),0)::bigint AS vacuums, coalesce(sum(autovacuum_count),0)::bigint AS autovacuums,
 max(last_vacuum) AS last_vacuum, max(last_autovacuum) AS last_autovacuum FROM pg_stat_user_tables
)
SELECT current_setting('track_counts')::boolean, d.stats_reset,
 d.xact_commit,d.xact_rollback,d.deadlocks,d.temp_files,d.temp_bytes,d.blks_read,d.blks_hit,
 a.clients,a.active,a.idle,a.idle_tx,a.locks,a.hidden,a.activity_disabled,a.autovacuum_workers,a.longest_tx,a.longest_idle_tx,
 t.live,t.dead,t.vacuums,t.autovacuums,t.last_vacuum,t.last_autovacuum
FROM pg_stat_database d CROSS JOIN activity a CROSS JOIN tables t WHERE d.datname=current_database()`

func (s *DatabaseStore) ReadDatabase(ctx context.Context) (Database, Reason) {
	var d Database
	var counts, activityDisabled bool
	err := s.transaction(ctx, false, true, func(ctx context.Context, tx pgx.Tx) error {
		return tx.QueryRow(ctx, databaseStatisticsSQL).Scan(&counts, &d.StatsReset,
			&d.Commits, &d.Rollbacks, &d.Deadlocks, &d.TempFiles, &d.TempBytes, &d.BlocksRead, &d.BlocksHit,
			&d.ClientConnections, &d.ActiveConnections, &d.IdleConnections, &d.IdleInTransactionConnections,
			&d.LockWaitingConnections, &d.HiddenConnections, &activityDisabled, &d.AutovacuumWorkers,
			&d.LongestTransactionSeconds, &d.LongestIdleTransactionSeconds,
			&d.EstimatedLiveTuples, &d.EstimatedDeadTuples, &d.VacuumCount, &d.AutovacuumCount,
			&d.LastVacuumAt, &d.LastAutovacuumAt)
	})
	if err != nil {
		return Database{}, databaseReason(err)
	}
	if !counts {
		d.StatsReset, d.Commits, d.Rollbacks, d.Deadlocks, d.TempFiles, d.TempBytes, d.BlocksRead, d.BlocksHit = nil, nil, nil, nil, nil, nil, nil, nil
		d.EstimatedLiveTuples, d.EstimatedDeadTuples, d.VacuumCount, d.AutovacuumCount = nil, nil, nil, nil
		d.LastVacuumAt, d.LastAutovacuumAt = nil, nil
		return d, StatisticsDisabled
	}
	if d.HiddenConnections != nil && *d.HiddenConnections > 0 {
		return d, PermissionDenied
	}
	if activityDisabled {
		return d, StatisticsDisabled
	}
	return d, ""
}

func (s *DatabaseStore) ReadSize(ctx context.Context) (DatabaseSize, Reason) {
	var size uint64
	err := s.transaction(ctx, true, true, func(ctx context.Context, tx pgx.Tx) error {
		return tx.QueryRow(ctx, `SELECT pg_database_size(current_database())`).Scan(&size)
	})
	if err != nil {
		return DatabaseSize{}, databaseReason(err)
	}
	return DatabaseSize{SizeBytes: &size}, ""
}

func databaseReason(err error) Reason {
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
		return BudgetExceeded
	}
	switch postgres.SQLState(err) {
	case "57014", "55P03":
		return BudgetExceeded
	case "42501":
		return PermissionDenied
	default:
		return DatabaseUnavailable
	}
}
