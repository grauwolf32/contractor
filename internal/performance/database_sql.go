package performance

// SQL statements for database.go.

// databaseStatisticsSQL reads one row of database-wide health statistics.
// Statistics views only, scoped to current_database. No SQL text or per-table,
// user or application identifiers leave PostgreSQL. Hidden states are counted
// separately, not fabricated as idle/active. No exact bloat verdict is inferred.
// Used by DatabaseStore.ReadDatabase.
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
