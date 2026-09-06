package performance

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/migrations"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func testDSNParameter(dsn, key, value string) string {
	if strings.HasPrefix(dsn, "postgres://") || strings.HasPrefix(dsn, "postgresql://") {
		u, _ := url.Parse(dsn)
		q := u.Query()
		q.Set(key, value)
		u.RawQuery = q.Encode()
		return u.String()
	}
	return dsn + " " + key + "='" + strings.ReplaceAll(strings.ReplaceAll(value, `\`, `\\`), `'`, `\'`) + "'"
}

func databaseFixture(t *testing.T) (*pgxpool.Pool, *DatabaseStore, string) {
	t.Helper()
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is required for real PostgreSQL tests")
	}
	ctx := context.Background()
	admin, err := postgres.OpenPool(ctx, dsn, postgres.PoolOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var random [8]byte
	if _, err = rand.Read(random[:]); err != nil {
		t.Fatal(err)
	}
	schema := "performance_test_" + hex.EncodeToString(random[:])
	if _, err = admin.Exec(ctx, `CREATE SCHEMA `+pgx.Identifier{schema}.Sanitize()); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(ctx, `DROP SCHEMA `+pgx.Identifier{schema}.Sanitize()+` CASCADE`); err != nil {
			t.Error(err)
		}
		admin.Close()
	})
	scoped := testDSNParameter(dsn, "search_path", schema)
	working, err := postgres.OpenPool(ctx, scoped, postgres.PoolOptions{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(working.Close)
	sql, err := migrations.Files.ReadFile("000049_performance_minutes.sql")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = working.Exec(ctx, string(sql)); err != nil {
		t.Fatal(err)
	}
	pool, err := NewDiagnosticPool(ctx, scoped)
	if err != nil {
		t.Fatal(err)
	}
	store := NewDatabaseStore(pool)
	t.Cleanup(store.Close)
	return working, store, scoped
}

func assertDiagnosticReusable(t *testing.T, s *DatabaseStore) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	c, err := s.pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Release()
	if c.Conn().PgConn().TxStatus() != 'I' {
		t.Fatal("open transaction after diagnostic work")
	}
	var statement, lock, app string
	if err = c.QueryRow(ctx, `SELECT current_setting('statement_timeout'),current_setting('lock_timeout'),current_setting('application_name')`).Scan(&statement, &lock, &app); err != nil {
		t.Fatal(err)
	}
	if statement != "750ms" || lock != "100ms" || app != DiagnosticApplicationName {
		t.Fatalf("settings leaked: %s %s %s", statement, lock, app)
	}
}

func TestPostgresDiagnosticsIsolationVisibilityAndOptionalPrivileges(t *testing.T) {
	working, store, dsn := databaseFixture(t)
	ctx := context.Background()
	before := working.Stat().AcquireCount()
	d, reason := store.ReadDatabase(ctx)
	if reason != "" || d.Commits == nil || d.EstimatedDeadTuples == nil || d.VacuumCount == nil {
		t.Fatalf("stats: %+v %s", d, reason)
	}
	size, reason := store.ReadSize(ctx)
	if reason != "" || size.SizeBytes == nil || *size.SizeBytes == 0 || working.Stat().AcquireCount() != before {
		t.Fatal("size or diagnostic used working pool")
	}
	assertDiagnosticReusable(t, store)
	var super bool
	if err := working.QueryRow(ctx, `SELECT rolsuper FROM pg_roles WHERE rolname=current_user`).Scan(&super); err != nil {
		t.Fatal(err)
	}
	if !super {
		t.Skip("visibility fixture requires a test superuser to create isolated roles")
	}
	var schema string
	if err := working.QueryRow(ctx, `SELECT current_schema()`).Scan(&schema); err != nil {
		t.Fatal(err)
	}
	role := schema + "_reader"
	quoted := pgx.Identifier{role}.Sanitize()
	if _, err := working.Exec(ctx, `CREATE ROLE `+quoted+` LOGIN PASSWORD 'performance-fixture-only'`); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if _, err := working.Exec(context.Background(), `DROP ROLE `+quoted); err != nil {
			t.Error(err)
		}
	})
	roleDSN := testDSNParameter(testDSNParameter(dsn, "user", role), "password", "performance-fixture-only")
	normal, err := NewDiagnosticPool(ctx, roleDSN)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(normal.Close)
	normalStore := NewDatabaseStore(normal)
	d, reason = normalStore.ReadDatabase(ctx)
	if reason != PermissionDenied || d.HiddenConnections == nil || *d.HiddenConnections == 0 || d.Commits == nil {
		t.Fatalf("hidden session misreported: %+v %s", d, reason)
	}
	raw, _ := json.Marshal(d)
	if strings.Contains(string(raw), "query") || strings.Contains(string(raw), schema) || strings.Contains(string(raw), role) {
		t.Fatal("sensitive dimensions leaked")
	}
	var granted bool
	if err = working.QueryRow(ctx, `SELECT pg_has_role($1,'pg_read_all_stats','member')`, role).Scan(&granted); err != nil {
		t.Fatal(err)
	}
	if granted {
		t.Fatal("collector granted privileges")
	}
	// Only this test fixture opts in; production code never performs a GRANT.
	if _, err = working.Exec(ctx, `GRANT pg_read_all_stats TO `+quoted); err != nil {
		t.Fatal(err)
	}
	d, reason = normalStore.ReadDatabase(ctx)
	if reason != "" || *d.HiddenConnections != 0 || *d.ClientConnections == 0 {
		t.Fatalf("all-stats view: %+v %s", d, reason)
	}
	// Idle transaction is visible with all-stats; the observer excludes itself.
	tx, err := working.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	d, reason = normalStore.ReadDatabase(ctx)
	if err = tx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	if reason != "" || *d.IdleInTransactionConnections < 1 || d.LongestTransactionSeconds == nil || d.LongestIdleTransactionSeconds == nil {
		t.Fatal("idle transaction missing")
	}
	var actual uint64
	if err = working.QueryRow(ctx, `SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND backend_type='client backend'`).Scan(&actual); err != nil {
		t.Fatal(err)
	}
	if *d.ClientConnections != actual-1 {
		t.Fatalf("own backend counted: got=%d all=%d", *d.ClientConnections, actual)
	}
}

func TestPostgresDiagnosticBudgetsAndRecovery(t *testing.T) {
	working, store, _ := databaseFixture(t)
	ctx := context.Background()
	for _, maintenance := range []bool{false, true} {
		t.Run(fmt.Sprintf("statement-maintenance-%t", maintenance), func(t *testing.T) {
			start := time.Now()
			err := store.transaction(ctx, maintenance, true, func(ctx context.Context, tx pgx.Tx) error { _, err := tx.Exec(ctx, `SELECT pg_sleep(10)`); return err })
			want := 750 * time.Millisecond
			if maintenance {
				want = 1500 * time.Millisecond
			}
			if postgres.SQLState(err) != "57014" || time.Since(start) < want-100*time.Millisecond || time.Since(start) > want+500*time.Millisecond {
				t.Fatalf("statement budget: %v %s", err, time.Since(start))
			}
			assertDiagnosticReusable(t, store)
		})
	}
	conn, err := store.pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	_, reason := store.ReadDatabase(ctx)
	conn.Release()
	if reason != BudgetExceeded || time.Since(start) > 500*time.Millisecond {
		t.Fatal("acquire budget not enforced")
	}
	assertDiagnosticReusable(t, store)
	tx, err := working.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = tx.Exec(ctx, `LOCK TABLE performance_minutes IN ACCESS EXCLUSIVE MODE`); err != nil {
		t.Fatal(err)
	}
	start = time.Now()
	err = store.Flush(ctx, nil, time.Now())
	if rollbackErr := tx.Rollback(ctx); rollbackErr != nil {
		t.Fatal(rollbackErr)
	}
	if postgres.SQLState(err) != "55P03" || time.Since(start) > 400*time.Millisecond {
		t.Fatalf("lock budget: %v %s", err, time.Since(start))
	}
	assertDiagnosticReusable(t, store)
	short, cancel := context.WithTimeout(ctx, 20*time.Millisecond)
	start = time.Now()
	err = store.transaction(short, true, true, func(ctx context.Context, tx pgx.Tx) error { _, err := tx.Exec(ctx, `SELECT pg_sleep(10)`); return err })
	cancel()
	if err == nil || time.Since(start) > 250*time.Millisecond {
		t.Fatal("earlier deadline extended")
	}
	assertDiagnosticReusable(t, store)
	// Kill only this fixture's one diagnostic connection, then observe recovery.
	conn, err = store.pool.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	pid := conn.Conn().PgConn().PID()
	conn.Release()
	if _, err = working.Exec(ctx, `SELECT pg_terminate_backend($1)`, pid); err != nil {
		t.Fatal(err)
	}
	_, _ = store.ReadDatabase(ctx) // failed read is allowed, no immediate internal retry
	if _, reason = store.ReadDatabase(ctx); reason != "" {
		t.Fatalf("did not reconnect: %s", reason)
	}
	assertDiagnosticReusable(t, store)
}

func TestPostgresDisabledStatisticsAndIndependentSizeFailure(t *testing.T) {
	working, store, dsn := databaseFixture(t)
	ctx := context.Background()
	var super bool
	if err := working.QueryRow(ctx, `SELECT rolsuper FROM pg_roles WHERE rolname=current_user`).Scan(&super); err != nil {
		t.Fatal(err)
	}
	if !super {
		t.Skip("statistics-disable fixture requires test superuser")
	}
	disabledPool, err := NewDiagnosticPool(ctx, testDSNParameter(dsn, "track_counts", "off"))
	if err != nil {
		t.Fatal(err)
	}
	defer disabledPool.Close()
	d, reason := NewDatabaseStore(disabledPool).ReadDatabase(ctx)
	if reason != StatisticsDisabled || d.Commits != nil || d.EstimatedLiveTuples != nil || d.ClientConnections == nil {
		t.Fatalf("disabled stats: %+v %s", d, reason)
	}
	// Shadow only the size function inside this isolated schema/connection.
	// No pg_catalog function or cluster-wide setting is changed.
	if _, err = working.Exec(ctx, `CREATE FUNCTION pg_database_size(name) RETURNS bigint LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'fixture size denied' USING ERRCODE='42501'; END $$`); err != nil {
		t.Fatal(err)
	}
	var schema string
	if err = working.QueryRow(ctx, `SELECT current_schema()`).Scan(&schema); err != nil {
		t.Fatal(err)
	}
	sizePool, err := NewDiagnosticPool(ctx, testDSNParameter(dsn, "search_path", schema+",pg_catalog"))
	if err != nil {
		t.Fatal(err)
	}
	defer sizePool.Close()
	sizeStore := NewDatabaseStore(sizePool)
	if _, reason = sizeStore.ReadSize(ctx); reason != PermissionDenied {
		t.Fatalf("size error hidden: %s", reason)
	}
	if d, reason = sizeStore.ReadDatabase(ctx); reason != "" || d.Commits == nil {
		t.Fatal("size failure poisoned statistics")
	}
	assertDiagnosticReusable(t, sizeStore)
	if _, err = working.Exec(ctx, `DROP FUNCTION `+pgx.Identifier{schema, "pg_database_size"}.Sanitize()+`(name)`); err != nil {
		t.Fatal(err)
	}
	if _, reason = sizeStore.ReadSize(ctx); reason != "" {
		t.Fatalf("size did not recover: %s", reason)
	}
	assertDiagnosticReusable(t, store)
}

func TestPostgresHistoryIdempotencyTTLBoundsAndPlans(t *testing.T) {
	working, store, _ := databaseFixture(t)
	ctx := context.Background()
	now := time.Now().UTC().Truncate(time.Minute)
	minute := Minute{Version: 1, Generation: "a", MinuteStart: now.Add(-time.Minute), Status: Partial, CoverageSeconds: 45}
	for i := 0; i < 2; i++ {
		if err := store.Flush(ctx, []Minute{minute}, now); err != nil {
			t.Fatal(err)
		}
	}
	var count int
	if err := working.QueryRow(ctx, `SELECT count(*) FROM performance_minutes`).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatal("duplicate minute not idempotent")
	}
	other := minute
	other.Generation = "b"
	if err := store.Flush(ctx, []Minute{other}, now); err != nil {
		t.Fatal(err)
	}
	repo := NewHistoryRepository(working, func() time.Time { return now })
	points, err := repo.Read(ctx, now.Add(-time.Minute), now, time.Minute)
	if err != nil || len(points) != 2 || points[0].Generation == points[1].Generation {
		t.Fatalf("generations: %+v %v", points, err)
	}
	if _, err = repo.Read(ctx, now.Add(-7*24*time.Hour), now, time.Minute); err != ErrHistoryRange {
		t.Fatal("unbounded output range")
	}
	if err = store.Flush(ctx, make([]Minute, 11), now); err == nil {
		t.Fatal("unbounded write batch")
	}
	// Payload/TTL checks are enforced independently of the Go writer.
	for _, query := range []string{
		`INSERT INTO performance_minutes VALUES ('bad',$1::timestamptz,1,0,decode(repeat('00',32769),'hex'),$1::timestamptz+interval '168 hours')`,
		`INSERT INTO performance_minutes VALUES ('bad',$1::timestamptz,1,61,'x',$1::timestamptz+interval '168 hours')`,
		`INSERT INTO performance_minutes VALUES ('bad',$1::timestamptz,1,0,'x',$1::timestamptz+interval '169 hours')`,
	} {
		if _, err := working.Exec(ctx, query, now); postgres.SQLState(err) != "23514" {
			t.Fatalf("missing DB bound: %v", err)
		}
	}
	// Representative seven-day, three-generation history, plus 1501 expired
	// rows. SQL builds valid bounded records; no large payload list in Go.
	_, err = working.Exec(ctx, `INSERT INTO performance_minutes
(server_generation,minute_start,schema_version,coverage_seconds,payload,expires_at)
SELECT generation,started,1,30,convert_to(json_build_object('version',1,'generation',generation,'minuteStart',started,'status','partial','coverageSeconds',30,'process',json_build_object(),'pool',json_build_object())::text,'UTF8'),started+interval '168 hours'
FROM (SELECT 'fixture-'||g AS generation, $1::timestamptz - n * interval '1 minute' AS started FROM generate_series(1,10080) n CROSS JOIN generate_series(1,3) g) x`, now)
	if err != nil {
		t.Fatal(err)
	}
	_, err = working.Exec(ctx, `INSERT INTO performance_minutes SELECT 'expired',started,1,0,'x',started+interval '168 hours' FROM (SELECT $1::timestamptz - interval '168 hours' - n*interval '1 minute' started FROM generate_series(1,1501)n)x`, now)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = working.Exec(ctx, `ANALYZE performance_minutes`); err != nil {
		t.Fatal(err)
	}
	for _, plan := range []struct {
		query, index string
		args         []any
	}{
		{"EXPLAIN (FORMAT JSON) " + historyReadSQL, "performance_minutes_range_idx", []any{now.Add(-5 * time.Minute), now, now, 1001}},
		{"EXPLAIN (FORMAT JSON) " + cleanupPerformanceSQL, "performance_minutes_expiry_idx", []any{now}},
	} {
		var raw []byte
		if err = working.QueryRow(ctx, plan.query, plan.args...).Scan(&raw); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(raw), plan.index) {
			t.Fatalf("missing intended index %s: %s", plan.index, raw)
		}
	}
	points, err = repo.Read(ctx, now.Add(-7*24*time.Hour), now, time.Hour)
	if err != nil || len(points) == 0 || len(points) > 1000 {
		t.Fatalf("seven-day read: points=%d err=%v", len(points), err)
	}
	for _, p := range points {
		if p.Generation == "expired" || p.CoverageSeconds > float64(p.StepSeconds) {
			t.Fatal("expired or invalid aggregate")
		}
	}
	// A disabled server only constructs the working-pool repository. Expired
	// rows remain physically present but are never exposed.
	if err = working.QueryRow(ctx, `SELECT count(*) FROM performance_minutes WHERE expires_at<=$1`, now).Scan(&count); err != nil {
		t.Fatal(err)
	}
	before := count
	if before != 1504 {
		t.Fatalf("fixture expiry: %d", before)
	}
	if err = store.Flush(ctx, nil, now); err != nil {
		t.Fatal(err)
	}
	if err = working.QueryRow(ctx, `SELECT count(*) FROM performance_minutes WHERE expires_at<=$1`, now).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != before-1000 {
		t.Fatalf("cleanup unbounded: %d -> %d", before, count)
	}
	assertDiagnosticReusable(t, store)
	late := NewHistoryRepository(working, func() time.Time { return now.Add(8 * 24 * time.Hour) })
	points, err = late.Read(ctx, now.Add(-time.Hour), now, time.Minute)
	if err != nil || len(points) != 0 {
		t.Fatal("expired data visible while metrics off")
	}
}
