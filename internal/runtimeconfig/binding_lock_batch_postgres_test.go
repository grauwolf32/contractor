package runtimeconfig

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// bindingStatementTrace counts the statements that read the binding and the
// version tables.
type bindingStatementTrace struct {
	mu                 sync.Mutex
	bindings, versions int
}

func (trace *bindingStatementTrace) TraceQueryStart(
	ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData,
) context.Context {
	trace.mu.Lock()
	defer trace.mu.Unlock()
	if strings.Contains(data.SQL, "runtime_label_bindings") {
		trace.bindings++
	}
	if strings.Contains(data.SQL, "runtime_config_versions") {
		trace.versions++
	}
	return ctx
}

func (*bindingStatementTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

// take returns and resets the counts since the last call.
func (trace *bindingStatementTrace) take() (int, int) {
	trace.mu.Lock()
	defer trace.mu.Unlock()
	bindings, versions := trace.bindings, trace.versions
	trace.bindings, trace.versions = 0, 0
	return bindings, versions
}

// bindingLockFixture is a migrated schema with count labels, each bound to
// its own published RuntimeConfig. The names interleave '-', digits, '_' and
// letters, whose bytewise order a linguistic collation does not keep.
type bindingLockFixture struct {
	pool    *pgxpool.Pool
	counted *pgxpool.Pool
	trace   *bindingStatementTrace
	labels  []string
	refs    map[string]Ref
}

func newBindingLockFixture(t *testing.T, ctx context.Context, count int) bindingLockFixture {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	pool := isolatedRuntimeConfigPool(t, ctx, databaseURL)
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 10, 23, 0, 0, 0, 0, time.UTC)
	publisher, err := NewPublisher(PublisherOptions{
		Pool: pool, RuntimeCredentials: allowRuntimeCredentialCatalog{}, Now: func() time.Time { return now },
		PlannerTelemetryAdapters: plannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
	})
	if err != nil {
		t.Fatal(err)
	}
	separators := []string{"-", "_", "", "x"}
	fixture := bindingLockFixture{pool: pool, refs: make(map[string]Ref, count)}
	repository := NewRepository(pool)
	for index := range count {
		label := fmt.Sprintf("a%s%02d", separators[index%len(separators)], index)
		document := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"config-` +
			fmt.Sprint(index) + `","version":"1"},"spec":{"worker":{"telemetry":null}}}`)
		published, publishErr := publisher.Publish(ctx, document, fmt.Sprintf("publish-%d", index), "operator")
		if publishErr != nil {
			t.Fatal(publishErr)
		}
		if _, err := repository.CreateBinding(ctx, label, published.Version.Ref, "operator", now); err != nil {
			t.Fatal(err)
		}
		fixture.labels = append(fixture.labels, label)
		fixture.refs[label] = published.Version.Ref
	}
	fixture.trace = &bindingStatementTrace{}
	configuration := pool.Config().Copy()
	configuration.ConnConfig.Tracer = fixture.trace
	fixture.counted, err = pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(fixture.counted.Close)
	return fixture
}

// reversed returns labels in reverse order, so no test passes labels in the
// order the lock statement needs.
func reversed(labels []string) []string {
	result := slices.Clone(labels)
	slices.Reverse(result)
	return result
}

// One Run pin of 16 explicit labels takes one binding lock statement and one
// version read, and an exclusive lock set takes one statement as well; each
// returns its bindings in bytewise label order. Invalid and duplicate sets are
// refused before any SQL, and a missing label keeps its classes. The
// per-label loops took 17 binding and 17 version statements for this pin.
func TestPostgresBindingLockSetsTakeOneStatement(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	f := newBindingLockFixture(t, ctx, 16)
	tx, err := f.counted.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(context.Background())
	f.trace.take()
	pin, err := PinRunSnapshot(ctx, tx, reversed(f.labels), allowRuntimeCredentialCatalog{}, TransactionLLMCredentialLookup{})
	if err != nil {
		t.Fatal(err)
	}
	if bindings, versions := f.trace.take(); bindings != 1 || versions != 1 {
		t.Fatalf("16-label pin = %d binding and %d version statements, want 1 and 1", bindings, versions)
	}
	sorted := slices.Sorted(slices.Values(f.labels))
	pinned := make([]string, len(pin.Labels))
	for index, label := range pin.Labels {
		pinned[index] = label.Label
		if label.Config != f.refs[label.Label] {
			t.Fatalf("pin of %s = %+v, want %+v", label.Label, label.Config, f.refs[label.Label])
		}
	}
	if !slices.Equal(pinned, sorted) {
		t.Fatalf("pinned labels = %v, want bytewise order %v", pinned, sorted)
	}

	all := append([]string{DefaultLabel}, f.labels...)
	locked, err := NewRepository(tx).LockBindingsForUpdate(ctx, reversed(all))
	if err != nil {
		t.Fatal(err)
	}
	if bindings, versions := f.trace.take(); bindings != 1 || versions != 0 {
		t.Fatalf("exclusive 17-label lock = %d binding and %d version statements, want 1 and 0", bindings, versions)
	}
	lockedLabels := make([]string, len(locked))
	for index, binding := range locked {
		lockedLabels[index] = binding.Label
	}
	if want := slices.Sorted(slices.Values(all)); !slices.Equal(lockedLabels, want) {
		t.Fatalf("locked labels = %v, want bytewise order %v", lockedLabels, want)
	}

	repository := NewRepository(tx)
	for name, labels := range map[string][]string{
		"duplicate label": {f.labels[0], f.labels[1], f.labels[0]},
		"invalid label":   {f.labels[0], "Invalid"},
	} {
		if _, err := repository.LockBindings(ctx, labels); !errors.Is(err, ErrInvalid) {
			t.Fatalf("%s lock = %v, want ErrInvalid", name, err)
		}
		if bindings, _ := f.trace.take(); bindings != 0 {
			t.Fatalf("%s lock sent %d statements, want none", name, bindings)
		}
	}
	if empty, err := repository.LockBindings(ctx, []string{}); err != nil || len(empty) != 0 {
		t.Fatalf("empty lock set = (%v, %v)", empty, err)
	}
	if bindings, _ := f.trace.take(); bindings != 0 {
		t.Fatalf("empty lock set sent %d statements, want none", bindings)
	}
	if _, err := repository.LockBindings(ctx, []string{f.labels[3], "missing", f.labels[5]}); !errors.Is(err, ErrNotFound) ||
		errors.Is(err, ErrUnknownLabel) {
		t.Fatalf("lock set with a missing label = %v, want ErrNotFound", err)
	}
}

// waitBlockedBy waits until the backend pid waits for a lock that blocker
// holds. The deadline only bounds a failure.
func waitBlockedBy(t *testing.T, ctx context.Context, pool *pgxpool.Pool, pid, blocker uint32) {
	t.Helper()
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(5 * time.Millisecond)
	defer tick.Stop()
	for {
		var blocked bool
		if err := pool.QueryRow(ctx, `SELECT $2::int = ANY(pg_blocking_pids($1::int))`, int(pid), int(blocker)).Scan(&blocked); err != nil {
			t.Fatal(err)
		}
		if blocked {
			return
		}
		select {
		case <-deadline.C:
			t.Fatalf("backend %d never waited for a lock of backend %d", pid, blocker)
		case <-tick.C:
		}
	}
}

// probeRowLock reports whether a transaction can take mode on the binding of
// label at once.
func probeRowLock(t *testing.T, ctx context.Context, pool *pgxpool.Pool, label, mode string) bool {
	t.Helper()
	probe, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer probe.Rollback(context.Background())
	_, err = probe.Exec(ctx, `SELECT 1 FROM runtime_label_bindings WHERE label = $1 `+mode+` NOWAIT`, label)
	if persistencepostgres.SQLState(err) == persistencepostgres.SQLStateLockNotAvailable {
		return false
	}
	if err != nil {
		t.Fatal(err)
	}
	return true
}

// Overlapping 16-label pins share their binding locks and both exclude a
// writer until each settles. The batched statement keeps the lock order: a
// pin or an exclusive set that waits on one label already holds every label
// before it in bytewise order and none after it.
func TestPostgresBatchedBindingLocksKeepModeAndOrder(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	f := newBindingLockFixture(t, ctx, 20)
	first, err := f.pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Rollback(context.Background())
	second, err := f.pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Rollback(context.Background())
	if _, err := PinRunSnapshot(ctx, first, f.labels[:16], allowRuntimeCredentialCatalog{}, TransactionLLMCredentialLookup{}); err != nil {
		t.Fatal(err)
	}
	// The deadline bounds a failure; the first pin stays open meanwhile.
	pinCtx, cancelPin := context.WithTimeout(ctx, 5*time.Second)
	_, err = PinRunSnapshot(pinCtx, second, reversed(f.labels[4:]), allowRuntimeCredentialCatalog{}, TransactionLLMCredentialLookup{})
	cancelPin()
	if err != nil {
		t.Fatalf("overlapping pin waited for or failed on a shared lock: %v", err)
	}
	writerExcluded := func(label string) bool {
		t.Helper()
		writer, err := f.pool.Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer writer.Rollback(context.Background())
		if _, err := writer.Exec(ctx, `SET LOCAL lock_timeout = '100ms'`); err != nil {
			t.Fatal(err)
		}
		_, err = NewRepository(writer).LockBindingsForUpdate(ctx, []string{label})
		if persistencepostgres.SQLState(err) == persistencepostgres.SQLStateLockNotAvailable {
			return true
		}
		if err != nil {
			t.Fatal(err)
		}
		return false
	}
	for _, label := range []string{f.labels[0], f.labels[8], f.labels[19], DefaultLabel} {
		if !writerExcluded(label) {
			t.Fatalf("writer of %s passed the shared pins", label)
		}
	}
	if err := first.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if writerExcluded(f.labels[0]) || !writerExcluded(f.labels[8]) {
		t.Fatal("the remaining pin does not decide writer exclusion alone")
	}
	if err := second.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	if writerExcluded(f.labels[8]) {
		t.Fatal("writer stayed excluded after both pins committed")
	}

	sorted := slices.Sorted(slices.Values(append([]string{DefaultLabel}, f.labels[:16]...)))
	middle := sorted[len(sorted)/2]
	for _, test := range []struct {
		name string
		lock func(context.Context, pgx.Tx) error
		// probe is the lock mode the waiting set's held rows refuse.
		probe string
	}{
		{"shared pin", func(ctx context.Context, tx pgx.Tx) error {
			_, err := PinRunSnapshot(ctx, tx, reversed(f.labels[:16]), allowRuntimeCredentialCatalog{}, TransactionLLMCredentialLookup{})
			return err
		}, "FOR UPDATE"},
		{"exclusive set", func(ctx context.Context, tx pgx.Tx) error {
			_, err := NewRepository(tx).LockBindingsForUpdate(ctx, reversed(sorted))
			return err
		}, "FOR SHARE"},
	} {
		t.Run(test.name, func(t *testing.T) {
			holder, err := f.pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer holder.Rollback(context.Background())
			if _, err := NewRepository(holder).LockBindingsForUpdate(ctx, []string{middle}); err != nil {
				t.Fatal(err)
			}
			waiter, err := f.pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer waiter.Rollback(context.Background())
			done := make(chan error, 1)
			go func() { done <- test.lock(ctx, waiter) }()
			waitBlockedBy(t, ctx, f.pool, waiter.Conn().PgConn().PID(), holder.Conn().PgConn().PID())
			// The labels sort differently under the database's linguistic
			// collation, so a collation-ordered lock fails one of these.
			for index, label := range sorted {
				switch free := probeRowLock(t, ctx, f.pool, label, test.probe); {
				case index < len(sorted)/2 && free:
					t.Fatalf("%s waiting on %s does not hold the earlier label %s", test.name, middle, label)
				case index > len(sorted)/2 && !free:
					t.Fatalf("%s waiting on %s already holds the later label %s", test.name, middle, label)
				}
			}
			if err := holder.Rollback(ctx); err != nil {
				t.Fatal(err)
			}
			if err := <-done; err != nil {
				t.Fatalf("%s after the holder ended = %v", test.name, err)
			}
		})
	}
}
