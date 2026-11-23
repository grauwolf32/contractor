package agentprincipal

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// catalogStatementCounts counts statements by the catalog table they name.
type catalogStatementCounts struct {
	bindings, versions, principals int
}

type catalogStatementTrace struct {
	mu     sync.Mutex
	counts catalogStatementCounts
}

func (trace *catalogStatementTrace) TraceQueryStart(
	ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData,
) context.Context {
	trace.mu.Lock()
	defer trace.mu.Unlock()
	if strings.Contains(data.SQL, "runtime_label_bindings") {
		trace.counts.bindings++
	}
	if strings.Contains(data.SQL, "runtime_config_versions") {
		trace.counts.versions++
	}
	if strings.Contains(data.SQL, "runtime_agent_principals") {
		trace.counts.principals++
	}
	return ctx
}

func (*catalogStatementTrace) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

// take returns and resets the counts since the last call.
func (trace *catalogStatementTrace) take() catalogStatementCounts {
	trace.mu.Lock()
	defer trace.mu.Unlock()
	counts := trace.counts
	trace.counts = catalogStatementCounts{}
	return counts
}

// agentLabelFixture is a migrated schema with count Agent labels, each bound
// to its own published RuntimeConfig with a Worker setting, and a counting
// pool on the same schema.
type agentLabelFixture struct {
	pool, counted *pgxpool.Pool
	trace         *catalogStatementTrace
	publisher     *runtimeconfig.Publisher
	labels        []string
	refs          []runtimeconfig.Ref
	now           time.Time
}

func newAgentLabelFixture(t *testing.T, ctx context.Context, count int) agentLabelFixture {
	t.Helper()
	pool := isolatedPrincipalPool(t, ctx)
	now := time.Date(2026, 10, 23, 0, 0, 0, 0, time.UTC)
	publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{
		Pool: pool, RuntimeCredentials: allowRuntimeCredentialCatalog{}, Now: func() time.Time { return now },
		PlannerTelemetryAdapters: runtimeconfigtest.PlannerTelemetryAdapterCatalogFunc(func(ref string) bool { return ref == "otlp-http@1" }),
	})
	if err != nil {
		t.Fatal(err)
	}
	f := agentLabelFixture{pool: pool, publisher: publisher, now: now, trace: &catalogStatementTrace{}}
	repository := runtimeconfig.NewRepository(pool)
	for index := range count {
		ref := f.publish(t, ctx, fmt.Sprintf("agent-config-%02d", index))
		label := fmt.Sprintf("agent-%02d", index)
		if _, err := repository.CreateBinding(ctx, label, ref, "operator", now); err != nil {
			t.Fatal(err)
		}
		f.labels = append(f.labels, label)
		f.refs = append(f.refs, ref)
	}
	configuration := pool.Config().Copy()
	configuration.ConnConfig.Tracer = f.trace
	f.counted, err = pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(f.counted.Close)
	return f
}

func (f agentLabelFixture) publish(t *testing.T, ctx context.Context, name string) runtimeconfig.Ref {
	t.Helper()
	document := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"` + name +
		`","version":"1"},"spec":{"worker":{"telemetry":null}}}`)
	published, err := f.publisher.Publish(ctx, document, "publish-"+name, "operator")
	if err != nil {
		t.Fatal(err)
	}
	return published.Version.Ref
}

func principalID(index int) string { return fmt.Sprintf("%064x", 0xa0+index) }

// Label replacement and the adapter projection of a principal with the full
// 32-label set read the bindings once and the versions once;
// the replacement locks the union of the old and new sets with one statement
// first. The per-label loops took 32 binding and 32 version statements for
// each of these, and 40 binding statements for the replacement's lock.
func TestPostgresAgentLabelValidationReadsEachCatalogOnce(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	f := newAgentLabelFixture(t, ctx, 40)
	service, err := NewService(ServiceOptions{
		Pool: f.counted, DeletionGuard: &recordingDeletionGuard{}, Now: func() time.Time { return f.now },
	})
	if err != nil {
		t.Fatal(err)
	}
	id := principalID(0)
	f.trace.take()
	registered, err := service.Register(ctx, id)
	if err != nil || registered.Labels == nil || len(registered.Labels) != 0 {
		t.Fatalf("register unlabeled principal = (%+v, %v)", registered, err)
	}
	if got := f.trace.take(); got.bindings != 0 || got.versions != 0 {
		t.Fatalf("registration read configuration catalogs: %+v", got)
	}
	registerPrincipalWithLabels(t, ctx, service, id, f.labels[:32])
	f.trace.take()

	replaced, err := service.ReplaceLabelsIdempotent(ctx, id, 2, f.labels[8:], "replace-labels", "operator", f.now)
	if err != nil || replaced.Principal == nil || !slices.Equal(replaced.Principal.Labels, f.labels[8:]) {
		t.Fatalf("replace 32 labels = (%+v, %v)", replaced, err)
	}
	if got := f.trace.take(); got.bindings != 2 || got.versions != 1 {
		t.Fatalf("label replacement = %+v, want one lock and one read of the bindings and 1 version statement", got)
	}

	adapters, err := service.RequiredRuntimeAdapters(ctx, f.labels[8:])
	if err != nil || adapters == nil || len(adapters) != 0 {
		t.Fatalf("adapter projection = (%v, %v)", adapters, err)
	}
	if got := f.trace.take(); got.bindings != 1 || got.versions != 1 || got.principals != 0 {
		t.Fatalf("adapter projection of 32 labels = %+v, want 1 binding and 1 version statement", got)
	}
	// A label that is no longer bound keeps the not-found class of the
	// single-label reads, not the request error of a selected label.
	if _, err := f.pool.Exec(ctx, `DELETE FROM runtime_label_bindings WHERE label = $1`, f.labels[0]); err != nil {
		t.Fatal(err)
	}
	_, err = service.RequiredRuntimeAdapters(ctx, f.labels[:2])
	if !errors.Is(err, runtimeconfig.ErrNotFound) || errors.Is(err, runtimeconfig.ErrUnknownLabel) {
		t.Fatalf("adapter projection of an unbound label = %v, want ErrNotFound", err)
	}
	_, err = service.ReplaceLabelsIdempotent(ctx, id, 3, f.labels[:2], "replace-unbound", "operator", f.now)
	if !errors.Is(err, runtimeconfig.ErrUnknownLabel) {
		t.Fatalf("replacement with an unbound label = %v, want ErrUnknownLabel", err)
	}
}

// A rebind validates every principal holding the label with the same two
// catalog reads whether one or four principals hold it. The per-holder loop
// read 32 bindings and 32 versions for each holder.
func TestPostgresRebindValidatesHoldersWithConstantReads(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	f := newAgentLabelFixture(t, ctx, 40)
	service, err := NewService(ServiceOptions{
		Pool: f.pool, DeletionGuard: &recordingDeletionGuard{}, Now: func() time.Time { return f.now },
	})
	if err != nil {
		t.Fatal(err)
	}
	bindingService, err := runtimeconfig.NewBindingService(allowRuntimeCredentialCatalog{})
	if err != nil {
		t.Fatal(err)
	}
	management, err := runtimeconfig.NewManagementService(f.counted, f.publisher, bindingService, LabelHolders{})
	if err != nil {
		t.Fatal(err)
	}
	// Every principal holds agent-00 and agent-01; the four sets overlap.
	for index := range 4 {
		labels := slices.Concat(f.labels[:2], f.labels[2+index*2:32+index*2])
		registerPrincipalWithLabels(t, ctx, service, principalID(index), labels)
	}
	rebind := func(label string, key string) catalogStatementCounts {
		t.Helper()
		target := f.publish(t, ctx, "target-"+key)
		f.trace.take()
		result, err := management.Rebind(ctx, label, 1, target, key, "operator", f.now.Add(time.Minute))
		if err != nil || result.Binding == nil || result.Binding.Ref != target {
			t.Fatalf("rebind %s = (%+v, %v)", label, result, err)
		}
		return f.trace.take()
	}
	// agent-01 has four holders; agent-37 has only the last principal.
	four, one := rebind(f.labels[1], "rebind-four"), rebind(f.labels[37], "rebind-one")
	if four != one {
		t.Fatalf("rebind statements with four holders %+v, with one holder %+v, want equal", four, one)
	}
	// Bindings: one exclusive lock of every holder label, the update of the
	// target binding and its read-back, and one read of every holder label.
	// Versions: the rebind target, then every holder label's version.
	// Principals: the optimistic holder list, the list again inside the
	// transaction and one lock of the holders.
	if want := (catalogStatementCounts{bindings: 4, versions: 2, principals: 3}); four != want {
		t.Fatalf("rebind statements = %+v, want %+v", four, want)
	}
}

// probePrincipalLock reports whether a transaction can share-lock principal
// id at once.
func probePrincipalLock(t *testing.T, ctx context.Context, pool *pgxpool.Pool, id string) bool {
	t.Helper()
	probe, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer probe.Rollback(context.Background())
	_, err = probe.Exec(ctx, `SELECT 1 FROM runtime_agent_principals WHERE runtime_agent_id = $1 FOR SHARE NOWAIT`, id)
	if persistencepostgres.SQLState(err) == persistencepostgres.SQLStateLockNotAvailable {
		return false
	}
	if err != nil {
		t.Fatal(err)
	}
	return true
}

// A principal lock set takes one statement, locks FOR UPDATE in ID order and
// keeps its validation and missing-row outcomes. While it waits for one
// principal it holds every earlier principal and no later one.
func TestPostgresPrincipalLockSetTakesOneStatementInOrder(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	f := newAgentLabelFixture(t, ctx, 0)
	ids := make([]string, 8)
	for index := range ids {
		ids[index] = principalID(index)
		inserted, err := NewRepository(f.pool).Insert(ctx, Principal{
			RuntimeAgentID: ids[index], Labels: []string{}, LabelRevision: 1,
			CreatedBy: "operator", CreatedAt: f.now, UpdatedBy: "operator", UpdatedAt: f.now,
		})
		if err != nil || !inserted {
			t.Fatalf("insert principal %d = (%v, %v)", index, inserted, err)
		}
	}
	reversedIDs := slices.Clone(ids)
	slices.Reverse(reversedIDs)
	tx, err := f.counted.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(context.Background())
	f.trace.take()
	locked, err := NewRepository(tx).LockMany(ctx, reversedIDs)
	if err != nil {
		t.Fatal(err)
	}
	if got := f.trace.take(); got.principals != 1 {
		t.Fatalf("lock of 8 principals = %d statements, want 1", got.principals)
	}
	for index, principal := range locked {
		if principal.RuntimeAgentID != ids[index] || principal.LabelRevision != 1 {
			t.Fatalf("locked principal %d = %+v, want %s in ID order", index, principal, ids[index])
		}
	}
	repository := NewRepository(tx)
	for name, set := range map[string][]string{
		"duplicate principal": {ids[0], ids[1], ids[0]},
		"invalid principal":   {ids[0], "runtime-agent"},
	} {
		if _, err := repository.LockMany(ctx, set); !errors.Is(err, runtimeconfig.ErrInvalid) {
			t.Fatalf("%s lock = %v, want ErrInvalid", name, err)
		}
	}
	if empty, err := repository.LockMany(ctx, nil); err != nil || len(empty) != 0 {
		t.Fatalf("empty lock set = (%v, %v)", empty, err)
	}
	if got := f.trace.take(); got.principals != 0 {
		t.Fatalf("refused and empty lock sets sent %d statements, want none", got.principals)
	}
	if _, err := repository.LockMany(ctx, []string{ids[2], principalID(99)}); !errors.Is(err, runtimeconfig.ErrNotFound) {
		t.Fatalf("lock set with a missing principal = %v, want ErrNotFound", err)
	}
	if err := tx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}

	holder, err := f.pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer holder.Rollback(context.Background())
	if _, err := NewRepository(holder).Lock(ctx, ids[4]); err != nil {
		t.Fatal(err)
	}
	waiter, err := f.pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer waiter.Rollback(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := NewRepository(waiter).LockMany(ctx, reversedIDs)
		done <- err
	}()
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	for blocked := false; !blocked; {
		if err := f.pool.QueryRow(ctx, `SELECT $2::int = ANY(pg_blocking_pids($1::int))`,
			int(waiter.Conn().PgConn().PID()), int(holder.Conn().PgConn().PID())).Scan(&blocked); err != nil {
			t.Fatal(err)
		}
		select {
		case <-deadline.C:
			t.Fatal("principal lock set never waited for the held principal")
		default:
		}
	}
	for index, id := range ids {
		switch free := probePrincipalLock(t, ctx, f.pool, id); {
		case index < 4 && free:
			t.Fatalf("lock set waiting on principal 4 does not hold principal %d", index)
		case index > 4 && !free:
			t.Fatalf("lock set waiting on principal 4 already holds principal %d", index)
		}
	}
	if err := holder.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatalf("lock set after the holder ended = %v", err)
	}
}
