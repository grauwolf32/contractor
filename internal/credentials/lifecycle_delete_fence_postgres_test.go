package credentials

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/persistence/leasetest"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// These tests reproduce the process-wide deletion fence at the boundary that
// raises it: the transaction that records a delete intent fails at its INSERT
// or its COMMIT. A Gateway failure after a durable intent is a different path
// (TestCredentialFailedDeleteFencesOnlyItsOwnCredential) and never raises it.

func TestCredentialDeleteIntentRollbackReopensGuardsAfterReconciliation(t *testing.T) {
	f, ctx := newDeleteFenceFixture(t)
	deletion := f.deleteRequest(f.deleting, "delete-rolled-back")
	f.failDeleteIntentInsert(t, ctx, deletion)
	if countOperations(t, ctx, f.pool, OperationDelete) != 0 {
		t.Fatal("a rolled-back delete intent is in the operation journal")
	}

	// Reconciliation proves the intent never became durable, so the same
	// Service admits references without a delete retry or a restart.
	f.assertGuardsAdmit(t, ctx, "run-after-rollback")
	if f.service.uncertainDelete.Load() != nil {
		t.Fatal("reconciled fence is still raised")
	}
	if err := f.service.WithRunCreation(ctx, func() error {
		return lookupManagedCredential(ctx, f.pool, f.deleting.CredentialID)
	}); err != nil {
		t.Fatalf("credential whose delete intent rolled back is not referenceable: %v", err)
	}
	if f.manager.deleteCalls() != 0 {
		t.Fatalf("rolled-back delete reached the Gateway %d times", f.manager.deleteCalls())
	}
}

func TestCredentialDeleteIntentStillCommittingKeepsGuardsClosedUntilItAborts(t *testing.T) {
	f, ctx := newDeleteFenceFixture(t)
	deletion := f.deleteRequest(f.deleting, "delete-held")
	f.faults.arm(commitFaultHold)
	if _, err := f.service.Delete(ctx, deletion); err == nil {
		t.Fatal("delete whose COMMIT got no reply succeeded")
	}
	held := f.faults.faultedConn(t)
	// The intent is not durable yet, and another session cannot see it.
	if countOperations(t, ctx, f.pool, OperationDelete) != 0 {
		t.Fatal("uncommitted delete intent is visible")
	}
	// A negative read now would race the open COMMIT, so both guards stay closed.
	f.assertGuardsFenced(t, ctx)

	// Another delete can neither run beside the open intent nor clear its fence.
	beside := f.deleteRequest(f.unrelated, "delete-beside-held")
	if _, err := f.service.Delete(ctx, beside); !errors.Is(err, ErrRecoveryRequired) {
		t.Fatalf("delete beside an unsettled intent error = %v", err)
	}
	if f.manager.deleteCalls() != 0 || countOperations(t, ctx, f.pool, OperationDelete) != 0 {
		t.Fatalf("delete beside an unsettled intent had side effects: calls=%d", f.manager.deleteCalls())
	}
	f.assertGuardsFenced(t, ctx)

	// Same-process recovery reads an empty journal but cannot settle the open
	// transaction, so it becomes ready with the fence still closed.
	if err := f.service.Recover(ctx); err != nil {
		t.Fatalf("recover beside an unsettled intent: %v", err)
	}
	f.assertGuardsFenced(t, ctx)

	if err := held.abortHeld(ctx, f.pool); err != nil {
		t.Fatal(err)
	}
	if countOperations(t, ctx, f.pool, OperationDelete) != 0 {
		t.Fatal("aborted delete intent is in the operation journal")
	}
	f.assertGuardsAdmit(t, ctx, "run-after-abort")
	if err := lookupManagedCredential(ctx, f.pool, f.deleting.CredentialID); err != nil {
		t.Fatalf("credential whose delete intent aborted is not referenceable: %v", err)
	}
}

func TestCredentialDeleteIntentCommittedLateStaysFencedUntilItsDeletionCompletes(t *testing.T) {
	f, ctx := newDeleteFenceFixture(t)
	deletion := f.deleteRequest(f.deleting, "delete-late")
	f.faults.arm(commitFaultHold)
	if _, err := f.service.Delete(ctx, deletion); err == nil {
		t.Fatal("delete whose COMMIT got no reply succeeded")
	}
	held := f.faults.faultedConn(t)
	f.assertGuardsFenced(t, ctx)
	if err := held.commitHeld(); err != nil {
		t.Fatal(err)
	}
	assertCredentialOperationPhase(t, ctx, f.pool, OperationDelete, deletion.IdempotencyKey, OperationPrepared)

	// A durable prepared deletion keeps both guards conservative, and the
	// credential's own lookups refuse it.
	f.assertGuardsFenced(t, ctx)
	if f.service.uncertainDelete.Load() == nil {
		t.Fatal("a durable prepared deletion cleared the conservative fence")
	}
	if err := lookupManagedCredential(ctx, f.pool, f.deleting.CredentialID); !errors.Is(err, ErrRecoveryRequired) {
		t.Fatalf("lookup of a credential with a prepared delete error = %v", err)
	}

	// The same request completes the deletion and reopens both guards.
	if result, err := f.service.Delete(ctx, deletion); err != nil || result.Replayed {
		t.Fatalf("delete completing the late intent = (%+v, %v)", result, err)
	}
	if f.manager.deleteCalls() != 1 || f.manager.remoteCount() != 1 {
		t.Fatalf("manager after completion: calls=%d remote=%d", f.manager.deleteCalls(), f.manager.remoteCount())
	}
	assertCredentialOperationPhase(t, ctx, f.pool, OperationDelete, deletion.IdempotencyKey, OperationCompleted)
	f.assertGuardsAdmit(t, ctx, "run-after-late-commit")
	if err := lookupManagedCredential(ctx, f.pool, f.deleting.CredentialID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("deleted credential lookup error = %v", err)
	}
}

func TestCredentialDeleteIntentWithLostCommitReplyIsCompletedBySameProcessRecovery(t *testing.T) {
	f, ctx := newDeleteFenceFixture(t)
	deletion := f.deleteRequest(f.deleting, "delete-lost-reply")
	f.faults.arm(commitFaultLoseResponse)
	if _, err := f.service.Delete(ctx, deletion); err == nil {
		t.Fatal("delete whose COMMIT reply was lost succeeded")
	}
	f.faults.faultedConn(t)
	assertCredentialOperationPhase(t, ctx, f.pool, OperationDelete, deletion.IdempotencyKey, OperationPrepared)
	if f.manager.deleteCalls() != 0 {
		t.Fatalf("delete with a lost COMMIT reply reached the Gateway %d times", f.manager.deleteCalls())
	}
	f.assertGuardsFenced(t, ctx)

	if err := f.service.Recover(ctx); err != nil {
		t.Fatalf("same-process recovery: %v", err)
	}
	if f.manager.deleteCalls() != 1 {
		t.Fatalf("recovery Gateway deletes = %d, want 1", f.manager.deleteCalls())
	}
	assertCredentialOperationPhase(t, ctx, f.pool, OperationDelete, deletion.IdempotencyKey, OperationCompleted)
	f.assertGuardsAdmit(t, ctx, "run-after-recovery")
	if err := lookupManagedCredential(ctx, f.pool, f.deleting.CredentialID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("recovered deleted credential lookup error = %v", err)
	}
}

func TestCredentialRecoveryCompletesAnIntentThatCommitsAfterItReadTheJournal(t *testing.T) {
	f, ctx := newDeleteFenceFixture(t)
	deletion := f.deleteRequest(f.deleting, "delete-commits-during-recovery")
	f.faults.arm(commitFaultHold)
	if _, err := f.service.Delete(ctx, deletion); err == nil {
		t.Fatal("delete whose COMMIT got no reply succeeded")
	}
	held := f.faults.faultedConn(t)
	// The held COMMIT lands right after recovery's first journal read
	// returned no prepared operation, before recovery settles the intent.
	var released atomic.Bool
	var releaseErr error
	f.faults.afterStatement(func(sql string) {
		if strings.Contains(sql, "WHERE phase = 'prepared'") && released.CompareAndSwap(false, true) {
			releaseErr = held.commitHeld()
		}
	})
	if err := f.service.Recover(ctx); err != nil {
		t.Fatalf("recover around a late COMMIT: %v", err)
	}
	if !released.Load() || releaseErr != nil {
		t.Fatalf("late COMMIT released=%v err=%v", released.Load(), releaseErr)
	}
	if f.manager.deleteCalls() != 1 {
		t.Fatalf("recovery Gateway deletes = %d, want 1", f.manager.deleteCalls())
	}
	assertCredentialOperationPhase(t, ctx, f.pool, OperationDelete, deletion.IdempotencyKey, OperationCompleted)
	f.assertGuardsAdmit(t, ctx, "run-after-late-recovery")
}

func TestCredentialDeleteFenceStaysClosedWhenReconciliationCannotRead(t *testing.T) {
	f, ctx := newDeleteFenceFixture(t)
	f.failDeleteIntentInsert(t, ctx, f.deleteRequest(f.deleting, "delete-before-outage"))

	// A cancelled caller keeps its cancellation, and the fence stays.
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if err := f.service.reconcileForReferences(cancelled); !errors.Is(err, context.Canceled) ||
		errors.Is(err, ErrRecoveryRequired) {
		t.Fatalf("cancelled reconciliation error = %v", err)
	}
	if f.service.uncertainDelete.Load() == nil {
		t.Fatal("cancelled reconciliation cleared the fence")
	}

	// A reconciliation that cannot read keeps the fence, and the public answer
	// is the retryable credential_recovery_required, not a storage failure.
	f.faults.setOutage(true)
	for name, guard := range f.guards() {
		called := false
		err := guard(ctx, func() error { called = true; return nil })
		if called {
			t.Fatalf("%s guard ran its reference during a database outage", name)
		}
		assertRecoveryRequiredAnswer(t, name+" guard during a database outage", err)
	}
	if f.service.uncertainDelete.Load() == nil {
		t.Fatal("failed reconciliation cleared the fence")
	}
	f.faults.setOutage(false)

	// A delete that cannot settle the earlier intent is rejected the same way
	// instead of running beside it. The outage starts after its reference
	// checks, so the reconciliation is the read that fails.
	f.faults.afterStatement(func(sql string) {
		if strings.Contains(sql, "credential_runs") {
			f.faults.setOutage(true)
		}
	})
	_, err := f.service.Delete(ctx, f.deleteRequest(f.unrelated, "delete-during-outage"))
	f.faults.afterStatement(nil)
	if !f.faults.inOutage() {
		t.Fatal("the outage did not start before the delete reconciled")
	}
	f.faults.setOutage(false)
	assertRecoveryRequiredAnswer(t, "delete during a reconciliation outage", err)
	if f.manager.deleteCalls() != 0 || countOperations(t, ctx, f.pool, OperationDelete) != 0 {
		t.Fatalf("delete during a reconciliation outage had side effects: calls=%d", f.manager.deleteCalls())
	}

	f.assertGuardsAdmit(t, ctx, "run-after-outage")
}

func TestCredentialDeleteFenceCleanGuardRunsNoQueryAndReconciliationIsBounded(t *testing.T) {
	f, ctx := newDeleteFenceFixture(t)
	noop := func() error { return nil }
	bothGuards := func() error {
		for _, guard := range f.guards() {
			if err := guard(ctx, noop); err != nil {
				return err
			}
		}
		return nil
	}
	if statements := f.faults.statementsOf(t, bothGuards); statements != 0 {
		t.Fatalf("clean guards ran %d statements", statements)
	}

	// One reconciliation is the same short transaction for a fresh journal
	// and for one that holds thousands of finished operations.
	f.failDeleteIntentInsert(t, ctx, f.deleteRequest(f.deleting, "delete-small-journal"))
	small := f.faults.statementsOf(t, bothGuards)
	f.insertFinishedDeletes(t, ctx, 5000)
	f.failDeleteIntentInsert(t, ctx, f.deleteRequest(f.deleting, "delete-large-journal"))
	large := f.faults.statementsOf(t, bothGuards)
	if small == 0 || large != small {
		t.Fatalf("reconciliation statements: fresh journal %d, large journal %d", small, large)
	}
	if statements := f.faults.statementsOf(t, bothGuards); statements != 0 {
		t.Fatalf("guards after reconciliation ran %d statements", statements)
	}

	// Guards that meet the fence together share one reconciliation.
	f.failDeleteIntentInsert(t, ctx, f.deleteRequest(f.deleting, "delete-concurrent-guards"))
	concurrent := f.faults.statementsOf(t, func() error {
		const callers = 16
		results := make(chan error, callers)
		var start sync.WaitGroup
		start.Add(1)
		for range callers {
			go func() {
				start.Wait()
				results <- f.service.WithRunCreation(ctx, noop)
			}()
		}
		start.Done()
		for range callers {
			if err := <-results; err != nil {
				return err
			}
		}
		return nil
	})
	if concurrent != small {
		t.Fatalf("concurrent guards ran %d statements, want one reconciliation of %d", concurrent, small)
	}

	// An ordinary Gateway failure leaves a durable intent fenced only for its
	// own credential, as before: no conservative fence and no guard query.
	f.manager.deleteErr = errors.New("gateway delete failed")
	if _, err := f.service.Delete(ctx, f.deleteRequest(f.unrelated, "delete-gateway-failure")); !errors.Is(err, ErrGatewayUnavailable) {
		t.Fatalf("Gateway failure error = %v", err)
	}
	if f.service.uncertainDelete.Load() != nil {
		t.Fatal("a Gateway failure raised the conservative fence")
	}
	if statements := f.faults.statementsOf(t, bothGuards); statements != 0 {
		t.Fatalf("guards after a Gateway failure ran %d statements", statements)
	}
	if err := f.service.WithRunCreation(ctx, func() error {
		return lookupManagedCredential(ctx, f.pool, f.deleting.CredentialID)
	}); err != nil {
		t.Fatalf("other credential fenced by a Gateway failure: %v", err)
	}
	if err := f.service.WithAllocationReferences(ctx, func() error {
		return lookupManagedCredential(ctx, f.pool, f.unrelated.CredentialID)
	}); !errors.Is(err, ErrRecoveryRequired) {
		t.Fatalf("credential with a prepared delete error = %v", err)
	}
}

type deleteFenceFixture struct {
	lifecycleFixture
	// pool reaches the same schema outside the injected faults.
	pool *pgxpool.Pool
	// lease is the Service's lease session, outside the injected faults.
	lease     *leasetest.Session
	faults    *connectionFaults
	manager   *fakeGatewayManager
	deleting  CreateRequest
	unrelated CreateRequest
}

func newDeleteFenceFixture(t *testing.T) (deleteFenceFixture, context.Context) {
	t.Helper()
	pool, ctx := lifecycleTestPool(t)
	faults := &connectionFaults{}
	config := pool.Config()
	config.ConnConfig.DialFunc = faults.dial
	config.ConnConfig.Tracer = faults
	config.AfterConnect = faults.afterConnect
	faulty, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		faulty.Close()
		faults.closeAll()
	})
	manager := newFakeGatewayManager()
	lease := leasetest.Hold(t, pool)
	f := deleteFenceFixture{
		lifecycleFixture: newLifecycleFixture(t, faulty, manager, ServiceOptions{Lease: lease.Lease}),
		pool:             pool, lease: lease, faults: faults, manager: manager,
	}
	if err := f.service.Recover(ctx); err != nil {
		t.Fatal(err)
	}
	f.deleting = f.createRequest("managed-deleting", "create-deleting")
	f.unrelated = f.createRequest("managed-unrelated", "create-unrelated")
	for _, request := range []CreateRequest{f.deleting, f.unrelated} {
		if _, err := f.service.Create(ctx, request); err != nil {
			t.Fatal(err)
		}
	}
	return f, ctx
}

func (f deleteFenceFixture) deleteRequest(credential CreateRequest, key string) DeleteRequest {
	return DeleteRequest{CredentialID: credential.CredentialID, IdempotencyKey: key, ActorID: "user-1"}
}

func (f deleteFenceFixture) guards() map[string]func(context.Context, func() error) error {
	return map[string]func(context.Context, func() error) error{
		"Run creation": f.service.WithRunCreation,
		"allocation":   f.service.WithAllocationReferences,
	}
}

// failDeleteIntentInsert makes PostgreSQL reject the delete intent's INSERT
// with an unclassified error, a definite rollback that still leaves the
// Service unable to tell which statement failed.
func (f deleteFenceFixture) failDeleteIntentInsert(t *testing.T, ctx context.Context, deletion DeleteRequest) {
	t.Helper()
	if _, err := f.pool.Exec(ctx, `
CREATE FUNCTION reject_delete_intent() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'injected delete intent failure'; END $$;
CREATE TRIGGER reject_delete_intent BEFORE INSERT ON credential_operations FOR EACH ROW
WHEN (NEW.operation_kind = 'delete') EXECUTE FUNCTION reject_delete_intent()`); err != nil {
		t.Fatal(err)
	}
	_, deleteErr := f.service.Delete(ctx, deletion)
	if _, err := f.pool.Exec(ctx, `
DROP TRIGGER reject_delete_intent ON credential_operations;
DROP FUNCTION reject_delete_intent()`); err != nil {
		t.Fatal(err)
	}
	if deleteErr == nil || errors.Is(deleteErr, ErrConflict) || errors.Is(deleteErr, ErrInvalid) {
		t.Fatalf("delete with a rejected intent = %v, want an unclassified store failure", deleteErr)
	}
	if f.service.uncertainDelete.Load() == nil {
		t.Fatal("a delete intent with an unknown outcome did not raise the conservative fence")
	}
}

// insertFinishedDeletes adds count completed delete operations of other
// credentials, the tombstones a long-lived journal accumulates.
func (f deleteFenceFixture) insertFinishedDeletes(t *testing.T, ctx context.Context, count int) {
	t.Helper()
	if _, err := f.pool.Exec(ctx, `
INSERT INTO credential_operations (operation_id, idempotency_key, request_hash, credential_id,
    operation_kind, phase, request_schema_version, request, created_at, updated_at)
SELECT 'credop_history_' || n, 'history-' || n, 'sha256:' || repeat('a', 64), 'history-' || n,
       'delete', 'completed', $1, '{}'::jsonb, now(), now()
FROM generate_series(1, $2::integer) AS n`, CredentialSchemaVersion, count); err != nil {
		t.Fatal(err)
	}
}

// assertGuardsFenced checks that both guards refuse a reference to any
// credential without running it.
func (f deleteFenceFixture) assertGuardsFenced(t *testing.T, ctx context.Context) {
	t.Helper()
	for name, guard := range f.guards() {
		called := false
		err := guard(ctx, func() error {
			called = true
			return lookupManagedCredential(ctx, f.pool, f.unrelated.CredentialID)
		})
		if !errors.Is(err, ErrRecoveryRequired) || called {
			t.Fatalf("%s guard = (called=%v, err=%v), want the conservative fence", name, called, err)
		}
	}
}

// assertGuardsAdmit checks that both guards admit references to the
// unrelated credential: a Run pinning it commits and an allocation resolves it.
func (f deleteFenceFixture) assertGuardsAdmit(t *testing.T, ctx context.Context, runID string) {
	t.Helper()
	runs := runstore.NewPostgresStore(f.pool, artifactpgtest.NewStorage())
	if err := f.service.WithRunCreation(ctx, func() error {
		if err := lookupManagedCredential(ctx, f.pool, f.unrelated.CredentialID); err != nil {
			return err
		}
		_, err := runstoretest.CreateRun(ctx, runs, pinnedRunParams(runID, f.unrelated.CredentialID))
		return err
	}); err != nil {
		t.Fatalf("Run pinning an unrelated credential: %v", err)
	}
	if err := f.service.WithAllocationReferences(ctx, func() error {
		return lookupManagedCredential(ctx, f.pool, f.unrelated.CredentialID)
	}); err != nil {
		t.Fatalf("allocation referencing an unrelated credential: %v", err)
	}
}

// assertRecoveryRequiredAnswer checks the public answer to err: the public API
// answers the first classified error of its chain, which must be the
// retryable 503 credential_recovery_required.
func assertRecoveryRequiredAnswer(t *testing.T, name string, err error) {
	t.Helper()
	classified, ok := apierror.Of(err)
	if !ok || classified.ErrorKind() != apierror.CredentialRecoveryRequired ||
		classified.ErrorKind().Status() != http.StatusServiceUnavailable || !classified.ErrorKind().Retryable() {
		t.Fatalf("%s answer = %v, want the retryable 503 credential_recovery_required", name, err)
	}
}

// commitFault selects how the next COMMIT on the fault pool fails.
type commitFault int

const (
	commitFaultNone commitFault = iota
	// commitFaultLoseResponse lets the server finish the COMMIT and loses its
	// reply: the intent is durable and the client cannot know it.
	commitFaultLoseResponse
	// commitFaultHold keeps the COMMIT from the server and fails the client,
	// while the server transaction stays open with its locks until the test
	// commits or aborts it.
	commitFaultHold
)

// simpleQueryCommit is the protocol message pgx sends for Tx.Commit.
var simpleQueryCommit = func() []byte {
	body := []byte("commit\x00")
	message := []byte{'Q', 0, 0, 0, 0}
	binary.BigEndian.PutUint32(message[1:], uint32(4+len(body)))
	return append(message, body...)
}()

var errInjectedConnectionLoss = fmt.Errorf("injected connection loss: %w", io.ErrUnexpectedEOF)

type statementSQLKey struct{}

// connectionFaults sits between pgx and PostgreSQL on the Service's pool. It
// fails one armed COMMIT, simulates an outage, and counts the statements the
// pool runs.
type connectionFaults struct {
	mu         sync.Mutex
	armed      commitFault
	faulted    []*faultConn
	conns      []*faultConn
	outage     bool
	after      func(sql string)
	statements atomic.Int64
}

func (f *connectionFaults) dial(ctx context.Context, network, address string) (net.Conn, error) {
	var dialer net.Dialer
	conn, err := dialer.DialContext(ctx, network, address)
	if err != nil {
		return nil, err
	}
	wrapped := &faultConn{Conn: conn, faults: f}
	f.mu.Lock()
	f.conns = append(f.conns, wrapped)
	f.mu.Unlock()
	return wrapped, nil
}

func (f *connectionFaults) afterConnect(_ context.Context, conn *pgx.Conn) error {
	if wrapped, ok := conn.PgConn().Conn().(*faultConn); ok {
		wrapped.pid.Store(conn.PgConn().PID())
	}
	return nil
}

func (f *connectionFaults) TraceQueryStart(
	ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData,
) context.Context {
	f.statements.Add(1)
	return context.WithValue(ctx, statementSQLKey{}, data.SQL)
}

func (f *connectionFaults) TraceQueryEnd(ctx context.Context, _ *pgx.Conn, _ pgx.TraceQueryEndData) {
	f.mu.Lock()
	after := f.after
	f.mu.Unlock()
	if sql, ok := ctx.Value(statementSQLKey{}).(string); ok && after != nil {
		after(sql)
	}
}

func (f *connectionFaults) arm(fault commitFault) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.armed = fault
}

func (f *connectionFaults) setOutage(outage bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.outage = outage
}

func (f *connectionFaults) inOutage() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.outage
}

func (f *connectionFaults) afterStatement(hook func(sql string)) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.after = hook
}

// claim fires the armed fault when p carries a COMMIT.
func (f *connectionFaults) claim(conn *faultConn, p []byte) commitFault {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.armed == commitFaultNone || !bytes.Contains(p, simpleQueryCommit) {
		return commitFaultNone
	}
	fault := f.armed
	f.armed = commitFaultNone
	f.faulted = append(f.faulted, conn)
	return fault
}

// faultedConn returns the connection whose COMMIT the armed fault took.
func (f *connectionFaults) faultedConn(t *testing.T) *faultConn {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.armed != commitFaultNone || len(f.faulted) == 0 {
		t.Fatal("the armed COMMIT fault did not fire")
	}
	return f.faulted[len(f.faulted)-1]
}

func (f *connectionFaults) statementsOf(t *testing.T, run func() error) int64 {
	t.Helper()
	before := f.statements.Load()
	if err := run(); err != nil {
		t.Fatal(err)
	}
	return f.statements.Load() - before
}

func (f *connectionFaults) closeAll() {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, conn := range f.conns {
		_ = conn.Conn.Close()
	}
}

type faultConn struct {
	net.Conn
	faults *connectionFaults
	pid    atomic.Uint32

	mu    sync.Mutex
	fault commitFault
	held  []byte
	lose  sync.Once
}

func (c *faultConn) state() commitFault {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.fault
}

func (c *faultConn) Write(p []byte) (int, error) {
	if c.state() != commitFaultNone {
		// The client side is gone: nothing more, not even Terminate, reaches
		// the server.
		return len(p), nil
	}
	if c.faults.inOutage() || isCancelRequest(p) {
		// pgx cancels the backend of a connection it abandons. Such a cancel
		// could land on a held COMMIT the test releases later, so none is sent.
		return 0, errInjectedConnectionLoss
	}
	switch fault := c.faults.claim(c, p); fault {
	case commitFaultLoseResponse:
		if _, err := c.Conn.Write(p); err != nil {
			return 0, err
		}
		c.mu.Lock()
		c.fault = fault
		c.mu.Unlock()
		return len(p), nil
	case commitFaultHold:
		c.mu.Lock()
		c.fault, c.held = fault, append([]byte(nil), p...)
		c.mu.Unlock()
		return len(p), nil
	}
	return c.Conn.Write(p)
}

func (c *faultConn) Read(p []byte) (int, error) {
	switch c.state() {
	case commitFaultNone:
		if c.faults.inOutage() {
			return 0, errInjectedConnectionLoss
		}
		return c.Conn.Read(p)
	case commitFaultLoseResponse:
		// The server finishes the COMMIT; its reply never reaches pgx.
		c.lose.Do(func() {
			_ = readUntilReady(c.Conn)
			_ = c.Conn.Close()
		})
	}
	return 0, errInjectedConnectionLoss
}

func (c *faultConn) Close() error {
	if c.state() == commitFaultHold {
		return nil // the test commits or aborts the held server transaction
	}
	return c.Conn.Close()
}

func (c *faultConn) SetDeadline(deadline time.Time) error {
	if c.state() != commitFaultNone {
		return nil
	}
	return c.Conn.SetDeadline(deadline)
}

func (c *faultConn) SetReadDeadline(deadline time.Time) error {
	if c.state() != commitFaultNone {
		return nil
	}
	return c.Conn.SetReadDeadline(deadline)
}

func (c *faultConn) SetWriteDeadline(deadline time.Time) error {
	if c.state() != commitFaultNone {
		return nil
	}
	return c.Conn.SetWriteDeadline(deadline)
}

// commitHeld lets the held COMMIT reach the server and waits for its reply.
func (c *faultConn) commitHeld() error {
	c.mu.Lock()
	held := c.held
	c.mu.Unlock()
	if held == nil {
		return errors.New("no COMMIT is held")
	}
	if _, err := c.Conn.Write(held); err != nil {
		return err
	}
	if err := readUntilReady(c.Conn); err != nil {
		return err
	}
	return c.Conn.Close()
}

// abortHeld ends the held server transaction without its COMMIT, as the
// server does when it loses the client, and waits until the backend is gone.
func (c *faultConn) abortHeld(ctx context.Context, admin *pgxpool.Pool) error {
	var terminated bool
	if err := admin.QueryRow(ctx, `SELECT pg_terminate_backend($1, 30000)`, int32(c.pid.Load())).
		Scan(&terminated); err != nil {
		return err
	}
	if !terminated {
		return errors.New("held backend did not terminate")
	}
	return c.Conn.Close()
}

// isCancelRequest reports whether p is a protocol CancelRequest packet.
func isCancelRequest(p []byte) bool {
	const cancelRequestCode = 80877102
	return len(p) >= 12 && int(binary.BigEndian.Uint32(p[0:4])) == len(p) &&
		binary.BigEndian.Uint32(p[4:8]) == cancelRequestCode
}

// readUntilReady reads server messages up to ReadyForQuery and fails when one
// of them is an ErrorResponse.
func readUntilReady(conn net.Conn) error {
	header := make([]byte, 5)
	failed := false
	for {
		if _, err := io.ReadFull(conn, header); err != nil {
			return err
		}
		length := binary.BigEndian.Uint32(header[1:])
		if length < 4 {
			return errors.New("invalid protocol message length")
		}
		if _, err := io.CopyN(io.Discard, conn, int64(length-4)); err != nil {
			return err
		}
		switch header[0] {
		case 'E':
			failed = true
		case 'Z':
			if failed {
				return errors.New("the server rejected the statement")
			}
			return nil
		}
	}
}
