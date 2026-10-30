package credentials

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"slices"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/persistence/leasetest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/jackc/pgx/v5/pgxpool"
)

// These tests characterize the availability of the current managed-credential
// lifecycle (RCF-R52, RCF-R53, RCF-R72) for proposal 0044. They record what
// the contract does today, not what a redesign would choose. A change that
// narrows a fence updates the availability assertions it changes and keeps
// every safety assertion: a credential whose deletion may have started is
// never referenced again.

const besideRuntimeCredential = "otel-beside-delete"

var besideRuntimeKinds = []string{string(RuntimeCredentialOTLPHeaders)}

// availabilityFixture is one Server's credential services: managed LLM
// credentials and Runtime credentials share one lifecycle barrier, as
// configureCredentials composes them.
type availabilityFixture struct {
	lifecycleFixture
	pool               *pgxpool.Pool
	manager            *fakeGatewayManager
	barrier            *LifecycleBarrier
	runtimeCredentials *RuntimeCredentialService
	deleting           CreateRequest
	unrelated          CreateRequest
}

func newAvailabilityFixture(t *testing.T) (availabilityFixture, context.Context) {
	t.Helper()
	pool, ctx := lifecycleTestPool(t)
	manager := newFakeGatewayManager()
	barrier := NewLifecycleBarrier()
	lease := leasetest.Hold(t, pool).Lease
	f := availabilityFixture{
		lifecycleFixture: newLifecycleFixture(t, pool, manager, ServiceOptions{Barrier: barrier, Lease: lease}),
		pool:             pool, manager: manager, barrier: barrier,
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
	f.runtimeCredentials = newRuntimeServiceBeside(t, ctx, pool, lease, f.cipher, barrier)
	return f, ctx
}

func (f availabilityFixture) deleteRequest(credential CreateRequest, key string) DeleteRequest {
	return DeleteRequest{CredentialID: credential.CredentialID, IdempotencyKey: key, ActorID: "user-1"}
}

// newRuntimeServiceBeside composes a Runtime credential service on barrier
// and stores one OTLP credential in it.
func newRuntimeServiceBeside(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, lease persistencepostgres.ServerLease,
	cipher *TokenCipher, barrier *LifecycleBarrier,
) *RuntimeCredentialService {
	t.Helper()
	service, err := NewRuntimeCredentialService(RuntimeCredentialServiceOptions{
		Pool: pool, Lease: lease, Cipher: cipher, Usage: NewRuntimeCredentialRepository(pool),
		Projects: projectstore.NewPostgresStore(pool), Audits: staticAuditDispatchHolds{}, Barrier: barrier,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.Create(ctx, runtimeOTLPRequest(t, besideRuntimeCredential, "runtime-create-beside")); err != nil {
		t.Fatal(err)
	}
	return service
}

func runtimeOTLPRequest(t *testing.T, credentialID, key string) RuntimeCredentialCreateRequest {
	t.Helper()
	material, err := NewOTLPHeadersCredential(map[string]string{"Authorization": "Bearer otlp-" + credentialID})
	if err != nil {
		t.Fatal(err)
	}
	return RuntimeCredentialCreateRequest{
		CredentialID: credentialID, Material: material, IdempotencyKey: key, ActorID: "operator",
	}
}

// eventLog records the order in which concurrent operations passed their
// critical points.
type eventLog struct {
	mu     sync.Mutex
	events []string
}

func (l *eventLog) add(event string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, event)
}

func (l *eventLog) snapshot() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.events...)
}

type operationResult struct {
	name string
	err  error
}

// start runs every operation in its own goroutine.
func start(operations map[string]func() error) <-chan operationResult {
	results := make(chan operationResult, len(operations))
	for name, operation := range operations {
		go func() { results <- operationResult{name: name, err: operation()} }()
	}
	return results
}

// assertBarrierHeldForWrite checks, without waiting, that neither a credential
// reference nor another credential mutation can enter the barrier.
func assertBarrierHeldForWrite(t *testing.T, barrier *LifecycleBarrier) {
	t.Helper()
	if barrier.mu.TryRLock() {
		barrier.mu.RUnlock()
		t.Fatal("a credential reference can enter the barrier")
	}
	if barrier.mu.TryLock() {
		barrier.mu.Unlock()
		t.Fatal("another credential mutation can enter the barrier")
	}
}

// awaitPendingWriter returns once a writer has asked for the barrier, which
// is when sync.RWMutex starts refusing new readers.
func awaitPendingWriter(t *testing.T, ctx context.Context, barrier *LifecycleBarrier) {
	t.Helper()
	for barrier.mu.TryRLock() {
		barrier.mu.RUnlock()
		if err := ctx.Err(); err != nil {
			t.Fatalf("the delete never asked for the barrier: %v", err)
		}
		runtime.Gosched()
	}
}

func TestCredentialDeleteBlocksReferencesAndMutationsOfOtherCredentialsAcrossItsRemoteCall(t *testing.T) {
	f, ctx := newAvailabilityFixture(t)
	inFlight := make(chan struct{})
	release := make(chan struct{})
	var signal sync.Once
	f.manager.deleteGate = func() {
		signal.Do(func() { close(inFlight) })
		<-release
	}
	deleted := make(chan error, 1)
	go func() {
		_, err := f.service.Delete(ctx, f.deleteRequest(f.deleting, "delete-in-flight"))
		deleted <- err
	}()
	<-inFlight

	// RCF-R52: while the Gateway request is in flight the delete holds the
	// write side of the barrier shared by every credential reference, and the
	// mutex of every managed-credential mutation.
	assertBarrierHeldForWrite(t, f.barrier)
	if f.service.lifecycleMu.TryLock() {
		f.service.lifecycleMu.Unlock()
		t.Fatal("a managed-credential mutation can start beside a remote delete")
	}

	var log eventLog
	const released = "remote delete released"
	runs := runstore.NewPostgresStore(f.pool)
	runtimeCreate := runtimeOTLPRequest(t, "otel-created-beside", "runtime-create-during-delete")
	operations := map[string]func() error{
		"Run creation pinning another LLM credential": func() error {
			return f.service.WithRunCreation(ctx, func() error {
				log.add("Run creation pinning another LLM credential")
				if err := lookupManagedCredential(ctx, f.pool, f.unrelated.CredentialID); err != nil {
					return err
				}
				_, err := runstoretest.CreateRun(ctx, runs, pinnedRunParams("run-beside-delete", f.unrelated.CredentialID))
				return err
			})
		},
		"allocation referencing another LLM credential": func() error {
			return f.service.WithAllocationReferences(ctx, func() error {
				log.add("allocation referencing another LLM credential")
				return lookupManagedCredential(ctx, f.pool, f.unrelated.CredentialID)
			})
		},
		"RuntimeConfig or Project reference to a Runtime credential": func() error {
			return f.runtimeCredentials.WithCredentialReferences(ctx, func() error {
				log.add("RuntimeConfig or Project reference to a Runtime credential")
				return f.runtimeCredentials.ValidateRuntimeCredential(ctx, besideRuntimeCredential, besideRuntimeKinds...)
			})
		},
		"Runtime credential use": func() error {
			return f.runtimeCredentials.Use(ctx, besideRuntimeCredential, besideRuntimeKinds, func(*RuntimeCredentialMaterial) error {
				log.add("Runtime credential use")
				return nil
			})
		},
		"Runtime credential create": func() error {
			_, err := f.runtimeCredentials.Create(ctx, runtimeCreate)
			log.add("Runtime credential create")
			return err
		},
		"managed credential create": func() error {
			_, err := f.service.Create(ctx, f.createRequest("managed-created-beside", "create-during-delete"))
			log.add("managed credential create")
			return err
		},
	}
	results := start(operations)
	log.add(released)
	close(release)

	if err := <-deleted; err != nil {
		t.Fatalf("delete held in flight: %v", err)
	}
	for range operations {
		if result := <-results; result.err != nil {
			t.Fatalf("%s beside the delete: %v", result.name, result.err)
		}
	}
	// Nothing passed the barrier or the mutation mutex before the Gateway
	// request returned, although none of it touched the deleted credential.
	events := log.snapshot()
	if len(events) != len(operations)+1 || events[0] != released {
		t.Fatalf("operations beside a remote delete ran in order %v, want every one after %q", events, released)
	}
	for name := range operations {
		if !slices.Contains(events, name) {
			t.Fatalf("operation %q did not run: %v", name, events)
		}
	}
	if f.manager.deleteCalls() != 1 {
		t.Fatalf("Gateway deletes = %d, want 1", f.manager.deleteCalls())
	}
	if err := lookupManagedCredential(ctx, f.pool, f.deleting.CredentialID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("deleted credential lookup error = %v", err)
	}
}

func TestCredentialDeleteWaitingForOneReferenceStopsEveryLaterReference(t *testing.T) {
	f, ctx := newAvailabilityFixture(t)
	var log eventLog
	f.manager.deleteGate = func() { log.add("remote delete") }
	runs := runstore.NewPostgresStore(f.pool)
	entered := make(chan struct{})
	release := make(chan struct{})
	first := make(chan error, 1)
	go func() {
		first <- f.service.WithRunCreation(ctx, func() error {
			close(entered)
			<-release
			_, err := runstoretest.CreateRun(ctx, runs, pinnedRunParams("run-slow-reference", f.unrelated.CredentialID))
			return err
		})
	}()
	<-entered
	deleted := make(chan error, 1)
	go func() {
		_, err := f.service.Delete(ctx, f.deleteRequest(f.deleting, "delete-behind-reference"))
		deleted <- err
	}()

	// sync.RWMutex queues new readers behind a waiting writer: once the delete
	// asks for the write side, a reference to any credential waits for the
	// slow reference and then for the whole delete, Gateway call included.
	awaitPendingWriter(t, ctx, f.barrier)
	if events := log.snapshot(); len(events) != 0 {
		t.Fatalf("delete passed a live reference: %v", events)
	}
	later := make(chan error, 1)
	go func() {
		later <- f.service.WithAllocationReferences(ctx, func() error {
			log.add("later reference")
			return lookupManagedCredential(ctx, f.pool, f.unrelated.CredentialID)
		})
	}()
	log.add("slow reference released")
	close(release)

	if err := <-first; err != nil {
		t.Fatalf("slow reference: %v", err)
	}
	if err := <-deleted; err != nil {
		t.Fatalf("delete behind the reference: %v", err)
	}
	if err := <-later; err != nil {
		t.Fatalf("later reference: %v", err)
	}
	want := []string{"slow reference released", "remote delete", "later reference"}
	if events := log.snapshot(); !slices.Equal(events, want) {
		t.Fatalf("order = %v, want %v", events, want)
	}
}

func TestCredentialPreparedDeleteFencesOwnReferencesAndEveryManagedMutation(t *testing.T) {
	f, ctx := newAvailabilityFixture(t)
	other := f.createRequest("managed-other", "create-other")
	if _, err := f.service.Create(ctx, other); err != nil {
		t.Fatal(err)
	}
	f.manager.deleteErr = errors.New("gateway delete failed")
	deletion := f.deleteRequest(f.deleting, "delete-gateway-failure")
	if _, err := f.service.Delete(ctx, deletion); !errors.Is(err, ErrGatewayUnavailable) {
		t.Fatalf("delete during a Gateway failure error = %v", err)
	}
	f.manager.deleteErr = nil
	assertCredentialOperationPhase(t, ctx, f.pool, OperationDelete, deletion.IdempotencyKey, OperationPrepared)
	if f.service.uncertainDelete.Load() != nil {
		t.Fatal("a Gateway failure raised the process-wide reference fence")
	}

	// References: only the credential with the durable intent is refused.
	if err := lookupManagedCredential(ctx, f.pool, f.deleting.CredentialID); !errors.Is(err, ErrRecoveryRequired) {
		t.Fatalf("lookup of the credential with a prepared delete error = %v", err)
	}
	runs := runstore.NewPostgresStore(f.pool)
	if err := f.service.WithRunCreation(ctx, func() error {
		if err := lookupManagedCredential(ctx, f.pool, f.unrelated.CredentialID); err != nil {
			return err
		}
		_, err := runstoretest.CreateRun(ctx, runs, pinnedRunParams("run-beside-prepared", f.unrelated.CredentialID))
		return err
	}); err != nil {
		t.Fatalf("Run pinning another credential beside a prepared delete: %v", err)
	}
	if err := f.runtimeCredentials.Use(ctx, besideRuntimeCredential, besideRuntimeKinds, func(*RuntimeCredentialMaterial) error {
		return nil
	}); err != nil {
		t.Fatalf("Runtime credential use beside a prepared delete: %v", err)
	}

	// Mutations: the journal's one prepared operation refuses every other
	// managed create and delete, a delete of the same credential under another
	// key included, before any Gateway call or journal write.
	validations, gatewayDeletes := f.manager.validationCalls(), f.manager.deleteCalls()
	createBeside := f.createRequest("managed-new", "create-beside-prepared")
	for name, mutation := range map[string]func() error{
		"delete of another credential": func() error {
			_, err := f.service.Delete(ctx, f.deleteRequest(other, "delete-beside-prepared"))
			return err
		},
		"delete of the same credential under another key": func() error {
			_, err := f.service.Delete(ctx, f.deleteRequest(f.deleting, "delete-prepared-new-key"))
			return err
		},
		"create of a new credential": func() error {
			_, err := f.service.Create(ctx, createBeside)
			return err
		},
	} {
		assertRecoveryRequiredAnswer(t, name, mutation())
	}
	if f.manager.validationCalls() != validations || f.manager.deleteCalls() != gatewayDeletes ||
		countOperations(t, ctx, f.pool, OperationDelete) != 1 || countOperations(t, ctx, f.pool, OperationCreate) != 3 {
		t.Fatalf("refused mutations had side effects: validations=%d deletes=%d",
			f.manager.validationCalls(), f.manager.deleteCalls())
	}

	// Runtime credentials keep their own local lifecycle.
	if _, err := f.runtimeCredentials.Create(ctx, runtimeOTLPRequest(t, "otel-beside-prepared", "runtime-create-beside-prepared")); err != nil {
		t.Fatalf("Runtime credential create beside a prepared delete: %v", err)
	}
	if _, err := f.runtimeCredentials.Delete(ctx, "otel-beside-prepared", "operator"); err != nil {
		t.Fatalf("Runtime credential delete beside a prepared delete: %v", err)
	}

	// Only the original key or startup recovery completes the deletion. Once
	// a restarted Server's recovery has completed it, the refused mutations
	// succeed with their original keys.
	restarted := newLifecycleFixtureWithCipher(t, f.pool, f.manager, f.cipher, ServiceOptions{})
	if err := restarted.service.Recover(ctx); err != nil {
		t.Fatalf("recovery of the prepared delete: %v", err)
	}
	assertCredentialOperationPhase(t, ctx, f.pool, OperationDelete, deletion.IdempotencyKey, OperationCompleted)
	if _, err := f.service.Delete(ctx, f.deleteRequest(other, "delete-beside-prepared")); err != nil {
		t.Fatalf("delete refused beside the prepared delete, after recovery: %v", err)
	}
	if _, err := f.service.Create(ctx, createBeside); err != nil {
		t.Fatalf("create refused beside the prepared delete, after recovery: %v", err)
	}
}

func TestCredentialDurableIntentFoundByReconciliationKeepsTheProcessWideFenceUntilRestart(t *testing.T) {
	f, ctx := newDeleteFenceFixture(t)
	runtimeCredentials := newRuntimeServiceBeside(t, ctx, f.pool, f.lease.Lease, f.cipher, f.service.barrier)
	deletion := f.deleteRequest(f.deleting, "delete-lost-reply")
	f.faults.arm(commitFaultLoseResponse)
	if _, err := f.service.Delete(ctx, deletion); err == nil {
		t.Fatal("delete whose COMMIT reply was lost succeeded")
	}
	f.faults.faultedConn(t)
	assertCredentialOperationPhase(t, ctx, f.pool, OperationDelete, deletion.IdempotencyKey, OperationPrepared)

	// Every reconciliation proves the intent durable and leaves both guards
	// closed for every credential. Nothing in this process completes the
	// deletion unless its client repeats the request with the same key.
	for range 3 {
		f.assertGuardsFenced(t, ctx)
		if f.service.uncertainDelete.Load() == nil {
			t.Fatal("a durable prepared delete cleared the process-wide fence")
		}
	}
	// The per-credential fence a Gateway failure relies on already holds.
	if err := lookupManagedCredential(ctx, f.pool, f.deleting.CredentialID); !errors.Is(err, ErrRecoveryRequired) {
		t.Fatalf("lookup of the credential with a prepared delete error = %v", err)
	}
	if err := lookupManagedCredential(ctx, f.pool, f.unrelated.CredentialID); err != nil {
		t.Fatalf("lookup of another credential: %v", err)
	}

	// References that take the shared side directly, not through the managed
	// guards, are not fenced: RuntimeConfig publication and binding, Project
	// targets and Runtime credential use.
	called := false
	if err := runtimeCredentials.WithCredentialReferences(ctx, func() error {
		called = true
		return runtimeCredentials.ValidateRuntimeCredential(ctx, besideRuntimeCredential, besideRuntimeKinds...)
	}); err != nil || !called {
		t.Fatalf("Runtime credential reference beside the fence = (called=%v, err=%v)", called, err)
	}
	if err := runtimeCredentials.Use(ctx, besideRuntimeCredential, besideRuntimeKinds, func(*RuntimeCredentialMaterial) error {
		return nil
	}); err != nil {
		t.Fatalf("Runtime credential use beside the fence: %v", err)
	}

	for name, mutation := range map[string]func() error{
		"delete of another credential": func() error {
			_, err := f.service.Delete(ctx, f.deleteRequest(f.unrelated, "delete-beside-durable"))
			return err
		},
		"delete of the same credential under another key": func() error {
			_, err := f.service.Delete(ctx, f.deleteRequest(f.deleting, "delete-durable-new-key"))
			return err
		},
		"create of a new credential": func() error {
			_, err := f.service.Create(ctx, f.createRequest("managed-new", "create-beside-durable"))
			return err
		},
	} {
		assertRecoveryRequiredAnswer(t, name, mutation())
	}
	if f.manager.deleteCalls() != 0 {
		t.Fatalf("the fenced process reached the Gateway %d times", f.manager.deleteCalls())
	}

	// A restarted Server completes the deletion in startup recovery and admits
	// references to other credentials again.
	restarted := newLifecycleFixtureWithCipher(t, f.pool, f.manager, f.cipher, ServiceOptions{})
	if err := restarted.service.Recover(ctx); err != nil {
		t.Fatalf("startup recovery: %v", err)
	}
	if f.manager.deleteCalls() != 1 {
		t.Fatalf("recovery Gateway deletes = %d, want 1", f.manager.deleteCalls())
	}
	assertCredentialOperationPhase(t, ctx, f.pool, OperationDelete, deletion.IdempotencyKey, OperationCompleted)
	runs := runstore.NewPostgresStore(f.pool)
	if err := restarted.service.WithRunCreation(ctx, func() error {
		if err := lookupManagedCredential(ctx, f.pool, f.unrelated.CredentialID); err != nil {
			return err
		}
		_, err := runstoretest.CreateRun(ctx, runs, pinnedRunParams("run-after-restart", f.unrelated.CredentialID))
		return err
	}); err != nil {
		t.Fatalf("Run after the restarted Server's recovery: %v", err)
	}
}

func TestCredentialDeleteWorkUnderTheBarrierDoesNotGrowWithHistory(t *testing.T) {
	f, ctx := newDeleteFenceFixture(t)
	deleteOf := func(credential CreateRequest, key string) func() error {
		return func() error {
			_, err := f.service.Delete(ctx, f.deleteRequest(credential, key))
			return err
		}
	}
	fresh := f.faults.statementsOf(t, deleteOf(f.deleting, "delete-fresh-history"))

	// Thousands of finished operations and hundreds of non-terminal Runs of
	// another credential add no statement to the work done under the barrier.
	f.insertFinishedDeletes(t, ctx, 5000)
	runs := runstore.NewPostgresStore(f.pool)
	for index := range 200 {
		createPinnedRun(t, ctx, runs, fmt.Sprintf("run-history-%03d", index), f.unrelated.CredentialID)
	}
	third := f.createRequest("managed-third", "create-third")
	if _, err := f.service.Create(ctx, third); err != nil {
		t.Fatal(err)
	}
	long := f.faults.statementsOf(t, deleteOf(third, "delete-long-history"))
	if fresh == 0 || long != fresh {
		t.Fatalf("delete statements: fresh history %d, long history %d", fresh, long)
	}
	if f.manager.deleteCalls() != 2 {
		t.Fatalf("Gateway deletes = %d, want one per delete", f.manager.deleteCalls())
	}
	t.Logf("one managed delete runs %d statements and one Gateway request under the barrier", fresh)
}

func TestCredentialStartupRecoveryNeedsTheGatewayOnlyForAPreparedOperation(t *testing.T) {
	pool, ctx := lifecycleTestPool(t)
	manager := newFakeGatewayManager()
	f := newLifecycleFixture(t, pool, manager, ServiceOptions{})
	if err := f.service.Recover(ctx); err != nil {
		t.Fatal(err)
	}
	deleting := f.createRequest("managed-deleting", "create-deleting")
	unrelated := f.createRequest("managed-unrelated", "create-unrelated")
	for _, request := range []CreateRequest{deleting, unrelated} {
		if _, err := f.service.Create(ctx, request); err != nil {
			t.Fatal(err)
		}
	}
	outage := errors.New("gateway unreachable")
	setOutage := func(err error) {
		manager.mu.Lock()
		defer manager.mu.Unlock()
		manager.validateErr, manager.createErr, manager.deleteErr, manager.recoverErr = err, err, err, err
	}
	gatewayCalls := func() int {
		return manager.validationCalls() + manager.createCalls() + manager.deleteCalls() + manager.recoverCalls()
	}
	setOutage(outage)

	// Without a prepared operation a Gateway outage does not affect startup:
	// recovery makes no Gateway call and the Server admits references.
	calls := gatewayCalls()
	clean := newLifecycleFixtureWithCipher(t, pool, manager, f.cipher, ServiceOptions{})
	if err := clean.service.Recover(ctx); err != nil {
		t.Fatalf("startup recovery during an outage with no prepared operation: %v", err)
	}
	if gatewayCalls() != calls {
		t.Fatalf("startup recovery without a prepared operation called the Gateway %d times", gatewayCalls()-calls)
	}
	if err := clean.service.WithRunCreation(ctx, func() error {
		return lookupManagedCredential(ctx, pool, unrelated.CredentialID)
	}); err != nil {
		t.Fatalf("Run guard after a clean start during an outage: %v", err)
	}

	// A delete that fails at the Gateway leaves a prepared operation, and
	// startup recovery must complete it before the Server is ready (RCF-R72).
	deletion := DeleteRequest{CredentialID: deleting.CredentialID, IdempotencyKey: "delete-during-outage", ActorID: "user-1"}
	if _, err := f.service.Delete(ctx, deletion); !errors.Is(err, ErrGatewayUnavailable) {
		t.Fatalf("delete during an outage error = %v", err)
	}
	blocked := newLifecycleFixtureWithCipher(t, pool, manager, f.cipher, ServiceOptions{})
	if err := blocked.service.Recover(ctx); !errors.Is(err, ErrGatewayUnavailable) {
		t.Fatalf("startup recovery of a prepared delete during an outage error = %v", err)
	}
	// configureCredentials stops startup on this error. The Service itself
	// stays unready and refuses every reference and every credential read.
	called := false
	if err := blocked.service.WithRunCreation(ctx, func() error { called = true; return nil }); called {
		t.Fatal("an unready Service admitted a reference")
	} else {
		assertRecoveryRequiredAnswer(t, "Run guard before recovery", err)
	}
	if _, err := blocked.service.ListCredentials(ctx, "", 10); !errors.Is(err, ErrRecoveryRequired) {
		t.Fatalf("credential list before recovery error = %v", err)
	}

	// Once the Gateway answers, the next start completes the deletion.
	setOutage(nil)
	recovered := newLifecycleFixtureWithCipher(t, pool, manager, f.cipher, ServiceOptions{})
	if err := recovered.service.Recover(ctx); err != nil {
		t.Fatalf("startup recovery after the outage: %v", err)
	}
	assertCredentialOperationPhase(t, ctx, pool, OperationDelete, deletion.IdempotencyKey, OperationCompleted)
	if err := recovered.service.WithRunCreation(ctx, func() error {
		return lookupManagedCredential(ctx, pool, unrelated.CredentialID)
	}); err != nil {
		t.Fatalf("Run guard after recovery: %v", err)
	}
}
