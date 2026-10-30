package credentials

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/leasetest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
)

// These tests run two Servers' credential services on one database across a
// handover of the advisory lease (PER-R34, RCF-R72). The previous owner is
// the delete fence fixture's Service; its lease session ends on the server
// side, as a terminated or failed lease connection does, and a successor
// takes the lease, waits for the handover and recovers, as configureCredentials
// does after awaitControlPlaneLease.

// The regression of the reproduced finding: on the code before the lease
// handover, the new owner recovered at once, admitted a Run of the credential,
// the held COMMIT then landed beside it and the next start's recovery
// answered credential_in_use.
func TestCredentialDeleteIntentOfAPreviousLeaseOwnerEndsBeforeTheNewOwnerRecovers(t *testing.T) {
	f, ctx := newDeleteFenceFixture(t)
	deletion := f.deleteRequest(f.deleting, "delete-previous-owner")
	f.faults.arm(commitFaultHold)
	if _, err := f.service.Delete(ctx, deletion); err == nil {
		t.Fatal("delete whose COMMIT got no reply succeeded")
	}
	// The previous owner's intent transaction is open with its COMMIT held:
	// it holds the credential's deletion lock and an uncommitted prepared row.
	held := f.faults.faultedConn(t)

	newOwner, recovered := f.recoverAfterHandover(t, ctx)
	recoveredEarly, recoveryErr := f.awaitRecoveryOrHandover(t, ctx, recovered, int32(held.pid.Load()))
	// The new owner admits a Run of the credential as soon as it is ready.
	runs := runstore.NewPostgresStore(f.pool)
	admitRun := func(runID string) error {
		return newOwner.service.WithRunCreation(ctx, func() error {
			if err := lookupManagedCredential(ctx, f.pool, f.deleting.CredentialID); err != nil {
				return err
			}
			_, err := runstoretest.CreateRun(ctx, runs, pinnedRunParams(runID, f.deleting.CredentialID))
			return err
		})
	}
	admittedEarly := admitRun("run-new-owner")
	if !recoveredEarly && !errors.Is(admittedEarly, ErrRecoveryRequired) {
		t.Fatalf("Run admission while recovery waits = %v, want credential_recovery_required", admittedEarly)
	}
	if err := held.commitHeld(); err != nil {
		t.Fatal(err)
	}
	if !recoveredEarly {
		recoveryErr = <-recovered
	}
	if recoveryErr != nil {
		t.Fatalf("new owner's recovery: %v", recoveryErr)
	}
	admittedLate := admitRun("run-after-recovery")

	// The next start recovers, and in no outcome does a prepared delete
	// coexist with a non-terminal Run that pins the credential.
	restarted := newLifecycleFixtureWithCipher(t, f.pool, f.manager, f.cipher, ServiceOptions{})
	restartErr := restarted.service.Recover(ctx)
	var inUse *CredentialInUseError
	if errors.As(restartErr, &inUse) {
		t.Fatalf("restart recovery answers credential_in_use: Runs %v pin %s", inUse.RunIDs, f.deleting.CredentialID)
	}
	if restartErr != nil {
		t.Fatalf("restart recovery: %v", restartErr)
	}
	pinning, err := runs.ListNonTerminalRunIDsByCredential(ctx, f.deleting.CredentialID, maximumCredentialRunReferences)
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := NewRepository(f.pool).HasPreparedDelete(ctx, f.deleting.CredentialID)
	if err != nil || prepared && len(pinning) != 0 {
		t.Fatalf("prepared delete %v (%v) beside Runs %v pinning the credential", prepared, err, pinning)
	}

	// The lease handover chose the first outcome: recovery waited for the
	// open intent, read it once it committed and completed the deletion, so
	// no Run of the credential was admitted.
	if recoveredEarly || !errors.Is(admittedLate, ErrNotFound) || len(pinning) != 0 {
		t.Fatalf("recovered before the intent ended %v; Runs after recovery %v (%v)",
			recoveredEarly, pinning, admittedLate)
	}
	assertCredentialOperationPhase(t, ctx, f.pool, OperationDelete, deletion.IdempotencyKey, OperationCompleted)
	if f.manager.deleteCalls() != 1 || countPreparedOperations(t, ctx, f) != 0 {
		t.Fatalf("Gateway deletes = %d, want the new owner's one", f.manager.deleteCalls())
	}
	f.assertGuardsOf(t, ctx, newOwner.service, "run-new-owner-unrelated")
}

func TestCredentialCreateOfAPreviousLeaseOwnerEndsBeforeTheNewOwnerRecovers(t *testing.T) {
	for _, held := range []struct {
		name string
		// arm holds the COMMIT of the intent or of the completion.
		arm func(deleteFenceFixture)
		// newOwnerRecovers: the held transaction is the intent, so only the
		// new owner's recovery reaches the Gateway.
		newOwnerRecovers bool
	}{
		{name: "intent", arm: func(f deleteFenceFixture) { f.faults.arm(commitFaultHold) }, newOwnerRecovers: true},
		{name: "completion", arm: func(f deleteFenceFixture) {
			f.faults.afterStatement(func(sql string) {
				if strings.Contains(sql, "INSERT INTO llm_credentials") {
					f.faults.arm(commitFaultHold)
				}
			})
		}},
	} {
		t.Run(held.name, func(t *testing.T) {
			f, ctx := newDeleteFenceFixture(t)
			request := f.createRequest("managed-handover", "create-handover")
			creates, remote := f.manager.createCalls(), f.manager.remoteCount()
			held.arm(f)
			if _, err := f.service.Create(ctx, request); err == nil {
				t.Fatal("create whose COMMIT got no reply succeeded")
			}
			f.faults.afterStatement(nil)
			conn := f.faults.faultedConn(t)

			newOwner, recovered := f.recoverAfterHandover(t, ctx)
			leasetest.AwaitWaiterBlockedBy(t, f.pool, int32(conn.pid.Load()))
			select {
			case err := <-recovered:
				t.Fatalf("the new owner recovered while the previous owner's create was open: %v", err)
			default:
			}
			if err := conn.commitHeld(); err != nil {
				t.Fatal(err)
			}
			if err := <-recovered; err != nil {
				t.Fatalf("new owner's recovery: %v", err)
			}
			assertCredentialOperationPhase(t, ctx, f.pool, OperationCreate, request.IdempotencyKey, OperationCompleted)
			if _, err := newOwner.service.GetCredential(ctx, request.CredentialID); err != nil {
				t.Fatalf("created credential after the handover: %v", err)
			}
			wantRecoveries := 0
			if held.newOwnerRecovers {
				wantRecoveries = 1
			}
			if f.manager.createCalls() != creates+1 || f.manager.recoverCalls() != wantRecoveries ||
				f.manager.remoteCount() != remote+1 {
				t.Fatalf("Gateway after the handover: creates %d (from %d), recoveries %d, remote keys %d (from %d)",
					f.manager.createCalls(), creates, f.manager.recoverCalls(), f.manager.remoteCount(), remote)
			}
			if countPreparedOperations(t, ctx, f) != 0 {
				t.Fatal("a prepared operation remains after the handover")
			}
		})
	}
}

func TestCredentialWritesOfAPreviousLeaseOwnerAfterTheHandoverCommitNothing(t *testing.T) {
	f, ctx := newDeleteFenceFixture(t)
	runtimeCredentials := newRuntimeServiceBeside(t, ctx, f.pool, f.lease.Lease, f.cipher, f.service.barrier)
	successor := f.handOver(t, ctx)
	creates, deletes, validations := f.manager.createCalls(), f.manager.deleteCalls(), f.manager.validationCalls()

	_, err := f.service.Delete(ctx, f.deleteRequest(f.deleting, "delete-after-handover"))
	assertLeaseLostAnswer(t, "delete intent of the previous owner", err)
	if f.service.uncertainDelete.Load() != nil {
		t.Fatal("a refused intent raised the uncertain-delete fence")
	}
	_, err = f.service.Create(ctx, f.createRequest("managed-after-handover", "create-after-handover"))
	assertLeaseLostAnswer(t, "create intent of the previous owner", err)
	if countOperations(t, ctx, f.pool, OperationDelete) != 0 || countPreparedOperations(t, ctx, f) != 0 ||
		f.manager.createCalls() != creates || f.manager.deleteCalls() != deletes {
		t.Fatalf("refused intents had effects: creates %d, deletes %d", f.manager.createCalls(), f.manager.deleteCalls())
	}
	if f.manager.validationCalls() != validations+1 {
		t.Fatalf("create validations = %d, want the one local validation before the intent", f.manager.validationCalls())
	}

	// A Runtime credential of the previous owner is neither deleted behind
	// a reference its successor may admit nor created behind its startup.
	_, err = runtimeCredentials.Delete(ctx, besideRuntimeCredential, "operator")
	if !errors.Is(err, persistencepostgres.ErrLeaseLost) {
		t.Fatalf("Runtime credential delete of the previous owner = %v", err)
	}
	material, err := NewOTLPHeadersCredential(map[string]string{"Authorization": "Bearer after-handover"})
	if err != nil {
		t.Fatal(err)
	}
	defer material.Destroy()
	_, err = runtimeCredentials.Create(ctx, RuntimeCredentialCreateRequest{
		CredentialID: "otel-after-handover", Material: material,
		IdempotencyKey: "create-otel-after-handover", ActorID: "operator",
	})
	if !errors.Is(err, persistencepostgres.ErrLeaseLost) {
		t.Fatalf("Runtime credential create of the previous owner = %v", err)
	}
	successorRuntime, err := NewRuntimeCredentialService(RuntimeCredentialServiceOptions{
		Pool: f.pool, Lease: successor.Lease, Cipher: f.cipher, Usage: NewRuntimeCredentialRepository(f.pool),
		Projects: projectstore.NewPostgresStore(f.pool), Audits: staticAuditDispatchHolds{}, Barrier: NewLifecycleBarrier(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := successorRuntime.Get(ctx, "otel-after-handover"); !errors.Is(err, ErrRuntimeCredentialNotFound) {
		t.Fatalf("refused Runtime credential create stored a credential: %v", err)
	}
	if result, err := successorRuntime.Delete(ctx, besideRuntimeCredential, "operator"); err != nil || result.Replayed {
		t.Fatalf("successor's delete of the credential the previous owner could not delete = (%+v, %v)", result, err)
	}
}

func TestCredentialCompletionOfAPreviousLeaseOwnerAfterTheHandoverIsLeftToTheNewOwner(t *testing.T) {
	f, ctx := newDeleteFenceFixture(t)
	// A Gateway failure leaves a durable prepared delete before the handover.
	deletion := f.deleteRequest(f.unrelated, "delete-prepared-before-handover")
	f.manager.deleteErr = errors.New("gateway delete failed")
	if _, err := f.service.Delete(ctx, deletion); !errors.Is(err, ErrGatewayUnavailable) {
		t.Fatalf("Gateway failure error = %v", err)
	}
	f.manager.deleteErr = nil
	successor := f.handOver(t, ctx)

	// The previous owner's retry may reach the Gateway, because the intent
	// was durable before the handover, but its completion commits nothing.
	_, err := f.service.Delete(ctx, deletion)
	assertLeaseLostAnswer(t, "completion of the previous owner", err)
	assertCredentialOperationPhase(t, ctx, f.pool, OperationDelete, deletion.IdempotencyKey, OperationPrepared)

	newOwner := newLifecycleFixtureWithCipher(t, f.pool, f.manager, f.cipher, ServiceOptions{Lease: successor.Lease})
	if err := newOwner.service.Recover(ctx); err != nil {
		t.Fatalf("new owner's recovery: %v", err)
	}
	assertCredentialOperationPhase(t, ctx, f.pool, OperationDelete, deletion.IdempotencyKey, OperationCompleted)
	if _, err := newOwner.service.GetCredential(ctx, f.unrelated.CredentialID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("credential after the new owner completed its deletion: %v", err)
	}
	if f.manager.remoteCount() != 1 {
		t.Fatalf("remote keys = %d, want the other credential's", f.manager.remoteCount())
	}
}

func TestCredentialCreateRetryOfAPreviousLeaseOwnerAfterTheHandoverMakesNoGatewayCall(t *testing.T) {
	f, ctx := newDeleteFenceFixture(t)
	// A Gateway failure leaves a durable prepared create before the handover.
	request := f.createRequest("managed-retried", "create-retried")
	f.manager.createErr = errors.New("gateway create failed")
	if _, err := f.service.Create(ctx, request); !errors.Is(err, ErrGatewayUnavailable) {
		t.Fatalf("Gateway failure error = %v", err)
	}
	f.manager.createErr = nil
	successor := f.handOver(t, ctx)
	creates, recoveries, remote := f.manager.createCalls(), f.manager.recoverCalls(), f.manager.remoteCount()

	// The retry would first remove the deterministic alias, which may name
	// the key the new owner creates for the same operation.
	_, err := f.service.Create(ctx, request)
	assertLeaseLostAnswer(t, "create retry of the previous owner", err)
	if f.manager.createCalls() != creates || f.manager.recoverCalls() != recoveries {
		t.Fatalf("create retry of the previous owner reached the Gateway: creates %d, recoveries %d",
			f.manager.createCalls()-creates, f.manager.recoverCalls()-recoveries)
	}
	assertCredentialOperationPhase(t, ctx, f.pool, OperationCreate, request.IdempotencyKey, OperationPrepared)

	newOwner := newLifecycleFixtureWithCipher(t, f.pool, f.manager, f.cipher, ServiceOptions{Lease: successor.Lease})
	if err := newOwner.service.Recover(ctx); err != nil {
		t.Fatalf("new owner's recovery: %v", err)
	}
	assertCredentialOperationPhase(t, ctx, f.pool, OperationCreate, request.IdempotencyKey, OperationCompleted)
	if f.manager.recoverCalls() != recoveries+1 || f.manager.createCalls() != creates+1 || f.manager.remoteCount() != remote+1 {
		t.Fatalf("new owner's recovery: recoveries %d, creates %d, remote keys %d",
			f.manager.recoverCalls()-recoveries, f.manager.createCalls()-creates, f.manager.remoteCount()-remote)
	}
	// Once the new owner completed it, the previous owner's retry only reads
	// the completed operation and replays it.
	replayed, err := f.service.Create(ctx, request)
	if err != nil || !replayed.Replayed || f.manager.createCalls() != creates+1 || f.manager.recoverCalls() != recoveries+1 {
		t.Fatalf("replay of the previous owner = (%+v, %v), Gateway creates %d, recoveries %d",
			replayed.Replayed, err, f.manager.createCalls()-creates, f.manager.recoverCalls()-recoveries)
	}
}

// handOver ends the fixture's lease session from the server side, lets a
// successor take the lease and waits for the handover, as a new active
// Server does before its startup reads durable state.
func (f deleteFenceFixture) handOver(t *testing.T, ctx context.Context) *leasetest.Session {
	t.Helper()
	f.lease.Lose(t)
	successor := f.lease.Successor(t)
	if err := persistencepostgres.AwaitLeaseHandover(ctx, f.pool, successor.Lease); err != nil {
		t.Fatal(err)
	}
	return successor
}

// recoverAfterHandover starts the successor of the fixture's Server: its
// lease session takes the lease once the fixture's ends, and in the
// background it waits for the handover and then recovers, as a starting
// Server does. The channel reports the first failure or the recovery's end.
func (f deleteFenceFixture) recoverAfterHandover(
	t *testing.T, ctx context.Context,
) (lifecycleFixture, <-chan error) {
	t.Helper()
	f.lease.Lose(t)
	successor := f.lease.Successor(t)
	newOwner := newLifecycleFixtureWithCipher(t, f.pool, f.manager, f.cipher, ServiceOptions{Lease: successor.Lease})
	recovered := make(chan error, 1)
	done := make(chan struct{})
	go func() {
		defer close(done)
		if err := persistencepostgres.AwaitLeaseHandover(ctx, f.pool, successor.Lease); err != nil {
			recovered <- err
			return
		}
		recovered <- newOwner.service.Recover(ctx)
	}()
	t.Cleanup(func() {
		// A failed test may leave the handover waiting for a held COMMIT
		// until the fixture's deadline.
		<-done
	})
	return newOwner, recovered
}

// awaitRecoveryOrHandover waits until the new owner's recovery ended or its
// lease handover waits for the open transaction of backend blocker, and
// reports whether recovery ended first, with its outcome.
func (f deleteFenceFixture) awaitRecoveryOrHandover(
	t *testing.T, ctx context.Context, recovered <-chan error, blocker int32,
) (bool, error) {
	t.Helper()
	for {
		select {
		case err := <-recovered:
			return true, err
		default:
		}
		var waiting bool
		if err := f.pool.QueryRow(ctx, `
SELECT EXISTS (
    SELECT 1 FROM pg_locks
    WHERE locktype = 'advisory' AND NOT granted AND $1::integer = ANY (pg_blocking_pids(pid)))`, blocker).
			Scan(&waiting); err != nil {
			t.Fatal(err)
		}
		if waiting {
			return false, nil
		}
		select {
		case err := <-recovered:
			return true, err
		case <-ctx.Done():
			t.Fatal("the new owner neither recovered nor waited for the previous owner's intent")
		case <-time.After(10 * time.Millisecond):
		}
	}
}

// assertGuardsOf checks that service admits a Run pinning the unrelated
// credential.
func (f deleteFenceFixture) assertGuardsOf(t *testing.T, ctx context.Context, service *Service, runID string) {
	t.Helper()
	runs := runstore.NewPostgresStore(f.pool)
	if err := service.WithRunCreation(ctx, func() error {
		if err := lookupManagedCredential(ctx, f.pool, f.unrelated.CredentialID); err != nil {
			return err
		}
		_, err := runstoretest.CreateRun(ctx, runs, pinnedRunParams(runID, f.unrelated.CredentialID))
		return err
	}); err != nil {
		t.Fatalf("Run pinning an unrelated credential after the handover: %v", err)
	}
}

func countPreparedOperations(t *testing.T, ctx context.Context, f deleteFenceFixture) int {
	t.Helper()
	operations, err := NewRepository(f.pool).ListPreparedOperations(ctx, maximumPreparedOperationBatch)
	if err != nil {
		t.Fatal(err)
	}
	return len(operations)
}

// assertLeaseLostAnswer checks that a managed credential write of a Server
// that lost the lease answers the retryable credential_recovery_required
// and keeps the lease loss as its cause.
func assertLeaseLostAnswer(t *testing.T, name string, err error) {
	t.Helper()
	assertRecoveryRequiredAnswer(t, name, err)
	if !errors.Is(err, persistencepostgres.ErrLeaseLost) {
		t.Fatalf("%s = %v, want the lease loss as its cause", name, err)
	}
}
