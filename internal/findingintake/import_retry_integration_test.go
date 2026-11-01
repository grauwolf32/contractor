//go:build integration

package findingintake

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Another finding writer of the same Project holds the Project's finding
// correlation fence, so the owner import's nonblocking fence aborts with the
// retryable 40001 (FND-R177). The other writer commits after that abort and
// before the next attempt; the import replays its whole transaction and
// commits exactly once. A replay of the request changes nothing.
func TestPostgresOwnerImportReplaysProjectFenceConflict(t *testing.T) {
	f := newDeletionImportFixture(t)
	writer := beginConcurrentFindingWriter(t, f)
	trace := &intakeTransactions{}
	trace.afterFence = func(fence int, _ error) {
		if fence == 1 {
			if err := writer.Commit(f.ctx); err != nil {
				t.Errorf("commit the concurrent finding writer: %v", err)
			}
		}
	}
	intake := intakeOnActorPool(t, f, trace)
	before := readIntakeState(t, f)
	hold, replayed, err := intake.ImportIntoAudit(f.ctx, f.request)
	if err != nil || replayed || hold.AuditID != f.request.AuditID || hold.ProjectID != "delete-project" ||
		hold.Proposal.Ref.Revision == nil {
		t.Fatalf("import under a contended Project fence = (%+v, %t, %v)", hold, replayed, err)
	}
	trace.requireAttempts(t, 2, true, false)
	committed := readIntakeState(t, f)
	if want := before.admitted(1); committed != want {
		t.Fatalf("state after the import = %+v, want %+v", committed, want)
	}

	again, replayed, err := intake.ImportIntoAudit(f.ctx, f.request)
	if err != nil || !replayed || !again.CreatedAt.Equal(hold.CreatedAt) ||
		!again.Proposal.Ref.SameExact(hold.Proposal.Ref) || len(again.Evidence) != len(hold.Evidence) {
		t.Fatalf("import replay = (%+v, %t, %v), want %+v", again, replayed, err, hold)
	}
	trace.requireAttempts(t, 3, true, false, false)
	if replayedState := readIntakeState(t, f); replayedState != committed {
		t.Fatalf("import replay changed state %+v to %+v", committed, replayedState)
	}
}

// Run creation, Audit start and Project artifact admission hold the Project
// row FOR SHARE until they commit, as Git import does. The finding correlation
// fence is a row of its own that only finding writers lock (FND-R177), so
// owner import and collection retention each commit in their first
// transaction with one fence step while such a holder is open, and the holder
// never waits on them.
func TestPostgresFindingIntakeSharesTheProjectWithConcurrentAdmissions(t *testing.T) {
	for _, contender := range []struct {
		name  string
		begin func(*testing.T, deletionImportFixture) pgx.Tx
	}{
		{"Run creation", func(t *testing.T, f deletionImportFixture) pgx.Tx {
			return beginConcurrentRunCreation(t, f, "concurrent-run")
		}},
		{"Audit start", beginConcurrentAuditStart},
		{"Project artifact admission", beginConcurrentProjectArtifactAdmission},
	} {
		t.Run(contender.name, func(t *testing.T) {
			f := newDeletionImportFixture(t)
			collected := f.request
			collected.Proposal = insertAuditChildReceipt(t, f, "collected-candidate").Proposal.Ref
			holder := contender.begin(t, f)
			trace := &intakeTransactions{}
			intake := intakeOnActorPool(t, f, trace)
			before := readIntakeState(t, f)
			if hold, replayed, err := intake.ImportIntoAudit(f.ctx, f.request); err != nil || replayed ||
				hold.AuditID != f.request.AuditID {
				t.Fatalf("import beside a concurrent %s = (%+v, %t, %v)", contender.name, hold, replayed, err)
			}
			trace.requireAttempts(t, 1, false)
			if err := intake.RetainAuditCollectionBatch(f.ctx, []ImportRequest{collected}); err != nil {
				t.Fatalf("collection beside a concurrent %s = %v", contender.name, err)
			}
			trace.requireAttempts(t, 2, false, false)
			if after, want := readIntakeState(t, f), before.admitted(2); after != want {
				t.Fatalf("state after import and collection = %+v, want %+v", after, want)
			}
			if err := holder.Commit(f.ctx); err != nil {
				t.Fatalf("commit the concurrent %s: %v", contender.name, err)
			}
		})
	}
}

// Only a definite transaction conflict starts another transaction, and never
// more than three. Cancellation, permanent failures and an uncertain commit
// return at once, after the transaction's rollback has finished.
func TestPostgresFindingIntakeReplaysOnlyDefiniteConflicts(t *testing.T) {
	for _, entry := range []struct {
		name string
		run  func(context.Context, *Service, ImportRequest) error
		// replayed reports whether a repeated request found the committed hold.
		replay func(context.Context, *Service, ImportRequest) (bool, error)
		// closed is the failure for a terminal Audit and closedFences whether
		// the entry reaches the Project fence before it finds that state.
		closed       error
		closedFences []bool
	}{
		{
			name: "owner import",
			run: func(ctx context.Context, intake *Service, request ImportRequest) error {
				_, _, err := intake.ImportIntoAudit(ctx, request)
				return err
			},
			replay: func(ctx context.Context, intake *Service, request ImportRequest) (bool, error) {
				_, replayed, err := intake.ImportIntoAudit(ctx, request)
				return replayed, err
			},
			closed: ErrNotFound, closedFences: []bool{false},
		},
		{
			name: "collection",
			run: func(ctx context.Context, intake *Service, request ImportRequest) error {
				return intake.RetainAuditCollectionBatch(ctx, []ImportRequest{request})
			},
			replay: func(ctx context.Context, intake *Service, request ImportRequest) (bool, error) {
				return true, intake.RetainAuditCollectionBatch(ctx, []ImportRequest{request})
			},
			closed: ErrAuditClosed,
		},
	} {
		t.Run(entry.name+"/persistent Project contention", func(t *testing.T) {
			f := newDeletionImportFixture(t)
			writer := beginConcurrentFindingWriter(t, f)
			trace := &intakeTransactions{}
			intake := intakeOnActorPool(t, f, trace)
			before := readIntakeState(t, f)
			err := entry.run(f.ctx, intake, f.request)
			if persistencepostgres.Classify(err) != persistencepostgres.ClassTransactionConflict {
				t.Fatalf("persistently contended %s = %v, want a transaction conflict", entry.name, err)
			}
			trace.requireAttempts(t, 3, true, true, true)
			requireAuditLockReleased(t, f)
			if after := readIntakeState(t, f); after != before {
				t.Fatalf("aborted %s changed state %+v to %+v", entry.name, before, after)
			}
			// The other finding writer never waited on the intake.
			if err := writer.Commit(f.ctx); err != nil {
				t.Fatal(err)
			}
		})

		t.Run(entry.name+"/cancelled after a conflict", func(t *testing.T) {
			f := newDeletionImportFixture(t)
			_ = beginConcurrentFindingWriter(t, f)
			ctx, cancel := context.WithCancel(f.ctx)
			defer cancel()
			trace := &intakeTransactions{}
			trace.afterFence = func(fence int, _ error) {
				if fence == 1 {
					cancel()
				}
			}
			intake := intakeOnActorPool(t, f, trace)
			before := readIntakeState(t, f)
			if err := entry.run(ctx, intake, f.request); !errors.Is(err, context.Canceled) {
				t.Fatalf("cancelled %s = %v", entry.name, err)
			}
			trace.requireAttempts(t, 1, true)
			requireAuditLockReleased(t, f)
			if after := readIntakeState(t, f); after != before {
				t.Fatalf("cancelled %s changed state %+v to %+v", entry.name, before, after)
			}
		})

		t.Run(entry.name+"/permanent failures", func(t *testing.T) {
			f := newDeletionImportFixture(t)
			trace := &intakeTransactions{}
			intake := intakeOnActorPool(t, f, trace)
			before := readIntakeState(t, f)
			invalid := f.request
			invalid.Proposal.Revision = nil
			if err := entry.run(f.ctx, intake, invalid); !errors.Is(err, ErrInvalid) {
				t.Fatalf("invalid %s = %v", entry.name, err)
			}
			trace.requireAttempts(t, 0)
			foreign := f.request
			foreign.OwnerID = "foreign-owner"
			if err := entry.run(f.ctx, intake, foreign); !errors.Is(err, ErrNotFound) {
				t.Fatalf("foreign %s = %v", entry.name, err)
			}
			trace.requireAttempts(t, 1)
			if _, err := f.pool.Exec(f.ctx, `
UPDATE audits
   SET state = 'completed', baseline_snapshot = '{}'::jsonb,
       started_at = clock_timestamp(), finished_at = clock_timestamp()
 WHERE audit_id = $1`, f.request.AuditID); err != nil {
				t.Fatal(err)
			}
			closedState := readIntakeState(t, f)
			if err := entry.run(f.ctx, intake, f.request); !errors.Is(err, entry.closed) {
				t.Fatalf("%s into a completed Audit = %v, want %v", entry.name, err, entry.closed)
			}
			trace.requireAttempts(t, 2, entry.closedFences...)
			if after := readIntakeState(t, f); after != closedState || closedState.holds != before.holds {
				t.Fatalf("refused %s changed state %+v to %+v", entry.name, closedState, after)
			}
		})

		t.Run(entry.name+"/lost commit response", func(t *testing.T) {
			f := newDeletionImportFixture(t)
			trace := &intakeTransactions{}
			fault := &lostCommitResponse{}
			intake := intakeOnLostCommitPool(t, f, trace, fault)
			before := readIntakeState(t, f)
			fault.armed.Store(true)
			// pgx reports the lost reply as a closed connection; the commit
			// failure is returned as it is, never as a transaction conflict.
			err := entry.run(f.ctx, intake, f.request)
			if fault.fired.Load() != 1 || err == nil || err.Error() != "commit PostgreSQL transaction" ||
				persistencepostgres.IsTransactionConflict(err) {
				t.Fatalf("%s with a lost COMMIT response = %v (fault fired %d)", entry.name, err, fault.fired.Load())
			}
			// The callback was not replayed blindly although PostgreSQL committed.
			trace.requireAttempts(t, 1, false)
			committed := readIntakeState(t, f)
			if want := before.admitted(1); committed != want {
				t.Fatalf("state after the uncertain commit = %+v, want %+v", committed, want)
			}
			// The caller reconciles through the receipt's natural key.
			replayed, err := entry.replay(f.ctx, f.intake, f.request)
			if err != nil || !replayed {
				t.Fatalf("%s reconciliation = (%t, %v)", entry.name, replayed, err)
			}
			if reconciled := readIntakeState(t, f); reconciled != committed {
				t.Fatalf("%s reconciliation changed state %+v to %+v", entry.name, committed, reconciled)
			}
		})
	}
}

// A 200-receipt collection page commits in bounded transactions. The page
// holds one proposal whose evidence is missing and, later, meets a Project
// fence conflict. The conflict replays only the transaction it aborted, with
// the same proposals and work as the uncontended transaction; a conflict that
// outlasts the attempts leaves the committed transactions, and a resumed
// collection retains only the rest. Every valid receipt is retained once and
// only the invalid one is rejected.
func TestPostgresCollectionPageReplaysOnlyItsConflictedTransaction(t *testing.T) {
	for _, persistent := range []bool{false, true} {
		name := "transient conflict"
		if persistent {
			name = "persistent conflict then resumed collection"
		}
		t.Run(name, func(t *testing.T) {
			f := newDeletionImportFixture(t)
			ctx, cancel := context.WithTimeout(t.Context(), 3*time.Minute)
			defer cancel()
			f.ctx = ctx
			const pageSize, invalidAt, evidencePerProposal = MaxAuditReceiptBatchSize, 20, 8
			evidence := writeFixtureEvidence(t, f, evidencePerProposal)
			missing := slices.Clone(evidence)
			missing[evidencePerProposal-1] = writeLostFixtureEvidence(t, f)
			// The fixture's own receipt, without evidence, comes first.
			var invalidReceipt string
			for index := 1; index < pageSize; index++ {
				cited := evidence
				if index == invalidAt {
					cited = missing
				}
				receipt := insertAuditChildReceiptWithEvidence(t, f, fmt.Sprintf("page-candidate-%03d", index), cited)
				if index == invalidAt {
					invalidReceipt = receipt.ReceiptID
				}
			}
			loseFixtureEvidence(t, f, missing[evidencePerProposal-1])
			page, err := f.intake.ListAuditCollection(ctx, f.request.OwnerID, f.request.AuditID, f.request.RunID,
				ListQuery{Limit: pageSize})
			if err != nil || len(page) != pageSize || page[0].Receipt.ReceiptID != f.receiptID ||
				page[invalidAt].Receipt.ReceiptID != invalidReceipt {
				t.Fatalf("collection page = %d receipts, %v", len(page), err)
			}
			requests := make([]ImportRequest, len(page))
			for index, candidate := range page {
				requests[index] = f.request
				requests[index].Proposal = candidate.Receipt.Proposal.Ref
			}

			// The page is [w2, 19 x w10, invalid, 179 x w10] in collection
			// work. The first transaction meets the invalid proposal, rolls
			// back and commits the 20 proposals before it with its rejection.
			// Later transactions start a proposal only below 512 work, so the
			// 52nd proposal of 10 crosses the threshold and ends each one.
			// Fences: 1 rolled back, 2 commits 20, 3 commits 52, 4 conflicts.
			const conflictedFence = 4
			contender, err := f.pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = contender.Rollback(context.Background()) }()
			var released atomic.Bool
			release := func() {
				if released.CompareAndSwap(false, true) {
					if err := contender.Rollback(ctx); err != nil {
						t.Errorf("release the Project's finding correlation fence: %v", err)
					}
				}
			}
			trace := &intakeTransactions{}
			trace.beforeFence = func(fence int) {
				if fence == conflictedFence {
					// Another finding writer of the Project takes its fence.
					if err := hookstore.NewFindingEventsTx(contender).LockFindingCorrelationFence(
						ctx, "delete-project"); err != nil {
						t.Errorf("contend for the Project's finding correlation fence: %v", err)
					}
				}
			}
			trace.afterFence = func(fence int, _ error) {
				if fence == conflictedFence && !persistent {
					release()
				}
			}
			f.intake = intakeOnActorPool(t, f, trace)
			before := readIntakeState(t, f)
			err = f.intake.RetainAuditCollectionBatch(ctx, requests)
			if persistent {
				if persistencepostgres.Classify(err) != persistencepostgres.ClassTransactionConflict {
					t.Fatalf("page under persistent contention = %v", err)
				}
				trace.requireAttempts(t, 6, false, false, false, true, true, true)
				trace.requireCommittedHolds(t, 20, 52)
				progress := readIntakeState(t, f)
				if want := before.admitted(72).rejected(1); progress != want {
					t.Fatalf("committed progress = %+v, want %+v", progress, want)
				}
				release()
				if err := collectFixtureProposals(ctx, f, false); err != nil {
					t.Fatalf("resumed collection: %v", err)
				}
			} else if err != nil {
				t.Fatalf("page with a transient conflict: %v", err)
			} else {
				trace.requireAttempts(t, 7, false, false, false, true, false, false, false)
			}
			trace.requireCommittedHolds(t, 20, 52, 52, 52, 23)
			collected := readIntakeState(t, f)
			if want := before.admitted(pageSize - 1).rejected(1); collected != want {
				t.Fatalf("collected state = %+v, want %+v", collected, want)
			}
			var held, distinct, invalidHolds int
			if err := f.pool.QueryRow(ctx, `
SELECT count(*), count(DISTINCT receipt_id), count(*) FILTER (WHERE receipt_id = $2)
  FROM finding_proposal_audit_holds WHERE audit_id = $1`, f.request.AuditID, invalidReceipt).Scan(
				&held, &distinct, &invalidHolds); err != nil || held != pageSize-1 || distinct != held || invalidHolds != 0 {
				t.Fatalf("holds = %d (%d distinct, %d of the invalid receipt), %v", held, distinct, invalidHolds, err)
			}
			var rejected string
			if err := f.pool.QueryRow(ctx, `
SELECT entity_id FROM audit_events WHERE audit_id = $1 AND kind = 'finding.proposal_rejected'`,
				f.request.AuditID).Scan(&rejected); err != nil || rejected != invalidReceipt {
				t.Fatalf("rejected receipt = %q, %v; want %q", rejected, err, invalidReceipt)
			}
			// Replaying the page through its natural keys changes nothing.
			if err := f.intake.RetainAuditCollectionBatch(ctx, requests); err != nil {
				t.Fatalf("page replay: %v", err)
			}
			if replayed := readIntakeState(t, f); replayed != collected {
				t.Fatalf("page replay changed state %+v to %+v", collected, replayed)
			}
		})
	}
}

// Project deletion holds the Project row and then waits for the Audit an
// intake transaction holds. Without the early nonblocking fence the intake's
// ProjectScope admission would wait for deletion, a parent/Audit lock cycle
// that PostgreSQL breaks with a deadlock abort. The fence aborts the intake
// instead, and its rollback lets deletion finish. The replayed attempt then
// finds the Project deleting and commits no late write. An intake that took
// the fence first commits before deletion.
func TestPostgresFindingIntakeSerializesWithProjectDeletion(t *testing.T) {
	for _, outcome := range []string{"deletion commits between attempts", "import cancelled after the conflict"} {
		t.Run("deletion waits on the import's Audit/"+outcome, func(t *testing.T) {
			f := newDeletionImportFixture(t)
			cancelled := outcome == "import cancelled after the conflict"
			importCtx, cancelImport := context.WithCancel(f.ctx)
			defer cancelImport()
			heldAudit, resumeImport := make(chan struct{}), make(chan struct{})
			deletionCommitted := make(chan struct{})
			var resumeOnce sync.Once
			resume := func() { resumeOnce.Do(func() { close(resumeImport) }) }
			defer resume()
			trace := &intakeTransactions{}
			trace.afterAuditLock = func(lock int) {
				if lock == 1 {
					close(heldAudit)
					<-resumeImport
				}
			}
			trace.afterFence = func(fence int, _ error) {
				if fence == 1 && cancelled {
					cancelImport()
				}
			}
			trace.beforeBegin = func(attempt int) {
				if attempt == 2 {
					select {
					case <-deletionCommitted:
					case <-f.ctx.Done():
					}
				}
			}
			actorPool, actorPID := deletionActorPool(t, f, trace)
			intake, err := New(actorPool)
			if err != nil {
				t.Fatal(err)
			}
			before := readIntakeState(t, f)
			imported := make(chan error, 1)
			go func() {
				_, _, err := intake.ImportIntoAudit(importCtx, f.request)
				imported <- err
			}()
			select {
			case <-heldAudit:
			case err := <-imported:
				t.Fatalf("import finished before it held the Audit: %v", err)
			case <-f.ctx.Done():
				t.Fatal("import never held the Audit")
			}
			// The import holds the source Run and the Audit. Deletion takes the
			// Project first and then waits for the Audit.
			deletionPool, deletionPID := deletionActorPool(t, f, nil)
			deletion, err := deletionPool.Begin(f.ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = deletion.Rollback(context.Background()) }()
			beginFixtureProjectDeletion(t, f, projectstore.NewPostgresStore(deletion))
			auditLocked := make(chan error, 1)
			go func() {
				_, err := deletion.Exec(f.ctx,
					`SELECT audit_id FROM audits WHERE project_id = $1 FOR UPDATE`, "delete-project")
				auditLocked <- err
			}()
			awaitDeletionDatabaseWait(t, f, deletionPID, actorPID, "FOR UPDATE")
			resume()
			// The fence fails without waiting, so no deadlock is detected:
			// the import's rollback hands the Audit to deletion.
			if err := receiveDeletionOperation(t, f.ctx, auditLocked); err != nil {
				t.Fatalf("deletion's Audit lock = %v", err)
			}
			if err := deletion.Commit(f.ctx); err != nil {
				t.Fatal(err)
			}
			close(deletionCommitted)
			err = receiveDeletionOperation(t, f.ctx, imported)
			if cancelled {
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("cancelled import = %v", err)
				}
				trace.requireAttempts(t, 1, true)
			} else {
				if !errors.Is(err, artifacts.ErrScopeDeleting) {
					t.Fatalf("import into a deleting Project = %v, want %v", err, artifacts.ErrScopeDeleting)
				}
				trace.requireAttempts(t, 2, true, false)
			}
			if after := readIntakeState(t, f); after != before {
				t.Fatalf("import around Project deletion changed state %+v to %+v", before, after)
			}
			requireFixtureProjectDeleting(t, f)
		})
	}

	t.Run("import holds the fence before deletion", func(t *testing.T) {
		f := newDeletionImportFixture(t)
		late := insertAuditChildReceipt(t, f, "late-candidate")
		fenceHeld, resumeImport := make(chan struct{}), make(chan struct{})
		var resumeOnce sync.Once
		resume := func() { resumeOnce.Do(func() { close(resumeImport) }) }
		defer resume()
		trace := &intakeTransactions{}
		trace.afterFence = func(fence int, err error) {
			if fence == 1 && err == nil {
				close(fenceHeld)
				<-resumeImport
			}
		}
		actorPool, actorPID := deletionActorPool(t, f, trace)
		intake, err := New(actorPool)
		if err != nil {
			t.Fatal(err)
		}
		before := readIntakeState(t, f)
		imported := make(chan error, 1)
		go func() {
			_, _, err := intake.ImportIntoAudit(f.ctx, f.request)
			imported <- err
		}()
		select {
		case <-fenceHeld:
		case err := <-imported:
			t.Fatalf("import finished without its Project fence: %v", err)
		case <-f.ctx.Done():
			t.Fatal("import never took its Project fence")
		}
		deletionPool, deletionPID := deletionActorPool(t, f, nil)
		deleted := make(chan error, 1)
		go func() {
			deleted <- beginFixtureProjectDeletionErr(f, projectstore.NewPostgresStore(deletionPool))
		}()
		awaitDeletionDatabaseWait(t, f, deletionPID, actorPID, "UPDATE projects")
		resume()
		awaitDeletionOperation(t, f.ctx, imported)
		awaitDeletionOperation(t, f.ctx, deleted)
		trace.requireAttempts(t, 1, false)
		if after, want := readIntakeState(t, f), before.admitted(1); after != want {
			t.Fatalf("state after the import = %+v, want %+v", after, want)
		}
		requireFixtureProjectDeleting(t, f)
		// After deletion begins, the committed hold replays, but no new
		// proposal is retained in the deleting Project.
		if _, replayed, err := intake.ImportIntoAudit(f.ctx, f.request); err != nil || !replayed {
			t.Fatalf("replay in a deleting Project = (%t, %v)", replayed, err)
		}
		lateRequest := f.request
		lateRequest.Proposal = late.Proposal.Ref
		if _, _, err := intake.ImportIntoAudit(f.ctx, lateRequest); !errors.Is(err, artifacts.ErrScopeDeleting) {
			t.Fatalf("late import into a deleting Project = %v", err)
		}
		trace.requireAttempts(t, 3, false, false, false)
		if after, want := readIntakeState(t, f), before.admitted(1); after != want {
			t.Fatalf("state after the late import = %+v, want %+v", after, want)
		}
	})
}

// intakeState is what an Audit admission changes: the Audit's holds,
// findings, events and revision and its Project's finding correlation
// revision.
type intakeState struct {
	holds, findings, events, rejections int64
	auditRevision, correlationRevision  int64
}

func readIntakeState(t *testing.T, f deletionImportFixture) intakeState {
	t.Helper()
	var state intakeState
	if err := f.pool.QueryRow(f.ctx, `
SELECT (SELECT count(*) FROM finding_proposal_audit_holds WHERE audit_id = $1),
       (SELECT count(*) FROM project_finding_audits WHERE audit_id = $1),
       (SELECT count(*) FROM audit_events WHERE audit_id = $1),
       (SELECT count(*) FROM audit_events WHERE audit_id = $1 AND kind = 'finding.proposal_rejected'),
       (SELECT revision::bigint FROM audits WHERE audit_id = $1),
       (SELECT revision FROM finding_correlation_fences WHERE project_id = $2)`,
		f.request.AuditID, "delete-project",
	).Scan(&state.holds, &state.findings, &state.events, &state.rejections,
		&state.auditRevision, &state.correlationRevision); err != nil {
		t.Fatal(err)
	}
	return state
}

// admitted is state after count more admissions: each adds one hold, finding
// and finding.proposed event and advances the Audit and Project revisions once.
func (s intakeState) admitted(count int64) intakeState {
	s.holds += count
	s.events += count
	s.auditRevision += count
	s.correlationRevision += count
	return s
}

// rejected is state after count more collection rejections, each one
// finding.proposal_rejected event.
func (s intakeState) rejected(count int64) intakeState {
	s.events += count
	s.rejections += count
	s.auditRevision += count
	return s
}

// beginConcurrentRunCreation creates another Run of the fixture Project in a
// transaction that the test ends. Like every Run creation, its active-Project
// check holds the Project row FOR SHARE until then, which the intake's
// finding correlation fence shares.
func beginConcurrentRunCreation(t *testing.T, f deletionImportFixture, runID string) pgx.Tx {
	t.Helper()
	source, err := runstore.NewPostgresStore(f.pool).GetRun(f.ctx, f.request.RunID)
	if err != nil {
		t.Fatal(err)
	}
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = tx.Rollback(context.Background()) })
	if _, err := runstoretest.CreateRun(f.ctx, runstore.NewPostgresStore(tx), runstore.CreateRunParams{
		LaneClass: resolved.LaneClassModel, RunID: runID, OwnerID: source.OwnerID, ProjectID: source.ProjectID,
		WorkflowName: source.WorkflowName, WorkflowVersion: source.WorkflowVersion,
		WorkflowSchemaVersion: source.WorkflowSchemaVersion, WorkflowSnapshot: source.WorkflowSnapshot,
		Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	return tx
}

// beginConcurrentAuditStart takes the shared Project lock that Audit start
// holds while it snapshots Project inputs, in a transaction that the test
// ends.
// beginConcurrentFindingWriter holds the fixture Project's finding correlation
// fence in a transaction that the test ends, as another finding writer of the
// Project does from its fence lock until it commits: an import or collection
// into another Audit takes it through the same auditstore call, and an
// analyst decision or an Audit purge through the finding triggers.
func beginConcurrentFindingWriter(t *testing.T, f deletionImportFixture) pgx.Tx {
	t.Helper()
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = tx.Rollback(context.Background()) })
	if err := hookstore.NewFindingEventsTx(tx).LockFindingCorrelationFence(f.ctx, "delete-project"); err != nil {
		t.Fatal(err)
	}
	return tx
}

func beginConcurrentAuditStart(t *testing.T, f deletionImportFixture) pgx.Tx {
	t.Helper()
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = tx.Rollback(context.Background()) })
	if _, err := projectstore.NewPostgresStore(tx).LockActiveAuditProject(
		f.ctx, f.request.OwnerID, "delete-project",
	); err != nil {
		t.Fatal(err)
	}
	return tx
}

// beginConcurrentProjectArtifactAdmission writes an Artifact into the
// fixture's existing ProjectScope in a transaction that the test ends. Its
// admission trigger holds the Project row FOR SHARE until then.
func beginConcurrentProjectArtifactAdmission(t *testing.T, f deletionImportFixture) pgx.Tx {
	t.Helper()
	write := func(db persistencepostgres.DBTX, name string) {
		t.Helper()
		projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(db)).Project("delete-project")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := projectArtifacts.Write(f.ctx, contracts.ArtifactRef{Namespace: "inputs", Name: name},
			artifacts.Payload{MediaType: "text/plain", Data: []byte(name)}, nil); err != nil {
			t.Fatal(err)
		}
	}
	// The scope exists before the contender, as it does for every Project
	// with inputs; only the contender's own admission stays open.
	write(f.pool, "existing")
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = tx.Rollback(context.Background()) })
	write(tx, "concurrent")
	return tx
}

func beginFixtureProjectDeletion(t *testing.T, f deletionImportFixture, projects *projectstore.PostgresStore) {
	t.Helper()
	if err := beginFixtureProjectDeletionErr(f, projects); err != nil {
		t.Fatal(err)
	}
}

func beginFixtureProjectDeletionErr(f deletionImportFixture, projects *projectstore.PostgresStore) error {
	project, err := projectstore.NewPostgresStore(f.pool).Get(f.ctx, f.request.OwnerID, "delete-project")
	if err != nil {
		return err
	}
	_, began, err := projects.BeginDeletion(f.ctx, projectstore.BeginDeletionParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision,
	})
	if err == nil && !began {
		return errors.New("Project deletion did not begin")
	}
	return err
}

func requireFixtureProjectDeleting(t *testing.T, f deletionImportFixture) {
	t.Helper()
	project, err := projectstore.NewPostgresStore(f.pool).Get(f.ctx, f.request.OwnerID, "delete-project")
	if err != nil || project.Lifecycle != projectstore.LifecycleDeleting {
		t.Fatalf("Project after deletion = %+v, %v", project, err)
	}
}

// requireAuditLockReleased fails unless another transaction can lock the
// fixture Audit without waiting: the intake finished its rollback before it
// returned.
func requireAuditLockReleased(t *testing.T, f deletionImportFixture) {
	t.Helper()
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	if _, err := tx.Exec(f.ctx, `SELECT audit_id FROM audits WHERE audit_id = $1 FOR UPDATE NOWAIT`,
		f.request.AuditID); err != nil {
		t.Fatalf("the intake kept its Audit lock: %v", err)
	}
}

func intakeOnActorPool(t *testing.T, f deletionImportFixture, trace pgx.QueryTracer) *Service {
	t.Helper()
	pool, _ := deletionActorPool(t, f, trace)
	intake, err := New(pool)
	if err != nil {
		t.Fatal(err)
	}
	return intake
}

// intakeTransactions observes the transactions of a single-connection actor
// pool that runs finding intake: each transaction start, each destination
// Audit lock, each early Project correlation fence with its outcome, and the
// collection work of every hold that each committed transaction created. Its
// hooks run on the actor's goroutine while the transaction is open, so a test
// can change concurrent state at an exact point of an attempt.
type intakeTransactions struct {
	mu         sync.Mutex
	begins     int
	auditLocks int
	fences     []error
	open       []int
	committed  [][]int

	beforeBegin    func(attempt int)
	afterAuditLock func(lock int)
	beforeFence    func(fence int)
	afterFence     func(fence int, err error)
}

type intakeStatementKey struct{}

type intakeStatement struct {
	auditLock, fence int
	commit           bool
}

func (c *intakeTransactions) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	statement := strings.ToLower(strings.TrimSpace(data.SQL))
	c.mu.Lock()
	switch {
	case strings.HasPrefix(statement, "begin"):
		c.begins++
		c.open = nil
		attempt, hook := c.begins, c.beforeBegin
		c.mu.Unlock()
		if hook != nil {
			hook(attempt)
		}
		return ctx
	case strings.HasPrefix(statement, "select state from audits"):
		c.auditLocks++
		lock := c.auditLocks
		c.mu.Unlock()
		return context.WithValue(ctx, intakeStatementKey{}, intakeStatement{auditLock: lock})
	case strings.Contains(statement, "contractor_lock_finding_correlation_fence"):
		c.fences = append(c.fences, nil)
		fence, hook := len(c.fences), c.beforeFence
		c.mu.Unlock()
		if hook != nil {
			hook(fence)
		}
		return context.WithValue(ctx, intakeStatementKey{}, intakeStatement{fence: fence})
	case strings.Contains(statement, "insert into finding_proposal_audit_holds"):
		// A fresh hold weighs one for its receipt and one per imported
		// proposal or evidence revision (auditdomain.MaximumCollectionTransactionWork).
		var evidence []json.RawMessage
		if len(data.Args) == 5 {
			if encoded, ok := data.Args[4].([]byte); ok {
				_ = json.Unmarshal(encoded, &evidence)
			}
		}
		c.open = append(c.open, 2+len(evidence))
	case statement == "rollback":
		c.open = nil
	case statement == "commit":
		c.mu.Unlock()
		return context.WithValue(ctx, intakeStatementKey{}, intakeStatement{commit: true})
	}
	c.mu.Unlock()
	return ctx
}

func (c *intakeTransactions) TraceQueryEnd(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryEndData) {
	statement, _ := ctx.Value(intakeStatementKey{}).(intakeStatement)
	c.mu.Lock()
	switch {
	case statement.auditLock != 0:
		hook := c.afterAuditLock
		c.mu.Unlock()
		if hook != nil && data.Err == nil {
			hook(statement.auditLock)
		}
		return
	case statement.fence != 0:
		c.fences[statement.fence-1] = data.Err
		hook := c.afterFence
		c.mu.Unlock()
		if hook != nil {
			hook(statement.fence, data.Err)
		}
		return
	case statement.commit && data.Err == nil:
		c.committed = append(c.committed, c.open)
		c.open = nil
	}
	c.mu.Unlock()
}

// requireAttempts fails unless the actor started begins transactions and its
// fences had the given outcomes: true for a definite transaction conflict,
// false for a granted fence.
func (c *intakeTransactions) requireAttempts(t *testing.T, begins int, conflicts ...bool) {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	outcomes := make([]bool, len(c.fences))
	for index, err := range c.fences {
		if err != nil && persistencepostgres.Classify(err) != persistencepostgres.ClassTransactionConflict {
			t.Fatalf("fence %d failed with %v", index+1, err)
		}
		outcomes[index] = err != nil
	}
	if c.begins != begins || !slices.Equal(outcomes, conflicts) {
		t.Fatalf("%d transaction(s) with fence conflicts %v, want %d with %v", c.begins, outcomes, begins, conflicts)
	}
}

// requireCommittedHolds fails unless the committed transactions created holds
// in these counts and none started a proposal at or above the work threshold:
// only its last proposal may cross it.
func (c *intakeTransactions) requireCommittedHolds(t *testing.T, holds ...int) {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	counts := make([]int, len(c.committed))
	for index, weights := range c.committed {
		counts[index] = len(weights)
		work := 0
		for _, weight := range weights {
			if work >= auditdomain.MaximumCollectionTransactionWork {
				t.Fatalf("transaction %d started a proposal at work %d: %v", index+1, work, weights)
			}
			work += weight
		}
	}
	if !slices.Equal(counts, holds) {
		t.Fatalf("committed transactions created %v holds, want %v", counts, holds)
	}
}

// lostCommitResponse sits between an actor pool and PostgreSQL. Once armed,
// it lets the next COMMIT reach the server, waits until the server finished
// it and then fails the connection: the commit is durable and the client
// cannot know it.
type lostCommitResponse struct {
	armed atomic.Bool
	fired atomic.Int32
}

var errLostCommitResponse = fmt.Errorf("injected lost COMMIT response: %w", io.ErrUnexpectedEOF)

// simpleQueryCommitMessage is the protocol message pgx sends for Tx.Commit.
var simpleQueryCommitMessage = func() []byte {
	body := []byte("commit\x00")
	message := []byte{'Q', 0, 0, 0, 0}
	binary.BigEndian.PutUint32(message[1:], uint32(4+len(body)))
	return append(message, body...)
}()

func (l *lostCommitResponse) dial(ctx context.Context, network, address string) (net.Conn, error) {
	var dialer net.Dialer
	conn, err := dialer.DialContext(ctx, network, address)
	if err != nil {
		return nil, err
	}
	return &lostCommitConn{Conn: conn, fault: l}, nil
}

type lostCommitConn struct {
	net.Conn
	fault *lostCommitResponse
	lost  atomic.Bool
	once  sync.Once
}

func (c *lostCommitConn) Write(p []byte) (int, error) {
	if c.lost.Load() {
		return len(p), nil // the client side is gone; nothing more reaches the server
	}
	if bytes.Contains(p, simpleQueryCommitMessage) && c.fault.armed.CompareAndSwap(true, false) {
		if _, err := c.Conn.Write(p); err != nil {
			return 0, err
		}
		c.lost.Store(true)
		c.fault.fired.Add(1)
		return len(p), nil
	}
	return c.Conn.Write(p)
}

func (c *lostCommitConn) Read(p []byte) (int, error) {
	if !c.lost.Load() {
		return c.Conn.Read(p)
	}
	c.once.Do(func() {
		_ = readUntilReadyForQuery(c.Conn)
		_ = c.Conn.Close()
	})
	return 0, errLostCommitResponse
}

func (c *lostCommitConn) SetDeadline(deadline time.Time) error {
	if c.lost.Load() {
		return nil
	}
	return c.Conn.SetDeadline(deadline)
}

func (c *lostCommitConn) SetReadDeadline(deadline time.Time) error {
	if c.lost.Load() {
		return nil
	}
	return c.Conn.SetReadDeadline(deadline)
}

func (c *lostCommitConn) SetWriteDeadline(deadline time.Time) error {
	if c.lost.Load() {
		return nil
	}
	return c.Conn.SetWriteDeadline(deadline)
}

// readUntilReadyForQuery reads server messages through ReadyForQuery.
func readUntilReadyForQuery(conn net.Conn) error {
	header := make([]byte, 5)
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
		if header[0] == 'Z' {
			return nil
		}
	}
}

func intakeOnLostCommitPool(
	t *testing.T, f deletionImportFixture, trace pgx.QueryTracer, fault *lostCommitResponse,
) *Service {
	t.Helper()
	configuration := f.pool.Config()
	configuration.MaxConns, configuration.MinConns = 1, 0
	configuration.ConnConfig.Tracer = trace
	configuration.ConnConfig.DialFunc = fault.dial
	pool, err := pgxpool.NewWithConfig(f.ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	intake, err := New(pool)
	if err != nil {
		t.Fatal(err)
	}
	return intake
}
