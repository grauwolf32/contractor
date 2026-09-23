//go:build integration

package auditimport

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestImporterReportFinalizationSurvivesRunDeletion(t *testing.T) {
	pool := completionPool(t)
	for _, acceptance := range []string{"automatic", "human-required"} {
		for _, point := range []string{"before-machine", "after-machine", "after-summary"} {
			t.Run(acceptance+"/"+point, func(t *testing.T) {
				f, snapshot := newReportDeletionFixture(t, pool, acceptance)
				access, err := NewArtifactAccess(f.artifacts)
				mustCompletion(t, err)
				deleted := false
				hooked := &reportDeletionArtifacts{ArtifactAccess: access, hook: func(observed string) {
					if observed == point && !deleted {
						mustCompletion(t, f.runs.DeleteReleasedTerminalRun(f.ctx, "owner", f.run.RunID))
						deleted = true
					}
				}}
				importer, err := New(f.audits, f.runs, hooked)
				mustCompletion(t, err)
				worked, err := importer.Finalize(f.ctx, f.claim, snapshot)
				if !deleted || worked || !errors.Is(err, auditstore.ErrPrecondition) || len(hooked.writes) != 2 {
					t.Fatalf("stale finalization: deleted=%t worked=%t writes=%d error=%v", deleted, worked, len(hooked.writes), err)
				}
				assertNoAcceptedReport(t, f)
				if _, err := f.audits.GetReportCandidate(f.ctx, f.id); !errors.Is(err, auditstore.ErrNotFound) {
					t.Fatalf("stale finalization froze a candidate: %v", err)
				}
				fresh, err := f.audits.GetReconcileSnapshot(f.ctx, f.claim)
				mustCompletion(t, err)
				if fresh.Audit.Revision != snapshot.Audit.Revision+1 || !fresh.Audit.UpdatedAt.Equal(snapshot.Audit.UpdatedAt) {
					t.Fatal("deletion must invalidate the projection revision without changing the finalizing report timestamp")
				}
				if _, err := f.runs.GetRun(f.ctx, f.run.RunID); !errors.Is(err, runstore.ErrNotFound) {
					t.Fatalf("source Run was not deleted: %v", err)
				}
				// Reconstruct the importer as after a process restart. Artifact writes
				// and report CAS remain real; the retry may only reuse identical bytes.
				importer, err = New(auditstore.NewPostgresStore(pool), runstore.NewPostgresStore(pool), access)
				mustCompletion(t, err)
				worked, err = importer.Finalize(f.ctx, f.claim, fresh)
				if err != nil || !worked {
					t.Fatalf("fresh finalization retry: worked=%t error=%v", worked, err)
				}
				var links []auditstore.ArtifactLink
				if acceptance == "human-required" {
					candidate, err := f.audits.GetReportCandidate(f.ctx, f.id)
					mustCompletion(t, err)
					links = []auditstore.ArtifactLink{candidate.Machine, candidate.Summary}
					assertNoAcceptedReport(t, f)
				} else {
					for _, key := range []string{auditstore.ReportMachineLogicalKey, auditstore.ReportSummaryLogicalKey} {
						link, err := f.audits.GetArtifactLink(f.ctx, f.id, key)
						mustCompletion(t, err)
						links = append(links, link)
					}
				}
				for index, link := range links {
					if !reflect.DeepEqual(link.Artifact, hooked.writes[index].artifact) {
						t.Fatal("retry replaced an immutable report descriptor")
					}
					data, err := access.ReadProjectExact(f.ctx, f.id, link.Artifact)
					mustCompletion(t, err)
					if !bytes.Equal(data, hooked.writes[index].data) {
						t.Fatal("retry changed immutable report bytes")
					}
				}
				var report machineReport
				mustCompletion(t, json.Unmarshal(hooked.writes[0].data, &report))
				if !report.GeneratedFrom.Equal(snapshot.Audit.UpdatedAt) || report.AttemptDispositions.ExecutionFailed != 1 || len(report.Items) != 2 {
					t.Fatal("retained report lost its timestamp, receipt or item facts after source deletion")
				}
			})
		}
	}
}

func TestImporterPendingReportReviewSurvivesRunDeletion(t *testing.T) {
	pool := completionPool(t)
	for _, action := range []auditservice.ReviewAction{auditservice.ReviewApprove, auditservice.ReviewReject} {
		t.Run(string(action), func(t *testing.T) {
			f, snapshot := newReportDeletionFixture(t, pool, "human-required")
			access, err := NewArtifactAccess(f.artifacts)
			mustCompletion(t, err)
			importer, err := New(f.audits, f.runs, access)
			mustCompletion(t, err)
			if worked, err := importer.Finalize(f.ctx, f.claim, snapshot); err != nil || !worked {
				t.Fatalf("initial report proposal: worked=%t error=%v", worked, err)
			}
			candidate, err := f.audits.GetReportCandidate(f.ctx, f.id)
			mustCompletion(t, err)
			before, err := f.audits.Get(f.ctx, "owner", f.id)
			mustCompletion(t, err)
			service := reportDeletionReviewService(t, f)
			review, err := service.GetReview(f.ctx, "owner", f.id, candidate.RequestID)
			mustCompletion(t, err)
			mustCompletion(t, f.runs.DeleteReleasedTerminalRun(f.ctx, "owner", f.run.RunID))
			after, err := f.audits.Get(f.ctx, "owner", f.id)
			mustCompletion(t, err)
			if after.State != auditstore.AuditWaitingReview || after.Revision != before.Revision+1 || !after.UpdatedAt.After(before.UpdatedAt) {
				t.Fatal("pending report deletion did not advance visible Audit revision and timestamp")
			}
			stored, err := f.audits.GetReportCandidate(f.ctx, f.id)
			mustCompletion(t, err)
			if !reflect.DeepEqual(stored, candidate) {
				t.Fatal("source deletion changed the frozen review subject")
			}
			// Lost proposal response: replay the original exact request after the
			// source deletion, including its older Audit subject revision.
			if worked, err := importer.Finalize(f.ctx, f.claim, snapshot); err != nil || !worked {
				t.Fatalf("report proposal replay after deletion: worked=%t error=%v", worked, err)
			}
			params := auditservice.DecideActionReviewParams{
				OwnerID: "owner", AuditID: f.id, RequestID: candidate.RequestID,
				ExpectedRequestRevision: review.Revision, DecisionID: f.id + "-decision",
				Action: action, Rationale: "Review the exact retained report after source deletion.",
				IdempotencyKey: f.id + "-decision", RequestDigest: auditdomain.DigestBytes([]byte(f.id + string(action))),
			}
			decision, err := service.DecideActionReview(f.ctx, params)
			mustCompletion(t, err)
			if decision.Replayed || decision.Decision.SubjectRevision != candidate.SubjectRevision || decision.Decision.SubjectDigest != candidate.SubjectDigest {
				t.Fatal("review decision lost exact candidate authority")
			}
			replayed, err := service.DecideActionReview(f.ctx, params)
			mustCompletion(t, err)
			if !replayed.Replayed || !reflect.DeepEqual(replayed.Decision, decision.Decision) {
				t.Fatal("review replay produced a different decision")
			}
			terminal, err := f.audits.Get(f.ctx, "owner", f.id)
			mustCompletion(t, err)
			if action == auditservice.ReviewApprove {
				if terminal.State != auditstore.AuditCompleted {
					t.Fatal("approved report did not complete")
				}
				for _, expected := range []auditstore.ArtifactLink{candidate.Machine, candidate.Summary} {
					link, err := f.audits.GetArtifactLink(f.ctx, f.id, expected.LogicalKey)
					mustCompletion(t, err)
					if !reflect.DeepEqual(link.Artifact, expected.Artifact) {
						t.Fatal("approval did not publish the exact frozen artifact")
					}
					_, err = access.ReadProjectExact(f.ctx, f.id, link.Artifact)
					mustCompletion(t, err)
				}
			} else {
				if terminal.State != auditstore.AuditFailed || terminal.StopReason == nil || terminal.StopReason.Code != "report_rejected" {
					t.Fatal("rejected report did not fail with the expected reason")
				}
				assertNoAcceptedReport(t, f)
			}
		})
	}
}

func TestImporterReportRetryRejectsChangedFinalizationTimestamp(t *testing.T) {
	pool := completionPool(t)
	f, snapshot := newReportDeletionFixture(t, pool, "automatic")
	access, err := NewArtifactAccess(f.artifacts)
	mustCompletion(t, err)
	hooked := &reportDeletionArtifacts{ArtifactAccess: access, hook: func(point string) {
		if point == "after-machine" {
			mustCompletion(t, f.runs.DeleteReleasedTerminalRun(f.ctx, "owner", f.run.RunID))
			// Counterexample: simulate a naive deletion timestamp update after
			// report.json has been durably written. Immutable collision stays fatal.
			_, err := pool.Exec(f.ctx, `UPDATE audits SET updated_at = updated_at + interval '1 second' WHERE audit_id = $1`, f.id)
			mustCompletion(t, err)
		}
	}}
	importer, err := New(f.audits, f.runs, hooked)
	mustCompletion(t, err)
	if worked, err := importer.Finalize(f.ctx, f.claim, snapshot); worked || !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("stale finalization: worked=%t error=%v", worked, err)
	}
	fresh, err := f.audits.GetReconcileSnapshot(f.ctx, f.claim)
	mustCompletion(t, err)
	importer, err = New(f.audits, f.runs, access)
	mustCompletion(t, err)
	if worked, err := importer.Finalize(f.ctx, f.claim, fresh); worked || !errors.Is(err, artifacts.ErrArtifactIntegrity) {
		t.Fatalf("timestamp mutation must not overwrite immutable report: worked=%t error=%v", worked, err)
	}
	assertNoAcceptedReport(t, f)
}

func TestImporterManagedRunDeletionRollbackAndRejection(t *testing.T) {
	pool := completionPool(t)
	t.Run("rollback-and-repeat", func(t *testing.T) {
		f, snapshot := newReportDeletionFixture(t, pool, "automatic")
		tx, err := pool.Begin(f.ctx)
		mustCompletion(t, err)
		defer tx.Rollback(context.Background())
		mustCompletion(t, runstore.NewPostgresStore(tx).DeleteReleasedTerminalRun(f.ctx, "owner", f.run.RunID))
		changed, err := auditstore.NewPostgresStore(tx).Get(f.ctx, "owner", f.id)
		mustCompletion(t, err)
		if changed.Revision != snapshot.Audit.Revision+1 || !changed.UpdatedAt.Equal(snapshot.Audit.UpdatedAt) {
			t.Fatal("transaction-backed deletion did not invalidate exactly one managed Audit")
		}
		if _, err := runstore.NewPostgresStore(tx).GetRun(f.ctx, f.run.RunID); !errors.Is(err, runstore.ErrNotFound) {
			t.Fatalf("transaction-backed deletion retained its source Run: %v", err)
		}
		mustCompletion(t, tx.Rollback(f.ctx))
		assertReportDeletionSourceIntact(t, f, snapshot.Audit)

		mustCompletion(t, f.runs.DeleteReleasedTerminalRun(f.ctx, "owner", f.run.RunID))
		changed, err = f.audits.Get(f.ctx, "owner", f.id)
		mustCompletion(t, err)
		if changed.Revision != snapshot.Audit.Revision+1 {
			t.Fatal("rolled-back deletion consumed a durable Audit revision")
		}
		if err := f.runs.DeleteReleasedTerminalRun(f.ctx, "owner", f.run.RunID); !errors.Is(err, runstore.ErrNotFound) {
			t.Fatalf("repeated source deletion: %v", err)
		}
		repeated, err := f.audits.Get(f.ctx, "owner", f.id)
		mustCompletion(t, err)
		if !reflect.DeepEqual(repeated, changed) {
			t.Fatal("repeated source deletion mutated the Audit again")
		}
	})
	t.Run("uncollected-terminal-rejection", func(t *testing.T) {
		f := newCompletionFixture(t, pool)
		_, err := f.runs.TransitionRun(f.ctx, f.run.RunID, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "fixture-terminal"})
		mustCompletion(t, err)
		before, err := f.audits.Get(f.ctx, "owner", f.id)
		mustCompletion(t, err)
		err = f.runs.DeleteReleasedTerminalRun(f.ctx, "owner", f.run.RunID)
		var blocked *runstore.RunNotDeletableError
		if !errors.As(err, &blocked) || blocked.Reason != runstore.RunAuditCollectionPending {
			t.Fatalf("uncollected source deletion: %v", err)
		}
		assertReportDeletionSourceIntact(t, f, before)
		var receipts int
		mustCompletion(t, pool.QueryRow(f.ctx, `SELECT count(*) FROM audit_collection_receipts WHERE audit_id = $1`, f.id).Scan(&receipts))
		if receipts != 0 {
			t.Fatal("rejected source deletion invented a collection receipt")
		}
	})
}

func assertReportDeletionSourceIntact(t *testing.T, f *completionFixture, before auditstore.Audit) {
	t.Helper()
	unchanged, err := f.audits.Get(f.ctx, "owner", f.id)
	mustCompletion(t, err)
	if !reflect.DeepEqual(unchanged, before) {
		t.Fatal("rolled-back or rejected deletion mutated the Audit")
	}
	run, err := f.runs.GetRun(f.ctx, f.run.RunID)
	mustCompletion(t, err)
	if run.State != runstore.RunFailed {
		t.Fatal("rolled-back or rejected deletion changed the source Run state")
	}
	var deleted bool
	mustCompletion(t, f.pool.QueryRow(f.ctx, `SELECT run_deleted_at IS NOT NULL FROM audit_executions WHERE execution_id = $1`, f.execution.ExecutionID).Scan(&deleted))
	if deleted {
		t.Fatal("rolled-back or rejected deletion left an execution tombstone")
	}
	store, err := f.artifacts.Run(f.run.RunID)
	mustCompletion(t, err)
	_, err = store.Read(f.ctx, f.run.AuditCompletion.Contract.Task)
	mustCompletion(t, err)
}

func newReportDeletionFixture(t *testing.T, pool *pgxpool.Pool, acceptance string) (*completionFixture, auditstore.ReconcileSnapshot) {
	t.Helper()
	f := newCompletionFixtureWithReportAcceptance(t, pool, acceptance)
	// No Runtime/model process is needed: a failed child contributes a real
	// collected receipt, then normal closure settles its undispatched retry.
	f.collect(t, runstore.RunFailed, false)
	audit, err := f.audits.Get(f.ctx, "owner", f.id)
	mustCompletion(t, err)
	_, err = f.audits.TransitionClaimed(f.ctx, auditstore.ClaimedTransitionParams{
		Claim: f.claim, ExpectedRevision: audit.Revision, ExpectedState: auditstore.AuditActive,
		TargetState: auditstore.AuditFinalizing,
	})
	mustCompletion(t, err)
	settled, err := f.audits.SettleUndispatched(f.ctx, f.claim, auditstore.MaxReconcileRows)
	mustCompletion(t, err)
	if settled != 2 {
		t.Fatalf("undispatched retry settlement count = %d", settled)
	}
	round, err := f.audits.GetRound(f.ctx, f.id, f.id)
	mustCompletion(t, err)
	_, err = f.audits.TransitionRound(f.ctx, auditstore.RoundTransitionParams{
		Claim: f.claim, RoundID: round.RoundID, ExpectedRevision: round.Revision,
		ExpectedState: round.State, TargetState: auditstore.RoundClosed,
	})
	mustCompletion(t, err)
	_, _, err = f.audits.ReleaseDispatchHold(f.ctx, f.claim)
	mustCompletion(t, err)
	snapshot, err := f.audits.GetReconcileSnapshot(f.ctx, f.claim)
	mustCompletion(t, err)
	return f, snapshot
}

func reportDeletionReviewService(t *testing.T, f *completionFixture) *auditservice.Service {
	t.Helper()
	manager, err := config.NewManager(config.ManagerOptions{
		OperatorRoot: "../config/testdata/valid", ManagedRoot: filepath.Join(t.TempDir(), "managed"),
		Descriptors: config.MVPDescriptors(),
	})
	mustCompletion(t, err)
	service, err := auditservice.New(auditservice.Options{
		Pool: f.pool, Profiles: manager, CredentialGuard: completionCredentials{},
		TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (config.CredentialLookup, error) {
			return completionCredentials{}, nil
		}), Now: time.Now,
	})
	mustCompletion(t, err)
	return service
}

func assertNoAcceptedReport(t *testing.T, f *completionFixture) {
	t.Helper()
	for _, key := range []string{auditstore.ReportMachineLogicalKey, auditstore.ReportSummaryLogicalKey} {
		if _, err := f.audits.GetArtifactLink(f.ctx, f.id, key); !errors.Is(err, auditstore.ErrNotFound) {
			t.Fatalf("unexpected accepted report link %q: %v", key, err)
		}
	}
}

type reportDeletionArtifacts struct {
	ArtifactAccess
	hook   func(string)
	writes []struct {
		artifact auditstore.ExactArtifact
		data     []byte
	}
}

func (a *reportDeletionArtifacts) PutImmutableProject(ctx context.Context, projectID string, target contracts.ArtifactRef, payload artifacts.Payload) (auditstore.ExactArtifact, error) {
	phase := "machine"
	if payload.MediaType == "text/markdown" {
		phase = "summary"
	}
	a.hook("before-" + phase)
	written, err := a.ArtifactAccess.PutImmutableProject(ctx, projectID, target, payload)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	a.writes = append(a.writes, struct {
		artifact auditstore.ExactArtifact
		data     []byte
	}{written, append([]byte(nil), payload.Data...)})
	a.hook("after-" + phase)
	return written, nil
}
