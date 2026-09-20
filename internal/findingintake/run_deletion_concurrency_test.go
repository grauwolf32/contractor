//go:build integration

package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresFindingImportSerializesWithRunDeletion(t *testing.T) {
	t.Run("import commits before waiting deletion", func(t *testing.T) {
		f := newDeletionImportFixture(t)
		gate := &importRunLockGate{held: make(chan struct{}), resume: make(chan struct{})}
		defer gate.release()
		importPool, importPID := deletionActorPool(t, f, gate)
		deletePool, deletePID := deletionActorPool(t, f, nil)
		intake, err := New(importPool)
		if err != nil {
			t.Fatal(err)
		}
		imported := make(chan error, 1)
		go func() {
			_, _, err := intake.ImportIntoAudit(f.ctx, f.request)
			imported <- err
		}()
		select {
		case <-gate.held:
		case err := <-imported:
			t.Fatalf("import completed without the source Run lock: %v", err)
		case <-f.ctx.Done():
			t.Fatal("import did not acquire the source Run lock")
		}
		deleted := make(chan error, 1)
		go func() {
			deleted <- runstore.NewPostgresStore(deletePool).DeleteReleasedTerminalRun(f.ctx, f.request.OwnerID, f.request.RunID)
		}()
		awaitDeletionDatabaseWait(t, f, deletePID, importPID, "FROM workflow_runs")
		gate.release()
		awaitDeletionOperation(t, f.ctx, imported)
		awaitDeletionOperation(t, f.ctx, deleted)
		receipt, err := f.intake.GetAuditReceipt(f.ctx, f.request.OwnerID, f.request.AuditID, f.receiptID)
		if err != nil || !receipt.Origin.RunDeleted || receipt.Retention != RetentionAuditHeld ||
			len(receipt.AuditHolds) != 1 || receipt.Document.ClientKey != "candidate" || !sameRef(receipt.Proposal.Ref, f.request.Proposal) {
			t.Fatalf("retained import after deletion = %+v, %v", receipt, err)
		}
		audit, err := auditstore.NewPostgresStore(f.pool).Get(f.ctx, f.request.OwnerID, f.request.AuditID)
		if err != nil || audit.Revision != 3 {
			t.Fatalf("draft + import + deletion revision = %d, %v", audit.Revision, err)
		}
	})

	for _, outcome := range []string{"commit", "rollback", "cancel waiter then retry"} {
		t.Run("deletion holds Run/"+outcome, func(t *testing.T) {
			f := newDeletionImportFixture(t)
			tx, err := f.pool.Begin(f.ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = tx.Rollback(context.Background()) }()
			if err := runstore.NewPostgresStore(tx).DeleteReleasedTerminalRun(f.ctx, f.request.OwnerID, f.request.RunID); err != nil {
				t.Fatal(err)
			}
			importPool, importPID := deletionActorPool(t, f, nil)
			intake, err := New(importPool)
			if err != nil {
				t.Fatal(err)
			}
			importCtx, cancel := context.WithCancel(f.ctx)
			defer cancel()
			imported := make(chan error, 1)
			go func() {
				_, _, err := intake.ImportIntoAudit(importCtx, f.request)
				imported <- err
			}()
			awaitDeletionDatabaseWait(t, f, importPID, tx.Conn().PgConn().PID(), "FOR KEY SHARE")
			if outcome == "cancel waiter then retry" {
				cancel()
				if err := receiveDeletionOperation(t, f.ctx, imported); !errors.Is(err, context.Canceled) {
					t.Fatalf("cancelled import = %v", err)
				}
			}
			if outcome == "commit" {
				if err := tx.Commit(f.ctx); err != nil {
					t.Fatal(err)
				}
				if err := receiveDeletionOperation(t, f.ctx, imported); !errors.Is(err, ErrNotFound) {
					t.Fatalf("import after committed deletion = %v", err)
				}
				var holds int
				if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM finding_proposal_audit_holds`).Scan(&holds); err != nil || holds != 0 {
					t.Fatalf("late import created %d holds: %v", holds, err)
				}
				audit, err := auditstore.NewPostgresStore(f.pool).Get(f.ctx, f.request.OwnerID, f.request.AuditID)
				if err != nil || audit.Revision != 1 {
					t.Fatalf("unaffected Audit revision = %d, %v", audit.Revision, err)
				}
				return
			}
			if err := tx.Rollback(f.ctx); err != nil {
				t.Fatal(err)
			}
			if outcome == "cancel waiter then retry" {
				if _, _, err := intake.ImportIntoAudit(f.ctx, f.request); err != nil {
					t.Fatalf("retry after cancellation and rollback = %v", err)
				}
			} else {
				awaitDeletionOperation(t, f.ctx, imported)
			}
			receipt, err := intake.GetAuditReceipt(f.ctx, f.request.OwnerID, f.request.AuditID, f.receiptID)
			if err != nil || receipt.Origin.RunDeleted || receipt.Retention != RetentionAuditHeld || len(receipt.AuditHolds) != 1 {
				t.Fatalf("import after rollback = %+v, %v", receipt, err)
			}
			audit, err := auditstore.NewPostgresStore(f.pool).Get(f.ctx, f.request.OwnerID, f.request.AuditID)
			if err != nil || audit.Revision != 2 {
				t.Fatalf("rollback retained a deletion revision: %d, %v", audit.Revision, err)
			}
		})
	}
}

func TestPostgresFindingRunDeletionSerializesWithAuditPurge(t *testing.T) {
	for _, first := range []string{"run deletion", "Audit purge"} {
		t.Run(first+" commits first", func(t *testing.T) {
			f := newDeletionImportFixture(t)
			if _, _, err := f.intake.ImportIntoAudit(f.ctx, f.request); err != nil {
				t.Fatal(err)
			}
			audits := auditstore.NewPostgresStore(f.pool)
			if _, _, err := audits.RequestDelete(f.ctx, auditstore.DeleteParams{
				OwnerID: f.request.OwnerID, AuditID: f.request.AuditID, ExpectedRevision: 2,
				IdempotencyKey: "delete-audit", RequestDigest: digestBytes([]byte("delete-audit")),
			}); err != nil {
				t.Fatal(err)
			}
			claims, err := audits.Claim(f.ctx, auditstore.ClaimParams{HolderID: "delete-controller", Lease: time.Minute, Limit: 1})
			if err != nil || len(claims) != 1 {
				t.Fatalf("claims = %+v, %v", claims, err)
			}
			claim := claims[0]
			namespace := auditdomain.ArtifactNamespace(f.request.AuditID)
			tx, err := f.pool.Begin(f.ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = tx.Rollback(context.Background()) }()
			actorPool, actorPID := deletionActorPool(t, f, nil)
			finished := make(chan error, 1)
			if first == "run deletion" {
				if err := runstore.NewPostgresStore(tx).DeleteReleasedTerminalRun(f.ctx, f.request.OwnerID, f.request.RunID); err != nil {
					t.Fatal(err)
				}
				go func() { finished <- auditstore.NewPostgresStore(actorPool).PurgeClaimed(f.ctx, claim, namespace) }()
			} else {
				if err := auditstore.NewPostgresStore(tx).PurgeClaimed(f.ctx, claim, namespace); err != nil {
					t.Fatal(err)
				}
				go func() {
					finished <- runstore.NewPostgresStore(actorPool).DeleteReleasedTerminalRun(f.ctx, f.request.OwnerID, f.request.RunID)
				}()
			}
			waitingQuery := "ORDER BY audit.audit_id"
			if first == "run deletion" {
				waitingQuery = "FOR UPDATE OF audit, claim"
			}
			// In particular, deletion must wait on the Audit before taking any
			// retention/artifact locks held by the purger. A later Audit UPDATE
			// after those locks would preserve outcomes here but invert the order.
			awaitDeletionDatabaseWait(t, f, actorPID, tx.Conn().PgConn().PID(), waitingQuery)
			if err := tx.Commit(f.ctx); err != nil {
				t.Fatal(err)
			}
			awaitDeletionOperation(t, f.ctx, finished)
			if _, err := audits.Get(f.ctx, f.request.OwnerID, f.request.AuditID); !errors.Is(err, auditstore.ErrNotFound) {
				t.Fatalf("Audit survived purge: %v", err)
			}
			if _, err := runstore.NewPostgresStore(f.pool).GetRun(f.ctx, f.request.RunID); !errors.Is(err, runstore.ErrNotFound) {
				t.Fatalf("source Run survived deletion: %v", err)
			}
			receipt, err := readReceiptByID(f.ctx, f.pool, f.receiptID)
			if err != nil || !receipt.Origin.RunDeleted || receipt.Retention != RetentionDiscarded || len(receipt.AuditHolds) != 0 {
				t.Fatalf("retention after both purges = %+v, %v", receipt, err)
			}
		})
	}
}

type deletionImportFixture struct {
	ctx       context.Context
	pool      *pgxpool.Pool
	intake    *Service
	request   ImportRequest
	receiptID string
}

func newDeletionImportFixture(t *testing.T) deletionImportFixture {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	t.Cleanup(cancel)
	pool := isolatedFindingPool(t, ctx)
	const owner, projectID, runID, auditID, receiptID = "delete-owner", "delete-project", "delete-run", "delete-audit", "delete-receipt"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: owner, Kind: projectstore.KindProject, Name: "Deletion fixture",
		IdempotencyKey: "project", RequestDigest: digestBytes([]byte("project")),
	}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := auditstore.NewPostgresStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: owner, ProjectID: projectID,
		Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: digestBytes([]byte("profile"))},
		ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`), InputSelection: json.RawMessage(`{}`),
		Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 100, MaxItemsTotal: 100,
			MaxSubmittedRuns: 100, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20},
		IdempotencyKey: "audit", RequestDigest: digestBytes([]byte("audit")),
	}); err != nil {
		t.Fatal(err)
	}
	snapshot, err := workflowconfig.Load("../config/testdata/valid", workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	project := projectID
	runs := runstore.NewPostgresStore(pool)
	if _, err := runs.CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: owner, ProjectID: &project, WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: encoded, Parameters: map[string]string{}, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	canonical, err := canonicalize(testSubmission("invocation", "candidate", []contracts.ArtifactRef{}))
	if err != nil {
		t.Fatal(err)
	}
	store := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	written, err := store.WriteFindingProposal(ctx, runID, "proposal", artifacts.Payload{MediaType: proposalMediaType, Data: canonical.proposalBytes})
	if err != nil {
		t.Fatal(err)
	}
	proposal := ExactArtifact{Ref: written.Ref, Digest: digestBytes(canonical.proposalBytes), MediaType: written.MediaType, SizeBytes: written.Size}
	origin := Origin{RunID: runID, Workflow: WorkflowOrigin{Name: workflow.Ref.Name, Version: workflow.Ref.Version,
		SchemaVersion: contracts.APIVersion, ClosureDigest: digestBytes(encoded)}}
	origin.Workflow.ConfigurationRef.Name, origin.Workflow.ConfigurationRef.Version = workflow.Ref.Name, workflow.Ref.Version
	grant := controlplane.AllocationGrant{RunID: runID, AllocationID: "allocation", StageExecutionID: "stage",
		RuntimeAgentID: "runtime", RuntimeInstanceID: "instance", LogicalAgentName: "worker"}
	if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		return insertReceipt(ctx, tx, receiptID, "proposal", canonical, proposal, []ExactArtifact{}, origin, owner, &project, grant)
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, runID, runstore.RunInitializing, runstore.RunFailed, runstore.Reason{Code: "fixture_done"}); err != nil {
		t.Fatal(err)
	}
	intake, err := New(pool)
	if err != nil {
		t.Fatal(err)
	}
	return deletionImportFixture{ctx: ctx, pool: pool, intake: intake, receiptID: receiptID,
		request: ImportRequest{OwnerID: owner, AuditID: auditID, RunID: runID, Proposal: proposal.Ref}}
}

// QueryEnd runs while the import transaction still owns its Run lock. Holding
// this barrier cannot change the database outcome; it exposes the wait order.
type importRunLockGate struct {
	held, resume chan struct{}
	once         sync.Once
	releaseOnce  sync.Once
}

type importRunLockContextKey struct{}

func (*importRunLockGate) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	return context.WithValue(ctx, importRunLockContextKey{}, strings.Contains(data.SQL, "FROM workflow_runs") && strings.Contains(data.SQL, "FOR KEY SHARE"))
}

func (gate *importRunLockGate) TraceQueryEnd(ctx context.Context, _ *pgx.Conn, _ pgx.TraceQueryEndData) {
	if matched, _ := ctx.Value(importRunLockContextKey{}).(bool); matched {
		gate.once.Do(func() {
			close(gate.held)
			select {
			case <-gate.resume:
			case <-ctx.Done():
			}
		})
	}
}

func (gate *importRunLockGate) release() { gate.releaseOnce.Do(func() { close(gate.resume) }) }

func deletionActorPool(t *testing.T, fixture deletionImportFixture, trace pgx.QueryTracer) (*pgxpool.Pool, uint32) {
	t.Helper()
	configuration := fixture.pool.Config()
	configuration.MaxConns, configuration.MinConns = 1, 0
	configuration.ConnConfig.Tracer = trace
	pool, err := pgxpool.NewWithConfig(fixture.ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	if gate, ok := trace.(*importRunLockGate); ok {
		t.Cleanup(gate.release)
	}
	connection, err := pool.Acquire(fixture.ctx)
	if err != nil {
		t.Fatal(err)
	}
	pid := connection.Conn().PgConn().PID()
	connection.Release()
	return pool, pid
}

func awaitDeletionDatabaseWait(t *testing.T, fixture deletionImportFixture, waiter, holder uint32, expectedQuery string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(fixture.ctx, 5*time.Second)
	defer cancel()
	for {
		var blocked bool
		var query string
		if err := fixture.pool.QueryRow(ctx, `
SELECT $2::integer = ANY(pg_blocking_pids($1::integer)), query
  FROM pg_stat_activity WHERE pid = $1`, waiter, holder).Scan(&blocked, &query); err != nil {
			t.Fatalf("backend %d never waited on backend %d: %v", waiter, holder, err)
		}
		if blocked {
			if !strings.Contains(query, expectedQuery) {
				t.Fatalf("backend %d waited at the wrong lock stage; expected query containing %q, got %s", waiter, expectedQuery, query)
			}
			return
		}
	}
}

func receiveDeletionOperation(t *testing.T, ctx context.Context, finished <-chan error) error {
	t.Helper()
	select {
	case err := <-finished:
		return err
	case <-ctx.Done():
		t.Fatal("concurrent operation did not finish:", ctx.Err())
		return ctx.Err()
	}
}

func awaitDeletionOperation(t *testing.T, ctx context.Context, finished <-chan error) {
	t.Helper()
	if err := receiveDeletionOperation(t, ctx, finished); err != nil {
		t.Fatal(err)
	}
}
