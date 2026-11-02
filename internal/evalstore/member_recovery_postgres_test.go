package evalstore

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// statementRecorder records the SQL text of every statement a traced pool
// sends.
type statementRecorder struct {
	mu         sync.Mutex
	statements []string
}

func (r *statementRecorder) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.statements = append(r.statements, data.SQL)
	return ctx
}
func (*statementRecorder) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

// take returns the statements recorded since the last take.
func (r *statementRecorder) take() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := r.statements
	r.statements = nil
	return out
}

func tracedPool(t *testing.T, pool *pgxpool.Pool) (*pgxpool.Pool, *statementRecorder) {
	t.Helper()
	recorder := &statementRecorder{}
	config := pool.Config()
	config.ConnConfig.Tracer = recorder
	traced, err := pgxpool.NewWithConfig(t.Context(), config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(traced.Close)
	return traced, recorder
}

// recoveryAuthority counts the statements of a recovery transaction that
// take or prove the recovery authority: the evaluation Project lock, the
// experiment lock, the claim lock and the later steps' refresh.
type recoveryAuthority struct{ project, experiment, claim, refresh int }

func countAuthority(statements []string) recoveryAuthority {
	var out recoveryAuthority
	for _, statement := range statements {
		compact := strings.Join(strings.Fields(statement), " ")
		switch {
		case strings.Contains(compact, "FROM projects WHERE") && strings.HasSuffix(compact, "FOR SHARE"):
			out.project++
		case strings.Contains(compact, "FROM eval_experiments WHERE") && strings.HasSuffix(compact, "FOR UPDATE"):
			out.experiment++
		case strings.Contains(compact, "FROM eval_controller_claims AS claim") && strings.HasSuffix(compact, "FOR UPDATE OF claim"):
			out.claim++
		case strings.Contains(compact, "JOIN eval_controller_claims claim USING (experiment_id)"):
			out.refresh++
		}
	}
	return out
}

// recoveryFixture is an admitted external-workflow member whose run-create
// intent is durable, under the coordinator's claim.
type recoveryFixture struct {
	pool       *pgxpool.Pool
	scope      Scope
	experiment Experiment
	member     string
	claim      Claim
	operation  Suboperation
}

func newRecoveryFixture(t *testing.T) recoveryFixture {
	t.Helper()
	pool := testPool(t)
	ctx := t.Context()
	scope := setupProject(t, pool, "owner", "eval")
	experiment := createExperiment(t, pool, scope, "exp", "trace-1", "external-workflow")
	member := members(t, pool, experiment)[0]
	if _, err := admit(t, pool, experiment, member.MemberID, nil, "submit"); err != nil {
		t.Fatal(err)
	}
	claim := oneClaim(t, pool)
	operation := Suboperation{Kind: "run-create", Key: "run-create-key", Request: json.RawMessage(`{"workflow":"trace-a@1"}`)}
	mustTx(t, pool, func(s *Store) error {
		return s.RecoverMember(scope, experiment.ID, member.MemberID, claim).PutSuboperation(ctx, operation)
	})
	return recoveryFixture{pool, scope, experiment, member.MemberID, claim, operation}
}

// run inserts the ordinary Run that the member's run-create intent created,
// as the Run service's creation commit would, in state.
func (f recoveryFixture) run(t *testing.T, runID, state string) {
	t.Helper()
	finished := "NULL"
	if state == "succeeded" || state == "failed" || state == "cancelled" {
		finished = "clock_timestamp()"
	}
	if _, err := f.pool.Exec(t.Context(), `
INSERT INTO workflow_runs (run_id, owner_id, project_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_snapshot, runtime_labels, runtime_config_snapshot, scheduler_lane_class,
    state, state_reason_code, finished_at, request_idempotency_key, request_digest)
VALUES ($1, $2, $3, 'trace', '1', '1', '{}', '{}', $4, 'model', $5, 'fixture', `+finished+`, $6, $7)`,
		runID, f.scope.OwnerID, f.scope.ProjectID, evalTestRuntimeSnapshot, state, f.operation.Key, evaldomain.Digest(f.operation.Request)); err != nil {
		t.Fatal(err)
	}
}

func (f recoveryFixture) recover(s *Store) *MemberRecovery {
	return s.RecoverMember(f.scope, f.experiment.ID, f.member, f.claim)
}

// A composed recovery transaction takes the recovery authority once, in its
// first step, and every later step proves the live claim and rereads the
// experiment's fence in one statement (EVM-R246): the acknowledgement of a
// created Run (BindExecution with ResolveSuboperation) sends 9 statements
// instead of 11, the observation and settlement of the terminal Run
// (ObserveTokens with Settle) 13 instead of 15. Both commit the same outcome
// as separate steps did, and a replayed acknowledgement replays.
func TestSharedPostgresEvalComposedRecoveryTakesItsAuthorityOnce(t *testing.T) {
	f := newRecoveryFixture(t)
	ctx := t.Context()
	traced, recorder := tracedPool(t, f.pool)
	const runID = "run-composed-recovery"
	f.run(t, runID, "succeeded")
	measure := func(name string, want int, wantAuthority recoveryAuthority, steps func(*MemberRecovery) error) {
		t.Helper()
		var statements []string
		mustTx(t, traced, func(s *Store) error {
			recovery := f.recover(s)
			recorder.take()
			err := steps(recovery)
			statements = recorder.take()
			return err
		})
		if len(statements) != want || countAuthority(statements) != wantAuthority {
			t.Fatalf("%s sent %d statements with authority %+v, want %d with %+v:\n%s", name, len(statements),
				countAuthority(statements), want, wantAuthority, strings.Join(statements, "\n---\n"))
		}
		t.Logf("%s: %d statements, authority %+v", name, len(statements), wantAuthority)
	}
	measure("BindExecution+ResolveSuboperation", 9, recoveryAuthority{project: 1, experiment: 1, claim: 1, refresh: 1}, func(r *MemberRecovery) error {
		if err := r.BindExecution(ctx, runID); err != nil {
			return err
		}
		return r.ResolveSuboperation(ctx, f.operation.Kind, []byte(`{"runId":"`+runID+`"}`), false)
	})
	// A replay after a lost commit response takes the authority again and
	// changes nothing.
	measure("replayed BindExecution+ResolveSuboperation", 8, recoveryAuthority{project: 1, experiment: 1, claim: 1, refresh: 1}, func(r *MemberRecovery) error {
		if err := r.BindExecution(ctx, runID); err != nil {
			return err
		}
		return r.ResolveSuboperation(ctx, f.operation.Kind, []byte(`{"runId":"`+runID+`"}`), false)
	})
	measure("ObserveTokens+Settle", 13, recoveryAuthority{project: 1, experiment: 1, claim: 1, refresh: 1}, func(r *MemberRecovery) error {
		if err := r.ObserveTokens(ctx, 7); err != nil {
			return err
		}
		return r.Settle(ctx)
	})
	submission, err := NewPostgresStore(f.pool, artifactpgtest.NewStorage()).Submission(ctx, f.scope.OwnerID, f.experiment.ID, f.member)
	if err != nil || submission.State != SubmissionTerminal || submission.ExecutionID == nil || *submission.ExecutionID != runID {
		t.Fatalf("settled submission %+v: %v", submission, err)
	}
	operation, err := NewPostgresStore(f.pool, artifactpgtest.NewStorage()).Suboperation(ctx, f.scope.OwnerID, f.experiment.ID, f.member, f.operation.Kind)
	if err != nil || operation.State != "succeeded" {
		t.Fatalf("acknowledged operation %+v: %v", operation, err)
	}
	settled, err := NewPostgresStore(f.pool, artifactpgtest.NewStorage()).Get(ctx, f.scope.OwnerID, f.experiment.ID)
	if err != nil || settled.Outstanding != 0 || settled.ObservedTokens != 7 {
		t.Fatalf("settled experiment %+v: %v", settled, err)
	}
}

// Every later step of a composed recovery transaction keeps the original
// fences: it fails when the claim expired or passed to another holder, and
// it sees a cancellation or deletion request that an earlier statement of
// the transaction wrote. The first step's locks stay held until the
// transaction ends, so no other transaction can fence the experiment, lease
// the claim or start the Project's deletion in between. The capability
// belongs to its transaction and needs one.
func TestSharedPostgresEvalComposedRecoveryRechecksItsFences(t *testing.T) {
	f := newRecoveryFixture(t)
	ctx := t.Context()
	inRolledBackTx := func(name string, fn func(tx pgx.Tx, r *MemberRecovery) error) {
		t.Helper()
		tx, err := f.pool.Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = tx.Rollback(ctx) }()
		recovery := NewTxStore(tx, artifactpgtest.NewStorage()).RecoverMember(f.scope, f.experiment.ID, f.member, f.claim)
		if err = recovery.ObserveTokens(ctx, 1); err != nil {
			t.Fatalf("%s: first step: %v", name, err)
		}
		if err = fn(tx, recovery); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
	exec := func(tx pgx.Tx, statement string) error {
		_, err := tx.Exec(ctx, statement, f.experiment.ID)
		return err
	}
	inRolledBackTx("expired claim", func(tx pgx.Tx, r *MemberRecovery) error {
		if err := exec(tx, `UPDATE eval_controller_claims SET claimed_at=clock_timestamp()-interval '2 seconds',
    expires_at=clock_timestamp()-interval '1 second' WHERE experiment_id=$1`); err != nil {
			return err
		}
		if err := r.ResolveSuboperation(ctx, f.operation.Kind, []byte(`{"rejected":true}`), true); !errors.Is(err, ErrClaimLost) {
			return errors.Join(errors.New("a step under an expired claim succeeded"), err)
		}
		return nil
	})
	inRolledBackTx("claim of another holder", func(tx pgx.Tx, r *MemberRecovery) error {
		if err := exec(tx, `UPDATE eval_controller_claims SET holder_id='other',epoch=epoch+1 WHERE experiment_id=$1`); err != nil {
			return err
		}
		if err := r.Settle(ctx); !errors.Is(err, ErrClaimLost) {
			return errors.Join(errors.New("a step under a lost claim succeeded"), err)
		}
		return nil
	})
	for _, fence := range []struct{ name, statement string }{
		{"cancellation", `UPDATE eval_experiments SET state='cancelling',` + advance + ` WHERE experiment_id=$1`},
		{"deletion request", `UPDATE eval_experiments SET deletion_requested_at=clock_timestamp(),state='cancelling',` + advance + ` WHERE experiment_id=$1`},
	} {
		inRolledBackTx(fence.name, func(tx pgx.Tx, r *MemberRecovery) error {
			if err := exec(tx, fence.statement); err != nil {
				return err
			}
			if err := r.ResolveSuboperation(ctx, f.operation.Kind, []byte(`{"rejected":true}`), true); err != nil {
				return err
			}
			late := Suboperation{Kind: "audit-start", Key: "late-start", Request: json.RawMessage(`{}`)}
			if err := r.PutSuboperation(ctx, late); !evaldomain.IsCode(err, "eval_project_deleting") {
				return errors.Join(errors.New("a new side effect was recorded after the fence"), err)
			}
			cancel := Suboperation{Kind: "cancel", Key: "cancel-key", Request: json.RawMessage(`{}`)}
			return r.PutSuboperation(ctx, cancel)
		})
	}
	inRolledBackTx("locks held", func(_ pgx.Tx, r *MemberRecovery) error {
		other, err := f.pool.Begin(ctx)
		if err != nil {
			return err
		}
		defer func() { _ = other.Rollback(ctx) }()
		for _, statement := range []string{
			`SAVEPOINT probe`,
			`SELECT 1 FROM eval_experiments WHERE experiment_id=$1 FOR UPDATE NOWAIT`,
			`ROLLBACK TO SAVEPOINT probe`,
			`SELECT 1 FROM eval_controller_claims WHERE experiment_id=$1 FOR UPDATE NOWAIT`,
			`ROLLBACK TO SAVEPOINT probe`,
			`SELECT 1 FROM projects WHERE project_id=$1 FOR UPDATE NOWAIT`,
		} {
			argument := []any{f.experiment.ID}
			switch {
			case strings.Contains(statement, "SAVEPOINT"):
				argument = nil
			case strings.Contains(statement, "projects"):
				argument = []any{f.scope.ProjectID}
			}
			_, err = other.Exec(ctx, statement, argument...)
			if argument != nil && pg.Classify(err) != pg.ClassTimeout {
				return errors.Join(errors.New("another transaction took a lock of the recovery authority: "+statement), err)
			}
		}
		return r.ObserveTokens(ctx, 2)
	})
	inRolledBackTx("lock order", func(_ pgx.Tx, r *MemberRecovery) error {
		if err := r.LockMember(ctx); !errors.Is(err, errRecoveryLockOrder) {
			return errors.Join(errors.New("a Project lock followed the experiment lock"), err)
		}
		return nil
	})

	var committed *MemberRecovery
	mustTx(t, f.pool, func(s *Store) error {
		committed = f.recover(s)
		return committed.ObserveTokens(ctx, 2)
	})
	if err := committed.Settle(ctx); !errors.Is(err, pgx.ErrTxClosed) {
		t.Fatalf("a recovery step outlived its transaction: %v", err)
	}
	if err := f.recover(NewPostgresStore(f.pool, artifactpgtest.NewStorage())).ObserveTokens(ctx, 3); !errors.Is(err, ErrTransaction) {
		t.Fatalf("a recovery step ran without a transaction: %v", err)
	}
	observed, err := NewPostgresStore(f.pool, artifactpgtest.NewStorage()).Get(ctx, f.scope.OwnerID, f.experiment.ID)
	if err != nil || observed.ObservedTokens != 2 || observed.State == evaldomain.StateCancelling || observed.DeletionRequestedAt != nil {
		t.Fatalf("rolled back fences or late steps left a trace: %+v, %v", observed, err)
	}
}
