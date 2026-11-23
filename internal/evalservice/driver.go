package evalservice

import (
	"context"
	"errors"
	"time"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type RunCreator interface {
	CreatePublic(context.Context, runservice.PublicCreateParams) (runservice.CreateResult, error)
}

type AuditExecutor interface {
	CreateDraft(context.Context, auditservice.CreateDraftParams) (auditstore.Audit, bool, error)
	Start(context.Context, auditservice.StartParams) (auditservice.StartedAudit, error)
	Get(context.Context, string, string) (auditstore.Audit, error)
	Cancel(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error)
	Delete(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error)
}

type RunNotifier interface {
	Wake()
	Cancel(string)
}

type Driver struct {
	Storage  *artifactpostgres.Storage
	Pool     *pgxpool.Pool
	Runs     RunCreator
	Audits   AuditExecutor
	Notifier RunNotifier
	// Now stamps a cancellation intent when it is journaled; every replay
	// uses the journaled time. Nil is the system clock.
	Now func() time.Time
}

type executionAdapter interface {
	Create(context.Context, evalstore.Experiment, evalstore.Member, evalstore.Claim) error
	Cancel(context.Context, evalstore.Experiment, evalstore.Member, evalstore.Claim, evalstore.Submission) error
}

type memberReconciler struct {
	operations *executionOperations
	adapter    executionAdapter
}

func (d *Driver) Reconcile(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim) error {
	if d.Storage == nil || d.Pool == nil || d.Runs == nil || d.Audits == nil {
		return errors.New("eval execution dependencies are incomplete")
	}
	now := d.Now
	if now == nil {
		now = time.Now
	}
	operations := &executionOperations{storage: d.Storage, pool: d.Pool, now: now}
	resources := &memberResources{operations: operations}
	var adapter executionAdapter
	switch m.ExecutionKind {
	case "run":
		adapter = &workflowAdapter{operations: operations, resources: resources, Runs: d.Runs, Notifier: d.Notifier}
	case "audit":
		adapter = &auditAdapter{operations: operations, resources: resources, Audits: d.Audits}
	default:
		return evaldomain.Failure("eval_invalid")
	}
	return (&memberReconciler{operations: operations, adapter: adapter}).Reconcile(ctx, e, m, c)
}

func (d *memberReconciler) Reconcile(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim) error {
	store := evalstore.NewPostgresStore(d.operations.pool, d.operations.storage)
	sub, err := store.Submission(ctx, e.OwnerID, e.ID, m.MemberID)
	if err != nil {
		return err
	}
	if sub.State == evalstore.SubmissionTerminal || sub.State == evalstore.SubmissionRejected {
		return nil
	}
	tombstone, tombErr := store.Tombstone(ctx, e.OwnerID, e.ID, m.MemberID)
	if tombErr == nil {
		return d.operations.tx(ctx, func(s *evalstore.Store, _ pgx.Tx) error {
			recovery := s.RecoverMember(scope(e), e.ID, m.MemberID, c)
			if !tombstone.NeverStarted {
				if err := recovery.BindTombstone(ctx); err != nil {
					return err
				}
			}
			// Recover any lost acknowledgement without calling a deleted execution's
			// creation API again. The tombstone came from its ordinary deletion path.
			for _, kind := range []string{"run-create", "audit-create", "audit-start", "cancel"} {
				op, err := s.Suboperation(ctx, e.OwnerID, e.ID, m.MemberID, kind)
				if notFound(err) {
					continue
				}
				if err != nil {
					return err
				}
				if op.State == "intent" {
					response, _ := jsonBytes(executionDeleted{ID: tombstone.ID, Deleted: true})
					if err = recovery.ResolveSuboperation(ctx, kind, response, tombstone.NeverStarted && kind == "audit-start"); err != nil {
						return err
					}
				}
			}
			return recovery.Settle(ctx)
		})
	}
	if !notFound(tombErr) {
		return tombErr
	}
	if sub.State == evalstore.SubmissionIntent {
		kind := "run-create"
		if m.ExecutionKind == "audit" {
			kind = "audit-create"
		}
		op, opErr := store.Suboperation(ctx, e.OwnerID, e.ID, m.MemberID, kind)
		if e.State == evaldomain.StateCancelling && notFound(opErr) {
			return d.operations.tx(ctx, func(s *evalstore.Store, _ pgx.Tx) error {
				recovery := s.RecoverMember(scope(e), e.ID, m.MemberID, c)
				for _, kind := range []string{"inputs", "project-create"} {
					op, err := s.Suboperation(ctx, e.OwnerID, e.ID, m.MemberID, kind)
					if notFound(err) {
						continue
					}
					if err != nil {
						return err
					}
					if op.State == "intent" {
						if err = recovery.ResolveSuboperation(ctx, kind, []byte(`{"reason":"cancelled_before_submission"}`), true); err != nil {
							return err
						}
					}
				}
				return recovery.Settle(ctx)
			})
		}
		if opErr == nil && op.State == "rejected" {
			return d.operations.tx(ctx, func(s *evalstore.Store, _ pgx.Tx) error {
				return s.RecoverMember(scope(e), e.ID, m.MemberID, c).Settle(ctx)
			})
		}
		if opErr != nil && !notFound(opErr) {
			return opErr
		}
		if err = d.adapter.Create(ctx, e, m, c); err != nil {
			return d.operations.rejectPreparation(ctx, e, m, c, kind, err)
		}
	}
	sub, err = store.Submission(ctx, e.OwnerID, e.ID, m.MemberID)
	if err != nil {
		return err
	}
	if sub.State != evalstore.SubmissionAccepted {
		return nil
	}
	if e.State == evaldomain.StateCancelling {
		if err = d.adapter.Cancel(ctx, e, m, c, sub); err != nil {
			return err
		}
	}
	tokens, err := store.KnownTokens(ctx, e.OwnerID, m.ExecutionKind, sub.ExecutionID)
	if err != nil {
		return err
	}
	return d.operations.tx(ctx, func(s *evalstore.Store, _ pgx.Tx) error {
		recovery := s.RecoverMember(scope(e), e.ID, m.MemberID, c)
		if err := recovery.ObserveTokens(ctx, tokens); err != nil {
			return err
		}
		err := recovery.Settle(ctx)
		if errors.Is(err, evalstore.ErrDrain) {
			return nil
		}
		return err
	})
}
