package evalservice

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type executionOperations struct{ pool *pgxpool.Pool }

func (d *executionOperations) tx(ctx context.Context, fn func(*evalstore.Store, pgx.Tx) error) error {
	return postgres.InTx(ctx, d.pool, pgx.TxOptions{}, func(tx pgx.Tx) error { return fn(evalstore.NewTxStore(tx), tx) })
}

func notFound(err error) bool {
	var e *evaldomain.Error
	return errors.As(err, &e) && e.Code == "eval_not_found"
}

func prepareOperation[T any](ctx context.Context, d *executionOperations, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, kind string, build func() (T, error)) (evalstore.Suboperation, error) {
	store := evalstore.NewPostgresStore(d.pool)
	op, err := store.Suboperation(ctx, e.OwnerID, e.ID, m.MemberID, kind)
	if err == nil {
		return op, nil
	}
	if !notFound(err) {
		return op, err
	}
	value, err := build()
	if err != nil {
		return op, err
	}
	raw, err := jsonBytes(value)
	if err != nil {
		return op, err
	}
	op = evalstore.Suboperation{Kind: kind, Key: m.SubmissionKey + "-" + kind, Request: raw, State: "intent"}
	err = d.tx(ctx, func(s *evalstore.Store, _ pgx.Tx) error {
		return s.PutSuboperation(ctx, scope(e), e.ID, m.MemberID, c, op)
	})
	return op, err
}

func resolveOperation[T any](ctx context.Context, d *executionOperations, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, op evalstore.Suboperation, response T, rejected bool, executionID string) error {
	raw, err := jsonBytes(response)
	if err != nil {
		return err
	}
	return d.tx(ctx, func(s *evalstore.Store, _ pgx.Tx) error {
		if executionID != "" {
			if err := s.BindExecution(ctx, scope(e), e.ID, m.MemberID, executionID, c); err != nil {
				return err
			}
		}
		return s.ResolveSuboperation(ctx, scope(e), e.ID, m.MemberID, op.Kind, c, raw, rejected)
	})
}

func (d *executionOperations) failDefinite(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, op evalstore.Suboperation, err error) error {
	// Only local validation/fence failures establish non-acceptance. Timeouts and
	// unknown database outcomes retain the original intent for idempotent replay.
	for _, definite := range []error{
		runservice.ErrInvalid, runservice.ErrPinnedSelectionChanged,
		auditservice.ErrInvalid, auditservice.ErrPinnedSelectionChanged,
		auditservice.ErrProfileNotFound, auditservice.ErrUnsupported,
		projectstore.ErrDeleting, auditstore.ErrProjectDeleting,
	} {
		if errors.Is(err, definite) {
			return resolveOperation(ctx, d, e, m, c, op, operationRejected{Reason: "eval_pin_mismatch"}, true, "")
		}
	}
	return err
}

// Only a local validation failure before any creation intent establishes that
// no remote effect can exist. Once an execution intent exists, replay owns it.
func (d *executionOperations) rejectPreparation(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, kind string, cause error) error {
	var domain *evaldomain.Error
	if !errors.As(cause, &domain) || (domain.Code != "eval_pin_mismatch" && domain.Code != "eval_evidence_unavailable" && domain.Code != "eval_not_found" && domain.Code != "eval_invalid") {
		return cause
	}
	return d.tx(ctx, func(s *evalstore.Store, _ pgx.Tx) error {
		if err := s.LockMemberRecovery(ctx, scope(e), e.ID, m.MemberID, c); err != nil {
			return err
		}
		if _, err := s.Suboperation(ctx, e.OwnerID, e.ID, m.MemberID, kind); !notFound(err) {
			return cause
		}
		for _, k := range []string{"inputs", "project-create"} {
			op, err := s.Suboperation(ctx, e.OwnerID, e.ID, m.MemberID, k)
			if notFound(err) {
				continue
			}
			if err != nil {
				return err
			}
			if op.State == "intent" {
				if err = s.ResolveSuboperation(ctx, scope(e), e.ID, m.MemberID, k, c, diagnostic(domain.Code), true); err != nil {
					return err
				}
			}
		}
		raw, _ := jsonBytes(preparationRejected{RejectedBeforeCreation: true, Code: domain.Code})
		if err := s.PutSuboperation(ctx, scope(e), e.ID, m.MemberID, c, evalstore.Suboperation{Kind: kind, Key: m.SubmissionKey + "-" + kind, Request: raw, State: "intent"}); err != nil {
			return err
		}
		if err := s.ResolveSuboperation(ctx, scope(e), e.ID, m.MemberID, kind, c, diagnostic(domain.Code), true); err != nil {
			return err
		}
		return s.Settle(ctx, scope(e), e.ID, m.MemberID, c)
	})
}
