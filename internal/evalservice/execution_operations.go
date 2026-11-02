package evalservice

import (
	"context"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type executionOperations struct {
	pool *pgxpool.Pool
	// now stamps the intents whose request records the time they were made.
	now func() time.Time
}

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
		return s.RecoverMember(scope(e), e.ID, m.MemberID, c).PutSuboperation(ctx, op)
	})
	return op, err
}

func resolveOperation[T any](ctx context.Context, d *executionOperations, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, op evalstore.Suboperation, response T, rejected bool, executionID string) error {
	raw, err := jsonBytes(response)
	if err != nil {
		return err
	}
	return d.tx(ctx, func(s *evalstore.Store, _ pgx.Tx) error {
		recovery := s.RecoverMember(scope(e), e.ID, m.MemberID, c)
		if executionID != "" {
			if err := recovery.BindExecution(ctx, executionID); err != nil {
				return err
			}
		}
		return recovery.ResolveSuboperation(ctx, op.Kind, raw, rejected)
	})
}

func (d *executionOperations) failDefinite(ctx context.Context, e evalstore.Experiment, m evalstore.Member, c evalstore.Claim, op evalstore.Suboperation, err error) error {
	// Only local validation/fence failures establish non-acceptance. Timeouts and
	// unknown database outcomes retain the original intent for idempotent replay.
	if definiteConfigurationError(err) {
		return resolveOperation(ctx, d, e, m, c, op, operationRejected{Reason: "eval_not_ready"}, true, "")
	}
	if artifacts.IsExactReferenceRejection(err) || errors.Is(err, blob.ErrMissing) || errors.Is(err, contracts.ErrExactArtifactMismatch) {
		return resolveOperation(ctx, d, e, m, c, op, operationRejected{Reason: "eval_evidence_unavailable"}, true, "")
	}
	for _, definite := range []error{
		runservice.ErrInvalid, runstore.ErrInvalid, auditservice.ErrInvalid, auditstore.ErrInvalid, contentdigest.ErrPinnedSelectionChanged,
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
		recovery := s.RecoverMember(scope(e), e.ID, m.MemberID, c)
		if err := recovery.LockMember(ctx); err != nil {
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
				if err = recovery.ResolveSuboperation(ctx, k, diagnostic(domain.Code), true); err != nil {
					return err
				}
			}
		}
		raw, _ := jsonBytes(preparationRejected{RejectedBeforeCreation: true, Code: domain.Code})
		if err := recovery.PutSuboperation(ctx, evalstore.Suboperation{Kind: kind, Key: m.SubmissionKey + "-" + kind, Request: raw, State: "intent"}); err != nil {
			return err
		}
		if err := recovery.ResolveSuboperation(ctx, kind, diagnostic(domain.Code), true); err != nil {
			return err
		}
		return recovery.Settle(ctx)
	})
}
