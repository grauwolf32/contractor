package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// RequestDelete durably records deletion independently from the stop reason.
// Active work first enters cancelling; drafts and terminal Audits can enter
// deleting directly. A new key against an already-deleting exact revision is
// accepted without manufacturing another lifecycle revision.
func (s *PostgresStore) RequestDelete(
	ctx context.Context, params DeleteParams,
) (Audit, bool, error) {
	if err := validateDelete(params); err != nil {
		return Audit{}, false, err
	}
	if replay, found, err := s.lookupAuditReplay(
		ctx, params.OwnerID, string(MutationDelete), params.IdempotencyKey, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	response, _ := json.Marshal(map[string]string{"auditId": params.AuditID})
	audit, err := scanAudit(s.db.QueryRow(ctx, requestDeleteSQL,
		params.OwnerID, params.AuditID, params.ExpectedRevision,
		params.IdempotencyKey, params.RequestDigest, response,
	))
	if err == nil {
		return audit, true, nil
	}
	if persistencepostgres.SQLState(err) == persistencepostgres.SQLStateUniqueViolation {
		return s.replayAudit(ctx, params.OwnerID, string(MutationDelete), params.IdempotencyKey, params.RequestDigest)
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Audit{}, false, fmt.Errorf("request Audit deletion: %w", err)
	}
	if replay, found, replayErr := s.lookupAuditReplay(
		ctx, params.OwnerID, string(MutationDelete), params.IdempotencyKey, params.RequestDigest,
	); replayErr != nil || found {
		return replay, false, replayErr
	}
	if _, getErr := s.Get(ctx, params.OwnerID, params.AuditID); errors.Is(getErr, ErrNotFound) {
		return Audit{}, false, ErrNotFound
	} else if getErr != nil {
		return Audit{}, false, getErr
	}
	return Audit{}, false, ErrPrecondition
}

// auditDeletionSource is who requested an Audit deletion: its owner or the
// deletion of its Project. Both record the request through
// deletionRequestCTEs and differ only in the stop reason and in the source the
// audit.delete_requested event names.
type auditDeletionSource int

const (
	auditDeletionByOwner auditDeletionSource = iota
	auditDeletionByProject
)
