package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// RequestDelete durably records deletion independently from the stop reason.
// Active work first enters cancelling; drafts and terminal Audits can enter
// deleting directly. A new key against an already-deleting exact revision is
// accepted without manufacturing another lifecycle revision.
func (s *OwnerStore) RequestDelete(
	ctx context.Context, params auditstore.DeleteParams,
) (auditstore.Audit, bool, error) {
	if err := validateDelete(params); err != nil {
		return auditstore.Audit{}, false, err
	}
	if replay, found, err := s.lookupAuditReplay(
		ctx, params.OwnerID, auditstore.MutationDelete, params.IdempotencyKey, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	response, _ := json.Marshal(map[string]string{"auditId": params.AuditID})
	audit, err := scanAudit(persistencepostgres.SavepointQueryRow(ctx, s.db, requestDeleteSQL,
		params.OwnerID, params.AuditID, params.ExpectedRevision,
		params.IdempotencyKey, params.RequestDigest, response,
	))
	if err == nil {
		return audit, true, nil
	}
	if persistencepostgres.Classify(err) == persistencepostgres.ClassDuplicate {
		return s.replayAudit(ctx, params.OwnerID, auditstore.MutationDelete, params.IdempotencyKey, params.RequestDigest)
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Audit{}, false, fmt.Errorf("request Audit deletion: %w", err)
	}
	if replay, found, replayErr := s.lookupAuditReplay(
		ctx, params.OwnerID, auditstore.MutationDelete, params.IdempotencyKey, params.RequestDigest,
	); replayErr != nil || found {
		return replay, false, replayErr
	}
	if _, getErr := s.get(ctx, params.OwnerID, params.AuditID); errors.Is(getErr, auditstore.ErrNotFound) {
		return auditstore.Audit{}, false, auditstore.ErrNotFound
	} else if getErr != nil {
		return auditstore.Audit{}, false, getErr
	}
	return auditstore.Audit{}, false, auditstore.ErrPrecondition
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
