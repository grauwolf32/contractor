package pgstore

import (
	"context"
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
	return s.keyedAuditMutation(ctx, params.OwnerID, auditstore.MutationDelete, params.IdempotencyKey, params.RequestDigest,
		func(tx pgx.Tx) (auditMutationResult, error) {
			audit, err := scanAudit(persistencepostgres.SavepointQueryRow(ctx, tx, requestDeleteSQL,
				params.OwnerID, params.AuditID, params.ExpectedRevision,
			))
			if err == nil {
				return auditMutationResult{
					audit: audit, resourceID: audit.AuditID, response: map[string]string{"auditId": params.AuditID},
				}, nil
			}
			if persistencepostgres.Classify(err) == persistencepostgres.ClassDuplicate {
				return auditMutationResult{}, auditstore.ErrConflict
			}
			if !errors.Is(err, pgx.ErrNoRows) {
				return auditMutationResult{}, fmt.Errorf("request Audit deletion: %w", err)
			}
			return auditMutationResult{}, ownerMutationRefusal(ctx, tx, params.OwnerID, params.AuditID)
		})
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
