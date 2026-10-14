package pgstore

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// ClaimedArtifacts is the narrow cross-context port for material published
// under an explicit controller claim. It grants no transition or dispatch.
type ClaimedArtifacts struct{ db persistencepostgres.DBTX }

func NewClaimedArtifacts(db persistencepostgres.DBTX) *ClaimedArtifacts {
	return &ClaimedArtifacts{db: db}
}

// WithArtifactWrite holds the live claim and Audit until the same transaction
// commits its binding. Takeover and purge therefore serialize with publication.
func (s *ClaimedArtifacts) WithArtifactWrite(
	ctx context.Context, claim auditstore.ControllerClaim, projectID, namespace string,
	write func(*artifacts.Service) error,
) error {
	if err := validateClaimIdentity(claim); err != nil {
		return err
	}
	if namespace != auditdomain.ArtifactNamespace(claim.AuditID) || write == nil {
		return invalidf("claimed Audit artifact namespace or writer is invalid")
	}
	return persistencepostgres.Within(ctx, s.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := auditClaims.Lock(ctx, tx, claim.AuditID, claim.HolderID, int64(claim.Epoch)); err != nil {
			if errors.Is(err, persistencepostgres.ErrControllerClaimLost) {
				return auditstore.ErrClaimLost
			}
			return err
		}
		var id string
		err := tx.QueryRow(ctx, `SELECT audit_id FROM audits
WHERE audit_id = $1 AND project_id = $2 AND state <> 'deleting'
FOR UPDATE`, claim.AuditID, projectID).Scan(&id)
		if errors.Is(err, pgx.ErrNoRows) {
			return auditstore.ErrClaimLost
		}
		if err != nil {
			return persistencepostgres.WrapError("lock Audit for artifact publication", err)
		}
		if err := write(artifacts.NewService(artifacts.NewPostgresRepository(tx))); err != nil {
			return err
		}
		live, err := auditClaims.Live(ctx, tx, claim.AuditID, claim.HolderID, int64(claim.Epoch))
		if err != nil {
			return err
		}
		if !live {
			return auditstore.ErrClaimLost
		}
		return nil
	})
}
