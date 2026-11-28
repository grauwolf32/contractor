package findingintake

import (
	"context"
	"errors"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/contracts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// CheckSource authorizes a pinned receipt without comparing a later analyst
// revision. Launch alone checks that revision; subsequent checks retain it.
func CheckSource(ctx context.Context, db postgres.DBTX, storage *artifactpostgres.Storage, owner, project string, check contracts.FindingCheck) error {
	if check.Validate() != nil {
		return ErrInvalid
	}
	if _, err := querystore.New(db, storage).GetProjectFinding(ctx, owner, project, check.FindingID); err != nil {
		return err
	}
	var found bool
	err := db.QueryRow(ctx, checkSourceMembershipSQL, owner, project, check.FindingID, check.ReceiptID).Scan(&found)
	if errors.Is(err, pgx.ErrNoRows) || err == nil && !found {
		return ErrNotFound
	}
	return err
}
