package findingintake

import (
	"context"
	"errors"
	"strings"

	"github.com/jackc/pgx/v5"
)

// MarkSourceRunDeleted tombstones, in tx, the retention state of every
// finding proposal received from Run runID while Run deletion removes that
// Run. A receipt that an Audit holds becomes audit-held and every other one
// discarded; each keeps its identity and safe provenance. internal/rundeletion
// calls it after it locks the Run and the Audits that refer to it and before
// it purges the Run's Artifacts, the proposal revisions among them (DEL-R22).
func MarkSourceRunDeleted(ctx context.Context, tx pgx.Tx, runID string) error {
	if tx == nil {
		return errors.New("PostgreSQL transaction is required")
	}
	if strings.TrimSpace(runID) == "" {
		return ErrInvalid
	}
	_, err := tx.Exec(ctx, markSourceRunDeletedSQL, runID)
	return err
}
