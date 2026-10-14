package auditstore

import (
	"context"

	"github.com/jackc/pgx/v5"
)

// PurgeDependencies releases foreign-context rows in the caller's locked,
// drained Audit transaction, returning the number removed within limit.
type PurgeDependencies func(ctx context.Context, tx pgx.Tx, auditID string, limit int) (int, error)
