package runcost

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

// ApplyReport projects one expected or received contribution into all owning
// scopes in the caller's transaction; it never changes another store's tables.
func ApplyReport(ctx context.Context, tx pgx.Tx, p runstore.LLMCostReport) error {
	change, err := runstore.NewPostgresStore(tx, nil).ApplyLLMCostReport(ctx, p)
	if err != nil || !change.Changed {
		return err
	}
	if change.AuditExecutionID != nil {
		if err := hookstore.NewRunCostsTx(tx).Apply(ctx, *change.AuditExecutionID, change.RunID, change.Role, change.Previous, &change.Next, change.RunTotal); err != nil {
			return err
		}
	}
	if change.ProjectID != nil {
		return projectstore.NewPostgresStore(tx).ApplyLLMCost(ctx, change.OwnerID, *change.ProjectID, change.Role, change.AuditExecutionID != nil, change.Previous, &change.Next)
	}
	return nil
}
