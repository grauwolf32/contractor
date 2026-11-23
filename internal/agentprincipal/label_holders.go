package agentprincipal

import (
	"context"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

// LabelHolders serves RuntimeConfig label management the principals that
// hold a label, read and locked through this registry's repository.
type LabelHolders struct{}

var _ runtimeconfig.LabelHolders = LabelHolders{}

func (LabelHolders) ListLabelHolders(
	ctx context.Context, db persistencepostgres.DBTX, label string,
) ([]runtimeconfig.LabelHolder, error) {
	principals, err := NewRepository(db).ListByLabel(ctx, label)
	if err != nil {
		return nil, err
	}
	return labelHolders(principals), nil
}

func (LabelHolders) LockLabelHolders(
	ctx context.Context, tx pgx.Tx, runtimeAgentIDs []string,
) ([]runtimeconfig.LabelHolder, error) {
	principals, err := NewRepository(tx).LockMany(ctx, runtimeAgentIDs)
	if err != nil {
		return nil, err
	}
	return labelHolders(principals), nil
}

func labelHolders(principals []Principal) []runtimeconfig.LabelHolder {
	result := make([]runtimeconfig.LabelHolder, len(principals))
	for index, principal := range principals {
		result[index] = runtimeconfig.LabelHolder{
			RuntimeAgentID: principal.RuntimeAgentID,
			Labels:         append([]string{}, principal.Labels...),
			LabelRevision:  principal.LabelRevision,
		}
	}
	return result
}
