package privateartifacts

import (
	"context"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type PlannerOperations interface {
	ExecuteOperation(context.Context, string, string, contracts.PlannerOperation) (contracts.PlannerOperationResult, error)
}
