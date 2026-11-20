// Package memorysnapshot binds one explicit Stage export to the Memory owner.
package memorysnapshot

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/memorycontract"
	"github.com/grauwolf32/contractor/internal/memory"
	"github.com/grauwolf32/contractor/internal/operations"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/jackc/pgx/v5"
)

type Module struct{ store *memory.PostgresStore }

func New(store *memory.PostgresStore) *Module { return &Module{store: store} }
func (*Module) Ref() string                   { return contracts.MemorySnapshotOperation }
func (m *Module) Run(ctx context.Context, call operations.Call) (stageresult.StageContentResult, error) {
	empty := stageresult.StageContentResult{}
	selection, err := memorycontract.SnapshotOptions(call.Arguments)
	if err != nil || len(call.Inputs) != 0 || len(call.Outputs) != 1 {
		return empty, invalid()
	}
	output, ok := call.Outputs["snapshot"]
	if !ok {
		return empty, invalid()
	}
	if m.store == nil {
		return empty, &operations.Failure{Code: "server_operation_unavailable", Message: "Memory snapshot storage is unavailable"}
	}
	metadata, err := m.store.SnapshotStage(ctx, memory.SnapshotStageRequest{RunID: call.RunID, StageExecutionID: call.StageExecutionID, ClaimID: call.ClaimID, Namespace: selection.Namespace, Notes: call.Arguments["notes"].(string), Output: output}, func(ctx context.Context, tx pgx.Tx, ownerID, projectID string) error {
		_, err := projectstore.NewPostgresStore(tx).LockActiveProject(ctx, ownerID, projectID)
		return err
	})
	if err != nil {
		if errors.Is(err, memory.ErrSnapshotInvalid) || errors.Is(err, contracts.ErrValidation) {
			return empty, invalid()
		}
		if classified, ok := apierror.Of(err); ok {
			kind := classified.ErrorKind()
			return empty, &operations.Failure{Code: kind.Code(), Message: kind.Message(), Retryable: kind.Retryable()}
		}
		return empty, err
	}
	return stageresult.StageContentResult{APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded, Summary: fmt.Sprintf("Exported %d explicitly selected Memory notes.", len(selection.Names)), Artifacts: map[string]contracts.ArtifactRef{"snapshot": metadata.Ref}}, nil
}
func invalid() error {
	return &operations.Failure{Code: "invalid_request", Message: "Memory snapshot selection, sources or output are invalid"}
}
