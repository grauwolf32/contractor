package findingintake

import (
	"context"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

func (s *Service) validateCandidateSetAuthority(ctx context.Context, runID, stageID, claimID string, selection contracts.ArtifactRef) error {
	return persistencepostgres.InTxWithRetry(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		stage, err := runstore.NewPostgresTxStore(tx, s.storage).LockStageOperation(ctx, runID, stageID, claimID)
		if err != nil {
			return err
		}
		spec, err := resolved.DecodeResolvedStageSnapshot(stage.StageSpecSnapshot)
		if err != nil || spec.Operation == nil || spec.Operation.Ref != contracts.PublishFindingsOperation || !samePinnedInput(stage, "materials", selection) {
			return ErrAccessDenied
		}
		run, err := runstore.NewPostgresStore(tx, s.storage).GetRun(ctx, runID)
		if err != nil {
			return err
		}
		if run.ProjectID == nil {
			return ErrAccessDenied
		}
		return nil
	})
}
