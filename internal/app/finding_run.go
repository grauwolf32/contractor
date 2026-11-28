package app

import (
	"context"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

type findingRunWriter struct {
	*runstore.PostgresTxStore
	tx      pgx.Tx
	storage *artifactpostgres.Storage
}

func (w findingRunWriter) PrepareFindingCheck(ctx context.Context, runID string, check contracts.FindingCheck) (findingintake.Receipt, error) {
	return findingintake.PrepareCheck(ctx, w.tx, w.storage, runID, check)
}
