// Package runcost composes expected participant reports and owning cost stores.
package runcost

import (
	"context"
	"time"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Participants composes participant creation with its expected cost report.
// Work may start only after the missing slot and all parent deltas commit.
type Participants struct {
	*runstore.PostgresStore
	pool *pgxpool.Pool
}

func NewParticipants(pool *pgxpool.Pool, storage *artifactpostgres.Storage) *Participants {
	return &Participants{PostgresStore: runstore.NewPostgresStore(pool, storage), pool: pool}
}

func (s *Participants) StartPlanner(ctx context.Context, p runstore.StartPlannerParams) error {
	expected := reporting.CostContribution{ReceivedAt: time.Now().UTC()}
	return persistencepostgres.Within(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := runstore.NewPostgresStore(tx, nil).StartPlanner(ctx, p); err != nil {
			return err
		}
		return ApplyReport(ctx, tx, runstore.LLMCostReport{StageExecutionID: p.StageExecutionID, Contribution: expected})
	})
}

// RecordStageAllocation joins the placement transaction, including all pins and
// reservations; a failed parent projection prevents allocation admission. The
// placement caller holds the Run lock before writing allocation rows (RUN-R29).
func (s *Participants) RecordStageAllocation(ctx context.Context, tx pgx.Tx, p runstore.StageAllocation) error {
	if tx == nil {
		return runstore.ErrInvalid
	}
	if err := runstore.NewPostgresStore(tx, nil).RecordStageAllocation(ctx, p); err != nil {
		return err
	}
	return ApplyReport(ctx, tx, runstore.LLMCostReport{StageExecutionID: p.StageExecutionID, AllocationID: p.AllocationID, Contribution: reporting.CostContribution{ReceivedAt: time.Now().UTC()}})
}
