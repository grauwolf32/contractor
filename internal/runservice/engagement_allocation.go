package runservice

import (
	"context"
	"fmt"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/engagement"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// ReadAllocationEngagement resolves ordinary Run authority from the original
// input fork in one read snapshot. ordinary is false only for an Audit-managed
// Run, whose owner must supply its existing Start authorization instead.
func ReadAllocationEngagement(ctx context.Context, pool *pgxpool.Pool, runID, stageID string, storage *artifactpostgres.Storage) (pin *engagement.Pin, ordinary bool, err error) {
	err = persistencepostgres.InTx(ctx, pool, pgx.TxOptions{AccessMode: pgx.ReadOnly, IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
		runs := runstore.NewPostgresStore(tx, storage)
		binding, err := runs.ReadAllocationAuditBinding(ctx, runID, stageID)
		if err != nil {
			return err
		}
		ordinary = binding.ExecutionID == nil
		if !ordinary {
			return nil
		}
		run, err := runs.GetRun(ctx, runID)
		if err != nil {
			return err
		}
		workflow, err := resolved.DecodeResolvedWorkflowSnapshot(run.WorkflowSnapshot)
		if err != nil || workflow.Ref.Name != run.WorkflowName || workflow.Ref.Version != run.WorkflowVersion {
			return fmt.Errorf("stored engagement Workflow authority is invalid")
		}
		slot, err := engagement.InputSlot(workflow.Inputs)
		if err != nil {
			return err
		}
		if slot == "" {
			if len(run.EngagementCredentialIDs) != 0 {
				return fmt.Errorf("engagement credentials have no declared input")
			}
			return nil
		}
		inputs, err := runs.DescribeOriginalInputs(ctx, runID, []string{slot})
		if err != nil {
			return err
		}
		if len(inputs) == 0 {
			if workflow.Inputs[slot].Required || len(run.EngagementCredentialIDs) != 0 {
				return fmt.Errorf("engagement original input is missing")
			}
			return nil
		}
		store, err := artifacts.NewService(storage.Repository(tx)).Run(runID)
		if err != nil {
			return err
		}
		document, err := engagement.Read(ctx, store, inputs[0])
		if err != nil {
			return err
		}
		if !slices.Equal(engagement.CredentialIDs(document), run.EngagementCredentialIDs) {
			return fmt.Errorf("engagement credential dependencies differ from accepted input")
		}
		pin = &engagement.Pin{Artifact: inputs[0], Document: document,
			AuthorizedBy: run.OwnerID, AuthorizedAt: run.CreatedAt}
		return nil
	})
	return
}
