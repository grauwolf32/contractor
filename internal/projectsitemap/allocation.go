package projectsitemap

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// AllocationAccess derives the shared map scope from the authenticated Run.
// Only the authenticated adapter sees owner/Project IDs.
type AllocationAccess struct {
	APIVersion     string                         `json:"apiVersion"`
	OwnerID        string                         `json:"-"`
	ProjectID      string                         `json:"-"`
	RunID          string                         `json:"-"`
	Namespace      string                         `json:"-"`
	Policy         contracts.ProjectSitemapPolicy `json:"policy"`
	ManualSourceID string                         `json:"manualSourceId"`
}

// AllocationAccess checks the current Project without reading artifact lineage
// or payloads. Snapshot attachments are resolved only during materialization.
func (s *Service) AllocationAccess(ctx context.Context, runID, namespace string, policy contracts.ProjectSitemapPolicy) (AllocationAccess, error) {
	if !contracts.ValidResourceID(runID) || !contracts.ValidName(namespace) || policy.Validate() != nil {
		return AllocationAccess{}, ErrInvalid
	}
	ctx, cancel := context.WithTimeout(ctx, OperationTimeout)
	defer cancel()
	access := AllocationAccess{APIVersion: contracts.APIVersion, RunID: runID, Namespace: namespace, Policy: *policy.Clone()}
	err := postgres.InTx(ctx, s.pool, pgx.TxOptions{AccessMode: pgx.ReadOnly}, func(tx pgx.Tx) error {
		err := tx.QueryRow(ctx, `SELECT owner_id,project_id FROM workflow_runs WHERE run_id=$1 AND project_id IS NOT NULL`, runID).Scan(&access.OwnerID, &access.ProjectID)
		if errors.Is(err, pgx.ErrNoRows) {
			return artifacts.ErrArtifactNotFound
		}
		if err != nil {
			return err
		}
		if err := checkProject(ctx, tx, access.OwnerID, access.ProjectID, false); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		return AllocationAccess{}, err
	}
	deployment := ""
	if policy.Deployment != nil {
		deployment = *policy.Deployment
	}
	access.ManualSourceID = key("manual", deployment, namespace)
	return access, nil
}

// ReadAllocation returns the same validated Project document as an owner read.
func (s *Service) ReadAllocation(ctx context.Context, access AllocationAccess, revision string) (View, error) {
	return s.Read(ctx, access.OwnerID, access.ProjectID, revision)
}

// MutateAllocation is the manual editor boundary. It never permits an
// observed fact or a source identity chosen by the model.
// The authenticated adapter checks the current allocation's selected operation,
// write policy and fence. Editors may select any member in their Project map.
func (s *Service) MutateAllocation(ctx context.Context, access AllocationAccess, idempotencyKey string, mutation Mutation) (Result, error) {
	if mutation.Operation == "observe" || mutation.Source != nil && mutation.Source.Artifact != nil {
		return Result{}, ErrInvalid
	}
	deployment := ""
	if access.Policy.Deployment != nil {
		deployment = *access.Policy.Deployment
	}
	authority := Authority{
		OwnerID: access.OwnerID, ProjectID: access.ProjectID, RunID: access.RunID,
		SourceID: access.ManualSourceID, Kind: "manual", Deployment: deployment,
	}
	if mutation.Source != nil && mutation.Source.SourceID != authority.SourceID {
		return Result{}, ErrInvalid
	}
	return s.Mutate(ctx, authority, idempotencyKey, mutation)
}
