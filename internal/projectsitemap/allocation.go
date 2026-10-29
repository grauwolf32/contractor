package projectsitemap

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

// AllocationAccess is trusted domain-command context derived from the
// authenticated Run. It is not an API response or a generic Artifact grant.
type AllocationAccess struct {
	OwnerID   string
	ProjectID string
	RunID     string
	Namespace string
	Policy    contracts.ProjectSitemapPolicy
}

// AllocationAccess checks the command's Project without reading artifact lineage
// or payloads. Snapshot attachments are resolved only during materialization.
func (s *Service) AllocationAccess(ctx context.Context, runID, namespace string, policy contracts.ProjectSitemapPolicy) (AllocationAccess, error) {
	if !contracts.ValidResourceID(runID) || !contracts.ValidName(namespace) || policy.Validate() != nil {
		return AllocationAccess{}, ErrInvalid
	}
	ctx, cancel := context.WithTimeout(ctx, OperationTimeout)
	defer cancel()
	access := AllocationAccess{RunID: runID, Namespace: namespace, Policy: *policy.Clone()}
	err := postgres.InTx(ctx, s.pool, pgx.TxOptions{AccessMode: pgx.ReadOnly}, func(tx pgx.Tx) error {
		owning, err := runstore.NewPostgresStore(tx).ProjectContext(ctx, runID)
		if errors.Is(err, runstore.ErrNotFound) {
			return artifacts.ErrArtifactNotFound
		}
		if err != nil {
			return err
		}
		access.OwnerID, access.ProjectID = owning.OwnerID, owning.ProjectID
		if err := checkProject(ctx, tx, access.OwnerID, access.ProjectID, false); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		return AllocationAccess{}, err
	}
	return access, nil
}

// MutateAllocation is the manual editor boundary. It never permits an
// observed fact or a source identity chosen by the model.
// The authenticated adapter checks the current allocation's selected operation,
// write policy and fence. Editors may select any member in their Project map.
func (s *Service) MutateAllocation(ctx context.Context, access AllocationAccess, idempotencyKey string, mutation EditorMutation) (Result, error) {
	if mutation.Operation == "observe" {
		return Result{}, ErrInvalid
	}
	deployment := ""
	if access.Policy.Deployment != nil {
		deployment = *access.Policy.Deployment
	}
	authority := Authority{
		OwnerID: access.OwnerID, ProjectID: access.ProjectID, RunID: access.RunID,
		SourceID: key("manual", deployment, access.Namespace), Kind: "manual", Deployment: deployment,
	}
	return s.Mutate(ctx, authority, idempotencyKey, mutation.WithSource(authority.SourceID))
}
