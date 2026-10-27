package projectsitemap

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// AllocationAccess exposes policy and authorized consumer descriptors, not
// another scope selector. Only the authenticated adapter sees owner/Project IDs.
type AllocationAccess struct {
	APIVersion     string                         `json:"apiVersion"`
	OwnerID        string                         `json:"-"`
	ProjectID      string                         `json:"-"`
	RunID          string                         `json:"-"`
	Namespace      string                         `json:"-"`
	Policy         contracts.ProjectSitemapPolicy `json:"policy"`
	PolicyDigest   string                         `json:"policyDigest"`
	ManualSourceID string                         `json:"manualSourceId"`
	Inputs         []contracts.ExactArtifact      `json:"inputs"`
}

// AllocationAccess resolves at most eight named original Run input revisions, never
// an unbounded Run artifact list or a caller-supplied foreign scope. The bound
// is independent of the containing Run's artifact count.
func (s *Service) AllocationAccess(ctx context.Context, runID, namespace string, policy contracts.ProjectSitemapPolicy) (AllocationAccess, error) {
	if !contracts.ValidResourceID(runID) || !contracts.ValidName(namespace) || policy.Validate() != nil {
		return AllocationAccess{}, ErrInvalid
	}
	ctx, cancel := context.WithTimeout(ctx, OperationTimeout)
	defer cancel()
	access := AllocationAccess{APIVersion: contracts.APIVersion, RunID: runID, Namespace: namespace, Policy: *policy.Clone(), Inputs: []contracts.ExactArtifact{}}
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
		store, err := artifacts.NewService(artifacts.NewPostgresRepository(tx)).Run(runID)
		if err != nil {
			return err
		}
		for _, name := range policy.OpenAPIInputs {
			rows, err := tx.Query(ctx, allocationInputRevisionSQL, runID, name)
			if err != nil {
				return err
			}
			revisions, err := pgx.CollectRows(rows, pgx.RowTo[string])
			if err != nil {
				return err
			}
			if len(revisions) == 0 {
				return artifacts.ErrArtifactNotFound
			}
			if len(revisions) != 1 {
				return ErrInvalid
			}
			metadata, err := store.Metadata(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: name, Revision: &revisions[0]})
			if err != nil {
				return err
			}
			exact := metadata.Exact()
			if err := exact.Validate(); err != nil {
				return ErrInvalid
			}
			access.Inputs = append(access.Inputs, exact)
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
	access.ManualSourceID = key("manual", policy.SourceMode, deployment, namespace)
	access.PolicyDigest, err = contentdigest.JSON(struct {
		Policy contracts.ProjectSitemapPolicy `json:"policy"`
		Inputs []contracts.ExactArtifact      `json:"inputs"`
	}{access.Policy, access.Inputs})
	return access, err
}

// ReadAllocation returns only the authorized projection. Raw mixed-map bytes
// stay inside the Server, including when the consumer requests an exact ref.
func (s *Service) ReadAllocation(ctx context.Context, access AllocationAccess, revision string) (View, error) {
	view, err := s.Read(ctx, access.OwnerID, access.ProjectID, revision)
	if err != nil {
		return View{}, err
	}
	view.Document, err = SafeProjection(view.Document, access.Policy, access.RunID, access.Inputs)
	return view, err
}

// MutateAllocation is the manual editor boundary. It never permits an
// observed fact or a source identity chosen by the model.
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
		SourceID: access.ManualSourceID, Kind: "manual", Mode: access.Policy.SourceMode, Deployment: deployment,
	}
	if access.Policy.SourceMode == "blackbox" {
		authority.Authorize = func(current Document, command Mutation) error {
			if len(command.MemberIDs) == 0 && command.Operation != "restore" {
				return nil
			}
			view, err := SafeProjection(current, access.Policy, access.RunID, access.Inputs)
			if err != nil {
				return err
			}
			allowed := map[string]bool{}
			for _, group := range view.Groups {
				for _, member := range group.Members {
					allowed[member.ID] = true
				}
			}
			for _, id := range command.MemberIDs {
				if !allowed[id] {
					return artifacts.ErrArtifactNotFound
				}
			}
			if command.Operation == "restore" {
				found := false
				for _, marker := range view.Markers {
					if marker.ID == command.MarkerID {
						found = true
					}
				}
				if !found {
					return artifacts.ErrArtifactNotFound
				}
			}
			return nil
		}
	}
	if mutation.Source != nil && mutation.Source.SourceID != authority.SourceID {
		return Result{}, ErrInvalid
	}
	return s.Mutate(ctx, authority, idempotencyKey, mutation)
}
