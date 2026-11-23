package auditservice

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

type MaterialPublicationRequest struct {
	AuditID string                `json:"auditId"`
	Source  contracts.ArtifactRef `json:"source"`
	Target  contracts.ArtifactRef `json:"target"`
}

type MaterialPublication struct {
	AuditID   string                `json:"auditId"`
	ProjectID string                `json:"projectId"`
	Source    contracts.ArtifactRef `json:"source"`
	Artifact  contracts.ArtifactRef `json:"artifact"`
	Replayed  bool                  `json:"-"`
}

// PublishMaterial retains a selected accepted output in its owning Project.
// The Project receipt outlives source Audit deletion and replays without
// reading/reuploading content or repeating the producer.
func (s *Service) PublishMaterial(ctx context.Context, ownerID, projectID, key string, request MaterialPublicationRequest, expected *string) (MaterialPublication, error) {
	if !contracts.ValidResourceID(projectID) || !contracts.ValidIdempotencyKey(key) || !contracts.ValidResourceID(request.AuditID) ||
		request.Source.ValidateExact() != nil || request.Target.Validate() != nil || request.Target.Revision != nil ||
		artifactpolicy.IsPurposeReservedProjectNamespace(request.Target.Namespace) || (expected != nil && *expected == "") {
		return MaterialPublication{}, ErrInvalid
	}
	encoded, err := json.Marshal(struct {
		Request  MaterialPublicationRequest
		Expected *string
	}{request, expected})
	if err != nil {
		return MaterialPublication{}, err
	}
	digest := auditdomain.DigestBytes(encoded)
	var result MaterialPublication
	err = persistencepostgres.Within(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		result = MaterialPublication{}
		if _, err := projectstore.NewPostgresStore(tx).LockActiveAuditProject(ctx, ownerID, projectID); err != nil {
			return err
		}
		receipt, replay, err := persistencepostgres.KeyedMutation(ctx, tx,
			persistencepostgres.ProjectReceiptKey(ownerID, projectID, request.AuditID, "audit-material-publication", key), digest,
			func() (persistencepostgres.Receipt, error) {
				owners := ownerstore.NewTx(tx, s.storage)
				if err := owners.LockAuditForResume(ctx, ownerID, request.AuditID); err != nil {
					return persistencepostgres.Receipt{}, err
				}
				audit, err := querystore.New(tx, s.storage).Get(ctx, ownerID, request.AuditID)
				if err != nil {
					return persistencepostgres.Receipt{}, err
				}
				if audit.ProjectID != projectID || audit.State == auditstore.AuditDeleting {
					return persistencepostgres.Receipt{}, auditstore.ErrNotFound
				}
				exact, err := querystore.New(tx, s.storage).Material(ctx, audit.AuditID, request.Source)
				if err != nil {
					return persistencepostgres.Receipt{}, err
				}
				service := artifacts.NewService(s.storage.Repository(tx))
				source, _ := artifacts.AuditScope(audit.AuditID)
				store, _ := service.Scoped(source)
				described, err := store.DescribeExact(ctx, []contracts.ArtifactRef{request.Source})
				if err != nil {
					return persistencepostgres.Receipt{}, err
				}
				if len(described) != 1 || exact.Verify(described[0]) != nil {
					return persistencepostgres.Receipt{}, blob.ErrIntegrity
				}
				target, _ := artifacts.ProjectScope(audit.ProjectID)
				fork, err := service.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageAuditPublish, Source: source, SourceRef: request.Source,
					Target: target, TargetRef: request.Target, ExpectedRevision: expected})
				if err != nil {
					return persistencepostgres.Receipt{}, err
				}
				result = MaterialPublication{AuditID: audit.AuditID, ProjectID: audit.ProjectID, Source: fork.SourceRef, Artifact: fork.TargetRef}
				body, err := json.Marshal(result)
				return persistencepostgres.Receipt{Operation: "audit-material-publication", ResourceID: audit.AuditID, OwnerID: ownerID, ProjectID: projectID, Response: body}, err
			})
		if errors.Is(err, persistencepostgres.ErrReceiptMismatch) {
			return contracts.ErrIdempotencyConflict
		}
		if err != nil {
			return err
		}
		if err := strictjson.Decode(receipt.Response, &result); err != nil {
			return err
		}
		if result.AuditID != request.AuditID || result.ProjectID != projectID || !result.Source.SameExact(request.Source) ||
			result.Artifact.ValidateExact() != nil || result.Artifact.Namespace != request.Target.Namespace || result.Artifact.Name != request.Target.Name {
			return ErrInvalid
		}
		result.Replayed = replay
		return nil
	})
	var pending *artifacts.HistoryMaintenanceError
	if errors.As(err, &pending) {
		err = s.storage.Repository(s.pool).MaintainHistory(ctx, pending)
	}
	return result, err
}
