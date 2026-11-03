package auditimport

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type ArtifactAccess interface {
	DescribeRunBinding(context.Context, string, contracts.ArtifactRef) (auditstore.ExactArtifact, bool, error)
	PutImmutableAudit(context.Context, string, contracts.ArtifactRef, artifacts.Payload) (auditstore.ExactArtifact, error)
	WithClaim(auditstore.ControllerClaim) ArtifactAccess
	ReadRunExact(context.Context, string, contracts.ArtifactRef) (auditstore.ExactArtifact, []byte, error)
	RetainRunExact(context.Context, string, string, []RunRetention) ([]auditstore.ExactArtifact, error)
}

func (a *ServiceArtifactAccess) DescribeRunBinding(ctx context.Context, runID string, ref contracts.ArtifactRef) (auditstore.ExactArtifact, bool, error) {
	store, err := a.service.Run(runID)
	if err != nil {
		return auditstore.ExactArtifact{}, false, err
	}
	metadata, err := store.Metadata(ctx, ref)
	if err != nil {
		return auditstore.ExactArtifact{}, false, err
	}
	return metadata.Exact(), metadata.Frozen, nil
}

func (a *ServiceArtifactAccess) PutImmutableAudit(ctx context.Context, projectID string, target contracts.ArtifactRef, payload artifacts.Payload) (auditstore.ExactArtifact, error) {
	scope, err := artifacts.AuditScope(a.claim.AuditID)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	var written artifacts.WriteResult
	err = a.writes.WithArtifactWrite(ctx, a.claim, projectID, target.Namespace, func(service *artifacts.Service) error {
		var err error
		written, err = service.CreateFrozen(ctx, scope, target, payload)
		return err
	})
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	return contracts.ExactContent(written.Ref, written.MediaType, payload.Data), nil
}

// RunRetention names an exact revision of an Audit child Run and the
// create-only Audit-managed binding that retains it.
type RunRetention struct {
	Source auditstore.ExactArtifact
	Target contracts.ArtifactRef
}

type ServiceArtifactAccess struct {
	service *artifacts.Service
	writes  *hookstore.ClaimedArtifacts
	claim   auditstore.ControllerClaim
}

func NewArtifactAccess(service *artifacts.Service, writes *hookstore.ClaimedArtifacts) (*ServiceArtifactAccess, error) {
	if service == nil || writes == nil {
		return nil, errors.New("Audit import Artifact service is required")
	}
	return &ServiceArtifactAccess{service: service, writes: writes}, nil
}

func (a *ServiceArtifactAccess) WithClaim(claim auditstore.ControllerClaim) ArtifactAccess {
	bound := *a
	bound.claim = claim
	return &bound
}

func (a *ServiceArtifactAccess) ReadRunExact(
	ctx context.Context, runID string, ref contracts.ArtifactRef,
) (auditstore.ExactArtifact, []byte, error) {
	if err := ref.ValidateExact(); err != nil {
		return auditstore.ExactArtifact{}, nil, artifacts.ErrExactRevisionRequired
	}
	store, err := a.service.Run(runID)
	if err != nil {
		return auditstore.ExactArtifact{}, nil, err
	}
	metadata, err := store.Metadata(ctx, ref)
	if err != nil {
		return auditstore.ExactArtifact{}, nil, err
	}
	read, err := store.ReadExact(ctx, metadata.Exact())
	if err != nil {
		return auditstore.ExactArtifact{}, nil, err
	}
	return metadata.Exact(), append([]byte(nil), read.Payload.Data...), nil
}

// RetainRunExact retains each source, an exact revision of the Audit child
// Run runID, as its create-only, frozen Audit-managed binding target of
// projectID, with audit_import lineage and without copying bytes, and returns
// the retained descriptors in request order. The bindings stay out of public
// Project projections until the Audit receipt that names them commits.
//
// Retentions commit in order in artifact-write transactions of at most
// auditdomain.MaximumCollectionTransactionWork retentions each (LIM-AUX-7,
// AUX-R98). Each transaction holds the live controller claim and the Audit
// row, like every claimed artifact write, so takeover and purge serialize
// with every retention. A failed transaction leaves earlier ones committed
// but unlinked; a retry after it, or after a lost acknowledgement, finds each
// binding retained from the same source and returns it.
func (a *ServiceArtifactAccess) RetainRunExact(
	ctx context.Context, runID, projectID string, retentions []RunRetention,
) ([]auditstore.ExactArtifact, error) {
	retained := make([]auditstore.ExactArtifact, 0, len(retentions))
	for start := 0; start < len(retentions); start += auditdomain.MaximumCollectionTransactionWork {
		chunk := retentions[start:min(start+auditdomain.MaximumCollectionTransactionWork, len(retentions))]
		namespace := chunk[0].Target.Namespace
		for _, retention := range chunk {
			if retention.Target.Namespace != namespace {
				return nil, fmt.Errorf("%w: retained Audit bindings span namespaces", auditstore.ErrInvalid)
			}
		}
		var written []auditstore.ExactArtifact
		err := a.writes.WithArtifactWrite(ctx, a.claim, projectID, namespace, func(service *artifacts.Service) error {
			// A definite transaction conflict replays this callback.
			written = make([]auditstore.ExactArtifact, 0, len(chunk))
			bound := *a
			bound.service = service
			for _, retention := range chunk {
				one, retainErr := bound.retainRunExact(ctx, runID, retention.Source, projectID, retention.Target)
				if retainErr != nil {
					return retainErr
				}
				written = append(written, one)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
		retained = append(retained, written...)
	}
	return retained, nil
}

func (a *ServiceArtifactAccess) retainRunExact(ctx context.Context, runID string, source auditstore.ExactArtifact, projectID string, target contracts.ArtifactRef) (auditstore.ExactArtifact, error) {
	run, err := artifacts.RunScope(runID)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	audit, err := artifacts.AuditScope(a.claim.AuditID)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	if source.Ref.Revision == nil {
		return auditstore.ExactArtifact{}, artifacts.ErrExactRevisionRequired
	}
	result, err := a.service.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageAuditOutput, Source: run, SourceRef: source.Ref, Target: audit, TargetRef: target, Frozen: true, Replay: true})
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	retained := auditstore.ExactArtifact{Ref: result.TargetRef, Digest: source.Digest, MediaType: result.MediaType, SizeBytes: result.Size}
	if retained.MediaType != source.MediaType || retained.SizeBytes != source.SizeBytes {
		return auditstore.ExactArtifact{}, blob.ErrIntegrity
	}
	return retained, nil
}
