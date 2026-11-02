package auditimport

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type ArtifactAccess interface {
	WithClaim(auditstore.ControllerClaim) ArtifactAccess
	ReadProjectExact(context.Context, string, auditstore.ExactArtifact) ([]byte, error)
	ReadRunBinding(context.Context, string, contracts.ArtifactRef) (auditstore.ExactArtifact, []byte, bool, error)
	ReadRunExact(context.Context, string, contracts.ArtifactRef) (auditstore.ExactArtifact, []byte, error)
	DescribeRunExact(context.Context, string, []contracts.ArtifactRef) ([]auditstore.ExactArtifact, error)
	RetainRunExact(context.Context, string, string, []RunRetention) ([]auditstore.ExactArtifact, error)
	PutImmutableProject(context.Context, string, contracts.ArtifactRef, artifacts.Payload) (auditstore.ExactArtifact, error)
}

// RunRetention names an exact revision of an Audit child Run and the
// create-only Audit-managed binding that retains it.
type RunRetention struct {
	AuditOwned bool
	Source     auditstore.ExactArtifact
	Target     contracts.ArtifactRef
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

// ReadProjectExact reads expected through the one exact read
// (artifacts.ScopedStore.ReadExact). Digest-only descriptors of rounds and
// executions pin the bytes; complete ones also pin media type and size.
func (a *ServiceArtifactAccess) ReadProjectExact(
	ctx context.Context, projectID string, expected auditstore.ExactArtifact,
) ([]byte, error) {
	store, err := a.service.Project(projectID)
	if err != nil {
		return nil, err
	}
	read, err := store.ReadExact(ctx, expected)
	if err != nil {
		return nil, err
	}
	return append([]byte(nil), read.Payload.Data...), nil
}

func (a *ServiceArtifactAccess) ReadRunBinding(
	ctx context.Context, runID string, ref contracts.ArtifactRef,
) (auditstore.ExactArtifact, []byte, bool, error) {
	store, err := a.service.Run(runID)
	if err != nil {
		return auditstore.ExactArtifact{}, nil, false, err
	}
	metadata, err := store.Metadata(ctx, ref)
	if err != nil {
		return auditstore.ExactArtifact{}, nil, false, err
	}
	read, err := store.ReadExact(ctx, metadata.Exact())
	if err != nil {
		return auditstore.ExactArtifact{}, nil, false, err
	}
	return metadata.Exact(), append([]byte(nil), read.Payload.Data...), metadata.Frozen, nil
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

// DescribeRunExact returns the exact descriptors of refs, exact revisions of
// the Audit child Run runID, in request order without reading their payloads
// (artifacts.ScopedStore.DescribeExact, LIM-ART-8). A reference that is not
// an exact revision is ErrExactRevisionRequired before any read; a revision
// the Run does not hold is ErrArtifactNotFound. Any other error leaves the
// references unverified, so the caller must not treat it as an invalid
// reference.
func (a *ServiceArtifactAccess) DescribeRunExact(
	ctx context.Context, runID string, refs []contracts.ArtifactRef,
) ([]auditstore.ExactArtifact, error) {
	for _, ref := range refs {
		if err := ref.ValidateExact(); err != nil {
			return nil, artifacts.ErrExactRevisionRequired
		}
	}
	store, err := a.service.Run(runID)
	if err != nil {
		return nil, err
	}
	return store.DescribeExact(ctx, refs)
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
				one, retainErr := bound.retainRunExact(ctx, runID, retention.Source, projectID, retention.Target, retention.AuditOwned)
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

func (a *ServiceArtifactAccess) retainRunExact(
	ctx context.Context, runID string, source auditstore.ExactArtifact, projectID string, target contracts.ArtifactRef, auditOwned bool,
) (auditstore.ExactArtifact, error) {
	run, err := artifacts.RunScope(runID)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	project, err := artifacts.ProjectScope(projectID)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	if source.Ref.Revision == nil {
		return auditstore.ExactArtifact{}, artifacts.ErrExactRevisionRequired
	}
	kind := artifacts.LineageAuditImport
	if auditOwned {
		project, err = artifacts.AuditScope(a.claim.AuditID)
		if err != nil {
			return auditstore.ExactArtifact{}, err
		}
		kind = artifacts.LineageAuditOutput
	}
	result, err := a.service.Fork(ctx, artifacts.Fork{
		Kind: kind, Source: run, SourceRef: source.Ref,
		Target: project, TargetRef: target, Frozen: true, Replay: auditOwned,
	})
	if err == nil {
		retained := auditstore.ExactArtifact{
			Ref: result.TargetRef, Digest: source.Digest,
			MediaType: result.MediaType, SizeBytes: result.Size,
		}
		if retained.MediaType != source.MediaType || retained.SizeBytes != source.SizeBytes {
			return auditstore.ExactArtifact{}, artifacts.ErrArtifactIntegrity
		}
		return retained, nil
	}
	if !errors.Is(err, artifacts.ErrArtifactConflict) {
		return auditstore.ExactArtifact{}, err
	}
	store, scopeErr := a.service.Scoped(project)
	if scopeErr != nil {
		return auditstore.ExactArtifact{}, scopeErr
	}
	metadata, metadataErr := store.Metadata(ctx, target)
	if metadataErr != nil {
		return auditstore.ExactArtifact{}, metadataErr
	}
	// Crash replay: the retained binding must hold the source's content.
	expected := source
	expected.Ref = metadata.Ref
	if err := expected.Verify(metadata.Exact()); err != nil {
		return auditstore.ExactArtifact{}, fmt.Errorf("%w: Audit retained binding collision", artifacts.ErrArtifactIntegrity)
	}
	lineage, lineageErr := store.ListLineage(ctx, metadata.Ref, artifacts.LineagePageQuery{Limit: 2})
	if lineageErr != nil {
		return auditstore.ExactArtifact{}, lineageErr
	}
	if len(lineage) != 1 || lineage[0].Kind != artifacts.LineageAuditImport ||
		lineage[0].SourceScope != artifacts.ScopeRun || lineage[0].SourceScopeID != runID ||
		lineage[0].TargetScope != artifacts.ScopeProject || lineage[0].TargetScopeID != projectID ||
		!lineage[0].Source.SameExact(source.Ref) || !lineage[0].Target.SameExact(metadata.Ref) {
		return auditstore.ExactArtifact{}, fmt.Errorf("%w: Audit retained lineage collision", artifacts.ErrArtifactIntegrity)
	}
	return metadata.Exact(), nil
}

// PutImmutableProject writes a content-named report or retained scan package
// as a frozen Audit artifact. CreateFrozen recovers a lost acknowledgement;
// another payload under the name is an integrity failure.
func (a *ServiceArtifactAccess) PutImmutableProject(
	ctx context.Context,
	projectID string,
	target contracts.ArtifactRef,
	payload artifacts.Payload,
) (auditstore.ExactArtifact, error) {
	project, err := artifacts.ProjectScope(projectID)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	var written artifacts.WriteResult
	err = a.writes.WithArtifactWrite(ctx, a.claim, projectID, target.Namespace, func(service *artifacts.Service) error {
		var writeErr error
		written, writeErr = service.CreateFrozen(ctx, project, target, payload)
		return writeErr
	})
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	return contracts.ExactContent(written.Ref, written.MediaType, payload.Data), nil
}
