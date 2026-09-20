// Package scan executes frozen, bounded scan plans through ordinary tool Workers.
package scan

import (
	"bytes"
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
)

// ArtifactStore receives a trusted Run ID from the immutable invocation. It
// never resolves a scope, filename or arbitrary revision from scanner output.
type ArtifactStore interface {
	Read(context.Context, string, contracts.ArtifactRef, int) (artifacts.Payload, error)
	Resolve(context.Context, string, contracts.ArtifactRef) (contracts.ArtifactRef, error)
	Create(context.Context, string, contracts.ArtifactRef, artifacts.Payload) (contracts.ArtifactRef, error)
}

type ServiceArtifacts struct{ service *artifacts.Service }

func NewServiceArtifacts(service *artifacts.Service) (*ServiceArtifacts, error) {
	if service == nil {
		return nil, fmt.Errorf("scan planner requires an Artifact service")
	}
	return &ServiceArtifacts{service: service}, nil
}

// Resolve pins the report binding authored for one specific stage/job. It is
// used only when a failed Worker response omits the report it already wrote.
func (s *ServiceArtifacts) Resolve(ctx context.Context, runID string, target contracts.ArtifactRef) (contracts.ArtifactRef, error) {
	if target.Validate() != nil || target.Revision != nil {
		return contracts.ArtifactRef{}, fmt.Errorf("invalid scan report binding")
	}
	store, err := s.service.Run(runID)
	if err != nil {
		return contracts.ArtifactRef{}, err
	}
	metadata, err := store.Metadata(ctx, target)
	if err != nil {
		return contracts.ArtifactRef{}, err
	}
	if metadata.Ref.ValidateExact() != nil || metadata.Ref.Namespace != target.Namespace || metadata.Ref.Name != target.Name {
		return contracts.ArtifactRef{}, fmt.Errorf("scan report binding differs")
	}
	return planner.CloneArtifactRef(metadata.Ref), nil
}

func (s *ServiceArtifacts) Read(ctx context.Context, runID string, ref contracts.ArtifactRef, limit int) (artifacts.Payload, error) {
	if ref.ValidateExact() != nil || limit < 1 || limit > 4*1024*1024 {
		return artifacts.Payload{}, fmt.Errorf("invalid exact scan artifact read")
	}
	store, err := s.service.Run(runID)
	if err != nil {
		return artifacts.Payload{}, err
	}
	metadata, err := store.Metadata(ctx, planner.CloneArtifactRef(ref))
	if err != nil {
		return artifacts.Payload{}, err
	}
	if !sameRef(metadata.Ref, ref) || metadata.Size < 0 || metadata.Size > int64(limit) {
		return artifacts.Payload{}, fmt.Errorf("scan artifact metadata differs or exceeds its bound")
	}
	result, err := store.Read(ctx, planner.CloneArtifactRef(ref))
	if err != nil {
		return artifacts.Payload{}, err
	}
	if !sameRef(result.Ref, ref) || result.Payload.MediaType != metadata.MediaType || int64(len(result.Payload.Data)) != metadata.Size || len(result.Payload.Data) > limit {
		return artifacts.Payload{}, fmt.Errorf("scan artifact revision or content differs")
	}
	return artifacts.Payload{MediaType: result.Payload.MediaType, Data: append([]byte(nil), result.Payload.Data...)}, nil
}

// Create uses stable bindings and create-only writes. A lost acknowledgement can
// be recovered by comparing the existing immutable payload, never by overwriting
// an artifact that another invocation or Worker may have changed.
func (s *ServiceArtifacts) Create(ctx context.Context, runID string, target contracts.ArtifactRef, payload artifacts.Payload) (contracts.ArtifactRef, error) {
	if target.Validate() != nil || target.Revision != nil || len(payload.Data) > 4*1024*1024 {
		return contracts.ArtifactRef{}, fmt.Errorf("invalid scan artifact creation")
	}
	store, err := s.service.Run(runID)
	if err != nil {
		return contracts.ArtifactRef{}, err
	}
	result, err := store.Write(ctx, target, payload, nil)
	if err == nil {
		if result.Ref.ValidateExact() != nil || result.Ref.Namespace != target.Namespace || result.Ref.Name != target.Name || result.MediaType != payload.MediaType || result.Size != int64(len(payload.Data)) {
			return contracts.ArtifactRef{}, fmt.Errorf("scan artifact creation returned inconsistent metadata")
		}
		return planner.CloneArtifactRef(result.Ref), nil
	}
	if !errors.Is(err, artifacts.ErrArtifactConflict) {
		return contracts.ArtifactRef{}, err
	}
	metadata, err := store.Metadata(ctx, target)
	if err != nil {
		return contracts.ArtifactRef{}, err
	}
	if metadata.Ref.ValidateExact() != nil || metadata.Ref.Namespace != target.Namespace || metadata.Ref.Name != target.Name || metadata.MediaType != payload.MediaType || metadata.Size != int64(len(payload.Data)) {
		return contracts.ArtifactRef{}, fmt.Errorf("existing scan artifact does not match its immutable input")
	}
	current, err := s.Read(ctx, runID, metadata.Ref, max(1, len(payload.Data)))
	if err != nil {
		return contracts.ArtifactRef{}, err
	}
	if current.MediaType != payload.MediaType || !bytes.Equal(current.Data, payload.Data) {
		return contracts.ArtifactRef{}, fmt.Errorf("existing scan artifact content differs")
	}
	return planner.CloneArtifactRef(metadata.Ref), nil
}

func sameRef(left, right contracts.ArtifactRef) bool {
	return left.Namespace == right.Namespace && left.Name == right.Name && left.Revision != nil && right.Revision != nil && *left.Revision == *right.Revision
}
