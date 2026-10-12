package runrepeat

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// recordRef is the repeat-request binding of a RunScope, in the hidden
// contractor-system Namespace that only this package writes (ART-R17).
// Generic Run writes and the private allocation Artifact API cannot reach it.
var recordRef = contracts.ArtifactRef{
	Namespace: artifactpolicy.RunSystemNamespace,
	Name:      artifactpolicy.RunRepeatRequestName,
}

// Retain stores encoded, the result of Encode, as the repeat-request record of
// runID. The record is created once, by the transaction that creates the Run,
// so pass a Service bound to that transaction.
func Retain(ctx context.Context, service *artifacts.Service, runID string, encoded []byte) error {
	run, err := artifacts.RunScope(runID)
	if err != nil {
		return err
	}
	_, err = service.Create(ctx, run, recordRef, artifacts.Payload{
		MediaType: artifactpolicy.RunRepeatRequestMediaType,
		Data:      append([]byte(nil), encoded...),
	})
	return err
}

// RunScopes opens the RunScope of a Run; *artifacts.Service implements it.
type RunScopes interface {
	Run(runID string) (artifacts.ScopedStore, error)
}

// Load reads and decodes the repeat-request record of runID. A Run without a
// record is artifacts.ErrArtifactNotFound, and a record of another media type
// is artifacts.ErrArtifactIntegrity.
func Load(ctx context.Context, scopes RunScopes, runID string) (Snapshot, error) {
	run, err := scopes.Run(runID)
	if err != nil {
		return Snapshot{}, err
	}
	retained, err := run.Read(ctx, recordRef)
	if err != nil {
		return Snapshot{}, err
	}
	if retained.Payload.MediaType != artifactpolicy.RunRepeatRequestMediaType {
		return Snapshot{}, artifacts.ErrArtifactIntegrity
	}
	return Decode(retained.Payload.Data)
}
