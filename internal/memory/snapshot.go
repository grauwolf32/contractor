package memory

import (
	"bytes"
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/memorycontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

const snapshotSchema = "contractor.memory-snapshot/v1"

var ErrSnapshotInvalid = errors.New("Memory snapshot selection, source or output is invalid")

// snapshotDocument contains intentionally exported note content and safe logical labels.
// Hidden revisions, physical refs, timestamps and ordinals are not exported.
type snapshotDocument struct {
	SchemaVersion string         `json:"schemaVersion" strictjson:"required"`
	Namespace     string         `json:"namespace" strictjson:"required"`
	Notes         []snapshotNote `json:"notes" strictjson:"required"`
}

type snapshotNote struct {
	Name        string   `json:"name" strictjson:"required"`
	Content     string   `json:"content" strictjson:"required"`
	Description string   `json:"description" strictjson:"required"`
	Tags        []string `json:"tags" strictjson:"required"`
}

func buildSnapshot(ctx context.Context, scope artifacts.ScopedStore, selection memorycontract.SnapshotSelection) ([]byte, error) {
	snapshot := snapshotDocument{SchemaVersion: snapshotSchema, Namespace: selection.Namespace, Notes: make([]snapshotNote, 0, len(selection.Names))}
	for _, name := range selection.Names {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		binding, err := ArtifactName(name)
		if err != nil {
			return nil, ErrSnapshotInvalid
		}
		metadata, err := scope.Metadata(ctx, contracts.ArtifactRef{Namespace: selection.Namespace, Name: binding})
		if errors.Is(err, artifacts.ErrArtifactNotFound) {
			return nil, ErrSnapshotInvalid
		}
		if err != nil {
			return nil, err
		}
		if metadata.MediaType != MediaType || metadata.Size < 0 || metadata.Size > MaximumPayloadBytes {
			return nil, ErrSnapshotInvalid
		}
		result, err := scope.ReadExact(ctx, metadata.Exact())
		if err != nil {
			return nil, err
		}
		note, err := Decode(binding, result.Payload.Data)
		if err != nil {
			return nil, ErrSnapshotInvalid
		}
		snapshot.Notes = append(snapshot.Notes, snapshotNote{Name: note.Name, Content: note.Content, Description: note.Description, Tags: note.Tags})
	}
	data, err := strictjson.Canonical(snapshot)
	if err != nil || len(data) > memorycontract.MaximumSnapshotBytes {
		return nil, ErrSnapshotInvalid
	}
	return data, nil
}

func validateSnapshot(data []byte, selection memorycontract.SnapshotSelection) error {
	if len(data) > memorycontract.MaximumSnapshotBytes {
		return ErrSnapshotInvalid
	}
	var snapshot snapshotDocument
	if strictjson.Decode(data, &snapshot) != nil || snapshot.SchemaVersion != snapshotSchema || snapshot.Namespace != selection.Namespace || len(snapshot.Notes) != len(selection.Names) {
		return ErrSnapshotInvalid
	}
	for i, note := range snapshot.Notes {
		if note.Name != selection.Names[i] || note.Tags == nil {
			return ErrSnapshotInvalid
		}
		binding, err := ArtifactName(note.Name)
		if err != nil {
			return ErrSnapshotInvalid
		}
		// Validate the original representation. Encode normalizes tags and can
		// hide malformed or oversized source notes before its byte check.
		original, err := strictjson.Canonical(StoredNote{SchemaVersion: SchemaVersion, Name: note.Name, Content: note.Content, Description: note.Description, Tags: note.Tags})
		if err != nil {
			return ErrSnapshotInvalid
		}
		if _, err := Decode(binding, original); err != nil {
			return ErrSnapshotInvalid
		}
	}
	canonical, err := strictjson.Canonical(snapshot)
	if err != nil || !bytes.Equal(data, canonical) {
		return ErrSnapshotInvalid
	}
	return nil
}
