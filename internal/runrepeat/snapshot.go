// Package runrepeat owns the bounded, non-secret fragment of an ordinary Run
// request needed to construct a future editable repeat draft.
package runrepeat

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/clone"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

const (
	SchemaVersion   = "contractor.run-repeat-request/v1"
	MaxSnapshotSize = 1 << 20
)

var ErrInvalidSnapshot = errors.New("Run repeat request snapshot is invalid")

// Snapshot is the retained authority for exact input sources and requested
// execution overrides. Credential values are identities, never token bytes.
// Inputs are the original source-scope refs, not RunScope forks.
type Snapshot struct {
	SchemaVersion   string                           `json:"schemaVersion"`
	Workflow        resolved.WorkflowRef             `json:"workflow"`
	ProjectID       *string                          `json:"projectId,omitempty"`
	Inputs          map[string]contracts.ArtifactRef `json:"inputs"`
	ExecutionConfig resolved.ExecutionConfigPatch    `json:"executionConfig"`
}

func Encode(snapshot Snapshot) ([]byte, error) {
	snapshot.SchemaVersion = SchemaVersion
	snapshot.ProjectID = clone.Pointer(snapshot.ProjectID)
	snapshot.Inputs = cloneInputs(snapshot.Inputs)
	if err := validate(snapshot); err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(snapshot)
	if err != nil {
		return nil, fmt.Errorf("%w: encode: %v", ErrInvalidSnapshot, err)
	}
	if len(encoded) > MaxSnapshotSize {
		return nil, fmt.Errorf("%w: snapshot exceeds %d bytes", ErrInvalidSnapshot, MaxSnapshotSize)
	}
	return encoded, nil
}

func Decode(encoded []byte) (Snapshot, error) {
	if len(encoded) == 0 || len(encoded) > MaxSnapshotSize {
		return Snapshot{}, fmt.Errorf("%w: invalid size", ErrInvalidSnapshot)
	}
	// A missing patch is not an explicitly retained empty patch.
	var stored struct {
		SchemaVersion   string                           `json:"schemaVersion"`
		Workflow        resolved.WorkflowRef             `json:"workflow"`
		ProjectID       *string                          `json:"projectId,omitempty"`
		Inputs          map[string]contracts.ArtifactRef `json:"inputs"`
		ExecutionConfig *resolved.ExecutionConfigPatch   `json:"executionConfig"`
	}
	if err := strictjson.Decode(encoded, &stored); errors.Is(err, strictjson.ErrTrailingData) {
		return Snapshot{}, fmt.Errorf("%w: trailing JSON value", ErrInvalidSnapshot)
	} else if err != nil {
		return Snapshot{}, fmt.Errorf("%w: decode: %v", ErrInvalidSnapshot, err)
	}
	if stored.ExecutionConfig == nil {
		return Snapshot{}, fmt.Errorf("%w: executionConfig must be an object", ErrInvalidSnapshot)
	}
	snapshot := Snapshot{
		SchemaVersion: stored.SchemaVersion, Workflow: stored.Workflow, ProjectID: stored.ProjectID,
		Inputs: stored.Inputs, ExecutionConfig: *stored.ExecutionConfig,
	}
	if err := validate(snapshot); err != nil {
		return Snapshot{}, err
	}
	snapshot.ProjectID = clone.Pointer(snapshot.ProjectID)
	snapshot.Inputs = cloneInputs(snapshot.Inputs)
	return snapshot, nil
}

func validate(snapshot Snapshot) error {
	if snapshot.SchemaVersion != SchemaVersion || strings.TrimSpace(snapshot.Workflow.Name) == "" ||
		strings.TrimSpace(snapshot.Workflow.Version) == "" {
		return fmt.Errorf("%w: schema or Workflow identity", ErrInvalidSnapshot)
	}
	if snapshot.ProjectID != nil && strings.TrimSpace(*snapshot.ProjectID) == "" {
		return fmt.Errorf("%w: empty Project identity", ErrInvalidSnapshot)
	}
	if snapshot.Inputs == nil {
		return fmt.Errorf("%w: inputs must be an object", ErrInvalidSnapshot)
	}
	for slot, ref := range snapshot.Inputs {
		if contracts.ValidateArtifactName(slot) != nil || ref.Validate() != nil || ref.Revision == nil {
			return fmt.Errorf("%w: input %q is not exact", ErrInvalidSnapshot, slot)
		}
	}
	return nil
}

func cloneInputs(source map[string]contracts.ArtifactRef) map[string]contracts.ArtifactRef {
	result := make(map[string]contracts.ArtifactRef, len(source))
	for slot, ref := range source {
		copy := ref
		copy.Revision = clone.Pointer(ref.Revision)
		result[slot] = copy
	}
	return result
}
