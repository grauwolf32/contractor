// Package runrepeat owns the bounded, non-secret fragment of an ordinary Run
// request needed to construct a future editable repeat draft.
package runrepeat

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	SchemaVersion   = "contractor.run-repeat-request/v1"
	MaxSnapshotSize = 1 << 20
)

var ErrInvalidSnapshot = errors.New("Run repeat request snapshot is invalid")

// Snapshot deliberately stores only request fields that WorkflowRun and input
// lineage cannot reproduce. Credential values are identities, never token
// bytes. Inputs are the original source-scope refs, not RunScope forks.
type Snapshot struct {
	SchemaVersion   string                           `json:"schemaVersion"`
	Workflow        config.WorkflowRef               `json:"workflow"`
	ProjectID       *string                          `json:"projectId,omitempty"`
	Inputs          map[string]contracts.ArtifactRef `json:"inputs"`
	ExecutionConfig config.ExecutionConfigPatch      `json:"executionConfig"`
}

func Encode(snapshot Snapshot) ([]byte, error) {
	snapshot.SchemaVersion = SchemaVersion
	snapshot.ProjectID = cloneString(snapshot.ProjectID)
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
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.DisallowUnknownFields()
	var snapshot Snapshot
	if err := decoder.Decode(&snapshot); err != nil {
		return Snapshot{}, fmt.Errorf("%w: decode: %v", ErrInvalidSnapshot, err)
	}
	if err := expectEOF(decoder); err != nil {
		return Snapshot{}, err
	}
	if err := validate(snapshot); err != nil {
		return Snapshot{}, err
	}
	snapshot.ProjectID = cloneString(snapshot.ProjectID)
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

func expectEOF(decoder *json.Decoder) error {
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return fmt.Errorf("%w: trailing JSON value", ErrInvalidSnapshot)
	}
	return nil
}

func cloneInputs(source map[string]contracts.ArtifactRef) map[string]contracts.ArtifactRef {
	result := make(map[string]contracts.ArtifactRef, len(source))
	for slot, ref := range source {
		copy := ref
		copy.Revision = cloneString(ref.Revision)
		result[slot] = copy
	}
	return result
}

func cloneString(value *string) *string {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}
