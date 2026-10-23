// Package runrepeat owns the bounded, non-secret fragment of an ordinary Run
// request needed to construct a future editable repeat draft.
package runrepeat

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/clone"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

const (
	SchemaVersion   = "contractor.run-repeat-request/v2"
	MaxSnapshotSize = 1 << 20
)

var ErrInvalidSnapshot = errors.New("Run repeat request snapshot is invalid")

// InputSource retains an exact original reference and its authorized scope kind.
type InputSource struct {
	SourceScope artifacts.ScopeKind   `json:"sourceScope"`
	Artifact    contracts.ArtifactRef `json:"artifact"`
}

// Snapshot is the retained authority for source inputs and requested execution
// overrides. Credential values are identities, never token bytes.
type Snapshot struct {
	SchemaVersion   string                        `json:"schemaVersion"`
	Workflow        resolved.WorkflowRef          `json:"workflow"`
	ProjectID       *string                       `json:"projectId,omitempty"`
	Inputs          map[string]InputSource        `json:"inputs"`
	ExecutionConfig resolved.ExecutionConfigPatch `json:"executionConfig"`
}

func Encode(snapshot Snapshot) ([]byte, error) {
	snapshot.SchemaVersion = SchemaVersion
	snapshot.ProjectID = clone.Pointer(snapshot.ProjectID)
	snapshot.Inputs = cloneInputSources(snapshot.Inputs)
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
		SchemaVersion   string                         `json:"schemaVersion"`
		Workflow        resolved.WorkflowRef           `json:"workflow"`
		ProjectID       *string                        `json:"projectId,omitempty"`
		Inputs          map[string]InputSource         `json:"inputs"`
		ExecutionConfig *resolved.ExecutionConfigPatch `json:"executionConfig"`
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
	snapshot.Inputs = cloneInputSources(snapshot.Inputs)
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
	for slot, source := range snapshot.Inputs {
		if (source.SourceScope != artifacts.ScopeUser && source.SourceScope != artifacts.ScopeProject) || (source.SourceScope == artifacts.ScopeProject && snapshot.ProjectID == nil) || (source.SourceScope == artifacts.ScopeUser && snapshot.ProjectID != nil && slot != "rulepack") {
			return fmt.Errorf("%w: invalid input scope", ErrInvalidSnapshot)
		}
		ref := source.Artifact
		if contracts.ValidateArtifactName(slot) != nil || ref.ValidateExact() != nil {
			return fmt.Errorf("%w: input %q is not exact", ErrInvalidSnapshot, slot)
		}
	}
	return nil
}

func cloneInputSources(source map[string]InputSource) map[string]InputSource {
	if source == nil {
		return nil
	}
	result := make(map[string]InputSource, len(source))
	for name, input := range source {
		input.Artifact = input.Artifact.Clone()
		result[name] = input
	}
	return result
}
