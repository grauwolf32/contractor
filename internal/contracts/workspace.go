package contracts

// Allocation workspace contracts: the capability surface a Runtime Agent
// advertises and the workspace spec an allocation pins.
// Mirrors the Python runtime's contracts/workspace.py.

import (
	"strings"

	"golang.org/x/text/unicode/norm"
)

type WorkspaceMode string

const (
	WorkspaceModeDirect  WorkspaceMode = "direct"
	WorkspaceModeOverlay WorkspaceMode = "overlay"
)

func (m WorkspaceMode) Validate() error {
	switch m {
	case WorkspaceModeDirect, WorkspaceModeOverlay:
		return nil
	default:
		return invalidf("workspace mode is invalid")
	}
}

type WorkspaceStorage string

const (
	WorkspaceStorageLocal  WorkspaceStorage = "local"
	WorkspaceStorageMemory WorkspaceStorage = "memory"
)

func (s WorkspaceStorage) Validate() error {
	switch s {
	case WorkspaceStorageLocal, WorkspaceStorageMemory:
		return nil
	default:
		return invalidf("workspace storage is invalid")
	}
}

type WorkspaceLimits struct {
	MaxFiles            int   `json:"maxFiles"`
	MaxExpandedBytes    int64 `json:"maxExpandedBytes"`
	MaxManagedTextBytes int64 `json:"maxManagedTextBytes"`
	MaxFileBytes        int64 `json:"maxFileBytes"`
}

func (l WorkspaceLimits) Validate() error {
	if l.MaxFiles <= 0 || l.MaxExpandedBytes <= 0 || l.MaxManagedTextBytes <= 0 || l.MaxFileBytes <= 0 ||
		l.MaxFileBytes > l.MaxExpandedBytes || l.MaxManagedTextBytes > l.MaxExpandedBytes {
		return invalidf("workspace limits are invalid")
	}
	return nil
}

type WorkspaceCapabilities struct {
	Storage WorkspaceStorage `json:"storage"`
	Modes   []WorkspaceMode  `json:"modes"`
	Limits  WorkspaceLimits  `json:"limits"`
}

func (c WorkspaceCapabilities) Validate() error {
	if err := c.Storage.Validate(); err != nil {
		return err
	}
	if len(c.Modes) == 0 || len(c.Modes) > 2 {
		return invalidf("workspace capability modes must be a non-empty bounded array")
	}
	previous := ""
	for _, mode := range c.Modes {
		if err := mode.Validate(); err != nil {
			return err
		}
		if string(mode) <= previous {
			return invalidf("workspace capability modes must be sorted and unique")
		}
		previous = string(mode)
	}
	return c.Limits.Validate()
}

type AllocationWorkspaceSource struct {
	Artifact ArtifactRef `json:"artifact"`
	Target   string      `json:"target"`
}

type AllocationWorkspaceState struct {
	Artifact ArtifactRef `json:"artifact"`
}

type AllocationWorkspaceExport struct {
	State string `json:"state"`
	Diff  string `json:"diff"`
}

type AllocationWorkspaceSpec struct {
	Mode    WorkspaceMode               `json:"mode"`
	Sources []AllocationWorkspaceSource `json:"sources"`
	State   *AllocationWorkspaceState   `json:"state,omitempty"`
	Export  *AllocationWorkspaceExport  `json:"export,omitempty"`
}

// CloneAllocationWorkspaceSpec returns a detached copy suitable for crossing
// ownership boundaries between Scheduler, Control Plane and Runtime clients.
func CloneAllocationWorkspaceSpec(source *AllocationWorkspaceSpec) *AllocationWorkspaceSpec {
	if source == nil {
		return nil
	}
	result := *source
	result.Sources = make([]AllocationWorkspaceSource, len(source.Sources))
	for index, item := range source.Sources {
		result.Sources[index] = item
		result.Sources[index].Artifact = cloneWorkspaceArtifactRef(item.Artifact)
	}
	if source.State != nil {
		state := *source.State
		state.Artifact = cloneWorkspaceArtifactRef(source.State.Artifact)
		result.State = &state
	}
	if source.Export != nil {
		export := *source.Export
		result.Export = &export
	}
	return &result
}

func cloneWorkspaceArtifactRef(source ArtifactRef) ArtifactRef {
	result := source
	if source.Revision != nil {
		revision := *source.Revision
		result.Revision = &revision
	}
	return result
}

func (s AllocationWorkspaceSpec) Validate() error {
	if err := s.Mode.Validate(); err != nil {
		return err
	}
	if len(s.Sources) == 0 || len(s.Sources) > 32 {
		return invalidf("workspace sources must be a non-empty bounded array")
	}
	targets := make([]string, 0, len(s.Sources))
	for _, source := range s.Sources {
		if err := source.Artifact.ValidateExact(); err != nil {
			return err
		}
		if err := validateWorkspaceTarget(source.Target); err != nil {
			return err
		}
		targets = append(targets, source.Target)
	}
	for index, target := range targets {
		for otherIndex, other := range targets {
			if index == otherIndex {
				continue
			}
			if target == other || target == "" || strings.HasPrefix(other, target+"/") {
				return invalidf("workspace source targets must be unique and non-overlapping")
			}
		}
	}
	if s.State != nil {
		if err := s.State.Artifact.ValidateExact(); err != nil {
			return err
		}
	}
	if s.Export != nil {
		if s.Mode != WorkspaceModeOverlay || s.Export.State == s.Export.Diff ||
			!idPattern.MatchString(s.Export.State) || !idPattern.MatchString(s.Export.Diff) {
			return invalidf("workspace export slots are invalid")
		}
	}
	return nil
}

func validateWorkspaceTarget(value string) error {
	if value == "" {
		return nil
	}
	if value != norm.NFC.String(value) || len([]byte(value)) > 1024 || strings.HasPrefix(value, "/") ||
		strings.ContainsAny(value, "\\\x00") || strings.Contains(value, "://") {
		return invalidf("workspace target is invalid")
	}
	parts := strings.Split(value, "/")
	if len(parts) > 32 {
		return invalidf("workspace target is invalid")
	}
	for _, part := range parts {
		if part == "" || part == "." || part == ".." || strings.ContainsAny(part, "\r\n\t") {
			return invalidf("workspace target is invalid")
		}
		for _, character := range part {
			if character < 0x20 || character == 0x7f {
				return invalidf("workspace target is invalid")
			}
		}
	}
	first := parts[0]
	if len(first) >= 2 && ((first[0] >= 'A' && first[0] <= 'Z') || (first[0] >= 'a' && first[0] <= 'z')) && first[1] == ':' {
		return invalidf("workspace target is invalid")
	}
	return nil
}
