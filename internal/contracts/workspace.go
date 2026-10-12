package contracts

// Allocation workspace contracts: the capability surface a Runtime Agent
// advertises and the workspace spec an allocation pins.
// Mirrors the Python runtime's contracts/workspace.py.

import (
	"strings"
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
		return Invalidf("workspace mode is invalid")
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
		return Invalidf("workspace storage is invalid")
	}
}

type WorkspaceLimits struct {
	MaxFiles            int   `json:"maxFiles" strictjson:"required"`
	MaxExpandedBytes    int64 `json:"maxExpandedBytes" strictjson:"required"`
	MaxManagedTextBytes int64 `json:"maxManagedTextBytes" strictjson:"required"`
	MaxFileBytes        int64 `json:"maxFileBytes" strictjson:"required"`
}

func (l WorkspaceLimits) Validate() error {
	if l.MaxFiles <= 0 || l.MaxExpandedBytes <= 0 || l.MaxManagedTextBytes <= 0 || l.MaxFileBytes <= 0 ||
		l.MaxFileBytes > l.MaxExpandedBytes || l.MaxManagedTextBytes > l.MaxExpandedBytes {
		return Invalidf("workspace limits are invalid")
	}
	return nil
}

type WorkspaceCapabilities struct {
	Storage WorkspaceStorage `json:"storage" strictjson:"required"`
	Modes   []WorkspaceMode  `json:"modes" strictjson:"required"`
	Limits  WorkspaceLimits  `json:"limits" strictjson:"required"`
}

func (c WorkspaceCapabilities) Validate() error {
	if err := c.Storage.Validate(); err != nil {
		return err
	}
	if len(c.Modes) == 0 || len(c.Modes) > 2 {
		return Invalidf("workspace capability modes must be a non-empty bounded array")
	}
	previous := ""
	for _, mode := range c.Modes {
		if err := mode.Validate(); err != nil {
			return err
		}
		if string(mode) <= previous {
			return Invalidf("workspace capability modes must be sorted and unique")
		}
		previous = string(mode)
	}
	return c.Limits.Validate()
}

type AllocationWorkspaceSource struct {
	Artifact ArtifactRef `json:"artifact" strictjson:"required"`
	Target   string      `json:"target" strictjson:"required"`
}

type AllocationWorkspaceState struct {
	Artifact ArtifactRef `json:"artifact" strictjson:"required"`
}

type AllocationWorkspaceExport struct {
	State string `json:"state" strictjson:"required"`
	Diff  string `json:"diff" strictjson:"required"`
}

type AllocationWorkspaceSpec struct {
	Mode    WorkspaceMode               `json:"mode" strictjson:"required"`
	Sources []AllocationWorkspaceSource `json:"sources" strictjson:"required"`
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
		result.Sources[index].Artifact = item.Artifact.Clone()
	}
	if source.State != nil {
		state := *source.State
		state.Artifact = source.State.Artifact.Clone()
		result.State = &state
	}
	if source.Export != nil {
		export := *source.Export
		result.Export = &export
	}
	return &result
}

func (s AllocationWorkspaceSpec) Validate() error {
	if err := s.Mode.Validate(); err != nil {
		return err
	}
	if len(s.Sources) == 0 || len(s.Sources) > 32 {
		return Invalidf("workspace sources must be a non-empty bounded array")
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
				return Invalidf("workspace source targets must be unique and non-overlapping")
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
			!ValidIdentifier(s.Export.State) || !ValidIdentifier(s.Export.Diff) {
			return Invalidf("workspace export slots are invalid")
		}
	}
	return nil
}

// validateWorkspaceTarget accepts "" (the workspace root) or a relative path
// of the workspace target grammar (WSP-R13).
func validateWorkspaceTarget(value string) error {
	if value == "" {
		return nil
	}
	if problem := RelativePathProblem(value, MaxWorkspaceTargetBytes, MaxWorkspaceTargetComponents); problem != "" {
		return Invalidf("workspace target %s", problem)
	}
	return nil
}
