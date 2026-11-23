package resolved

import (
	"fmt"
	"slices"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// ValidateSandboxToolCompatibility enforces registered profile requirements
// without consulting fleet state or granting an execution capability.
func ValidateSandboxToolCompatibility(template contracts.ResolvedAgentTemplate) error {
	return sandboxToolCompatibility(registeredDescriptors(), template)
}

func sandboxToolCompatibility(registered descriptorTable, template contracts.ResolvedAgentTemplate) error {
	profile := template.SandboxProfile.SandboxProfileID + "@" + template.SandboxProfile.Version
	for _, selected := range template.Toolsets {
		ref := selected.Ref.ToolsetID + "@" + selected.Ref.Version
		required := registered.Toolsets[ref].RequiredSandboxProfiles
		if len(required) > 0 && !slices.Contains(required, profile) {
			return fmt.Errorf("Toolset %s requires one of SandboxProfiles %v", ref, required)
		}
	}
	return nil
}

// SandboxWorkspaceCompatible is binding-specific: other Workers retain their
// usual storage-independent workspace placement. It runs for every Worker and
// Runtime pairing during placement, so it reads the registry without copying.
func SandboxWorkspaceCompatible(template contracts.ResolvedAgentTemplate, workspace *contracts.AllocationWorkspaceSpec, capability *contracts.WorkspaceCapabilities) bool {
	registered := registeredDescriptors()
	if sandboxToolCompatibility(registered, template) != nil {
		return false
	}
	profile := template.SandboxProfile.SandboxProfileID + "@" + template.SandboxProfile.Version
	required := registered.SandboxProfiles[profile]
	if required.WorkspaceMode != "" && (workspace == nil || workspace.Mode != required.WorkspaceMode) {
		return false
	}
	return required.WorkspaceStorage == "" || (capability != nil && capability.Storage == required.WorkspaceStorage)
}
