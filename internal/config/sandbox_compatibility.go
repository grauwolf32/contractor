package config

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// Private immutable lookup on the per-candidate placement hot path. Do not
// rebuild all registered descriptor maps for every Worker/Runtime pairing.
var sandboxPlacementDescriptors = MVPDescriptors()

// ValidateSandboxToolCompatibility enforces registered profile requirements
// without consulting fleet state or granting an execution capability.
func (d Descriptors) ValidateSandboxToolCompatibility(template contracts.ResolvedAgentTemplate) error {
	profile := template.SandboxProfile.SandboxProfileID + "@" + template.SandboxProfile.Version
	for _, selected := range template.Toolsets {
		ref := selected.Ref.ToolsetID + "@" + selected.Ref.Version
		required := d.Toolsets[ref].RequiredSandboxProfile
		if required != "" && profile != required {
			return fmt.Errorf("Toolset %s requires SandboxProfile %s", ref, required)
		}
	}
	return nil
}

// SandboxWorkspaceCompatible is binding-specific: other Workers retain their
// usual storage-independent workspace placement.
func SandboxWorkspaceCompatible(template contracts.ResolvedAgentTemplate, workspace *contracts.AllocationWorkspaceSpec, capability *contracts.WorkspaceCapabilities) bool {
	d := sandboxPlacementDescriptors
	if d.ValidateSandboxToolCompatibility(template) != nil {
		return false
	}
	profile := template.SandboxProfile.SandboxProfileID + "@" + template.SandboxProfile.Version
	required := d.SandboxProfiles[profile]
	if required.WorkspaceMode != "" && (workspace == nil || workspace.Mode != required.WorkspaceMode) {
		return false
	}
	return required.WorkspaceStorage == "" || (capability != nil && capability.Storage == required.WorkspaceStorage)
}
