package scan

import "github.com/grauwolf32/contractor/internal/contracts"

// IsNucleiSelectionBinding recognizes the closed set of scan-planner inputs.
// Their values must come from pinned Audit parameters or a retained input artifact.
func IsNucleiSelectionBinding(name string, binding contracts.ToolArgumentBinding) bool {
	expected := contracts.ToolArgumentBinding{Source: "parameter", Name: name}
	switch name {
	case "template_ids", "tags", "severity":
	case "rate_limit":
		expected.Parse = "integer"
	case "rulepack_ref":
		expected.Source, expected.Name = "artifact", "rulepack"
		expected.Optional = binding.Optional
	default:
		return false
	}
	return binding.Source == expected.Source && binding.Name == expected.Name && binding.Parse == expected.Parse &&
		binding.Value == nil && binding.Default == nil && binding.Optional == expected.Optional
}

// IsScanCredentialBinding selects an optional engagement identity for a scanner.
func IsScanCredentialBinding(tool, name string, binding contracts.ToolArgumentBinding) bool {
	return (tool == "scan_nuclei" || tool == "scan_sqlmap") && name == "credential_id" && binding.Source == "parameter" && binding.Name == name && binding.Optional && binding.Parse == "" && binding.Default == nil && binding.Value == nil
}
