package contracts

// ValidateResponseRecovery fences the document retry to material processing.
// Tool names are scoped by their owning toolset, so unrelated tools cannot
// acquire eligibility by exporting an identically named function.
func (t ResolvedAgentTemplate) ValidateResponseRecovery() error {
	if t.ResponseRecovery == "" {
		return nil
	}
	if t.ResponseRecovery != "document@1" || t.Runtime.RuntimeID != "adk" || t.Runtime.Version != "1" || len(t.Skills) != 0 || len(t.Toolsets) == 0 {
		return Invalidf("responseRecovery requires document@1 with ADK document tools")
	}
	allowed := map[string]map[string]bool{
		"memory-tools": {
			"list_memories": true, "read_memory": true,
			"write_memory": true, "append_memory": true,
			"search_memory": true, "list_memory_tags": true,
		},
		"run-artifacts":  {"read_artifact": true},
		"text-artifacts": {"write_text_artifact": true},
		"security-findings": {
			"list_project_findings": true, "read_project_finding": true,
			"select_project_findings": true, "read_finding_selection": true,
		},
	}
	for _, selection := range t.Toolsets {
		for _, name := range selection.Tools {
			if selection.Ref.Version != "1" || !allowed[selection.Ref.ToolsetID][name] {
				return Invalidf("document@1 requires document-only tools")
			}
		}
	}
	return nil
}
