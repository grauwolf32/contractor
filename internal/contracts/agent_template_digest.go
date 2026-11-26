package contracts

import "github.com/grauwolf32/contractor/internal/helpers/contentdigest"

// ContentDigest returns the canonical digest of a normalized template body.
func (template ResolvedAgentTemplate) ContentDigest() (string, error) {
	toolsets := make([]any, 0, len(template.Toolsets))
	for _, selection := range template.Toolsets {
		tools := make([]any, len(selection.Tools))
		for index, tool := range selection.Tools {
			tools[index] = tool
		}
		toolsets = append(toolsets, map[string]any{
			"ref": map[string]any{
				"toolsetId": selection.Ref.ToolsetID,
				"version":   selection.Ref.Version,
			},
			"tools": tools,
		})
	}
	spec := map[string]any{
		"description": template.Description,
		"runtime": map[string]any{
			"runtimeId": template.Runtime.RuntimeID,
			"version":   template.Runtime.Version,
		},
		"instructions": map[string]any{
			"ref":    template.Instructions.Ref,
			"digest": template.Instructions.Digest,
		},
		"toolsets": toolsets,
		"sandboxProfile": map[string]any{
			"sandboxProfileId": template.SandboxProfile.SandboxProfileID,
			"version":          template.SandboxProfile.Version,
		},
	}
	if template.ResponseRecovery != "" {
		spec["responseRecovery"] = template.ResponseRecovery
	}
	if template.IsToolWorker() {
		delete(spec, "instructions")
		spec["execution"] = template.Execution
	}
	if template.Sitemap != nil {
		spec["sitemap"] = template.Sitemap
	}
	if len(template.Skills) > 0 {
		skills := make([]any, len(template.Skills))
		for index, skill := range template.Skills {
			skills[index] = map[string]any{"namespace": skill.Namespace, "name": skill.Name}
		}
		spec["skills"] = skills
	}
	if template.Summarizer != nil {
		summarizer := map[string]any{}
		if template.Summarizer.CumulativeBudget != nil {
			summarizer["cumulativeBudget"] = *template.Summarizer.CumulativeBudget
		}
		summarizer["contextWindowRatio"] = template.Summarizer.ContextWindowRatio
		summarizer["includeReasoning"] = template.Summarizer.IncludeReasoning
		if instructions := template.Summarizer.Instructions; instructions != nil {
			summarizer["instructions"] = map[string]any{
				"ref": instructions.Ref, "digest": instructions.Digest,
			}
		}
		spec["summarizer"] = summarizer
	}

	manifest := map[string]any{
		"apiVersion": APIVersion,
		"kind":       "AgentTemplate",
		"metadata": map[string]any{
			"name":    template.Ref.TemplateID,
			"version": template.Ref.Version,
		},
		"spec": spec,
	}
	return contentdigest.CanonicalJSON(manifest)
}
