package scanplan

// ReferenceBudget bounds local structural resolution across a whole metadata
// extraction. It contains no document, credential or cross-call cache.
type ReferenceBudget struct {
	Resolutions int
	Nodes       int
}

// ResolveLocalObject reuses preparation's local-reference and sibling-overlay
// semantics without preparing target requests or following concrete hints.
func ResolveLocalObject(root map[string]any, value any, budget *ReferenceBudget) (map[string]any, string) {
	if budget == nil {
		return nil, "invalid_reference"
	}
	p := preparer{root: root, refs: budget.Resolutions, schemaNodes: budget.Nodes}
	resolved, code := p.resolve(value, nil)
	budget.Resolutions, budget.Nodes = p.refs, p.schemaNodes
	return resolved, code
}
