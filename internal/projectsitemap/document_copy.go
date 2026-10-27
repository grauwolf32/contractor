package projectsitemap

import (
	"slices"
)

// cloneDocument gives a reducer or authorized projection private mutable
// collections without encoding and reparsing the full document. Strings are
// immutable; all slices and exact-reference pointers belong to the new value.
func cloneDocument(document Document) Document {
	copy := document
	copy.Groups = slices.Clone(document.Groups)
	for i := range copy.Groups {
		group := &copy.Groups[i]
		group.Structural = cloneStructural(group.Structural)
		group.Declarations = slices.Clone(group.Declarations)
		for j := range group.Declarations {
			group.Declarations[j].Parameters = slices.Clone(group.Declarations[j].Parameters)
		}
		group.Members = slices.Clone(group.Members)
		for j := range group.Members {
			member := &group.Members[j]
			member.Summaries = slices.Clone(member.Summaries)
			for k := range member.Summaries {
				member.Summaries[k].Parameters = slices.Clone(member.Summaries[k].Parameters)
			}
			member.Examples = slices.Clone(member.Examples)
			member.Evidence = slices.Clone(member.Evidence)
			for k := range member.Evidence {
				member.Evidence[k].Artifact = member.Evidence[k].Artifact.Clone()
			}
			member.Annotations = slices.Clone(member.Annotations)
			for k := range member.Annotations {
				member.Annotations[k].Tags = slices.Clone(member.Annotations[k].Tags)
			}
		}
	}
	copy.Sources = slices.Clone(document.Sources)
	for i := range copy.Sources {
		if copy.Sources[i].Artifact != nil {
			exact := copy.Sources[i].Artifact.Clone()
			copy.Sources[i].Artifact = &exact
		}
	}
	copy.Rules = slices.Clone(document.Rules)
	for i := range copy.Rules {
		rule := &copy.Rules[i]
		rule.Parameters = slices.Clone(rule.Parameters)
		rule.Structural = slices.Clone(rule.Structural)
		for j := range rule.Structural {
			rule.Structural[j].Allowed = slices.Clone(rule.Structural[j].Allowed)
		}
	}
	copy.Markers = slices.Clone(document.Markers)
	for i := range copy.Markers {
		copy.Markers[i].Members = slices.Clone(copy.Markers[i].Members)
	}
	copy.Gaps = slices.Clone(document.Gaps)
	return copy
}

func cloneStructural(values []StructuralValue) []StructuralValue {
	copy := slices.Clone(values)
	for i := range copy {
		copy[i].Values = slices.Clone(copy[i].Values)
	}
	return copy
}
