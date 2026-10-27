package projectsitemap

import (
	"errors"
	"slices"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// ErrProjectionUnavailable refuses a view whose allowed membership cannot be
// reconstructed within the lossless domain bounds. No partial raw map escapes.
var ErrProjectionUnavailable = errors.New("authorized sitemap projection is unavailable")

// SafeProjection removes disallowed contributions and their grouping effects
// on the Server, before any bytes are sent to a narrower allocation. Inputs
// are exact Run input revisions resolved from pinned configuration by the Server.
func SafeProjection(document Document, policy contracts.ProjectSitemapPolicy, runID string, inputs []contracts.ExactArtifact) (Document, error) {
	if policy.Validate() != nil || len(inputs) > contracts.MaxSitemapOpenAPIInputs {
		return Document{}, ErrInvalid
	}
	for _, input := range inputs {
		if input.Validate() != nil {
			return Document{}, ErrInvalid
		}
	}
	if validateDocument(document) != nil || len(jsonBytes(document)) > MaxDocumentBytes {
		return Document{}, ErrInvalid
	}
	copy := cloneDocument(document)
	if policy.SourceMode == "source-assisted" {
		return copy, nil
	}
	allowed := map[string]bool{}
	restricted := false
	sources := []Source{}
	for _, source := range copy.Sources {
		permit := source.Provenance.Mode == "blackbox" && source.Provenance.Kind != "openapi"
		if source.Provenance.Kind == "openapi" && source.Artifact != nil {
			for _, input := range inputs {
				if source.Artifact.Digest == input.Digest && source.Artifact.MediaType == input.MediaType && source.Artifact.SizeBytes == input.SizeBytes {
					permit = true
					// Expose the authorized consumer ref, never an ungranted source alias.
					selected := input
					source.Artifact = &selected
					break
				}
			}
		}
		allowed[source.Provenance.SourceID] = permit
		if permit {
			sources = append(sources, source)
		} else {
			restricted = true
		}
	}
	copy.Sources = sources
	copy.Rules = slices.DeleteFunc(copy.Rules, func(rule Rule) bool { return !allowed[rule.SourceID] })
	// Manual declarations remain concrete groups when later template rules
	// regroup their members. Recover their fixed member identities across
	// groups; the current template's route is not declaration provenance.
	declaredMembers := map[string]bool{}
	for _, group := range copy.Groups {
		for _, declaration := range group.Declarations {
			if !declaration.Rule && allowed[declaration.Provenance.SourceID] {
				declaredMembers[memberID(group.Origin, group.Method, group.Route, group.Structural...)] = true
			}
		}
	}
	keptMembers := map[string]bool{}
	for i := range copy.Groups {
		group := &copy.Groups[i]
		group.Declarations = slices.DeleteFunc(group.Declarations, func(d Declaration) bool { return !allowed[d.Provenance.SourceID] })
		members := []Member{}
		for _, member := range group.Members {
			member.Summaries = slices.DeleteFunc(member.Summaries, func(s Summary) bool { return !allowed[s.Provenance.SourceID] })
			member.Annotations = slices.DeleteFunc(member.Annotations, func(a Annotation) bool {
				return !allowed[a.Provenance.SourceID] || a.Provenance.Mode != "blackbox"
			})
			declared := declaredMembers[member.ID]
			if len(member.Summaries) == 0 && !declared {
				continue
			}
			route, _ := routeFor(copy.Rules, group.Origin, group.Method, member.Path)
			if !safeStructuralProjection(copy.Rules, group.Origin, group.Method, route, group.Structural) {
				restricted = true
				continue
			}
			// An example has no independent source attribution; regenerate it
			// from the allowed concrete observation rather than retain its origin.
			member.Examples = []string{}
			if len(member.Summaries) > 0 {
				example := group.Method + " " + group.Origin + member.Path
				if len(example) <= MaxExampleBytes {
					member.Examples = append(member.Examples, example)
				}
			}
			member.Evidence = slices.DeleteFunc(member.Evidence, func(e Evidence) bool {
				return e.RunID != runID || !slices.ContainsFunc(inputs, func(input contracts.ExactArtifact) bool {
					return input.Ref.SameExact(e.Artifact.Ref) && input.Digest == e.Artifact.Digest && input.MediaType == e.Artifact.MediaType && input.SizeBytes == e.Artifact.SizeBytes
				})
			})
			if member.Omitted > 0 {
				restricted = true
				member.Omitted = 0
			}
			members = append(members, member)
			keptMembers[member.ID] = true
		}
		group.Members = members
		if group.Omitted > 0 {
			restricted = true
			group.Omitted = 0
		}
	}
	markers := []Marker{}
	for _, marker := range copy.Markers {
		if !allowed[marker.Provenance.SourceID] {
			continue
		}
		marker.Members = slices.DeleteFunc(marker.Members, func(id string) bool { return !keptMembers[id] })
		if len(marker.Members) > 0 {
			markers = append(markers, marker)
		}
	}
	copy.Markers = markers
	copy.Gaps = slices.DeleteFunc(copy.Gaps, func(gap Gap) bool { return !allowed[gap.SourceID] })
	if restricted {
		addGap(&copy, "", "projection", 1)
	}
	if err := regroup(&copy); err != nil {
		return Document{}, ErrProjectionUnavailable
	}
	if _, err := encode(copy); err != nil {
		return Document{}, ErrProjectionUnavailable
	}
	return copy, nil
}

func safeStructuralProjection(rules []Rule, origin, method, route string, values []StructuralValue) bool {
	if !sameStructuralNames(rules, origin, method, route, values) {
		return false
	}
	for _, rule := range rules {
		if rule.Origin != origin || rule.Method != method || rule.Path != route {
			continue
		}
		for _, selector := range rule.Structural {
			for _, selected := range values {
				if selector.Name != selected.Name {
					continue
				}
				for _, value := range selected.Values {
					if !slices.Contains(selector.Allowed, value) {
						return false
					}
				}
			}
		}
	}
	return true
}
