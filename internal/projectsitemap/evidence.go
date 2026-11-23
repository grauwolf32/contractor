package projectsitemap

import "slices"

func attachEvidence(document *Document, observations []Observation, sourceID string, evidence Evidence) {
	selected := map[string]bool{}
	for _, observation := range observations {
		origin, path, _, err := projectURL(observation.URL)
		if err != nil {
			continue // The generic mutation projector already validated this URL.
		}
		route, _ := routeFor(document.Rules, origin, observation.Method, path)
		structural, err := projectStructural(document.Rules, origin, observation.Method, route, observation.URL)
		if err == nil {
			selected[memberID(origin, observation.Method, path, structural...)] = true
		}
	}
	for i := range document.Groups {
		for j := range document.Groups[i].Members {
			member := &document.Groups[i].Members[j]
			if !selected[member.ID] || slices.ContainsFunc(member.Evidence, func(existing Evidence) bool {
				return existing.RunID == evidence.RunID && existing.Artifact.Ref.SameExact(evidence.Artifact.Ref)
			}) {
				continue
			}
			if len(member.Evidence) == MaxProvenance {
				addGap(document, sourceID, "evidence", 1)
				continue
			}
			member.Evidence = append(member.Evidence, evidence)
		}
	}
}
