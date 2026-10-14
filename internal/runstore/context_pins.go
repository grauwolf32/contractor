package runstore

import (
	"sort"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// StageContextPin names an exact version required by attempt creation/resume.
type StageContextPin struct {
	Name string
	Ref  contracts.ArtifactRef
}

// Pins includes original inputs and all explicit saved continuation material.
// Private labels contain slashes, which authored aliases cannot contain.
func (s StageContextSnapshot) Pins() []StageContextPin {
	result := make([]StageContextPin, 0, len(s.Artifacts))
	for name, value := range s.Artifacts {
		if value.Artifact != nil {
			result = append(result, StageContextPin{Name: name, Ref: value.Artifact.Clone()})
		}
	}
	if value := s.Continuation; value != nil {
		result = append(result, StageContextPin{Name: "continuation/summary", Ref: value.SummaryArtifact.Clone()})
		for name, ref := range value.Artifacts {
			result = append(result, StageContextPin{Name: "continuation/result/" + name, Ref: ref.Clone()})
		}
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	return result
}
