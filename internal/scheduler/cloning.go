package scheduler

import (
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func cloneParameters(source map[string]string) map[string]string {
	result := make(map[string]string, len(source))
	for name, value := range source {
		result[name] = value
	}
	return result
}

func cloneStringMap(source map[string]string) map[string]string {
	result := make(map[string]string, len(source))
	for name, value := range source {
		result[name] = value
	}
	return result
}

func cloneArtifactSlots(
	source map[string]workflowconfig.ArtifactSlot,
) map[string]workflowconfig.ArtifactSlot {
	result := make(map[string]workflowconfig.ArtifactSlot, len(source))
	for name, slot := range source {
		slot.MediaTypes = append([]string(nil), slot.MediaTypes...)
		if slot.From != nil {
			from := *slot.From
			slot.From = &from
		}
		result[name] = slot
	}
	return result
}

func cloneArtifactRef(source contracts.ArtifactRef) contracts.ArtifactRef {
	result := source
	if source.Revision != nil {
		revision := *source.Revision
		result.Revision = &revision
	}
	return result
}

func cloneCredentialRef(source *contracts.LLMCredentialRef) *contracts.LLMCredentialRef {
	if source == nil {
		return nil
	}
	result := *source
	return &result
}

func cloneModelPolicy(source contracts.ResolvedModelPolicy) contracts.ResolvedModelPolicy {
	result := source
	if source.Temperature != nil {
		temperature := *source.Temperature
		result.Temperature = &temperature
	}
	return result
}

func cloneStageResult(source contracts.StageContentResult) contracts.StageContentResult {
	result := source
	result.Artifacts = make(map[string]contracts.ArtifactRef, len(source.Artifacts))
	for name, ref := range source.Artifacts {
		result.Artifacts[name] = cloneArtifactRef(ref)
	}
	if source.Error != nil {
		cloned := *source.Error
		result.Error = &cloned
	}
	return result
}

func sameExactRef(left, right contracts.ArtifactRef) bool {
	return left.Namespace == right.Namespace && left.Name == right.Name &&
		left.Revision != nil && right.Revision != nil && *left.Revision == *right.Revision
}

func stringPointer(value string) *string { return &value }

func intPointer(value int) *int { return &value }
