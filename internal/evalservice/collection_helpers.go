package evalservice

import (
	"reflect"
	"slices"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func sortedUnique(values []string) []string {
	slices.Sort(values)
	return slices.Compact(values)
}

func selectedCollectionComplete(result evaldomain.ResultInput, observed *evaldomain.ResultInput, outputs map[string]evaldomain.Output) bool {
	if result.Collection.Status != "complete" || len(result.Collection.Gaps) > 0 || observed == nil || observed.Collection.Status != "complete" {
		return false
	}
	for role, contract := range outputs {
		ref, exists := result.Outputs[role]
		if contract.Required && !exists || exists && !slices.Contains(contract.MediaTypes, ref.MediaType) {
			return false
		}
	}
	return true
}

func usageMeasures(usage *evaldomain.Usage) []*evaldomain.Measure {
	return []*evaldomain.Measure{&usage.InputTokens, &usage.OutputTokens, &usage.TotalTokens, &usage.CachedInputTokens, &usage.ModelCalls, &usage.ToolCalls, &usage.ToolFailures, &usage.WallMS}
}

// Invalidate only a changed dimension. Missing child token accounting does not
// erase an unchanged authoritative parent duration.
func reconcileSelectedUsage(selected evaldomain.Usage, observed *evaldomain.ResultInput) evaldomain.Usage {
	var current evaldomain.Usage
	if observed != nil {
		current = observed.Usage
	}
	measures := usageMeasures(&current)
	for i, measure := range usageMeasures(&selected) {
		if observed != nil && reflect.DeepEqual(*measure, *measures[i]) {
			continue
		}
		measure.Completeness = "unavailable"
		measure.Value = nil
		measure.Scope.Missing = evaldomain.BoundedGaps(append(append([]string{}, measure.Scope.Missing...), "Selected usage no longer matches available execution evidence."))
	}
	return selected
}
