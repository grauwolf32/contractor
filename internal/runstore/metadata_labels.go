package runstore

import (
	"sort"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	MaxRunMetadataLabels     = contracts.MaxRunMetadataLabels
	MaxRunMetadataLabelKey   = contracts.MaxRunMetadataLabelKey
	MaxRunMetadataLabelValue = contracts.MaxRunMetadataLabelValue
)

// RunMetadataLabels is immutable descriptive metadata attached to one
// WorkflowRun. Callers must clone it at repository boundaries because Go maps
// themselves are mutable reference values.
type RunMetadataLabels = contracts.RunMetadataLabels

type RunMetadataLabelSelector struct {
	Key   string
	Value string
}

// NormalizeRunMetadataLabels validates and clones a caller-owned map. Nil and
// an empty map have the same canonical non-nil representation.
func NormalizeRunMetadataLabels(source map[string]string) (RunMetadataLabels, error) {
	result, err := contracts.NormalizeRunMetadataLabels(source)
	if err != nil {
		return nil, invalidf("Run metadata labels are invalid: %v", err)
	}
	return result, nil
}

// NormalizeRunMetadataLabelSelectors validates, deduplicates and orders exact
// key/value requirements. Different values for one key intentionally remain:
// they form a valid conjunction that no immutable one-value binding can meet.
func NormalizeRunMetadataLabelSelectors(
	source []RunMetadataLabelSelector,
) ([]RunMetadataLabelSelector, error) {
	if len(source) > MaxRunMetadataLabels {
		return nil, invalidf("Run metadata label selectors exceed %d entries", MaxRunMetadataLabels)
	}
	seen := make(map[RunMetadataLabelSelector]struct{}, len(source))
	result := make([]RunMetadataLabelSelector, 0, len(source))
	for _, selector := range source {
		if err := validateRunMetadataLabel(selector.Key, selector.Value); err != nil {
			return nil, err
		}
		if _, duplicate := seen[selector]; duplicate {
			continue
		}
		seen[selector] = struct{}{}
		result = append(result, selector)
	}
	sort.Slice(result, func(left, right int) bool {
		if result[left].Key == result[right].Key {
			return result[left].Value < result[right].Value
		}
		return result[left].Key < result[right].Key
	})
	return result, nil
}

func validateRunMetadataLabel(key, value string) error {
	if err := contracts.ValidateRunMetadataLabel(key, value); err != nil {
		return invalidf("Run metadata label is invalid: %v", err)
	}
	return nil
}
