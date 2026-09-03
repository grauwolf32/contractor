package runstore

import (
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

const (
	MaxRunMetadataLabels     = 32
	MaxRunMetadataLabelKey   = 63
	MaxRunMetadataLabelValue = 256
)

var runMetadataLabelKeyPattern = regexp.MustCompile(`^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$`)

// RunMetadataLabels is immutable descriptive metadata attached to one
// WorkflowRun. Callers must clone it at repository boundaries because Go maps
// themselves are mutable reference values.
type RunMetadataLabels map[string]string

type RunMetadataLabelSelector struct {
	Key   string
	Value string
}

// NormalizeRunMetadataLabels validates and clones a caller-owned map. Nil and
// an empty map have the same canonical non-nil representation.
func NormalizeRunMetadataLabels(source map[string]string) (RunMetadataLabels, error) {
	if len(source) > MaxRunMetadataLabels {
		return nil, invalidf("Run metadata labels exceed %d entries", MaxRunMetadataLabels)
	}
	result := make(RunMetadataLabels, len(source))
	for key, value := range source {
		if err := validateRunMetadataLabel(key, value); err != nil {
			return nil, err
		}
		result[key] = value
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
	if !utf8.ValidString(key) || len(key) == 0 || len(key) > MaxRunMetadataLabelKey ||
		!runMetadataLabelKeyPattern.MatchString(key) {
		return invalidf("Run metadata label key %q is invalid", key)
	}
	if strings.HasPrefix(key, "contractor.") {
		return invalidf("Run metadata label key %q uses the reserved contractor. namespace", key)
	}
	if !utf8.ValidString(value) || len(value) == 0 || len(value) > MaxRunMetadataLabelValue ||
		strings.ContainsRune(value, '\x00') {
		return invalidf("Run metadata label %q value must contain 1 to %d database-safe UTF-8 bytes", key, MaxRunMetadataLabelValue)
	}
	return nil
}

func (labels RunMetadataLabels) Validate() error {
	_, err := NormalizeRunMetadataLabels(labels)
	return err
}

func (labels RunMetadataLabels) Clone() RunMetadataLabels {
	result := make(RunMetadataLabels, len(labels))
	for key, value := range labels {
		result[key] = value
	}
	return result
}

func (labels RunMetadataLabels) SortedKeys() []string {
	result := make([]string, 0, len(labels))
	for key := range labels {
		result = append(result, key)
	}
	sort.Strings(result)
	return result
}
