package contracts

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

// RunMetadataLabels is bounded immutable descriptive metadata for one
// WorkflowRun. A non-nil value is required on private wire DTOs.
type RunMetadataLabels map[string]string

// NormalizeRunMetadataLabels validates and detaches a caller-owned map. Nil is
// accepted at internal normalization boundaries and becomes an explicit empty
// map; wire DTO validation rejects nil so JSON always carries an object.
func NormalizeRunMetadataLabels(source map[string]string) (RunMetadataLabels, error) {
	if len(source) > MaxRunMetadataLabels {
		return nil, invalidf("Run metadata labels exceed %d entries", MaxRunMetadataLabels)
	}
	result := make(RunMetadataLabels, len(source))
	for key, value := range source {
		if err := ValidateRunMetadataLabel(key, value); err != nil {
			return nil, err
		}
		result[key] = value
	}
	return result, nil
}

// ValidateRunMetadataLabel checks one exact key/value pair against the shared
// public, persistence, private-wire and telemetry limits.
func ValidateRunMetadataLabel(key, value string) error {
	if !utf8.ValidString(key) || len(key) == 0 || len(key) > MaxRunMetadataLabelKey ||
		!runMetadataLabelKeyPattern.MatchString(key) {
		return invalidf("Run metadata label key %q is invalid", key)
	}
	if strings.HasPrefix(key, "contractor.") {
		return invalidf("Run metadata label key %q uses the reserved contractor. namespace", key)
	}
	if !utf8.ValidString(value) || len(value) == 0 || len(value) > MaxRunMetadataLabelValue ||
		strings.ContainsRune(value, '\x00') {
		return invalidf(
			"Run metadata label %q value must contain 1 to %d database-safe UTF-8 bytes",
			key, MaxRunMetadataLabelValue,
		)
	}
	return nil
}

func (labels RunMetadataLabels) Validate() error {
	if labels == nil {
		return invalidf("runMetadataLabels must be an object")
	}
	_, err := NormalizeRunMetadataLabels(labels)
	return err
}

// Clone returns a non-nil detached map suitable for an explicit wire object.
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
