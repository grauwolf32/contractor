package contracts

import (
	"slices"
	"strings"
)

// NormalizePlacementLabels detaches and sorts a caller's location selection.
// Omitted and empty selections have the same canonical empty representation.
func NormalizePlacementLabels(labels []string) ([]string, error) {
	result := append([]string{}, labels...)
	slices.Sort(result)
	if err := ValidatePlacementLabels(result); err != nil {
		return nil, err
	}
	return result, nil
}

// PlacementLabelsMatch evaluates the required parent paths against a Runtime's
// actual paths. Both sets must satisfy the canonical placement grammar.
func PlacementLabelsMatch(required, actual []string) bool {
	if ValidatePlacementLabels(required) != nil || ValidatePlacementLabels(actual) != nil {
		return false
	}
	if len(required) == 0 {
		return true
	}
	if len(actual) == 0 {
		return false
	}
	// Every set is one ancestor chain, so its most specific path subsumes all
	// its parents. Only these two leaves determine the all-requirements match.
	want, have := required[len(required)-1], actual[len(actual)-1]
	return have == want || strings.HasPrefix(have, want+"/")
}

// ValidatePlacementLabels checks a canonical location declaration or requirement.
// Labels form one ancestor chain in the cloud or local namespace. An empty set
// declares no location (or, on a launch, imposes no location restriction).
func ValidatePlacementLabels(labels []string) error {
	if len(labels) > 32 {
		return Invalidf("placement labels must contain at most 32 values")
	}
	previous := ""
	for _, label := range labels {
		if len(label) == 0 || len(label) > 63 {
			return Invalidf("placement labels must be 1-63 ASCII bytes")
		}
		segments := strings.Split(label, "/")
		if segments[0] != "cloud" && segments[0] != "local" {
			return Invalidf("placement labels must use the cloud or local namespace")
		}
		for _, segment := range segments {
			if !ValidIdentifier(segment) {
				return Invalidf("placement label segments must match [a-z][a-z0-9_-]*")
			}
		}
		if previous != "" && (label <= previous || !strings.HasPrefix(label, previous+"/")) {
			return Invalidf("placement labels must be sorted, unique and form one ancestor chain")
		}
		previous = label
	}
	return nil
}
