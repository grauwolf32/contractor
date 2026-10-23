package contracts

import "strings"

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
