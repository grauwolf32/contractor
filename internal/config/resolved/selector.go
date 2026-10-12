package resolved

import (
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// Selector is an exact, versioned configuration or code descriptor lookup.
type Selector struct {
	ID      string
	Version string
}

func (s Selector) String() string { return s.ID + "@" + s.Version }

// ParseSelector validates and separates an exact <id>@<version> selector.
func ParseSelector(raw string) (Selector, error) {
	if strings.Count(raw, "@") != 1 {
		return Selector{}, fmt.Errorf("selector %q must use exact <id>@<version> syntax", raw)
	}
	id, version, _ := strings.Cut(raw, "@")
	if !contracts.ValidConfigID(id) {
		return Selector{}, fmt.Errorf("selector %q has invalid id", raw)
	}
	if !contracts.ValidConfigVersion(version) {
		return Selector{}, fmt.Errorf("selector %q has invalid version", raw)
	}
	return Selector{ID: id, Version: version}, nil
}
