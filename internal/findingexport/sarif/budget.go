package sarif

import (
	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"sort"
)

type object = map[string]any

// Count each piece once at its actual output position. Container headers
// exclude separately counted children. The lower bound stops repeated large
// symbols/locations before an unbounded flow is constructed; final encoding
// independently enforces the exact byte cap.
type byteBudget struct{ used int }

func (b *byteBudget) add(value any) error {
	encoded, err := strictjson.Canonical(value)
	if err != nil {
		return findingexport.ErrInvalid
	}
	if len(encoded) > findingexport.MaxBytes-b.used {
		return findingexport.ErrLimit
	}
	b.used += len(encoded)
	return nil
}

func sortedKeys[V any](values map[string]V) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
