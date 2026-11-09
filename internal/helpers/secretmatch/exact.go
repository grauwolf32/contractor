package secretmatch

import (
	"slices"
	"strings"
)

// ExactReplacer replaces every original occurrence of known nonempty values,
// including short embedded values. Equal-start matches prefer the longest
// value. Replacement text is never scanned again. The returned closure keeps
// its match set private to the owning execution.
func ExactReplacer(values []string) func(string) string {
	values = slices.Clone(values)
	slices.SortFunc(values, func(a, b string) int {
		if len(a) != len(b) {
			return len(b) - len(a)
		}
		return strings.Compare(a, b)
	})
	values = slices.Compact(values)
	pairs := make([]string, 0, 2*len(values))
	for _, value := range values {
		if value != "" {
			pairs = append(pairs, value, "[REDACTED]")
		}
	}
	replacer := strings.NewReplacer(pairs...)
	return replacer.Replace
}
