// Package clone copies values that would otherwise alias caller-owned memory.
package clone

import "maps"

// Pointer returns a pointer to a copy of *value, or nil for a nil pointer.
func Pointer[T any](value *T) *T {
	if value == nil {
		return nil
	}
	result := *value
	return &result
}

// Map returns a shallow copy of value. Unlike maps.Clone the result is never
// nil, so callers may add entries and encoding/json emits an empty object.
func Map[M ~map[K]V, K comparable, V any](value M) M {
	result := make(M, len(value))
	maps.Copy(result, value)
	return result
}
