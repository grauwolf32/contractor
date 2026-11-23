package evalservice

import (
	"slices"
	"testing"
)

func TestSortedUniqueKeepsCallerSlice(t *testing.T) {
	values := []string{"b", "a", "b"}
	if got := sortedUnique(values); !slices.Equal(got, []string{"a", "b"}) {
		t.Fatalf("sortedUnique = %v", got)
	}
	if !slices.Equal(values, []string{"b", "a", "b"}) {
		t.Fatalf("sortedUnique mutated its input: %v", values)
	}
}
