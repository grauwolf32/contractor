package evalstore

import (
	"slices"
	"testing"
)

func TestFreezeBatchesBoundRowsAndBytes(t *testing.T) {
	sized := func(sizes ...int) func(int) int { return func(i int) int { return sizes[i] } }
	each := func(size int) func(int) int { return func(int) int { return size } }
	for _, tc := range []struct {
		name string
		n    int
		size func(int) int
		want []int
	}{
		{"no rows", 0, each(1), nil},
		{"row bound", 2*freezeBatchRows + 500, each(1), []int{freezeBatchRows, 2 * freezeBatchRows, 2*freezeBatchRows + 500}},
		{"byte bound reached exactly", 9, each(freezeBatchBytes / 4), []int{4, 8, 9}},
		{"one byte over the bound", 5, sized(freezeBatchBytes/4, freezeBatchBytes/4, freezeBatchBytes/4, freezeBatchBytes/4+1, 1), []int{3, 5}},
		{"oversized row alone", 3, sized(10, freezeBatchBytes+1, 10), []int{1, 2, 3}},
	} {
		if got := freezeBatches(tc.n, tc.size); !slices.Equal(got, tc.want) {
			t.Errorf("%s: batch ends %v, want %v", tc.name, got, tc.want)
		}
	}
}
