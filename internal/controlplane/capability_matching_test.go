package controlplane

import (
	"reflect"
	"testing"
)

func TestMatchingAllSmallCandidateGraphs(t *testing.T) {
	for bindings := 0; bindings <= 3; bindings++ {
		for agents := 0; agents <= 3; agents++ {
			for mask := 0; mask < 1<<(bindings*agents); mask++ {
				graph := make([][]int, bindings)
				for b := range bindings {
					for a := range agents {
						if mask&(1<<(b*agents+a)) != 0 {
							graph[b] = append(graph[b], a)
						}
					}
				}
				want := bruteCompleteAssignment(graph, 0, 0)
				assignment, ok := matchCapabilityCandidates(graph, agents)
				if ok != want {
					t.Fatalf("graph %v: complete=%t, want %t", graph, ok, want)
				}
				used := map[int]bool{}
				for b, a := range assignment {
					edge := false
					for _, candidate := range graph[b] {
						if candidate == a {
							edge = true
						}
					}
					if !edge || used[a] {
						t.Fatalf("invalid assignment %v for %v", assignment, graph)
					}
					used[a] = true
				}
			}
		}
	}
}
func bruteCompleteAssignment(graph [][]int, binding int, used uint) bool {
	if binding == len(graph) {
		return true
	}
	for _, agent := range graph[binding] {
		if used&(1<<agent) == 0 && bruteCompleteAssignment(graph, binding+1, used|1<<agent) {
			return true
		}
	}
	return false
}
func TestMatchingStableAugmentationAndReadOnlyGraph(t *testing.T) {
	graph := [][]int{{0, 1}, {0}}
	for range 3 {
		got, ok := matchCapabilityCandidates(graph, 2)
		if !ok || !reflect.DeepEqual(got, []int{1, 0}) {
			t.Fatalf("assignment=%v complete=%t", got, ok)
		}
		if !reflect.DeepEqual(graph, [][]int{{0, 1}, {0}}) {
			t.Fatal("matching mutated graph")
		}
	}
}
