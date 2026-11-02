package controlplane

import (
	"fmt"
	"github.com/grauwolf32/contractor/internal/contracts/control"
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

func TestCapabilityIndexRetainsExactDetachedTools(t *testing.T) {
	registration := testRegistration("indexed-agent")
	index := indexToolCapabilities(registration.SupportedToolsets)
	binding := testBinding(t, "worker", "worker", testTemplate(t))
	registration.SupportedToolsets[0].Tools[0] = "caller-changed"
	fresh := testRegistration("indexed-agent")
	if !isBindingCompatibleWithTools(fresh, binding, index) {
		t.Fatal("index shared registration tool memory")
	}
	binding.AgentTemplate.Toolsets[0].Tools = append(binding.AgentTemplate.Toolsets[0].Tools, "unsupported_tool")
	if isBindingCompatibleWithTools(fresh, binding, index) {
		t.Fatal("index widened the exact selected tool set")
	}
}

// Compare repeated compatibility checks with the immutable registration index;
// the graph subtest includes complete matching for 32 bindings and 256 slots.
func BenchmarkCapabilityMatching(b *testing.B) {
	template := testTemplate(b)
	binding := BindingRequirement{LogicalAgentName: "worker", Namespace: "worker", AgentTemplate: template}
	for _, toolsets := range []int{1, 16} {
		registration := testRegistration("benchmark-agent")
		for extra := 1; extra < toolsets; extra++ {
			capability := control.ToolsetCapability{Ref: fmt.Sprintf("extra%d@1", extra)}
			for tool := range 32 {
				capability.Tools = append(capability.Tools, fmt.Sprintf("tool%d", tool))
			}
			registration.SupportedToolsets = append(registration.SupportedToolsets, capability)
		}
		index := indexToolCapabilities(registration.SupportedToolsets)
		b.Run(fmt.Sprintf("toolsets%d/rebuild", toolsets), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if !isBindingCompatible(registration, binding) {
					b.Fatal("incompatible fixture")
				}
			}
		})
		b.Run(fmt.Sprintf("toolsets%d/indexed", toolsets), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if !isBindingCompatibleWithTools(registration, binding, index) {
					b.Fatal("incompatible fixture")
				}
			}
		})
	}
	registration := testRegistration("benchmark-agent")
	agents := make([]*agentEntry, 256)
	for index := range agents {
		agents[index] = &agentEntry{registration: registration, toolCapabilities: indexToolCapabilities(registration.SupportedToolsets)}
	}
	bindings := make([]BindingRequirement, 32)
	for index := range bindings {
		bindings[index] = binding
	}
	b.Run("graph32x256", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, ok := completeCapabilityAssignmentWithEdges(agents, bindings, nil); !ok {
				b.Fatal("incomplete assignment")
			}
		}
	})
}

func TestCapabilityMatchingPlacesThirtyTwoBindingsAcrossFleet(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	for index := range 256 {
		registerReady(t, registry, fmt.Sprintf("fleet-%03d", index))
	}
	template := testTemplate(t)
	request := ReservationRequest{RunID: "run-fleet", StageExecutionID: "stage-fleet"}
	for index := range 32 {
		name := fmt.Sprintf("worker-%02d", index)
		request.Bindings = append(request.Bindings, testBinding(t, name, name, template))
	}
	reservations, err := registry.ReserveAll(request)
	if err != nil || len(reservations) != 32 {
		t.Fatalf("fleet matching: count=%d err=%v", len(reservations), err)
	}
	instances := map[string]bool{}
	for _, reservation := range reservations {
		if instances[reservation.Grant.RuntimeInstanceID] {
			t.Fatal("matching reused a slot")
		}
		instances[reservation.Grant.RuntimeInstanceID] = true
	}
}
