package controlplane

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/control"
)

// runtimeCapabilityCases mirrors api/testdata/v1alpha1/runtime-capability-cases.json.
type runtimeCapabilityCases struct {
	ToolName struct {
		MaxLength int      `json:"maxLength"`
		Valid     []string `json:"valid"`
		Invalid   []string `json:"invalid"`
	} `json:"toolName"`
	ToolsPerToolset struct {
		MaxItems int `json:"maxItems"`
	} `json:"toolsPerToolset"`
	CapabilityRefs struct {
		MaxItems  int `json:"maxItems"`
		MaxLength int `json:"maxLength"`
	} `json:"capabilityRefs"`
	Identity struct {
		MaxLength int      `json:"maxLength"`
		Valid     []string `json:"valid"`
		Invalid   []string `json:"invalid"`
	} `json:"identity"`
}

func readRuntimeCapabilityCases(t *testing.T) runtimeCapabilityCases {
	t.Helper()
	data := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "runtime-capability-cases.json")
	var cases runtimeCapabilityCases
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if cases.ToolName.MaxLength == 0 || cases.ToolsPerToolset.MaxItems == 0 || cases.CapabilityRefs.MaxItems == 0 ||
		cases.CapabilityRefs.MaxLength == 0 || cases.Identity.MaxLength == 0 ||
		len(cases.ToolName.Invalid) == 0 || len(cases.Identity.Invalid) == 0 {
		t.Fatal("runtime capability cases are incomplete")
	}
	return cases
}

func numberedValues(count int, format string) []string {
	values := make([]string, count)
	for index := range values {
		values[index] = fmt.Sprintf(format, index)
	}
	return values
}

// TestSharedRegistrationRejectsWhatOperationsCannotShow is the regression for a
// registration that passed validation while the Operations snapshot rejected
// it, which failed every Operations list for every viewer.
func TestSharedRegistrationRejectsWhatOperationsCannotShow(t *testing.T) {
	t.Parallel()
	cases := readRuntimeCapabilityCases(t)
	tests := map[string]func(*control.AgentRegistration){
		"too many tools": func(registration *control.AgentRegistration) {
			registration.SupportedToolsets[0].Tools = numberedValues(cases.ToolsPerToolset.MaxItems+1, "tool_%03d")
		},
		"colon tool name": func(registration *control.AgentRegistration) {
			registration.SupportedToolsets[0].Tools = []string{"server:tool"}
		},
		"oversized tool name": func(registration *control.AgentRegistration) {
			registration.SupportedToolsets[0].Tools = []string{strings.Repeat("a", cases.ToolName.MaxLength+1)}
		},
		"oversized toolset ref": func(registration *control.AgentRegistration) {
			registration.SupportedToolsets[0].Ref = strings.Repeat("a", cases.CapabilityRefs.MaxLength-1) + "@1"
		},
		"oversized runtime ref": func(registration *control.AgentRegistration) {
			registration.SupportedRuntimes = []string{strings.Repeat("a", cases.CapabilityRefs.MaxLength-1) + "@1"}
		},
		"instance id with a space": func(registration *control.AgentRegistration) {
			registration.InstanceID = "runtime 1"
		},
		"allocation id with a slash": func(registration *control.AgentRegistration) {
			allocationID := "allocation/1"
			registration.ObservedState = control.AgentAllocated
			registration.AllocationID = &allocationID
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			registry, err := NewRegistry(RegistryOptions{})
			if err != nil {
				t.Fatal(err)
			}
			registration := testRegistration("runtime-1")
			mutate(&registration)
			if _, err := registry.Register(registration); !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("Register error = %v, want ErrInvalidRequest", err)
			}
			if err := registry.SnapshotOperations().Validate(); err != nil {
				t.Fatalf("Operations snapshot after a rejected registration: %v", err)
			}
		})
	}
}

// TestSharedAcceptedRegistrationAlwaysYieldsValidOperationsSnapshot draws
// registrations and heartbeats from the shared Runtime capability cases and
// their bounds: whatever the Registry accepts, Operations can show.
func TestSharedAcceptedRegistrationAlwaysYieldsValidOperationsSnapshot(t *testing.T) {
	t.Parallel()
	cases := readRuntimeCapabilityCases(t)
	random := rand.New(rand.NewPCG(367, 7))
	pick := func(values []string) string { return values[random.IntN(len(values))] }
	// choose draws an invalid value one time in ten, so registrations reach
	// the Registry both accepted and rejected with fields near their bounds.
	choose := func(valid, invalid []string) string {
		if random.IntN(10) == 0 {
			return pick(invalid)
		}
		return pick(valid)
	}

	validTools := append([]string{strings.Repeat("a", cases.ToolName.MaxLength)}, cases.ToolName.Valid...)
	invalidTools := append([]string{strings.Repeat("a", cases.ToolName.MaxLength+1)}, cases.ToolName.Invalid...)
	validIdentities := append([]string{strings.Repeat("a", cases.Identity.MaxLength)}, cases.Identity.Valid...)
	invalidIdentities := append([]string{strings.Repeat("a", cases.Identity.MaxLength+1)}, cases.Identity.Invalid...)
	validRefs := []string{
		"adk@1", "local-workdir@1", "run-artifacts@1", "a@1.0.0-rc.1+build.5",
		strings.Repeat("a", cases.CapabilityRefs.MaxLength-2) + "@1",
	}
	invalidRefs := []string{"Adk@1", "adk", strings.Repeat("a", cases.CapabilityRefs.MaxLength-1) + "@1"}
	// list draws 1..3 distinct values, each invalid one time in ten, or none,
	// or a list at or just past its count bound.
	list := func(maximum int, format string, valid, invalid []string) []string {
		switch random.IntN(12) {
		case 0:
			return nil
		case 1:
			return numberedValues(maximum, format)
		case 2:
			return numberedValues(maximum+1, format)
		default:
			order := random.Perm(len(valid))
			values := make([]string, 1+random.IntN(3))
			for index := range values {
				values[index] = valid[order[index]]
				if random.IntN(10) == 0 {
					values[index] = pick(invalid)
				}
			}
			return values
		}
	}
	refList := func() []string {
		return list(cases.CapabilityRefs.MaxItems, "ref-%03d@1", validRefs, invalidRefs)
	}
	toolList := func() []string {
		return list(cases.ToolsPerToolset.MaxItems, "tool_%03d", validTools, invalidTools)
	}
	observation := func() (control.AgentObservedState, *string) {
		switch random.IntN(3) {
		case 0:
			return control.AgentIdle, nil
		case 1:
			return control.AgentFenced, nil
		default:
			allocationID := choose(validIdentities, invalidIdentities)
			return control.AgentAllocated, &allocationID
		}
	}

	var accepted, rejected, heartbeats int
	for iteration := range 3000 {
		registration := testRegistration("runtime-property")
		registration.InstanceID = choose(validIdentities, invalidIdentities)
		registration.SupportedRuntimes = refList()
		registration.SupportedSandboxProfiles = refList()
		toolsetRefs := refList()
		registration.SupportedToolsets = make([]control.ToolsetCapability, len(toolsetRefs))
		for index, ref := range toolsetRefs {
			registration.SupportedToolsets[index] = control.ToolsetCapability{Ref: ref, Tools: toolList()}
		}
		registration.ObservedState, registration.AllocationID = observation()

		registry, err := NewRegistry(RegistryOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := registry.Register(registration); err != nil {
			if !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("iteration %d: Register error = %v", iteration, err)
			}
			rejected++
			continue
		}
		accepted++
		if err := registry.SnapshotOperations().Validate(); err != nil {
			t.Fatalf("iteration %d: accepted registration of %q yields invalid Operations snapshot: %v", iteration, registration.InstanceID, err)
		}
		next := heartbeat(registration.InstanceID, 1, 0)
		next.ObservedState, next.AllocationID = observation()
		if _, err := registry.Heartbeat(next); err != nil {
			continue
		}
		heartbeats++
		if err := registry.SnapshotOperations().Validate(); err != nil {
			t.Fatalf("iteration %d: accepted heartbeat of %q yields invalid Operations snapshot: %v", iteration, next.InstanceID, err)
		}
	}
	t.Logf("accepted=%d rejected=%d heartbeats=%d", accepted, rejected, heartbeats)
	if accepted < 300 || rejected < 300 || heartbeats < 100 {
		t.Fatalf("degenerate property sample: accepted=%d rejected=%d heartbeats=%d", accepted, rejected, heartbeats)
	}
}

// TestSharedOperationsCapabilityRulesAreTheRegistrationRules pins the Operations read
// model to the shared registration rules rather than a copy of them.
func TestSharedOperationsCapabilityRulesAreTheRegistrationRules(t *testing.T) {
	t.Parallel()
	cases := readRuntimeCapabilityCases(t)
	valid := func() RuntimeAgentObservation {
		return RuntimeAgentObservation{
			InstanceID: "runtime-1", SoftwareVersion: "0.1.0",
			SupportedRuntimes:        []string{"adk@1"},
			SupportedToolsets:        []RuntimeToolsetCapability{{Ref: "run-artifacts@1", Tools: []string{"read_artifact"}}},
			SupportedSandboxProfiles: []string{"local-workdir@1"},
			ObservedState:            control.AgentIdle, SlotState: SlotIdle,
		}
	}
	for _, group := range []struct {
		values []string
		valid  bool
	}{{cases.ToolName.Valid, true}, {cases.ToolName.Invalid, false}} {
		for _, tool := range group.values {
			agent := valid()
			agent.SupportedToolsets[0].Tools = []string{tool}
			if err := agent.Validate(); (err == nil) != group.valid {
				t.Errorf("Operations tool %q: valid=%v, error=%v", tool, group.valid, err)
			}
		}
	}
	for _, group := range []struct {
		values []string
		valid  bool
	}{{cases.Identity.Valid, true}, {cases.Identity.Invalid, false}} {
		for _, identity := range group.values {
			agent := valid()
			agent.InstanceID = identity
			if err := agent.Validate(); (err == nil) != group.valid {
				t.Errorf("Operations instanceId %q: valid=%v, error=%v", identity, group.valid, err)
			}
		}
	}
	agent := valid()
	agent.SupportedToolsets[0].Tools = numberedValues(cases.ToolsPerToolset.MaxItems, "tool_%03d")
	if err := agent.Validate(); err != nil {
		t.Errorf("Operations rejects the largest Toolset capability: %v", err)
	}
	agent.SupportedToolsets[0].Tools = numberedValues(cases.ToolsPerToolset.MaxItems+1, "tool_%03d")
	if err := agent.Validate(); err == nil {
		t.Error("Operations accepts an oversized Toolset capability")
	}
	agent = valid()
	agent.SupportedRuntimes = []string{strings.Repeat("a", cases.CapabilityRefs.MaxLength-2) + "@1"}
	if err := agent.Validate(); err != nil {
		t.Errorf("Operations rejects the longest capability ref: %v", err)
	}
	agent.SupportedRuntimes = []string{strings.Repeat("a", cases.CapabilityRefs.MaxLength-1) + "@1"}
	if err := agent.Validate(); err == nil {
		t.Error("Operations accepts an oversized capability ref")
	}
	agent = valid()
	agent.SupportedRuntimes = numberedValues(cases.CapabilityRefs.MaxItems+1, "ref-%03d@1")
	if err := agent.Validate(); err == nil {
		t.Error("Operations accepts too many capability refs")
	}
	if !contracts.ValidName(strings.Repeat("a", cases.ToolName.MaxLength)) ||
		contracts.ValidName(strings.Repeat("a", cases.ToolName.MaxLength+1)) ||
		contracts.MaxNameLength != cases.ToolName.MaxLength ||
		control.MaxToolsetCapabilityTools != cases.ToolsPerToolset.MaxItems ||
		control.MaxRuntimeCapabilityRefs != cases.CapabilityRefs.MaxItems ||
		control.MaxRuntimeCapabilityRefLength != cases.CapabilityRefs.MaxLength ||
		contracts.MaxResourceIDLength != cases.Identity.MaxLength {
		t.Fatal("shared Runtime capability bounds differ from the shared cases")
	}
}

// TestSharedPlacementReservesExactlyTheLogicalWorkersOperationsShows is the
// regression for a logical Agent name such as "my worker": the catalog and
// placement accepted it while the Operations snapshot rejected the allocation,
// which failed every Operations allocation list. Placement now reserves
// exactly the names the catalog accepts and Operations shows (WFL-R84).
func TestSharedPlacementReservesExactlyTheLogicalWorkersOperationsShows(t *testing.T) {
	t.Parallel()
	cases := readRuntimeCapabilityCases(t)
	binding := testBinding(t, "builder", "builder", testTemplate(t))
	longest := strings.Repeat("a", cases.ToolName.MaxLength)
	for _, group := range []struct {
		values []string
		valid  bool
	}{
		{append([]string{longest}, cases.ToolName.Valid...), true},
		{append([]string{longest + "a"}, cases.ToolName.Invalid...), false},
	} {
		for _, name := range group.values {
			registry := newTestRegistry(t, newTestClock())
			registerReady(t, registry, "agent-names")
			candidate := binding
			candidate.LogicalAgentName = name
			_, err := registry.ReserveAll(ReservationRequest{
				RunID: "run-names", StageExecutionID: "stage-names", Bindings: []BindingRequirement{candidate},
			})
			snapshot := registry.SnapshotOperations()
			if validateErr := snapshot.Validate(); validateErr != nil {
				t.Fatalf("logical Worker %q: Operations snapshot is invalid: %v", name, validateErr)
			}
			if group.valid {
				if err != nil || len(snapshot.Allocations) != 1 || snapshot.Allocations[0].LogicalWorker != name {
					t.Errorf("logical Worker %q: ReserveAll error = %v, allocations = %+v", name, err, snapshot.Allocations)
				}
				continue
			}
			if !errors.Is(err, ErrInvalidRequest) || len(snapshot.Allocations) != 0 {
				t.Errorf("logical Worker %q: ReserveAll error = %v, allocations = %+v; want ErrInvalidRequest and none",
					name, err, snapshot.Allocations)
			}
		}
	}
}
