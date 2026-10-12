package resolved

import (
	"bytes"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

// TestRegisteredDescriptorsNormalize reports a defect in the built-in table
// as a test failure instead of the panic of the registry's first reader.
func TestRegisteredDescriptorsNormalize(t *testing.T) {
	t.Parallel()
	if _, err := normalizeDescriptors(builtInDescriptors()); err != nil {
		t.Fatalf("built-in descriptor table: %v", err)
	}
}

// Every check of a selection against registered code reads one normalized
// registry, so no caller can be handed, or build, a table of its own: the
// table and its types are unexported, the loader and the decoders take no
// descriptor argument, and inside this package only registeredDescriptors
// reads the built-in table.
func TestBuiltInDescriptorTableHasOneReader(t *testing.T) {
	t.Parallel()
	paths, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	var readers []string
	for _, path := range paths {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		source, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		file, err := parser.ParseFile(fset, path, source, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		for _, declaration := range file.Decls {
			ast.Inspect(declaration, func(node ast.Node) bool {
				call, ok := node.(*ast.CallExpr)
				if !ok {
					return true
				}
				if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "builtInDescriptors" {
					readers = append(readers, enclosingName(declaration)+" at "+fset.Position(call.Pos()).String())
				}
				return true
			})
		}
	}
	if len(readers) != 1 || !strings.HasPrefix(readers[0], "registeredDescriptors ") {
		t.Fatalf("builtInDescriptors is read by %v; read the registry through registeredDescriptors instead", readers)
	}
}

func enclosingName(declaration ast.Decl) string {
	switch value := declaration.(type) {
	case *ast.FuncDecl:
		return value.Name.Name
	case *ast.GenDecl:
		for _, spec := range value.Specs {
			if variable, ok := spec.(*ast.ValueSpec); ok && len(variable.Names) > 0 {
				return variable.Names[0].Name
			}
		}
	}
	return "unknown declaration"
}

// The Planner facts live on the registered PlannerFactory descriptor; the
// catalog load, every snapshot decoder and therefore the Scheduler read them
// from there (LIM-PLN-1, LIM-PLN-2).
func TestPlannerDescriptorsDecideAgentCountAndModelUse(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		planner   PlannerRef
		modelFree bool
		accepts   map[int]bool
	}{
		{PlannerRef{PlannerID: "passthrough", Version: "1"}, true, map[int]bool{0: false, 1: true, 2: false}},
		{PlannerRef{PlannerID: "streamline", Version: "1"}, false, map[int]bool{0: false, 1: true, 2: false}},
		{PlannerRef{PlannerID: "router", Version: "1"}, false, map[int]bool{0: false, 1: true, 2: true}},
		{PlannerRef{PlannerID: "scan-plan", Version: "1"}, true, map[int]bool{0: false, 1: true, 2: true}},
	} {
		if !KnownPlanner(test.planner) || IsModelFreePlanner(test.planner) != test.modelFree {
			t.Errorf("%+v: known = %v, model-free = %v", test.planner, KnownPlanner(test.planner), IsModelFreePlanner(test.planner))
		}
		for agents, accepted := range test.accepts {
			if err := ValidatePlannerAgents(test.planner, agents); (err == nil) != accepted {
				t.Errorf("%+v with %d logical Agents: %v", test.planner, agents, err)
			}
		}
	}
	retired := PlannerRef{PlannerID: "passthrough", Version: "2"}
	if KnownPlanner(retired) || IsModelFreePlanner(retired) {
		t.Fatal("an unregistered Planner is known")
	}
	if err := ValidatePlannerAgents(retired, 1); err == nil || !strings.Contains(err.Error(), `unknown PlannerFactory "passthrough@2"`) {
		t.Fatalf("unregistered Planner error = %v", err)
	}
}

// TestRequiredRuntimeAdaptersFollowToolChannels checks how the channels the
// descriptors declare become placement requirements: every http-tools@1 tool
// uses the direct or optionally proxied HTTP client and needs no adapter,
// while every Caido tool needs the typed Caido client. The tool lists and
// their channels are the ones TestSharedToolsetDescriptorParity pins.
func TestRequiredRuntimeAdaptersFollowToolChannels(t *testing.T) {
	t.Parallel()
	for ref, want := range map[contracts.ToolsetRef][]runtimesettings.RuntimeAdapterRef{
		{ToolsetID: "http-tools", Version: "1"}: nil,
		{ToolsetID: "caido", Version: "1"}:      {runtimesettings.RuntimeAdapterCaidoGraphQL},
	} {
		tools := registeredDescriptors().Toolsets[ref.ToolsetID+"@"+ref.Version].Tools
		if len(tools) == 0 {
			t.Fatalf("Toolset %s has no tools", ref.ToolsetID)
		}
		for _, tool := range tools {
			selected := contracts.ResolvedAgentTemplate{Toolsets: []contracts.ToolsetSelection{{Ref: ref, Tools: []string{tool}}}}
			adapters, err := RequiredRuntimeAdaptersForTemplate(selected)
			if err != nil || !slices.Equal(adapters, want) {
				t.Errorf("%s adapter requirements = (%v, %v), want %v", tool, adapters, err, want)
			}
		}
	}
}

func TestToolsetDescriptorRejectsInvalidInfrastructureChannels(t *testing.T) {
	t.Parallel()

	for name, channels := range map[string]map[string][]infrastructureChannel{
		"unknown tool": {"missing": {channelRuntimeHTTPClient}},
		"empty":        {"read_artifact": {}},
		"unknown channel": {
			"read_artifact": {infrastructureChannel("ambient-network")},
		},
		"duplicate": {"read_artifact": {channelRuntimeHTTPClient, channelRuntimeHTTPClient}},
	} {
		t.Run(name, func(t *testing.T) {
			descriptors := builtInDescriptors()
			descriptor := descriptors.Toolsets["run-artifacts@1"]
			descriptor.InfrastructureChannels = channels
			descriptors.Toolsets["run-artifacts@1"] = descriptor
			if _, err := normalizeDescriptors(descriptors); err == nil {
				t.Fatal("normalizeDescriptors accepted invalid infrastructure channels")
			}
		})
	}
}

// toolsetDescriptorFixture is one Toolset of api/testdata/toolset-descriptors.json.
type toolsetDescriptorFixture struct {
	Tools                  []string                           `json:"tools"`
	InfrastructureChannels map[string][]infrastructureChannel `json:"infrastructureChannels"`
	RequiresWorkspace      bool                               `json:"requiresWorkspace"`
	RequiredSandboxProfile *string                            `json:"requiredSandboxProfile"`
	ActiveCheckTools       []string                           `json:"activeCheckTools"`
	FindingProposalTools   []string                           `json:"findingProposalTools"`
	RequiredWorkflowInputs map[string]struct {
		Input     string `json:"input"`
		MediaType string `json:"mediaType"`
	} `json:"requiredWorkflowInputs"`
}

// TestSharedToolsetDescriptorParity holds every Server Toolset descriptor to
// api/testdata/toolset-descriptors.json, which the Runtime's factories are
// held to as well (runtime/tests/test_capabilities.py), so a tool or Toolset
// added on one side only fails.
func TestSharedToolsetDescriptorParity(t *testing.T) {
	t.Parallel()
	var fixture struct {
		Description string                              `json:"description"`
		Toolsets    map[string]toolsetDescriptorFixture `json:"toolsets"`
	}
	decoder := json.NewDecoder(bytes.NewReader(contractstest.ReadFile(t, "api", "testdata", "toolset-descriptors.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatal(err)
	}
	normalized := registeredDescriptors()
	if got, want := slices.Sorted(maps.Keys(normalized.Toolsets)), slices.Sorted(maps.Keys(fixture.Toolsets)); !slices.Equal(got, want) {
		t.Fatalf("Toolset descriptors = %v, shared fixture = %v", got, want)
	}
	for ref, shared := range fixture.Toolsets {
		profile := ""
		if shared.RequiredSandboxProfile != nil {
			profile = *shared.RequiredSandboxProfile
		}
		// The fixture spells an absent collection as an empty one.
		want := toolsetDescriptor{
			RequiredSandboxProfile: profile, RequiresWorkspace: shared.RequiresWorkspace,
			Tools: shared.Tools, InfrastructureChannels: shared.InfrastructureChannels,
		}
		if len(shared.ActiveCheckTools) > 0 {
			want.ActiveCheckTools = shared.ActiveCheckTools
		}
		if len(shared.FindingProposalTools) > 0 {
			want.FindingProposalTools = shared.FindingProposalTools
		}
		for tool, input := range shared.RequiredWorkflowInputs {
			if want.RequiredWorkflowInputs == nil {
				want.RequiredWorkflowInputs = map[string]workflowInputRequirement{}
			}
			want.RequiredWorkflowInputs[tool] = workflowInputRequirement{Input: input.Input, MediaType: input.MediaType}
		}
		if got := normalized.Toolsets[ref]; !reflect.DeepEqual(got, want) {
			t.Errorf("Toolset %s descriptor = %+v, shared fixture = %+v", ref, got, want)
		}
	}
}

// The Agent Skills toolset owns its tool names whenever a template selects
// skills, so the registry refuses a Toolset that exports one.
func TestToolsetDescriptorRejectsNativeSkillToolNames(t *testing.T) {
	t.Parallel()
	descriptors := builtInDescriptors()
	descriptor := descriptors.Toolsets["run-artifacts@1"]
	descriptor.Tools = append(descriptor.Tools, "load_skill")
	descriptors.Toolsets["run-artifacts@1"] = descriptor
	if _, err := normalizeDescriptors(descriptors); err == nil ||
		!strings.Contains(err.Error(), `exports tool "load_skill", which is reserved by Agent Skills`) {
		t.Fatalf("reserved tool descriptor error = %v", err)
	}
}
