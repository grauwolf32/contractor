package contracts_test

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/dlclark/regexp2"
	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

// fixtureIndexEntry mirrors api/testdata/v1alpha1/index.json, which Go and
// Python both read instead of keeping their own fixture tables.
type fixtureIndexEntry struct {
	Type   string `json:"type"`
	Schema string `json:"schema"`
	Strict bool   `json:"strict"`
	Reason string `json:"reason"`
}

type fixtureIndex struct {
	Valid   map[string]fixtureIndexEntry `json:"valid"`
	Invalid map[string]fixtureIndexEntry `json:"invalid"`
}

type fixtureCodec struct {
	roundTrip        func([]byte) ([]byte, error)
	reject           func([]byte) error
	privateRoundTrip func([]byte) ([]byte, error)
	privateReject    func([]byte) error
}

func codecFor[T contracts.Validatable]() fixtureCodec {
	return fixtureCodec{roundTrip[T], reject[T], privateRoundTrip[T], privateReject[T]}
}

var fixtureCodecs = map[string]fixtureCodec{
	"AbortAllocationRequest":          codecFor[control.AbortAllocationRequest](),
	"AgentHeartbeat":                  codecFor[control.AgentHeartbeat](),
	"AgentRegistration":               codecFor[control.AgentRegistration](),
	"AgentRegistrationResponse":       codecFor[control.AgentRegistrationResponse](),
	"AgentStateSnapshot":              codecFor[reporting.AgentStateSnapshot](),
	"AllocationFinalResponse":         codecFor[reporting.AllocationFinalResponse](),
	"AllocationSpec":                  codecFor[control.AllocationSpec](),
	"AllocationWorkspaceSpec":         codecFor[contracts.AllocationWorkspaceSpec](),
	"ArtifactListResult":              codecFor[contracts.ArtifactListResult](),
	"ArtifactReadResult":              codecFor[contracts.ArtifactReadResult](),
	"FinalizeAllocationRequest":       codecFor[control.FinalizeAllocationRequest](),
	"HeartbeatResponse":               codecFor[control.HeartbeatResponse](),
	"ReleaseAllocationRequest":        codecFor[control.ReleaseAllocationRequest](),
	"ResolvedLLMGatewayConfig":        codecFor[llmgateway.ResolvedLLMGatewayConfig](),
	"ResolvedRuntimeConfigProvenance": codecFor[runtimesettings.ResolvedRuntimeConfigProvenance](),
	"RuntimeReport":                   codecFor[reporting.RuntimeReport](),
	"RuntimeSettings":                 codecFor[runtimesettings.RuntimeSettings](),
	"StageContentRequest":             codecFor[contracts.StageContentRequest](),
	"StageContentResult":              codecFor[contracts.StageContentResult](),
	"WorkerCompletion":                codecFor[contracts.WorkerCompletion](),
	"WorkspaceCapabilities":           codecFor[contracts.WorkspaceCapabilities](),
}

func readFixtureIndex(t *testing.T) fixtureIndex {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "api", "testdata", "v1alpha1", "index.json"))
	if err != nil {
		t.Fatal(err)
	}
	var index fixtureIndex
	if err := json.Unmarshal(data, &index); err != nil {
		t.Fatal(err)
	}
	return index
}

func sortedFixtureNames(entries map[string]fixtureIndexEntry) []string {
	names := make([]string, 0, len(entries))
	for name := range entries {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func TestFixtureIndexListsEveryGoldenFixture(t *testing.T) {
	t.Parallel()
	index := readFixtureIndex(t)
	for kind, entries := range map[string]map[string]fixtureIndexEntry{"valid": index.Valid, "invalid": index.Invalid} {
		paths, err := filepath.Glob(filepath.Join("..", "..", "api", "testdata", "v1alpha1", kind, "*.json"))
		if err != nil {
			t.Fatal(err)
		}
		files := make([]string, len(paths))
		for i, path := range paths {
			files[i] = filepath.Base(path)
		}
		if got := sortedFixtureNames(entries); strings.Join(got, ",") != strings.Join(files, ",") {
			t.Fatalf("%s index = %v, files = %v", kind, got, files)
		}
		for name, entry := range entries {
			if _, ok := fixtureCodecs[entry.Type]; !ok {
				t.Errorf("%s: unknown type %q", name, entry.Type)
			}
			if kind == "invalid" && entry.Schema == "" && !entry.Strict && entry.Reason == "" {
				t.Errorf("%s: invalid fixture asserts nothing", name)
			}
		}
	}
}

// ecmaRegexp evaluates schema patterns as ECMA-262 regular expressions, which
// use \uXXXX escapes and lookaheads that RE2 does not support.
type ecmaRegexp struct{ *regexp2.Regexp }

func (r ecmaRegexp) MatchString(value string) bool {
	matched, err := r.Regexp.MatchString(value)
	return err == nil && matched
}

func compileECMAPattern(pattern string) (jsonschema.Regexp, error) {
	compiled, err := regexp2.Compile(pattern, regexp2.ECMAScript)
	if err != nil {
		return nil, err
	}
	return ecmaRegexp{compiled}, nil
}

func TestPrivateSchemasAcceptAndRejectGoldenFixtures(t *testing.T) {
	t.Parallel()
	compiler := jsonschema.NewCompiler()
	compiler.UseRegexpEngine(compileECMAPattern)
	paths, err := filepath.Glob(filepath.Join("..", "..", "api", "v1alpha1", "*.schema.json"))
	if err != nil {
		t.Fatal(err)
	}
	ids := map[string]string{}
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		document, err := jsonschema.UnmarshalJSON(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		id, _ := document.(map[string]any)["$id"].(string)
		if err := compiler.AddResource(id, document); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		ids[filepath.Base(path)] = id
	}
	index := readFixtureIndex(t)
	for kind, entries := range map[string]map[string]fixtureIndexEntry{"valid": index.Valid, "invalid": index.Invalid} {
		for _, name := range sortedFixtureNames(entries) {
			entry := entries[name]
			if entry.Schema == "" {
				continue
			}
			file, fragment, _ := strings.Cut(entry.Schema, "#")
			location := ids[file]
			if fragment != "" {
				location += "#" + fragment
			}
			schema, err := compiler.Compile(location)
			if err != nil {
				t.Fatalf("%s: compile %s: %v", name, entry.Schema, err)
			}
			value, err := jsonschema.UnmarshalJSON(bytes.NewReader(contractstest.ReadFixture(t, kind, name)))
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if err := schema.Validate(value); (err == nil) != (kind == "valid") {
				t.Errorf("%s/%s against %s: %v", kind, name, entry.Schema, err)
			}
		}
	}
}
