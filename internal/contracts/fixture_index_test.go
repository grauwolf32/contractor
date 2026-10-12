package contracts_test

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/dlclark/regexp2"
	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"
	"golang.org/x/text/message"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/strictjson"
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
	typ              reflect.Type
	roundTrip        func([]byte) ([]byte, error)
	reject           func([]byte) error
	privateRoundTrip func([]byte) ([]byte, error)
	privateReject    func([]byte) error
}

func codecFor[T contracts.Validatable]() fixtureCodec {
	return fixtureCodec{reflect.TypeFor[T](), roundTrip[T], reject[T], privateRoundTrip[T], privateReject[T]}
}

// revealingCodecFor round-trips a type that carries RuntimeSettings secrets
// through its allocation-wire encoder, because default JSON encoding redacts
// them and the fixtures hold their plaintext.
func revealingCodecFor[T contracts.Validatable](reveal func(T) ([]byte, error)) fixtureCodec {
	return fixtureCodec{
		typ: reflect.TypeFor[T](),
		roundTrip: func(data []byte) ([]byte, error) {
			value, err := contracts.DecodeStrict[T](data)
			if err != nil {
				return nil, err
			}
			return reveal(value)
		},
		reject: reject[T],
		privateRoundTrip: func(data []byte) ([]byte, error) {
			value, err := contracts.DecodePrivateStrict[T](data)
			if err != nil {
				return nil, err
			}
			revealed, err := reveal(value)
			if err != nil {
				return nil, err
			}
			return strictjson.Canonical(json.RawMessage(revealed))
		},
		privateReject: privateReject[T],
	}
}

var fixtureCodecs = map[string]fixtureCodec{
	"AbortAllocationRequest":          codecFor[control.AbortAllocationRequest](),
	"AgentHeartbeat":                  codecFor[control.AgentHeartbeat](),
	"AgentRegistration":               codecFor[control.AgentRegistration](),
	"AgentRegistrationResponse":       codecFor[control.AgentRegistrationResponse](),
	"AgentStateSnapshot":              codecFor[reporting.AgentStateSnapshot](),
	"AllocationFinalResponse":         codecFor[reporting.AllocationFinalResponse](),
	"AllocationSpec":                  revealingCodecFor(control.AllocationSpec.RevealJSON),
	"AllocationWorkspaceSpec":         codecFor[contracts.AllocationWorkspaceSpec](),
	"ArtifactListResult":              codecFor[contracts.ArtifactListResult](),
	"ArtifactReadResult":              codecFor[contracts.ArtifactReadResult](),
	"FinalizeAllocationRequest":       codecFor[control.FinalizeAllocationRequest](),
	"HeartbeatResponse":               codecFor[control.HeartbeatResponse](),
	"ReleaseAllocationRequest":        codecFor[control.ReleaseAllocationRequest](),
	"ResolvedLLMGatewayConfig":        codecFor[llmgateway.ResolvedLLMGatewayConfig](),
	"ResolvedRuntimeConfigProvenance": codecFor[runtimesettings.ResolvedRuntimeConfigProvenance](),
	"RuntimeReport":                   codecFor[reporting.RuntimeReport](),
	"RuntimeSettings":                 revealingCodecFor(runtimesettings.RuntimeSettings.RevealJSON),
	"StageContentRequest":             codecFor[contracts.StageContentRequest](),
	"WorkerCompletion":                codecFor[contracts.WorkerCompletion](),
	"WorkspaceCapabilities":           codecFor[contracts.WorkspaceCapabilities](),
}

func readFixtureIndex(t *testing.T) fixtureIndex {
	t.Helper()
	data := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "index.json")
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

func TestSharedFixtureIndexListsEveryGoldenFixture(t *testing.T) {
	t.Parallel()
	index := readFixtureIndex(t)
	for kind, entries := range map[string]map[string]fixtureIndexEntry{"valid": index.Valid, "invalid": index.Invalid} {
		paths, err := filepath.Glob(filepath.Join(contractstest.Path(t, "api", "testdata", "v1alpha1", kind), "*.json"))
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

// maxBytesKeyword bounds a string's UTF-8 length where Go and the Runtime
// count bytes: JSON Schema's maxLength counts characters, so it alone would
// accept what both implementations reject. The Python schema tests enforce
// the same keyword (runtime/tests/private_schemas.py).
const maxBytesKeyword = "x-maxBytes"

type maxBytes struct{ bound int }

func (m maxBytes) Validate(ctx *jsonschema.ValidatorContext, value any) {
	if text, ok := value.(string); ok && len(text) > m.bound {
		ctx.AddError(&maxBytesError{Got: len(text), Want: m.bound})
	}
}

type maxBytesError struct{ Got, Want int }

func (*maxBytesError) KeywordPath() []string { return []string{maxBytesKeyword} }

func (e *maxBytesError) LocalizedString(printer *message.Printer) string {
	return printer.Sprintf("%s: got %d bytes, want at most %d", maxBytesKeyword, e.Got, e.Want)
}

func maxBytesVocabulary(t *testing.T) *jsonschema.Vocabulary {
	t.Helper()
	const url = "https://contractor.local/schema/vocabulary/max-bytes"
	meta, err := jsonschema.UnmarshalJSON(strings.NewReader(
		`{"properties": {"` + maxBytesKeyword + `": {"type": "integer", "minimum": 1}}}`,
	))
	if err != nil {
		t.Fatal(err)
	}
	compiler := jsonschema.NewCompiler()
	if err := compiler.AddResource(url, meta); err != nil {
		t.Fatal(err)
	}
	schema, err := compiler.Compile(url)
	if err != nil {
		t.Fatal(err)
	}
	return &jsonschema.Vocabulary{
		URL: url, Schema: schema,
		Compile: func(_ *jsonschema.CompilerContext, object map[string]any) (jsonschema.SchemaExt, error) {
			bound, ok := object[maxBytesKeyword].(json.Number)
			if !ok {
				return nil, nil
			}
			value, err := bound.Int64()
			if err != nil {
				return nil, err
			}
			return maxBytes{bound: int(value)}, nil
		},
	}
}

// privateSchemaCompiler loads every api/v1alpha1 schema with ECMA-262
// patterns and the x-maxBytes keyword and returns the compiler and each
// file's $id.
func privateSchemaCompiler(t *testing.T) (*jsonschema.Compiler, map[string]string) {
	t.Helper()
	compiler := jsonschema.NewCompiler()
	compiler.UseRegexpEngine(compileECMAPattern)
	compiler.AssertVocabs()
	compiler.RegisterVocabulary(maxBytesVocabulary(t))
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
	return compiler, ids
}

func TestSharedPrivateSchemasAcceptAndRejectGoldenFixtures(t *testing.T) {
	t.Parallel()
	compiler, ids := privateSchemaCompiler(t)
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
