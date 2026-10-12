package contracts_test

import (
	"bytes"
	"encoding/json"
	"flag"
	"os"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"

	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
)

var updateRequiredMemberCases = flag.Bool(
	"update-required-member-cases", false,
	"rewrite api/testdata/v1alpha1/required-member-cases.json from the schemas and valid fixtures",
)

const requiredMemberCasesDescription = "Generated from api/v1alpha1 and the valid golden fixtures; do not edit. " +
	"For every valid fixture, omit lists the JSON Pointer of each object member whose omission makes the " +
	"fixture's JSON Schema reject the document, and null those of them whose replacement by null the schema " +
	"rejects too. Go and Python recompute both lists with their own schema validators and must reject every " +
	"case. Regenerate with: go test ./internal/contracts -run TestSharedRequiredMemberCasesAreRejected " +
	"-args -update-required-member-cases"

// requiredMemberCases mirrors api/testdata/v1alpha1/required-member-cases.json.
type requiredMemberCases struct {
	Description string                         `json:"description"`
	Cases       map[string]requiredMemberMutes `json:"cases"`
}

type requiredMemberMutes struct {
	Omit []string `json:"omit"`
	Null []string `json:"null"`
}

// TestSharedRequiredMemberCasesAreRejected drops every member of every valid
// fixture in turn, and sets each required one to null. Each document the
// schema rejects must also be rejected by the private decoder, and the shared
// case file must list exactly those documents, so the Python side tests the
// same cases. Only the parts of a RuntimeReport that the Server isolates are
// discarded instead of failing the report (isolatedRuntimeTelemetry).
func TestSharedRequiredMemberCasesAreRejected(t *testing.T) {
	compiler, ids := privateSchemaCompiler(t)
	index := readFixtureIndex(t)
	computed := map[string]requiredMemberMutes{}
	for _, name := range sortedFixtureNames(index.Valid) {
		entry := index.Valid[name]
		file, fragment, _ := strings.Cut(entry.Schema, "#")
		location := ids[file]
		if fragment != "" {
			location += "#" + fragment
		}
		schema, err := compiler.Compile(location)
		if err != nil {
			t.Fatalf("%s: compile %s: %v", name, entry.Schema, err)
		}
		raw := contractstest.ReadFixture(t, "valid", name)
		cases := requiredMemberMutes{Omit: []string{}, Null: []string{}}
		for _, pointer := range memberPointers(t, decodeNumbers(t, raw), "") {
			if schema.Validate(withoutMember(t, raw, pointer)) == nil {
				continue
			}
			cases.Omit = append(cases.Omit, pointer)
			if schema.Validate(withNullMember(t, raw, pointer)) != nil {
				cases.Null = append(cases.Null, pointer)
			}
		}
		sort.Strings(cases.Omit)
		sort.Strings(cases.Null)
		computed[name] = cases
	}

	path := contractstest.Path(t, "api", "testdata", "v1alpha1", "required-member-cases.json")
	if *updateRequiredMemberCases {
		encoded, err := json.MarshalIndent(requiredMemberCases{
			Description: requiredMemberCasesDescription, Cases: computed,
		}, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(encoded, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var committed requiredMemberCases
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "required-member-cases.json"), &committed); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(committed.Cases, computed) || committed.Description != requiredMemberCasesDescription {
		t.Fatalf("required-member-cases.json differs from the schemas; regenerate it with -update-required-member-cases")
	}

	total := 0
	for _, name := range sortedFixtureNames(index.Valid) {
		typeName := index.Valid[name].Type
		reject := fixtureCodecs[typeName].privateReject
		raw := contractstest.ReadFixture(t, "valid", name)
		if err := reject(raw); err != nil {
			t.Fatalf("%s: valid fixture rejected: %v", name, err)
		}
		for kind, pointers := range map[string][]string{"without": computed[name].Omit, "with null": computed[name].Null} {
			for _, pointer := range pointers {
				total++
				document := withoutMember(t, raw, pointer)
				if kind == "with null" {
					document = withNullMember(t, raw, pointer)
				}
				encoded, err := json.Marshal(document)
				if err != nil {
					t.Fatal(err)
				}
				if check, isolated := isolatedRuntimeTelemetry(typeName, pointer); isolated {
					if problem := check(encoded); problem != "" {
						t.Errorf("%s %s %s: %s", name, kind, pointer, problem)
					}
					continue
				}
				if reject(encoded) == nil {
					t.Errorf("%s %s %s was accepted", name, kind, pointer)
				}
			}
		}
	}
	if total == 0 {
		t.Fatal("no required member case was tested")
	}
}

// isolatedRuntimeTelemetryPattern matches the RuntimeReport parts the Server
// discards as incomplete telemetry instead of rejecting the whole report: one
// adapter's metrics (OBS-R36, FAIL-OBS-1) and the resources block (CTL-R136,
// FAIL-CTL-10).
var isolatedRuntimeTelemetryPattern = regexp.MustCompile(`^(/report/runtime)?/(adapters/[^/]+|resources)/`)

// isolatedRuntimeTelemetry returns, for a case inside an isolated part, the
// check that the decoded report dropped that part and is marked incomplete.
func isolatedRuntimeTelemetry(typeName, pointer string) (func([]byte) string, bool) {
	match := isolatedRuntimeTelemetryPattern.FindStringSubmatch(pointer)
	if match == nil {
		return nil, false
	}
	var runtime func([]byte) (reporting.RuntimeReport, error)
	switch {
	case typeName == "RuntimeReport" && match[1] == "":
		runtime = contracts.DecodePrivateStrict[reporting.RuntimeReport]
	case typeName == "AllocationFinalResponse" && match[1] != "":
		runtime = func(data []byte) (reporting.RuntimeReport, error) {
			response, err := contracts.DecodePrivateStrict[reporting.AllocationFinalResponse](data)
			return response.Report.Runtime, err
		}
	default:
		return nil, false
	}
	return func(data []byte) string {
		report, err := runtime(data)
		switch {
		case err != nil:
			return "rejected the whole report: " + err.Error()
		case match[2] == "resources":
			if report.Resources != nil || report.ResourcesError == nil {
				return "kept a malformed resources block"
			}
		case report.Complete:
			return "kept the report complete"
		default:
			ref := strings.NewReplacer("~1", "/", "~0", "~").Replace(strings.TrimPrefix(match[2], "adapters/"))
			if _, kept := report.Adapters[contracts.RuntimeAdapterRef(ref)]; kept {
				return "kept the malformed adapter metrics"
			}
		}
		return ""
	}, true
}

// decodeNumbers decodes JSON keeping every number exact.
func decodeNumbers(t *testing.T, raw []byte) any {
	t.Helper()
	value, err := jsonschema.UnmarshalJSON(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	return value
}

// memberPointers lists the JSON Pointer of every object member in value.
func memberPointers(t *testing.T, value any, prefix string) []string {
	t.Helper()
	var result []string
	switch typed := value.(type) {
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			pointer := prefix + "/" + strings.NewReplacer("~", "~0", "/", "~1").Replace(key)
			result = append(result, pointer)
			result = append(result, memberPointers(t, typed[key], pointer)...)
		}
	case []any:
		for position, item := range typed {
			result = append(result, memberPointers(t, item, prefix+"/"+strconv.Itoa(position))...)
		}
	}
	return result
}

// withoutMember returns raw decoded with the member at pointer removed.
func withoutMember(t *testing.T, raw []byte, pointer string) any {
	t.Helper()
	document := decodeNumbers(t, raw)
	object, name := memberParent(t, document, pointer)
	delete(object, name)
	return document
}

// withNullMember returns raw decoded with the member at pointer set to null.
func withNullMember(t *testing.T, raw []byte, pointer string) any {
	t.Helper()
	document := decodeNumbers(t, raw)
	object, name := memberParent(t, document, pointer)
	object[name] = nil
	return document
}

// memberParent returns the object holding the member at pointer and its name.
func memberParent(t *testing.T, document any, pointer string) (map[string]any, string) {
	t.Helper()
	parts := strings.Split(pointer, "/")[1:]
	for index := range parts {
		parts[index] = strings.NewReplacer("~1", "/", "~0", "~").Replace(parts[index])
	}
	target := document
	for _, part := range parts[:len(parts)-1] {
		switch typed := target.(type) {
		case map[string]any:
			target = typed[part]
		case []any:
			position, err := strconv.Atoi(part)
			if err != nil {
				t.Fatalf("pointer %s: %v", pointer, err)
			}
			target = typed[position]
		}
	}
	object, ok := target.(map[string]any)
	if !ok {
		t.Fatalf("pointer %s does not name an object member", pointer)
	}
	return object, parts[len(parts)-1]
}
