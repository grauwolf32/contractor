package contracts_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"

	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"
	"github.com/santhosh-tekuri/jsonschema/v6/kind"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

var updateMutationCases = flag.Bool(
	"update-mutation-cases", false,
	"rewrite the shared fixture v1alpha1/mutation-cases.json from the schemas and valid fixtures",
)

const mutationCasesDescription = "Generated from api/v1alpha1 and the valid golden fixtures; do not edit. " +
	"For every valid fixture: enum lists the members whose value, replaced by enumSentinels, fails an enum or " +
	"const of the fixture's JSON Schema; maxLength lists the strings the schema bounds, which as maxLength+1 " +
	"repetitions of \"a\" the schema rejects; unknownMember lists the objects that reject an added member " +
	"\"contractorUnknownMember\": true; duplicateKey lists every object with a member, whose first member (in " +
	"key order) is written twice. Go, Python and the schema reject every such case. maxBytes holds the same " +
	"strings as maxLength-1 repetitions of \"a\" followed by U+00E9, maxLength characters in maxLength+1 UTF-8 " +
	"bytes, with the verdict the schema (x-maxBytes), Go and Python all reach. Regenerate with: go test " +
	"./internal/contracts -run TestSharedMutationCasesAreRejected -args -update-mutation-cases"

// mutationEnumSentinels are values no enum or const of the private schemas
// holds. Both languages put them in place of a member to fail its enum.
var mutationEnumSentinels = map[string]any{"string": "contractor-mutation-probe", "number": 987654321}

// mutationUnknownMember is the member an unknownMember case adds.
const mutationUnknownMember = "contractorUnknownMember"

// mutationCases mirrors api/testdata/v1alpha1/mutation-cases.json.
type mutationCases struct {
	Description   string                          `json:"description"`
	EnumSentinels map[string]any                  `json:"enumSentinels"`
	Cases         map[string]fixtureMutationCases `json:"cases"`
}

type fixtureMutationCases struct {
	Enum          []string           `json:"enum"`
	MaxLength     []lengthMutation   `json:"maxLength"`
	MaxBytes      []byteSizeMutation `json:"maxBytes"`
	UnknownMember []string           `json:"unknownMember"`
	DuplicateKey  []string           `json:"duplicateKey"`
}

type lengthMutation struct {
	Pointer   string `json:"pointer"`
	MaxLength int    `json:"maxLength"`
}

type byteSizeMutation struct {
	Pointer   string `json:"pointer"`
	MaxLength int    `json:"maxLength"`
	Verdict   string `json:"verdict"`
}

// TestSharedMutationCasesAreRejected mutates every valid fixture where its
// JSON Schema constrains it: a value outside an enum, a string one character
// over its maxLength, an unknown member and a duplicate key. The schema and
// the private decoder must reject each case, and the Python Runtime checks the
// same committed cases, so a mutation accepted by one side only fails a
// test. The maxBytes cases stay within maxLength characters but exceed it in
// UTF-8 bytes: where Go and Python count bytes the schema states x-maxBytes,
// so all three reach the same verdict. Only the parts of a RuntimeReport that
// the Server isolates are discarded instead of failing the report
// (isolatedRuntimeTelemetry).
func TestSharedMutationCasesAreRejected(t *testing.T) {
	compiler, ids := privateSchemaCompiler(t)
	index := readFixtureIndex(t)
	computed := map[string]fixtureMutationCases{}
	total := 0
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
		reject := fixtureCodecs[entry.Type].privateReject
		if err := reject(raw); err != nil {
			t.Fatalf("%s: valid fixture rejected: %v", name, err)
		}
		cases := discoverMutations(t, schema, raw)
		for index, item := range cases.MaxBytes {
			document := applyMutation(t, raw, "maxBytes", item.Pointer, item.MaxLength)
			schemaVerdict := verdict(schema.Validate(decodeNumbers(t, document)) == nil)
			if decoderVerdict := verdict(reject(document) == nil); decoderVerdict != schemaVerdict {
				t.Errorf("%s maxBytes %s: the schema and the decoder disagree (%s, %s)", name, item.Pointer, schemaVerdict, decoderVerdict)
			}
			cases.MaxBytes[index].Verdict = schemaVerdict
		}
		computed[name] = cases
	}

	path := contractstest.Path(t, "api", "testdata", "v1alpha1", "mutation-cases.json")
	if *updateMutationCases {
		encoded, err := json.MarshalIndent(mutationCases{
			Description: mutationCasesDescription, EnumSentinels: mutationEnumSentinels, Cases: computed,
		}, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(encoded, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var committed mutationCases
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "mutation-cases.json"), &committed); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(committed.Cases, computed) || committed.Description != mutationCasesDescription ||
		!reflect.DeepEqual(normalizeSentinels(committed.EnumSentinels), normalizeSentinels(mutationEnumSentinels)) {
		t.Fatalf("mutation-cases.json differs from the schemas and decoders; regenerate it with -update-mutation-cases")
	}

	for _, name := range sortedFixtureNames(index.Valid) {
		typeName := index.Valid[name].Type
		reject := fixtureCodecs[typeName].privateReject
		raw := contractstest.ReadFixture(t, "valid", name)
		cases := computed[name]
		check := func(kind, pointer string, document []byte) {
			t.Helper()
			total++
			if check, isolated := isolatedRuntimeTelemetry(typeName, pointer+"/"); isolated && kind != "duplicateKey" {
				if problem := check(document); problem != "" {
					t.Errorf("%s %s %s: %s", name, kind, pointer, problem)
				}
				return
			}
			if reject(document) == nil {
				t.Errorf("%s %s %s was accepted", name, kind, pointer)
			}
		}
		for _, pointer := range cases.Enum {
			check("enum", pointer, applyMutation(t, raw, "enum", pointer, 0))
		}
		for _, item := range cases.MaxLength {
			check("maxLength", item.Pointer, applyMutation(t, raw, "maxLength", item.Pointer, item.MaxLength))
		}
		for _, pointer := range cases.UnknownMember {
			check("unknownMember", pointer, applyMutation(t, raw, "unknownMember", pointer, 0))
		}
		for _, pointer := range cases.DuplicateKey {
			check("duplicateKey", pointer, applyMutation(t, raw, "duplicateKey", pointer, 0))
		}
		total += len(cases.MaxBytes)
	}
	if total == 0 {
		t.Fatal("no mutation case was tested")
	}
}

// discoverMutations lists the mutations the schema rejects in raw, the
// strings it bounds and every object of raw.
func discoverMutations(t *testing.T, schema *jsonschema.Schema, raw []byte) fixtureMutationCases {
	t.Helper()
	cases := fixtureMutationCases{
		Enum: []string{}, MaxLength: []lengthMutation{}, MaxBytes: []byteSizeMutation{},
		UnknownMember: []string{}, DuplicateKey: []string{},
	}
	rejects := func(document []byte) bool { return schema.Validate(decodeNumbers(t, document)) != nil }
	for _, site := range valueSites(decodeNumbers(t, raw), "") {
		switch value := site.value.(type) {
		case map[string]any:
			if len(value) > 0 {
				cases.DuplicateKey = append(cases.DuplicateKey, site.pointer)
			}
			probe := applyMutation(t, raw, "unknownMember", site.pointer, 0)
			if forbidsUnknownMember(schema, decodeNumbers(t, probe), site.pointer) && rejects(probe) {
				cases.UnknownMember = append(cases.UnknownMember, site.pointer)
			}
		case string, json.Number:
			if site.pointer == "" {
				continue
			}
			probe := applyMutation(t, raw, "enum", site.pointer, 0)
			if hasEnumErrorAt(schema, decodeNumbers(t, probe), site.pointer) && rejects(probe) {
				cases.Enum = append(cases.Enum, site.pointer)
			}
			if _, isString := value.(string); !isString {
				continue
			}
			long := withValue(t, raw, site.pointer, strings.Repeat("a", 1<<18))
			for _, bound := range maxLengthsAt(schema, decodeNumbers(t, long), site.pointer) {
				if rejects(applyMutation(t, raw, "maxLength", site.pointer, bound)) {
					cases.MaxLength = append(cases.MaxLength, lengthMutation{Pointer: site.pointer, MaxLength: bound})
					cases.MaxBytes = append(cases.MaxBytes, byteSizeMutation{Pointer: site.pointer, MaxLength: bound})
				}
			}
		}
	}
	sort.Strings(cases.Enum)
	sort.Strings(cases.UnknownMember)
	sort.Strings(cases.DuplicateKey)
	sort.Slice(cases.MaxLength, func(i, j int) bool { return cases.MaxLength[i].Pointer < cases.MaxLength[j].Pointer })
	sort.Slice(cases.MaxBytes, func(i, j int) bool { return cases.MaxBytes[i].Pointer < cases.MaxBytes[j].Pointer })
	return cases
}

type valueSite struct {
	pointer string
	value   any
}

// valueSites lists every value in document with its JSON Pointer, parents
// before children and object members in key order.
func valueSites(value any, pointer string) []valueSite {
	sites := []valueSite{{pointer, value}}
	switch typed := value.(type) {
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			sites = append(sites, valueSites(typed[key], pointer+"/"+escapePointer(key))...)
		}
	case []any:
		for position, item := range typed {
			sites = append(sites, valueSites(item, pointer+"/"+strconv.Itoa(position))...)
		}
	}
	return sites
}

func escapePointer(token string) string {
	return strings.NewReplacer("~", "~0", "/", "~1").Replace(token)
}

func pointerTokens(pointer string) []string {
	if pointer == "" {
		return []string{}
	}
	tokens := strings.Split(pointer, "/")[1:]
	for index := range tokens {
		tokens[index] = strings.NewReplacer("~1", "/", "~0", "~").Replace(tokens[index])
	}
	return tokens
}

// leafErrors flattens a validation error into the errors without causes.
func leafErrors(err error) []*jsonschema.ValidationError {
	var validation *jsonschema.ValidationError
	if !errors.As(err, &validation) {
		return nil
	}
	var result []*jsonschema.ValidationError
	var walk func(*jsonschema.ValidationError)
	walk = func(item *jsonschema.ValidationError) {
		if len(item.Causes) == 0 {
			result = append(result, item)
		}
		for _, cause := range item.Causes {
			walk(cause)
		}
	}
	walk(validation)
	return result
}

// hasEnumErrorAt reports whether an enum or const fails at pointer.
func hasEnumErrorAt(schema *jsonschema.Schema, document any, pointer string) bool {
	location := pointerTokens(pointer)
	for _, item := range leafErrors(schema.Validate(document)) {
		switch item.ErrorKind.(type) {
		case *kind.Enum, *kind.Const:
			if slicesEqual(item.InstanceLocation, location) {
				return true
			}
		}
	}
	return false
}

// forbidsUnknownMember reports whether the object at pointer rejects the
// added mutationUnknownMember as an additional or false-schema member, rather
// than as a value of a map it allows.
func forbidsUnknownMember(schema *jsonschema.Schema, document any, pointer string) bool {
	location := pointerTokens(pointer)
	member := append(append([]string{}, location...), mutationUnknownMember)
	for _, item := range leafErrors(schema.Validate(document)) {
		switch typed := item.ErrorKind.(type) {
		case *kind.AdditionalProperties:
			if slicesEqual(item.InstanceLocation, location) && slicesEqual(typed.Properties, []string{mutationUnknownMember}) {
				return true
			}
		case *kind.FalseSchema:
			if slicesEqual(item.InstanceLocation, member) {
				return true
			}
		}
	}
	return false
}

// maxLengthsAt returns the distinct maxLength bounds failing at pointer.
func maxLengthsAt(schema *jsonschema.Schema, document any, pointer string) []int {
	location := pointerTokens(pointer)
	bounds := map[int]bool{}
	for _, item := range leafErrors(schema.Validate(document)) {
		if bound, ok := item.ErrorKind.(*kind.MaxLength); ok && slicesEqual(item.InstanceLocation, location) {
			bounds[bound.Want] = true
		}
	}
	result := make([]int, 0, len(bounds))
	for bound := range bounds {
		result = append(result, bound)
	}
	sort.Ints(result)
	return result
}

func slicesEqual(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func verdict(accepted bool) string {
	if accepted {
		return "accept"
	}
	return "reject"
}

func normalizeSentinels(values map[string]any) map[string]string {
	result := map[string]string{}
	for key, value := range values {
		encoded, _ := json.Marshal(value)
		result[key] = string(encoded)
	}
	return result
}

// applyMutation returns raw with one mutation applied, as compact JSON. The
// Python Runtime builds the same documents from the committed cases.
func applyMutation(t *testing.T, raw []byte, mutation, pointer string, bound int) []byte {
	t.Helper()
	switch mutation {
	case "enum":
		document := decodeNumbers(t, raw)
		sentinel := mutationEnumSentinels["string"]
		if _, number := valueAt(t, document, pointer).(json.Number); number {
			sentinel = json.Number(strconv.Itoa(mutationEnumSentinels["number"].(int)))
		}
		return encodeMutation(t, setValue(t, document, pointer, sentinel), noDuplicate)
	case "maxLength":
		return withValue(t, raw, pointer, strings.Repeat("a", bound+1))
	case "maxBytes":
		return withValue(t, raw, pointer, strings.Repeat("a", bound-1)+"é")
	case "unknownMember":
		document := decodeNumbers(t, raw)
		valueAt(t, document, pointer).(map[string]any)[mutationUnknownMember] = true
		return encodeMutation(t, document, noDuplicate)
	case "duplicateKey":
		return encodeMutation(t, decodeNumbers(t, raw), pointer)
	}
	t.Fatalf("unknown mutation %q", mutation)
	return nil
}

func withValue(t *testing.T, raw []byte, pointer string, value any) []byte {
	t.Helper()
	return encodeMutation(t, setValue(t, decodeNumbers(t, raw), pointer, value), noDuplicate)
}

func valueAt(t *testing.T, document any, pointer string) any {
	t.Helper()
	target := document
	for _, token := range pointerTokens(pointer) {
		switch typed := target.(type) {
		case map[string]any:
			target = typed[token]
		case []any:
			position, err := strconv.Atoi(token)
			if err != nil {
				t.Fatalf("pointer %s: %v", pointer, err)
			}
			target = typed[position]
		default:
			t.Fatalf("pointer %s does not name a value", pointer)
		}
	}
	return target
}

func setValue(t *testing.T, document any, pointer string, value any) any {
	t.Helper()
	tokens := pointerTokens(pointer)
	if len(tokens) == 0 {
		return value
	}
	parent := valueAt(t, document, "/"+strings.Join(escapeTokens(tokens[:len(tokens)-1]), "/"))
	if len(tokens) == 1 {
		parent = document
	}
	last := tokens[len(tokens)-1]
	switch typed := parent.(type) {
	case map[string]any:
		typed[last] = value
	case []any:
		position, err := strconv.Atoi(last)
		if err != nil {
			t.Fatalf("pointer %s: %v", pointer, err)
		}
		typed[position] = value
	default:
		t.Fatalf("pointer %s does not name a member", pointer)
	}
	return document
}

func escapeTokens(tokens []string) []string {
	result := make([]string, len(tokens))
	for index, token := range tokens {
		result[index] = escapePointer(token)
	}
	return result
}

// noDuplicate is no JSON Pointer, so encodeMutation duplicates no member.
const noDuplicate = "-"

// encodeMutation writes document as compact JSON with object members in key
// order; the object at the JSON Pointer duplicate writes its first member
// twice.
func encodeMutation(t *testing.T, document any, duplicate string) []byte {
	t.Helper()
	var buffer bytes.Buffer
	var write func(value any, pointer string)
	write = func(value any, pointer string) {
		switch typed := value.(type) {
		case map[string]any:
			keys := make([]string, 0, len(typed))
			for key := range typed {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			if pointer == duplicate && len(keys) > 0 {
				keys = append([]string{keys[0]}, keys...)
			}
			buffer.WriteByte('{')
			for index, key := range keys {
				if index > 0 {
					buffer.WriteByte(',')
				}
				encoded, err := json.Marshal(key)
				if err != nil {
					t.Fatal(err)
				}
				buffer.Write(encoded)
				buffer.WriteByte(':')
				write(typed[key], pointer+"/"+escapePointer(key))
			}
			buffer.WriteByte('}')
		case []any:
			buffer.WriteByte('[')
			for index, item := range typed {
				if index > 0 {
					buffer.WriteByte(',')
				}
				write(item, pointer+"/"+strconv.Itoa(index))
			}
			buffer.WriteByte(']')
		default:
			encoder := json.NewEncoder(&buffer)
			encoder.SetEscapeHTML(false)
			if err := encoder.Encode(typed); err != nil {
				t.Fatal(err)
			}
			buffer.Truncate(buffer.Len() - 1)
		}
	}
	write(document, "")
	return buffer.Bytes()
}
