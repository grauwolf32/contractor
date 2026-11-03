package scanplan_test

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

// The fixtures below drive the Audit scan inventory, the batch caller of
// preparation, through auditdomain.BuildOpenAPIScanInventory.

// reuseDocument mixes a runnable request, a POST with an authenticated JSON
// body, an operation without concrete values, two selected operations whose
// reference work is legal only per operation (3000 resolutions each, 6000
// together, LIM-SCN-43 allows 4096) and an unselected operation whose
// reference work alone exceeds that limit.
func reuseDocument() map[string]any {
	wide, heavy := map[string]any{}, map[string]any{}
	for i := range 3000 {
		wide[fmt.Sprintf("p%04d", i)] = map[string]any{"$ref": "#/components/schemas/Leaf"}
	}
	for i := range 5000 {
		heavy[fmt.Sprintf("h%04d", i)] = map[string]any{"$ref": "#/components/schemas/Leaf"}
	}
	query := map[string]any{"in": "query", "name": "q", "schema": map[string]any{"type": "string"}, "example": "abc"}
	jsonBody := func(schema string) map[string]any {
		return map[string]any{"required": true, "content": map[string]any{
			"application/json": map[string]any{"schema": map[string]any{"$ref": "#/components/schemas/" + schema}},
		}}
	}
	return map[string]any{
		"openapi": "3.0.3",
		"info":    map[string]any{"title": "Reuse", "version": "1"},
		"servers": []any{map[string]any{"url": "https://unselected.test"}},
		"paths": map[string]any{
			"/pets/{id}": map[string]any{
				"parameters": []any{map[string]any{"$ref": "#/components/parameters/ID"}},
				"get":        map[string]any{"parameters": []any{query}},
				"post": map[string]any{
					"security":    []any{map[string]any{"bearer": []any{}}},
					"parameters":  []any{map[string]any{"in": "query", "name": "q", "example": "post"}},
					"requestBody": jsonBody("Pet"),
				},
			},
			"/missing/{id}": map[string]any{"get": map[string]any{"parameters": []any{
				map[string]any{"in": "path", "name": "id", "required": true, "schema": map[string]any{"type": "integer"}},
				map[string]any{"in": "query", "name": "q", "example": "m"},
			}}},
			"/wide/a": map[string]any{"post": map[string]any{"parameters": []any{query}, "requestBody": jsonBody("Wide")}},
			"/wide/b": map[string]any{"post": map[string]any{"parameters": []any{query}, "requestBody": jsonBody("Wide")}},
			"/heavy":  map[string]any{"post": map[string]any{"parameters": []any{query}, "requestBody": jsonBody("Heavy")}},
		},
		"components": map[string]any{
			"parameters": map[string]any{"ID": map[string]any{"in": "path", "name": "id", "required": true, "schema": map[string]any{"type": "integer"}}},
			"schemas": map[string]any{
				"Leaf":  map[string]any{"type": "string", "example": "x"},
				"Pet":   map[string]any{"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string", "example": "Milo"}}, "required": []any{"name"}},
				"Wide":  map[string]any{"type": "object", "properties": wide},
				"Heavy": map[string]any{"type": "object", "properties": heavy},
			},
			"securitySchemes": map[string]any{"bearer": map[string]any{"type": "http", "scheme": "bearer"}},
		},
	}
}

const reuseSQLMapSettings = `{"schema":"contractor.audit.openapi-scan-settings.v1","scanner":"sqlmap","server":"https://target.test/api",
 "authentication":{"bearer":"golden-credential"},"testParameters":["q"],
 "operations":{
  "#/paths/~1pets~1{id}/get":{"parameters":{"path:id":7}},
  "#/paths/~1pets~1{id}/post":{"parameters":{"path:id":8}},
  "#/paths/~1missing~1{id}/get":{},
  "#/paths/~1wide~1a/post":{},
  "#/paths/~1wide~1b/post":{}
 }}`

const reuseNucleiSettings = `{"schema":"contractor.audit.openapi-scan-settings.v1","scanner":"nuclei","server":"https://target.test/api",
 "operations":{
  "#/paths/~1pets~1{id}/get":{"parameters":{"path:id":7}},
  "#/paths/~1pets~1{id}/post":{"parameters":{"path:id":8}},
  "#/paths/~1missing~1{id}/get":{},
  "#/paths/~1wide~1a/post":{}
 }}`

// sourceFormats maps a fixture format to the media type the inventory
// receives. The YAML form is never valid JSON, so it takes the YAML parser.
var sourceFormats = map[string]string{"json": "application/json", "yaml": "application/yaml"}

func encodeSource(t testing.TB, document map[string]any, format string) []byte {
	t.Helper()
	if format == "yaml" {
		var out strings.Builder
		writeBlockYAML(t, &out, document, "")
		data := []byte(out.String())
		if json.Valid(data) {
			t.Fatal("YAML fixture is also JSON")
		}
		return data
	}
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// writeBlockYAML writes block-style YAML with sorted keys and double-quoted
// strings, so the golden fixture bytes never depend on a YAML library's
// formatting. A nested collection starts on the line after its key or dash.
func writeBlockYAML(t testing.TB, out *strings.Builder, value any, indent string) {
	t.Helper()
	nested := func(value any) {
		switch typed := value.(type) {
		case map[string]any:
			if len(typed) == 0 {
				out.WriteString(" {}\n")
				return
			}
		case []any:
			if len(typed) == 0 {
				out.WriteString(" []\n")
				return
			}
		default:
			out.WriteString(" ")
			writeBlockYAML(t, out, value, indent)
			return
		}
		out.WriteString("\n")
		writeBlockYAML(t, out, value, indent+"  ")
	}
	switch typed := value.(type) {
	case map[string]any:
		for _, key := range slices.Sorted(maps.Keys(typed)) {
			out.WriteString(indent + strconv.Quote(key) + ":")
			nested(typed[key])
		}
	case []any:
		for _, item := range typed {
			out.WriteString(indent + "-")
			nested(item)
		}
	case string:
		out.WriteString(strconv.Quote(typed) + "\n")
	case int:
		out.WriteString(strconv.Itoa(typed) + "\n")
	case bool:
		out.WriteString(strconv.FormatBool(typed) + "\n")
	default:
		t.Fatalf("unsupported YAML fixture value %T", value)
	}
}

func inventorySourceRef(revision string) contracts.ArtifactRef {
	return contracts.ArtifactRef{Namespace: "inputs", Name: "openapi", Revision: &revision}
}

func prepareScanBatch(t testing.TB, source []byte, mediaType string, ref contracts.ArtifactRef, settingsText string) ([]scanplan.PreparedAuditOperation, error) {
	t.Helper()
	settings, err := scanplan.DecodeAuditScanSettings([]byte(settingsText))
	if err != nil {
		return nil, err
	}
	result := []scanplan.PreparedAuditOperation{}
	for item, err := range settings.PrepareOperations(source, mediaType, ref) {
		if err != nil {
			return nil, err
		}
		result = append(result, item)
	}
	return result, nil
}

// largeOperations selects the maximum number of operations one settings
// document may name (LIM-SCN: MaxOperations).
const largeOperations = scanplan.MaxOperations

// largeSource is an accepted 1000-operation source whose stored bytes lie
// within 64 KiB of the 2 MiB source limit (LIM-SCN-39). Odd operations use a
// shared parameter reference.
func largeSource(t testing.TB, format string) []byte {
	t.Helper()
	build := func(padding int) []byte {
		description := strings.Repeat("a", padding)
		paths := make(map[string]any, largeOperations)
		for i := range largeOperations {
			var query any = map[string]any{"in": "query", "name": "q", "schema": map[string]any{"type": "string"}, "example": fmt.Sprintf("v%04d", i)}
			if i%2 == 1 {
				query = map[string]any{"$ref": "#/components/parameters/Query"}
			}
			paths[fmt.Sprintf("/r%04d/{id}", i)] = map[string]any{"get": map[string]any{
				"description": description,
				"parameters": []any{
					map[string]any{"in": "path", "name": "id", "required": true, "schema": map[string]any{"type": "integer"}},
					query,
				},
			}}
		}
		return encodeSource(t, map[string]any{
			"openapi": "3.0.3", "info": map[string]any{"title": "Large", "version": "1"},
			"servers": []any{map[string]any{"url": "https://unselected.test"}},
			"paths":   paths,
			"components": map[string]any{"parameters": map[string]any{"Query": map[string]any{
				"in": "query", "name": "q", "schema": map[string]any{"type": "string"}, "example": "shared",
			}}},
		}, format)
	}
	target := scanplan.MaxSourceBytes - 32*1024
	data := build((target - len(build(0))) / largeOperations)
	if len(data) > scanplan.MaxSourceBytes || len(data) < scanplan.MaxSourceBytes-64*1024 {
		t.Fatalf("large %s source has %d bytes", format, len(data))
	}
	return data
}

// largeSettings selects every operation of largeSource. Every tenth one has
// no binding for its required path value and stays a non-runnable item.
func largeSettings(t testing.TB, scanner string) string {
	t.Helper()
	operations := make(map[string]any, largeOperations)
	for i := range largeOperations {
		input := map[string]any{}
		if i%10 != 0 {
			input["parameters"] = map[string]any{"path:id": i}
		}
		operations[fmt.Sprintf("#/paths/~1r%04d~1{id}/get", i)] = input
	}
	settings := map[string]any{
		"schema": scanplan.AuditScanSettingsSchema, "scanner": scanner,
		"server": "https://target.test/api", "operations": operations,
	}
	if scanner == "sqlmap" {
		settings["testParameters"] = []string{"q"}
	}
	data, err := json.Marshal(settings)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}
