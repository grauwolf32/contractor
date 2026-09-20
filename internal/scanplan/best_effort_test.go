package scanplan_test

import (
	"bytes"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

func bestEffortDocument(paths map[string]any) map[string]any {
	doc := document(paths)
	doc["openapi"] = "3.1.0"
	return doc
}

func requireBestEffortRequest(t *testing.T, view preparedView, query, body string) {
	t.Helper()
	if len(view.Requests) != 1 || view.Coverage.Prepared != 1 || view.Coverage.Skipped != 0 {
		t.Fatalf("concrete data did not produce one request: %+v", view)
	}
	parsed, err := url.Parse(view.Requests[0].Request.URL)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Query().Get("q") != query || view.Requests[0].Request.Body != body {
		t.Fatalf("concrete request data changed: %+v", view.Requests[0].Request)
	}
}

func TestBestEffort31ConcreteValuesIgnoreUnusedSchemaProblems(t *testing.T) {
	guard := &forbiddenPreparationTransport{}
	previous := http.DefaultTransport
	http.DefaultTransport = guard
	t.Cleanup(func() { http.DefaultTransport = previous })
	for name, schema := range map[string]any{
		"missing":            nil,
		"boolean true":       true,
		"boolean false":      false,
		"union":              map[string]any{"type": []any{"integer", "null"}},
		"invalid type":       map[string]any{"type": "not-a-json-schema-type"},
		"missing reference":  map[string]any{"$ref": "#/components/schemas/Missing"},
		"cyclic reference":   map[string]any{"$ref": "#/components/schemas/Cycle"},
		"external reference": map[string]any{"$ref": "https://external.example.invalid/not-fetched"},
		"file reference":     map[string]any{"$ref": "file:///not-fetched.json"},
		"advanced keywords":  map[string]any{"$dynamicRef": "#unknown", "unevaluatedProperties": false, "if": false, "then": false, "vendorKeyword": map[string]any{"ignored": true}},
	} {
		t.Run(name, func(t *testing.T) {
			for _, explicitBinding := range []bool{false, true} {
				param := map[string]any{"name": "q", "in": "query", "required": true}
				media := map[string]any{}
				if schema != nil {
					param["schema"] = schema
					media["schema"] = schema
				}
				options := scanplan.Options{}
				if explicitBinding {
					options.Operations = map[string]scanplan.OperationInput{"#/paths/~1x/post": {
						Parameters: map[string]any{"query:q": "actual value"},
						Body:       &scanplan.BodyInput{MediaType: "application/json", Value: map[string]any{"actual": 17}},
					}}
				} else {
					param["example"] = "actual value"
					media["example"] = map[string]any{"actual": 17}
				}
				doc := bestEffortDocument(map[string]any{"/x": map[string]any{"post": map[string]any{
					"parameters":  []any{param},
					"requestBody": map[string]any{"required": true, "content": map[string]any{"application/json": media}},
				}}})
				doc["components"] = map[string]any{"schemas": map[string]any{"Cycle": map[string]any{"$ref": "#/components/schemas/Cycle"}}}
				view := mustDocument(t, doc, options)
				requireBestEffortRequest(t, view, "actual value", `{"actual":17}`)
			}
		})
	}
	if calls := guard.calls.Load(); calls != 0 {
		t.Fatalf("unused schema references caused %d network calls", calls)
	}
}

func TestBestEffort31ExtractsConcreteSchemaHints(t *testing.T) {
	for name, schema := range map[string]any{
		"example":  map[string]any{"example": "from-hint"},
		"examples": map[string]any{"examples": []any{"from-hint", "later"}},
		"default":  map[string]any{"default": "from-hint"},
		"const":    map[string]any{"const": "from-hint"},
		"enum":     map[string]any{"enum": []any{"from-hint", "later"}},
	} {
		t.Run(name, func(t *testing.T) {
			doc := bestEffortDocument(map[string]any{"/x": map[string]any{"post": map[string]any{
				"parameters":  []any{map[string]any{"name": "q", "in": "query", "required": true, "schema": schema}},
				"requestBody": map[string]any{"required": true, "content": map[string]any{"application/json": map[string]any{"schema": schema}}},
			}}})
			view := mustDocument(t, doc, scanplan.Options{})
			requireBestEffortRequest(t, view, "from-hint", `"from-hint"`)
		})
	}
}

func TestBestEffortExplicitDataPrecedence(t *testing.T) {
	param := map[string]any{
		"name": "q", "in": "query", "required": true,
		"example":  "direct",
		"examples": map[string]any{"z": map[string]any{"value": "later"}, "a": map[string]any{"value": "named"}},
		"schema":   map[string]any{"example": "schema"},
	}
	media := map[string]any{
		"example":  "direct",
		"examples": map[string]any{"z": map[string]any{"value": "later"}, "a": map[string]any{"value": "named"}},
		"schema":   map[string]any{"example": "schema"},
	}
	doc := bestEffortDocument(map[string]any{"/x": map[string]any{"post": map[string]any{
		"parameters":  []any{param},
		"requestBody": map[string]any{"required": true, "content": map[string]any{"application/json": media}},
	}}})
	options := scanplan.Options{Operations: map[string]scanplan.OperationInput{"#/paths/~1x/post": {
		Parameters: map[string]any{"query:q": "bound"},
		Body:       &scanplan.BodyInput{MediaType: "application/json", Value: "bound"},
	}}}
	requireBestEffortRequest(t, mustDocument(t, doc, options), "bound", `"bound"`)
	requireBestEffortRequest(t, mustDocument(t, doc, scanplan.Options{}), "direct", `"direct"`)
	delete(param, "example")
	delete(media, "example")
	requireBestEffortRequest(t, mustDocument(t, doc, scanplan.Options{}), "named", `"named"`)
	delete(param, "examples")
	delete(media, "examples")
	requireBestEffortRequest(t, mustDocument(t, doc, scanplan.Options{}), "schema", `"schema"`)
}

func TestBestEffortAssemblesBodyFromConcretePropertyData(t *testing.T) {
	schema := map[string]any{
		"type": []any{"object", "null"}, "required": []any{"name", "count"},
		"properties": map[string]any{
			"name":     map[string]any{"example": "actual-name"},
			"count":    map[string]any{"default": 0},
			"optional": map[string]any{"type": "string"},
		},
	}
	doc := bestEffortDocument(map[string]any{"/x": map[string]any{"post": map[string]any{
		"requestBody": map[string]any{"required": true, "content": map[string]any{"application/json": map[string]any{"schema": schema}}},
	}}})
	view := mustDocument(t, doc, scanplan.Options{})
	requireBestEffortRequest(t, view, "", `{"count":0,"name":"actual-name"}`)
}

func TestBestEffortMissingDataDoesNotDiscardOtherOperations(t *testing.T) {
	missing := map[string]any{"type": "string"}
	doc := bestEffortDocument(map[string]any{
		"/optional": map[string]any{"post": map[string]any{
			"parameters":  []any{map[string]any{"name": "q", "in": "query", "schema": missing}},
			"requestBody": map[string]any{"content": map[string]any{"text/plain": map[string]any{"schema": missing}}},
		}},
		"/required": map[string]any{"get": map[string]any{
			"parameters": []any{map[string]any{"name": "q", "in": "query", "required": true, "schema": missing}},
		}},
		"/ready": map[string]any{"get": map[string]any{
			"parameters": []any{map[string]any{"name": "q", "in": "query", "required": true, "example": "actual"}},
		}},
	})
	view := mustDocument(t, doc, scanplan.Options{})
	if len(view.Requests) != 2 || view.Coverage.Operations != 3 || view.Coverage.Prepared != 2 || view.Coverage.Skipped != 1 || view.Coverage.Complete {
		t.Fatalf("missing data discarded useful operations or hid gaps: %+v", view)
	}
	foundOptionalGap := false
	for _, gap := range view.Gaps {
		foundOptionalGap = foundOptionalGap || strings.HasPrefix(gap.Pointer, "#/paths/~1optional/post")
	}
	if !foundOptionalGap {
		t.Fatal("omitted optional inputs were not reported")
	}
	for _, entry := range view.Requests {
		parsed, err := url.Parse(entry.Request.URL)
		if err != nil {
			t.Fatal(err)
		}
		switch parsed.Path {
		case "/base/optional":
			if parsed.RawQuery != "" || entry.Request.Body != "" {
				t.Fatal("invented optional input")
			}
		case "/base/ready":
			if parsed.Query().Get("q") != "actual" {
				t.Fatal("changed available input")
			}
		default:
			t.Fatal("prepared operation lacking required input")
		}
	}
}

func TestBestEffortBrokenNamedExamplesFallThroughWithoutFetching(t *testing.T) {
	guard := &forbiddenPreparationTransport{}
	previous := http.DefaultTransport
	http.DefaultTransport = guard
	t.Cleanup(func() { http.DefaultTransport = previous })
	examples := map[string]any{
		"a-external": map[string]any{"externalValue": "https://external.example.invalid/secret-canary"},
		"b-missing":  map[string]any{"$ref": "#/components/examples/Missing"},
		"c-concrete": map[string]any{"value": "available"},
	}
	doc := bestEffortDocument(map[string]any{"/x": map[string]any{"post": map[string]any{
		"parameters":  []any{map[string]any{"name": "q", "in": "query", "required": true, "examples": examples}},
		"requestBody": map[string]any{"required": true, "content": map[string]any{"text/plain": map[string]any{"examples": examples}}},
	}}})
	view := mustDocument(t, doc, scanplan.Options{})
	requireBestEffortRequest(t, view, "available", "available")
	if guard.calls.Load() != 0 {
		t.Fatal("preparation fetched an external example")
	}
	if strings.Contains(string(encoded(t, view.Gaps)), "secret-canary") {
		t.Fatal("gap exposed example URL")
	}
}

func TestBestEffortUnknownOperationMetadataPreservesRequest(t *testing.T) {
	doc := bestEffortDocument(map[string]any{"/x": map[string]any{"get": map[string]any{
		"customMetadata": map[string]any{"meaning": "annotation"},
		"parameters":     []any{map[string]any{"name": "q", "in": "query", "example": "actual"}},
	}}})
	view := mustDocument(t, doc, scanplan.Options{})
	requireBestEffortRequest(t, view, "actual", "")
	if len(view.Gaps) == 0 || view.Coverage.Complete {
		t.Fatal("unknown metadata was not reported")
	}
}

func TestBestEffortHintPreparationIsDeterministicAndBounded(t *testing.T) {
	schema := map[string]any{"properties": map[string]any{
		"z": map[string]any{"enum": []any{false, true}},
		"a": map[string]any{"examples": []any{"actual", "other"}},
	}}
	doc := bestEffortDocument(map[string]any{"/x": map[string]any{"post": map[string]any{
		"requestBody": map[string]any{"required": true, "content": map[string]any{"application/json": map[string]any{"schema": schema}}},
	}}})
	data := encoded(t, doc)
	view, baseline := mustPrepare(t, data, "application/json", sourceRef(), scanplan.Options{})
	requireBestEffortRequest(t, view, "", `{"a":"actual","z":false}`)
	for i := 0; i < 12; i++ {
		_, current := mustPrepare(t, data, "application/json", sourceRef(), scanplan.Options{})
		if !bytes.Equal(baseline, current) {
			t.Fatal("hint extraction changed canonical output")
		}
	}
	doc["paths"].(map[string]any)["/x"].(map[string]any)["post"].(map[string]any)["requestBody"] = map[string]any{
		"required": true, "content": map[string]any{"text/plain": map[string]any{"schema": map[string]any{"default": strings.Repeat("x", contracts.MaxHTTPRequestBodyBytes+1)}}},
	}
	view = mustDocument(t, doc, scanplan.Options{})
	requireSkipped(t, view)
}

func TestBestEffortExplicitBodySurvivesBrokenDefinition(t *testing.T) {
	for name, definition := range map[string]any{
		"absent":            nil,
		"malformed":         false,
		"missing reference": map[string]any{"$ref": "#/components/requestBodies/Missing"},
		"undeclared media":  map[string]any{"content": map[string]any{"text/plain": map[string]any{}}},
	} {
		t.Run(name, func(t *testing.T) {
			op := map[string]any{}
			if definition != nil {
				op["requestBody"] = definition
			}
			doc := bestEffortDocument(map[string]any{"/x": map[string]any{"post": op}})
			options := scanplan.Options{Operations: map[string]scanplan.OperationInput{"#/paths/~1x/post": {
				Body: &scanplan.BodyInput{MediaType: "application/json", Value: map[string]any{"supplied": true}},
			}}}
			view := mustDocument(t, doc, options)
			requireBestEffortRequest(t, view, "", `{"supplied":true}`)
			if len(view.Gaps) == 0 {
				t.Fatal("body definition problem was not reported")
			}
			if headerValue(t, view, "content-type") != "application/json" {
				t.Fatal("supplied body media type changed")
			}
		})
	}
}

func TestBestEffortBodySelectionTriesLaterConcreteMedia(t *testing.T) {
	for name, first := range map[string]any{
		"no data":              map[string]any{"schema": map[string]any{"type": "object"}},
		"invalid media object": false,
		"unresolved hint":      map[string]any{"schema": map[string]any{"$ref": "#/components/schemas/Missing"}},
		"oversized data":       map[string]any{"example": strings.Repeat("x", contracts.MaxHTTPRequestBodyBytes+1)},
	} {
		t.Run(name, func(t *testing.T) {
			doc := bestEffortDocument(map[string]any{"/x": map[string]any{"post": map[string]any{
				"requestBody": map[string]any{"required": true, "content": map[string]any{
					"application/json": first,
					"text/plain":       map[string]any{"example": "concrete text"},
				}},
			}}})
			view := mustDocument(t, doc, scanplan.Options{})
			requireBestEffortRequest(t, view, "", "concrete text")
			if headerValue(t, view, "content-type") != "text/plain" {
				t.Fatal("selected unavailable first media type")
			}
		})
	}
}

func TestBestEffortPathNeedsDataWithoutRequiredFlag(t *testing.T) {
	for _, declaredRequired := range []bool{false, true} {
		param := map[string]any{"name": "id", "in": "path", "example": "actual/id"}
		if declaredRequired {
			param["required"] = false
		}
		doc := bestEffortDocument(map[string]any{"/x/{id}": map[string]any{"get": map[string]any{"parameters": []any{param}}}})
		view := mustDocument(t, doc, scanplan.Options{})
		requireBestEffortRequest(t, view, "", "")
		if !strings.HasSuffix(view.Requests[0].Request.URL, "/x/actual%2Fid") {
			t.Fatal("path example was not substituted and encoded")
		}
		delete(param, "example")
		view = mustDocument(t, doc, scanplan.Options{})
		requireSkipped(t, view)
	}
}

func TestBestEffortOptionalUnsupportedSerializationIsOmitted(t *testing.T) {
	doc := bestEffortDocument(map[string]any{"/x": map[string]any{"get": map[string]any{
		"parameters": []any{map[string]any{"name": "q", "in": "query", "style": "deepObject", "example": map[string]any{"known": "value"}}},
	}}})
	view := mustDocument(t, doc, scanplan.Options{})
	requireBestEffortRequest(t, view, "", "")
	if len(view.Gaps) == 0 || view.Coverage.Complete {
		t.Fatal("unsupported omitted input was not reported")
	}
	view = mustDocument(t, doc, scanplan.Options{Operations: map[string]scanplan.OperationInput{"#/paths/~1x/get": {
		Parameters: map[string]any{"query:q": map[string]any{"bound": "value"}},
	}}})
	requireSkipped(t, view)
}

func TestBestEffortSchemaHintsDoNotCreateCredentials(t *testing.T) {
	doc := bestEffortDocument(map[string]any{"/x": map[string]any{"get": map[string]any{
		"security": []any{map[string]any{"key": []any{}}},
	}}})
	doc["components"] = map[string]any{"securitySchemes": map[string]any{"key": map[string]any{
		"type": "apiKey", "in": "header", "name": "x-api-key", "example": "not-a-credential", "default": "not-a-credential",
	}}}
	view := mustDocument(t, doc, scanplan.Options{})
	requireSkipped(t, view)
}

func TestBestEffortOptionalInvalidHTTPValuesAreOmitted(t *testing.T) {
	for _, location := range []string{"header", "cookie"} {
		t.Run(location, func(t *testing.T) {
			param := map[string]any{"name": "x-input", "in": location, "example": "invalid\r\nvalue"}
			doc := bestEffortDocument(map[string]any{"/x": map[string]any{"get": map[string]any{"parameters": []any{param}}}})
			view := mustDocument(t, doc, scanplan.Options{})
			requireBestEffortRequest(t, view, "", "")
			if len(view.Gaps) == 0 || len(view.Requests[0].Request.Headers) != 0 {
				t.Fatal("invalid optional HTTP value was not omitted with a gap")
			}
			options := scanplan.Options{Operations: map[string]scanplan.OperationInput{"#/paths/~1x/get": {
				Parameters: map[string]any{location + ":x-input": "invalid\r\nvalue"},
			}}}
			view = mustDocument(t, doc, options)
			requireSkipped(t, view)
		})
	}
}

func TestBestEffortSuppliedServerVariableNeedsNoDefinition(t *testing.T) {
	doc := bestEffortDocument(map[string]any{"/x": map[string]any{"get": map[string]any{}}})
	doc["servers"] = []any{map[string]any{"url": "https://{host}/api"}}
	view := mustDocument(t, doc, scanplan.Options{ServerVariables: map[string]string{"host": "bound.example.test"}})
	requireBestEffortRequest(t, view, "", "")
	if view.Requests[0].Request.URL != "https://bound.example.test/api/x" {
		t.Fatal("concrete server variable was not bound")
	}
}
