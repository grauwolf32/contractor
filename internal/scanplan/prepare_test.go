package scanplan_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

// Assert the public artifact shape independently of the preparer's Go structs.
type preparedView struct {
	SchemaVersion int `json:"schemaVersion"`
	Source        struct {
		Artifact      contracts.ArtifactRef `json:"artifact"`
		ContentDigest string                `json:"contentDigest"`
	} `json:"source"`
	PreparationDigest string `json:"preparationDigest"`
	Requests          []struct {
		ID            string `json:"id"`
		ContentDigest string `json:"contentDigest"`
		Request       struct {
			Method  string `json:"method"`
			URL     string `json:"url"`
			Headers []struct {
				Name  string `json:"name"`
				Value string `json:"value"`
			} `json:"headers"`
			Body string `json:"body"`
		} `json:"request"`
		Origins []struct {
			Pointer string `json:"pointer"`
		} `json:"origins"`
	} `json:"requests"`
	Gaps []struct {
		Pointer string `json:"pointer"`
		Code    string `json:"code"`
	} `json:"gaps"`
	Coverage struct {
		Operations int  `json:"operations"`
		Prepared   int  `json:"prepared"`
		Skipped    int  `json:"skipped"`
		Complete   bool `json:"complete"`
	} `json:"coverage"`
}

func sourceRef() contracts.ArtifactRef {
	revision := "source-1"
	return contracts.ArtifactRef{Namespace: "inputs", Name: "openapi", Revision: &revision}
}

func encoded(t *testing.T, value any) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func digest(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func document(paths map[string]any) map[string]any {
	return map[string]any{
		"openapi": "3.0.3", "info": map[string]any{"title": "Fixture", "version": "1"},
		"servers": []any{map[string]any{"url": "https://api.example.test/base"}}, "paths": paths,
	}
}

func mustPrepare(t *testing.T, data []byte, media string, ref contracts.ArtifactRef, options scanplan.Options) (preparedView, []byte) {
	t.Helper()
	result, err := scanplan.Prepare(data, media, ref, options)
	if err != nil {
		t.Fatal(err)
	}
	wire, err := contracts.MarshalPrivateCanonical(result)
	if err != nil {
		t.Fatal(err)
	}
	var view preparedView
	decoder := json.NewDecoder(bytes.NewReader(wire))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&view); err != nil {
		t.Fatal(err)
	}
	if view.SchemaVersion != 1 || view.Requests == nil || view.Gaps == nil {
		t.Fatalf("invalid RequestSet envelope: %s", wire)
	}
	if view.Coverage.Operations != view.Coverage.Prepared+view.Coverage.Skipped {
		t.Fatalf("unaccounted coverage: %+v", view.Coverage)
	}
	for i, entry := range view.Requests {
		canonical, err := contracts.MarshalPrivateCanonical(entry.Request)
		if err != nil {
			t.Fatal(err)
		}
		want := digest(canonical)
		if entry.ContentDigest != want || entry.ID != "request-"+strings.TrimPrefix(want, "sha256:") {
			t.Fatalf("request identity does not bind its content: %s", entry.ID)
		}
		if i > 0 && view.Requests[i-1].ID >= entry.ID {
			t.Fatal("request IDs not sorted/unique")
		}
		for j, header := range entry.Request.Headers {
			if header.Name != strings.ToLower(header.Name) || (j > 0 && entry.Request.Headers[j-1].Name >= header.Name) {
				t.Fatal("headers not sorted, lowercase and unique")
			}
		}
		for j, origin := range entry.Origins {
			if j > 0 && entry.Origins[j-1].Pointer >= origin.Pointer {
				t.Fatal("origins not sorted/unique")
			}
		}
	}
	return view, wire
}

func mustDocument(t *testing.T, doc map[string]any, options scanplan.Options) preparedView {
	t.Helper()
	view, _ := mustPrepare(t, encoded(t, doc), "application/json", sourceRef(), options)
	return view
}

func headerValue(t *testing.T, view preparedView, name string) string {
	t.Helper()
	if len(view.Requests) != 1 {
		t.Fatalf("request count = %d", len(view.Requests))
	}
	for _, header := range view.Requests[0].Request.Headers {
		if header.Name == name {
			return header.Value
		}
	}
	return ""
}

func requireSkipped(t *testing.T, view preparedView) {
	t.Helper()
	if len(view.Requests) != 0 || view.Coverage.Operations != 1 || view.Coverage.Skipped != 1 || view.Coverage.Complete || len(view.Gaps) == 0 {
		t.Fatalf("unresolved operation was not explicitly skipped: %+v", view)
	}
}

func TestPrepareStableIdentityAndExactInputProvenance(t *testing.T) {
	first := []byte(`{"openapi":"3.0.3","servers":[{"url":"https://api.example.test"}],"paths":{"/b":{"get":{}},"/a":{"get":{}}}}`)
	reordered := []byte(`{"paths":{"/a":{"get":{}},"/b":{"get":{}}},"servers":[{"url":"https://api.example.test"}],"openapi":"3.0.3"}`)
	options := scanplan.Options{Authentication: map[string]contracts.SecretString{}}
	options.Authentication["z"] = contracts.NewSecretString("unused-z")
	options.Authentication["a"] = contracts.NewSecretString("unused-a")
	otherOptions := scanplan.Options{Authentication: map[string]contracts.SecretString{}}
	otherOptions.Authentication["a"] = contracts.NewSecretString("unused-a")
	otherOptions.Authentication["z"] = contracts.NewSecretString("unused-z")
	left, leftBytes := mustPrepare(t, first, "application/json", sourceRef(), options)
	right, rightBytes := mustPrepare(t, first, "application/json", sourceRef(), otherOptions)
	if !bytes.Equal(leftBytes, rightBytes) {
		t.Fatal("identical exact inputs or map insertion order changed canonical output")
	}
	if left.Source.ContentDigest != digest(first) || !reflect.DeepEqual(left.Source.Artifact, sourceRef()) {
		t.Fatal("source provenance is not exact")
	}
	if !left.Coverage.Complete || left.Coverage.Prepared != 2 {
		t.Fatalf("coverage: %+v", left.Coverage)
	}
	right, _ = mustPrepare(t, reordered, "application/json", sourceRef(), options)
	if !reflect.DeepEqual(left.Requests, right.Requests) {
		t.Fatal("JSON property order changed request identity")
	}
	if left.Source.ContentDigest == right.Source.ContentDigest || left.PreparationDigest == right.PreparationDigest {
		t.Fatal("exact source bytes are missing from provenance/digest")
	}
	ref := sourceRef()
	revision := "source-2"
	ref.Revision = &revision
	right, _ = mustPrepare(t, first, "application/json", ref, options)
	if !reflect.DeepEqual(left.Requests, right.Requests) || left.PreparationDigest == right.PreparationDigest {
		t.Fatal("source revision must affect preparation, not request identity")
	}
	otherOptions.Authentication["a"] = contracts.NewSecretString("changed-binding")
	right, _ = mustPrepare(t, first, "application/json", sourceRef(), otherOptions)
	if !reflect.DeepEqual(left.Requests, right.Requests) || left.PreparationDigest == right.PreparationDigest {
		t.Fatal("preparation digest omitted explicit options")
	}
}

func TestPrepareDeduplicatesWithoutDroppingOriginsOrLimitCoverage(t *testing.T) {
	doc := document(map[string]any{
		"/a":    map[string]any{"get": map[string]any{}},
		"/b":    map[string]any{"get": map[string]any{}},
		"/{id}": map[string]any{"get": map[string]any{"parameters": []any{map[string]any{"name": "id", "in": "path", "required": true, "schema": map[string]any{"type": "string"}, "example": "a"}}}},
	})
	view := mustDocument(t, doc, scanplan.Options{MaxRequests: 1})
	if len(view.Requests) != 1 || len(view.Requests[0].Origins) != 2 || view.Coverage.Operations != 3 || view.Coverage.Prepared != 2 || view.Coverage.Skipped != 1 || view.Coverage.Complete {
		t.Fatalf("deduplication or request bound lost origins: %+v", view)
	}
	if view.Requests[0].Origins[0].Pointer != "#/paths/~1a/get" || view.Requests[0].Origins[1].Pointer != "#/paths/~1{id}/get" {
		t.Fatal("wrong retained origins")
	}
	if len(view.Gaps) != 1 || view.Gaps[0].Pointer != "#/paths/~1b/get" {
		t.Fatalf("request limit is not accounted for: %+v", view.Gaps)
	}
}

func TestPrepareParameterReferencesEncodingOverridesAndNamedExamples(t *testing.T) {
	doc := document(map[string]any{
		"/items/{id}": map[string]any{
			"parameters": []any{map[string]any{"name": "q", "in": "query", "schema": map[string]any{"type": "string"}, "example": "path-level"}},
			"get": map[string]any{"parameters": []any{
				map[string]any{"$ref": "#/components/parameters/ID"},
				map[string]any{"name": "q", "in": "query", "schema": map[string]any{"type": "string"}, "example": "operation-level"},
				map[string]any{"name": "limit", "in": "query", "schema": map[string]any{"type": "integer"}, "examples": map[string]any{"z": map[string]any{"value": 9}, "a": map[string]any{"value": 2}}},
				map[string]any{"name": "X-Enabled", "in": "header", "schema": map[string]any{"type": "boolean", "example": true}},
				map[string]any{"name": "locale", "in": "cookie", "schema": map[string]any{"type": "string"}, "example": "en"},
			}},
		},
	})
	doc["components"] = map[string]any{"parameters": map[string]any{"ID": map[string]any{"name": "id", "in": "path", "required": true, "schema": map[string]any{"$ref": "#/components/schemas/Identifier"}, "example": "unused"}}, "schemas": map[string]any{"Identifier": map[string]any{"type": "string"}}}
	value := "a/b +雪"
	view := mustDocument(t, doc, scanplan.Options{Operations: map[string]scanplan.OperationInput{
		"#/paths/~1items~1{id}/get": {Parameters: map[string]any{"path:id": value, "query:q": value}},
	}})
	if len(view.Requests) != 1 || !view.Coverage.Complete {
		t.Fatalf("preparation incomplete: %+v", view)
	}
	request := view.Requests[0].Request
	parsed, err := url.Parse(request.URL)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.EscapedPath() != "/base/items/a%2Fb%20%2B%E9%9B%AA" || parsed.Query().Get("q") != value || parsed.Query().Get("limit") != "2" || len(parsed.Query()["q"]) != 1 {
		t.Fatalf("incorrect value encoding/precedence: %s", request.URL)
	}
	if headerValue(t, view, "x-enabled") != "true" || headerValue(t, view, "cookie") != "locale=en" {
		t.Fatal("header/cookie parameters were not serialized")
	}
}

func TestPrepareServerPrecedenceVariablesAndBasePath(t *testing.T) {
	for _, test := range []struct {
		name            string
		server          string
		operation, path bool
		want            string
	}{
		{"root", "", false, false, "https://api.example.test/base/x"},
		{"path", "", false, true, "https://path.example.test/root/x"},
		{"operation", "", true, true, "https://operation.example.test/root/x"},
		{"override", "https://override.example.test/raw/../base", true, true, "https://override.example.test/raw/../base/x"},
	} {
		t.Run(test.name, func(t *testing.T) {
			op := map[string]any{}
			path := map[string]any{"get": op}
			if test.path {
				path["servers"] = []any{map[string]any{"url": "https://path.example.test/root"}}
			}
			if test.operation {
				op["servers"] = []any{map[string]any{"url": "https://operation.example.test/root"}}
			}
			view := mustDocument(t, document(map[string]any{"/x": path}), scanplan.Options{Server: test.server})
			if len(view.Requests) != 1 || view.Requests[0].Request.URL != test.want {
				t.Fatalf("server precedence/path cleaning changed request: %+v", view)
			}
		})
	}
	doc := document(map[string]any{"/x": map[string]any{"get": map[string]any{}}})
	doc["servers"] = []any{map[string]any{"url": "https://{region}.example.test/{version}", "variables": map[string]any{
		"region": map[string]any{"default": "eu", "enum": []any{"eu", "us"}}, "version": map[string]any{"default": "v1"},
	}}}
	view := mustDocument(t, doc, scanplan.Options{ServerVariables: map[string]string{"region": "us"}})
	if len(view.Requests) != 1 || view.Requests[0].Request.URL != "https://us.example.test/v1/x" {
		t.Fatal("server variables were not bound")
	}
	view = mustDocument(t, doc, scanplan.Options{ServerVariables: map[string]string{"region": "other-region"}})
	if len(view.Requests) != 1 || view.Requests[0].Request.URL != "https://other-region.example.test/v1/x" {
		t.Fatal("server enum constraint rejected concrete data")
	}
}

func TestPrepareBodiesUseDeterministicExamplesAndExplicitInput(t *testing.T) {
	op := map[string]any{"requestBody": map[string]any{"required": true, "content": map[string]any{
		"application/json": map[string]any{"schema": map[string]any{"type": "object", "properties": map[string]any{"id": map[string]any{"type": "integer"}, "name": map[string]any{"type": "string"}}, "required": []any{"id"}}, "examples": map[string]any{"z": map[string]any{"value": map[string]any{"id": 9}}, "a": map[string]any{"$ref": "#/components/examples/Body"}}},
		"text/plain":       map[string]any{"schema": map[string]any{"type": "string"}, "example": "plain-body"},
	}}}
	doc := document(map[string]any{"/items": map[string]any{"post": op}})
	doc["components"] = map[string]any{"examples": map[string]any{"Body": map[string]any{"value": map[string]any{"name": "雪", "id": 2}}}}
	view := mustDocument(t, doc, scanplan.Options{})
	if len(view.Requests) != 1 || view.Requests[0].Request.Body != `{"id":2,"name":"雪"}` || headerValue(t, view, "content-type") != "application/json" {
		t.Fatalf("body example selection/canonicalization failed: %+v", view)
	}
	options := scanplan.Options{Operations: map[string]scanplan.OperationInput{"#/paths/~1items/post": {Body: &scanplan.BodyInput{MediaType: "text/plain", Value: "explicit-body\n"}}}}
	view = mustDocument(t, doc, options)
	if len(view.Requests) != 1 || view.Requests[0].Request.Body != "explicit-body\n" || headerValue(t, view, "content-type") != "text/plain" {
		t.Fatal("neutral body input was changed or not selected")
	}
	options.Operations["#/paths/~1items/post"] = scanplan.OperationInput{Body: &scanplan.BodyInput{MediaType: "application/json", Value: map[string]any{"name": "missing-required-id"}}}
	view = mustDocument(t, doc, options)
	if len(view.Requests) != 1 || view.Requests[0].Request.Body != `{"name":"missing-required-id"}` {
		t.Fatal("schema validation rejected explicit body data")
	}
}

func TestPrepareMissingInputsUnsupportedMethodsAndCallbacksAreCoverageGaps(t *testing.T) {
	parameter := func(required bool) any {
		return map[string]any{"name": "q", "in": "query", "required": required, "schema": map[string]any{"type": "string"}}
	}
	body := func(required bool) any {
		return map[string]any{"required": required, "content": map[string]any{"application/json": map[string]any{"schema": map[string]any{"type": "object"}}}}
	}
	doc := document(map[string]any{
		"/required-parameter": map[string]any{"get": map[string]any{"parameters": []any{parameter(true)}}},
		"/optional-parameter": map[string]any{"get": map[string]any{"parameters": []any{parameter(false)}}},
		"/required-body":      map[string]any{"post": map[string]any{"requestBody": body(true)}},
		"/optional-body":      map[string]any{"post": map[string]any{"requestBody": body(false)}},
		"/plain":              map[string]any{"get": map[string]any{}},
		"/trace":              map[string]any{"trace": map[string]any{}},
		"/callback":           map[string]any{"get": map[string]any{"callbacks": map[string]any{"event": map[string]any{"{$request.body#/url}": map[string]any{"post": map[string]any{}}}}}},
	})
	view := mustDocument(t, doc, scanplan.Options{})
	if view.Coverage.Operations != 7 || view.Coverage.Prepared != 4 || view.Coverage.Skipped != 3 || view.Coverage.Complete || len(view.Gaps) < 6 {
		t.Fatalf("missing/unsupported input was hidden: %+v", view)
	}
	for _, entry := range view.Requests {
		if strings.Contains(entry.Request.URL, "do-not-invent") || entry.Request.Body != "" {
			t.Fatal("missing data was invented")
		}
	}
}

func authenticationDocument() map[string]any {
	doc := document(map[string]any{"/x": map[string]any{"get": map[string]any{}}})
	doc["components"] = map[string]any{"securitySchemes": map[string]any{
		"bearer": map[string]any{"type": "http", "scheme": "bearer"},
		"key":    map[string]any{"type": "apiKey", "in": "query", "name": "api_key"},
		"cookie": map[string]any{"type": "apiKey", "in": "cookie", "name": "session"},
		"header": map[string]any{"type": "apiKey", "in": "header", "name": "X-API-Key"},
	}}
	return doc
}

func TestPrepareAuthenticationAlternativesAndOperationOverrides(t *testing.T) {
	doc := authenticationDocument()
	doc["security"] = []any{map[string]any{"bearer": []any{}, "key": []any{}}, map[string]any{"cookie": []any{}}}
	view := mustDocument(t, doc, scanplan.Options{Authentication: map[string]contracts.SecretString{"bearer": contracts.NewSecretString("bearer-canary"), "key": contracts.NewSecretString("key /+canary")}})
	if headerValue(t, view, "authorization") != "Bearer bearer-canary" {
		t.Fatal("AND bearer binding was dropped")
	}
	parsed, _ := url.Parse(view.Requests[0].Request.URL)
	if parsed.Query().Get("api_key") != "key /+canary" {
		t.Fatal("AND query key was dropped or incorrectly encoded")
	}
	view = mustDocument(t, doc, scanplan.Options{Authentication: map[string]contracts.SecretString{"bearer": contracts.NewSecretString("unused-canary"), "cookie": contracts.NewSecretString("cookie-canary")}})
	if headerValue(t, view, "authorization") != "" || headerValue(t, view, "cookie") != "session=cookie-canary" {
		t.Fatal("partially bound OR alternative leaked into selected one")
	}
	requireSkipped(t, mustDocument(t, doc, scanplan.Options{}))
	for _, security := range []any{[]any{}, []any{map[string]any{}}} {
		doc["paths"].(map[string]any)["/x"].(map[string]any)["get"].(map[string]any)["security"] = security
		view = mustDocument(t, doc, scanplan.Options{})
		if len(view.Requests) != 1 || !view.Coverage.Complete || headerValue(t, view, "authorization") != "" {
			t.Fatal("operation anonymous security did not override root")
		}
	}
}

func TestPrepareAuthenticationCollisionAndUnsupportedSchemesCannotBypassSecurity(t *testing.T) {
	for _, test := range []struct{ name, scheme, location, parameter string }{
		{"query", "key", "query", "api_key"}, {"header", "header", "header", "x-api-key"}, {"cookie", "cookie", "cookie", "session"},
	} {
		t.Run(test.name, func(t *testing.T) {
			doc := authenticationDocument()
			doc["security"] = []any{map[string]any{test.scheme: []any{}}}
			op := doc["paths"].(map[string]any)["/x"].(map[string]any)["get"].(map[string]any)
			op["parameters"] = []any{map[string]any{"name": test.parameter, "in": test.location, "schema": map[string]any{"type": "string"}, "example": "ordinary-canary"}}
			view := mustDocument(t, doc, scanplan.Options{Authentication: map[string]contracts.SecretString{test.scheme: contracts.NewSecretString("credential-canary")}})
			requireSkipped(t, view)
			if strings.Contains(string(encoded(t, view.Gaps)), "canary") {
				t.Fatal("authentication collision leaked credentials/examples")
			}
		})
	}
	doc := authenticationDocument()
	doc["components"].(map[string]any)["securitySchemes"].(map[string]any)["oauth"] = map[string]any{"type": "oauth2", "flows": map[string]any{}}
	doc["security"] = []any{map[string]any{"oauth": []any{}}}
	requireSkipped(t, mustDocument(t, doc, scanplan.Options{Authentication: map[string]contracts.SecretString{"oauth": contracts.NewSecretString("canary")}}))
	doc["security"] = []any{map[string]any{"oauth": []any{}}, map[string]any{"header": []any{}}}
	view := mustDocument(t, doc, scanplan.Options{Authentication: map[string]contracts.SecretString{"header": contracts.NewSecretString("header-canary")}})
	if headerValue(t, view, "x-api-key") != "header-canary" {
		t.Fatal("supported satisfiable security alternative was not selected")
	}
}

type forbiddenPreparationTransport struct{ calls atomic.Int32 }

func (transport *forbiddenPreparationTransport) RoundTrip(*http.Request) (*http.Response, error) {
	transport.calls.Add(1)
	return nil, errors.New("preparation must not perform network I/O")
}

func TestPrepareOnlyResolvesReferencesNeededForDataWithoutHTTP(t *testing.T) {
	guard := &forbiddenPreparationTransport{}
	previous := http.DefaultTransport
	http.DefaultTransport = guard
	t.Cleanup(func() { http.DefaultTransport = previous })
	for _, ref := range []string{"https://external.example.invalid/secret-canary", "local-secret-canary.yaml#/schema", "#/components/schemas/Missing", "#/components/schemas/Cycle"} {
		t.Run(ref, func(t *testing.T) {
			doc := document(map[string]any{"/x": map[string]any{"get": map[string]any{"parameters": []any{map[string]any{"name": "q", "in": "query", "required": true, "schema": map[string]any{"$ref": ref}, "example": "value-canary"}}}}})
			doc["components"] = map[string]any{"schemas": map[string]any{"Cycle": map[string]any{"$ref": "#/components/schemas/Cycle"}}}
			view := mustDocument(t, doc, scanplan.Options{})
			if len(view.Requests) != 1 || !strings.Contains(view.Requests[0].Request.URL, "q=value-canary") {
				t.Fatal("unused schema reference blocked concrete example")
			}
			parameter := doc["paths"].(map[string]any)["/x"].(map[string]any)["get"].(map[string]any)["parameters"].([]any)[0].(map[string]any)
			delete(parameter, "example")
			view = mustDocument(t, doc, scanplan.Options{})
			requireSkipped(t, view)
			if strings.Contains(string(encoded(t, view.Gaps)), "canary") {
				t.Fatal("reference diagnostic leaked supplied values")
			}
		})
	}
	doc := document(map[string]any{"/x": map[string]any{"get": map[string]any{"responses": map[string]any{"200": map[string]any{"content": map[string]any{"application/json": map[string]any{"schema": map[string]any{"$ref": "https://external.example.invalid/unused"}}}}}}}})
	view := mustDocument(t, doc, scanplan.Options{})
	if len(view.Requests) != 1 || !view.Coverage.Complete {
		t.Fatal("unused response schemas affected request preparation")
	}
	if guard.calls.Load() != 0 {
		t.Fatalf("preparation performed %d HTTP requests", guard.calls.Load())
	}
}

func TestPrepareDistinguishesSerializationLimitsFromSchemaConstraints(t *testing.T) {
	for name, operation := range map[string]any{
		"style":            map[string]any{"parameters": []any{map[string]any{"name": "q", "in": "query", "required": true, "style": "deepObject", "schema": map[string]any{"type": "object"}, "example": map[string]any{"x": 1}}}},
		"array":            map[string]any{"parameters": []any{map[string]any{"name": "q", "in": "query", "required": true, "schema": map[string]any{"type": "array", "items": map[string]any{"type": "string"}}, "example": []any{"x"}}}},
		"wrong-type":       map[string]any{"parameters": []any{map[string]any{"name": "q", "in": "query", "required": true, "schema": map[string]any{"type": "integer"}, "example": "secret-canary"}}},
		"external-example": map[string]any{"parameters": []any{map[string]any{"name": "q", "in": "query", "required": true, "schema": map[string]any{"type": "string"}, "examples": map[string]any{"x": map[string]any{"externalValue": "https://example.invalid/secret-canary"}}}}},
		"ref-sibling":      map[string]any{"parameters": []any{map[string]any{"$ref": "#/components/parameters/Q", "example": "secret-canary"}}},
		"multipart":        map[string]any{"requestBody": map[string]any{"required": true, "content": map[string]any{"multipart/form-data": map[string]any{"schema": map[string]any{"type": "object"}, "example": map[string]any{"file": "secret-canary"}}}}},
		"binary":           map[string]any{"requestBody": map[string]any{"required": true, "content": map[string]any{"text/plain": map[string]any{"schema": map[string]any{"type": "string", "format": "binary"}, "example": "secret-canary"}}}},
		"ambiguous-one-of": map[string]any{"requestBody": map[string]any{"required": true, "content": map[string]any{"application/json": map[string]any{"schema": map[string]any{"oneOf": []any{map[string]any{"type": "string"}, map[string]any{"type": "string"}}}, "example": "secret-canary"}}}},
	} {
		t.Run(name, func(t *testing.T) {
			doc := document(map[string]any{"/x": map[string]any{"post": operation}})
			doc["components"] = map[string]any{"parameters": map[string]any{"Q": map[string]any{"name": "q", "in": "query", "required": true, "schema": map[string]any{"type": "string"}, "example": "value"}}}
			view := mustDocument(t, doc, scanplan.Options{})
			switch name {
			case "wrong-type", "ref-sibling", "binary", "ambiguous-one-of":
				if len(view.Requests) != 1 || !view.Coverage.Complete {
					t.Fatal("schema constraint rejected serializable request data")
				}
			default:
				requireSkipped(t, view)
			}
			if strings.Contains(string(encoded(t, view.Gaps)), "canary") {
				t.Fatal("coverage gaps leaked supplied examples")
			}
		})
	}
}

func TestPrepareJSONAndYAMLProduceSameRequests(t *testing.T) {
	jsonDoc := document(map[string]any{"/x": map[string]any{"get": map[string]any{"parameters": []any{map[string]any{"name": "q", "in": "query", "schema": map[string]any{"type": "string"}, "example": "snow 雪"}}}}})
	yamlDoc := []byte("openapi: 3.0.3\ninfo: {title: Fixture, version: '1'}\nservers:\n  - url: https://api.example.test/base\npaths:\n  /x:\n    get:\n      parameters:\n        - name: q\n          in: query\n          schema: {type: string}\n          example: snow 雪\n")
	left := mustDocument(t, jsonDoc, scanplan.Options{})
	for _, media := range []string{"application/yaml", "application/x-yaml", "text/yaml"} {
		right, _ := mustPrepare(t, yamlDoc, media, sourceRef(), scanplan.Options{})
		if !reflect.DeepEqual(left.Requests, right.Requests) || !right.Coverage.Complete {
			t.Fatalf("JSON/YAML preparation differs for %s", media)
		}
		if right.Source.ContentDigest != digest(yamlDoc) {
			t.Fatal("source digest is not exact YAML bytes")
		}
	}
}

func TestPrepareLiteralReferenceDataAndComposedBodySchemaDoNotMutateInputs(t *testing.T) {
	doc := document(map[string]any{"/x": map[string]any{"post": map[string]any{"requestBody": map[string]any{
		"required": true, "content": map[string]any{"application/json": map[string]any{"schema": map[string]any{
			"oneOf": []any{map[string]any{"type": "object", "additionalProperties": true}, map[string]any{"type": "integer"}},
		}}},
	}}}})
	literal := map[string]any{"$ref": "https://example.invalid/private-reference-canary", "items": []any{"example-canary"}}
	options := scanplan.Options{Operations: map[string]scanplan.OperationInput{"#/paths/~1x/post": {Body: &scanplan.BodyInput{MediaType: "application/json", Value: literal}}}}
	beforeOptions := encoded(t, options)
	beforeDocument := encoded(t, doc)
	view := mustDocument(t, doc, options)
	if len(view.Requests) != 1 || !view.Coverage.Complete {
		t.Fatalf("supported composed body schema was not prepared: %+v", view.Coverage)
	}
	var body map[string]any
	if err := json.Unmarshal([]byte(view.Requests[0].Request.Body), &body); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(body, literal) {
		t.Fatal("literal example $ref was resolved or changed")
	}
	if !bytes.Equal(beforeOptions, encoded(t, options)) || !bytes.Equal(beforeDocument, encoded(t, doc)) {
		t.Fatal("preparation mutated caller-owned inputs")
	}
}

func TestPrepareHeaderParameterBindingsAreCaseInsensitive(t *testing.T) {
	doc := document(map[string]any{"/x": map[string]any{
		"parameters": []any{map[string]any{"name": "X-Enabled", "in": "header", "schema": map[string]any{"type": "boolean"}, "example": true}},
		"get":        map[string]any{"parameters": []any{map[string]any{"name": "x-enabled", "in": "header", "schema": map[string]any{"type": "boolean"}, "example": true}}},
	}})
	view := mustDocument(t, doc, scanplan.Options{Operations: map[string]scanplan.OperationInput{
		"#/paths/~1x/get": {Parameters: map[string]any{"header:X-ENABLED": false}},
	}})
	if headerValue(t, view, "x-enabled") != "false" || !view.Coverage.Complete {
		t.Fatal("case-insensitive header binding did not override its example")
	}
	options := scanplan.Options{Operations: map[string]scanplan.OperationInput{
		"#/paths/~1x/get": {Parameters: map[string]any{"header:X-ENABLED": true, "header:x-enabled": false}},
	}}
	_, err := scanplan.Prepare(encoded(t, doc), "application/json", sourceRef(), options)
	var preparationError *scanplan.PreparationError
	if !errors.As(err, &preparationError) || preparationError.Code != "duplicate_parameter_binding" {
		t.Fatalf("case-duplicate header bindings accepted or misclassified: %v", err)
	}
}

func TestPrepareAuthenticationAlternativeValidatesHeaderBeforeSelection(t *testing.T) {
	doc := authenticationDocument()
	schemes := doc["components"].(map[string]any)["securitySchemes"].(map[string]any)
	schemes["malformed"] = map[string]any{"type": "apiKey", "in": "header", "name": "not a header"}
	schemes["valid"] = map[string]any{"type": "apiKey", "in": "header", "name": "X-Key"}
	doc["security"] = []any{map[string]any{"malformed": []any{}}, map[string]any{"valid": []any{}}}
	view := mustDocument(t, doc, scanplan.Options{Authentication: map[string]contracts.SecretString{
		"malformed": contracts.NewSecretString("discarded-canary"),
		"valid":     contracts.NewSecretString("selected-token"),
	}})
	if headerValue(t, view, "x-key") != "selected-token" || !view.Coverage.Complete {
		t.Fatal("malformed first alternative prevented selection of valid authentication")
	}
	if strings.Contains(string(encoded(t, view)), "discarded-canary") {
		t.Fatal("failed authentication alternative leaked into the prepared request or gaps")
	}
}

func TestPrepareHTTPBearerSchemeIsCaseInsensitive(t *testing.T) {
	doc := authenticationDocument()
	doc["components"].(map[string]any)["securitySchemes"].(map[string]any)["bearer"] = map[string]any{"type": "http", "scheme": "Bearer"}
	doc["security"] = []any{map[string]any{"bearer": []any{}}}
	view := mustDocument(t, doc, scanplan.Options{Authentication: map[string]contracts.SecretString{
		"bearer": contracts.NewSecretString("explicit-token"),
	}})
	if headerValue(t, view, "authorization") != "Bearer explicit-token" || !view.Coverage.Complete {
		t.Fatal("case variant of bearer scheme was not recognized")
	}
}

func TestPrepareRequestBodiesRespectSupportedOperationMethods(t *testing.T) {
	for _, method := range []string{"get", "head", "delete", "post"} {
		t.Run(method, func(t *testing.T) {
			doc := document(map[string]any{"/x": map[string]any{method: map[string]any{"requestBody": map[string]any{
				"required": true,
				"content": map[string]any{"text/plain": map[string]any{
					"schema": map[string]any{"type": "string"}, "example": "explicit-body",
				}},
			}}}})
			view := mustDocument(t, doc, scanplan.Options{})
			if len(view.Requests) != 1 || !view.Coverage.Complete || view.Requests[0].Request.Body != "explicit-body" || view.Requests[0].Request.Method != strings.ToUpper(method) {
				t.Fatal("concrete request body was not preserved")
			}
		})
	}
}

func TestPrepareAcceptsOpenAPI30And31PatchVersions(t *testing.T) {
	for _, version := range []string{"3.0.0", "3.0.4", "3.0.5", "3.1.0", "3.1.1"} {
		t.Run(version, func(t *testing.T) {
			doc := document(map[string]any{"/x": map[string]any{"get": map[string]any{}}})
			doc["openapi"] = version
			view := mustDocument(t, doc, scanplan.Options{})
			if len(view.Requests) != 1 || !view.Coverage.Complete {
				t.Fatal("supported OpenAPI patch version did not produce a request")
			}
		})
	}
}

func TestPrepareInvalidWholeInputsAndBindingsHavePrivateErrors(t *testing.T) {
	valid := encoded(t, document(map[string]any{"/x": map[string]any{"get": map[string]any{}}}))
	for name, test := range map[string]struct {
		data    []byte
		media   string
		ref     contracts.ArtifactRef
		options scanplan.Options
	}{
		"unsupported-version": {data: bytes.Replace(valid, []byte("3.0.3"), []byte("2.0.0"), 1)},
		"duplicate-json":      {data: []byte(`{"openapi":"3.0.3","openapi":"secret-canary","paths":{}}`)},
		"nonfinite":           {data: []byte(`{"openapi":"3.0.3","paths":{},"secret-canary":NaN}`)},
		"large-integer":       {data: []byte(`{"openapi":"3.0.3","paths":{},"secret-canary":9007199254740993}`)},
		"invalid-utf8":        {data: append([]byte("secret-canary"), 0xff)},
		"yaml-alias":          {data: []byte("openapi: 3.0.3\npaths: &secret-canary {}\ncopy: *secret-canary\n"), media: "application/yaml"},
		"unknown-media":       {data: valid, media: "text/secret-canary"},
		"source-size":         {data: append([]byte("secret-canary"), bytes.Repeat([]byte(" "), scanplan.MaxSourceBytes)...)},
		"versionless-ref":     {data: valid, ref: contracts.ArtifactRef{Namespace: "inputs", Name: "secret-canary"}},
		"negative-limit":      {data: valid, options: scanplan.Options{MaxRequests: -1}},
		"oversized-limit":     {data: valid, options: scanplan.Options{MaxRequests: scanplan.MaxRequests + 1}},
		"unknown-operation":   {data: valid, options: scanplan.Options{Operations: map[string]scanplan.OperationInput{"#/paths/~1unknown/get": {Parameters: map[string]any{"query:q": "secret-canary"}}}}},
		"options-size":        {data: valid, options: scanplan.Options{Authentication: map[string]contracts.SecretString{"unused": contracts.NewSecretString(strings.Repeat("secret-canary", scanplan.MaxOptionsBytes))}}},
	} {
		t.Run(name, func(t *testing.T) {
			if test.media == "" {
				test.media = "application/json"
			}
			if test.ref.Namespace == "" {
				test.ref = sourceRef()
			}
			_, err := scanplan.Prepare(test.data, test.media, test.ref, test.options)
			if err == nil {
				t.Fatal("invalid input accepted")
			}
			for _, diagnostic := range []string{err.Error(), fmt.Sprintf("%+v", err), fmt.Sprintf("%#v", err)} {
				if strings.Contains(diagnostic, "canary") {
					t.Fatalf("error disclosed input: %s", diagnostic)
				}
			}
		})
	}
}

func TestPrepareOperationAndGlobalReferenceWorkBounds(t *testing.T) {
	paths := make(map[string]any)
	for i := 0; i <= scanplan.MaxOperations; i++ {
		paths[fmt.Sprintf("/p%04d", i)] = map[string]any{"get": map[string]any{}}
	}
	if _, err := scanplan.Prepare(encoded(t, document(paths)), "application/json", sourceRef(), scanplan.Options{}); err == nil {
		t.Fatal("operation limit accepted")
	}
	paths = make(map[string]any)
	parameters := make(map[string]any)
	refs := make([]any, 5)
	for i := range refs {
		name := fmt.Sprintf("Q%d", i)
		parameters[name] = map[string]any{"name": name, "in": "query", "schema": map[string]any{"type": "string"}, "example": "value"}
		refs[i] = map[string]any{"$ref": "#/components/parameters/" + name}
	}
	for i := 0; i < scanplan.MaxOperations; i++ {
		paths[fmt.Sprintf("/p%04d", i)] = map[string]any{"get": map[string]any{"parameters": refs}}
	}
	doc := document(paths)
	doc["components"] = map[string]any{"parameters": parameters}
	if _, err := scanplan.Prepare(encoded(t, doc), "application/json", sourceRef(), scanplan.Options{}); err == nil {
		t.Fatal("reference budget was reset for each operation")
	}
}

func TestPrepareRetainedOutputHasAnIndependentByteBound(t *testing.T) {
	paths := make(map[string]any)
	for i := 0; i < scanplan.MaxOperations; i++ {
		paths[fmt.Sprintf("/p%04d", i)] = map[string]any{"post": map[string]any{"requestBody": map[string]any{"$ref": "#/components/requestBodies/Shared"}}}
	}
	doc := document(paths)
	doc["components"] = map[string]any{"requestBodies": map[string]any{"Shared": map[string]any{
		"required": true, "content": map[string]any{"text/plain": map[string]any{"schema": map[string]any{"type": "string"}, "example": strings.Repeat("x", 6000)}},
	}}}
	data := encoded(t, doc)
	if len(data) >= scanplan.MaxSourceBytes {
		t.Fatal("fixture exceeds source bound")
	}
	result, err := scanplan.Prepare(data, "application/json", sourceRef(), scanplan.Options{})
	if err == nil {
		t.Fatalf("oversized retained output accepted with %d requests", len(result.Requests))
	}
	if len(result.Requests) != 0 {
		t.Fatal("output bound returned unaccounted partial requests")
	}
}
