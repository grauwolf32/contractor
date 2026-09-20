package contracts

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"
)

func requestSetFixture(t *testing.T) []byte {
	t.Helper()
	raw, err := os.ReadFile("../../api/scan/v1/testdata/valid.json")
	if err != nil {
		t.Fatal(err)
	}
	return bytes.TrimSpace(raw)
}

func validRequestSet(t *testing.T) HTTPRequestSet {
	t.Helper()
	value, err := DecodeHTTPRequestSet(requestSetFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func requestSetSchema(t *testing.T) *jsonschema.Schema {
	t.Helper()
	path, err := filepath.Abs("../../api/scan/v1/http-request-set.schema.json")
	if err != nil {
		t.Fatal(err)
	}
	compiler := jsonschema.NewCompiler()
	schema, err := compiler.Compile(path)
	if err != nil {
		t.Fatal(err)
	}
	return schema
}

func schemaAccepts(t *testing.T, schema *jsonschema.Schema, data []byte) bool {
	t.Helper()
	value, err := jsonschema.UnmarshalJSON(bytes.NewReader(data))
	if err != nil {
		return false
	}
	return schema.Validate(value) == nil
}

func TestHTTPRequestSetCanonicalFixture(t *testing.T) {
	raw := requestSetFixture(t)
	value := validRequestSet(t)
	if !schemaAccepts(t, requestSetSchema(t), raw) {
		t.Fatal("schema rejected normative fixture")
	}
	encoded, err := MarshalHTTPRequestSet(value)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, encoded) {
		t.Fatalf("fixture is not canonical: %s", encoded)
	}
	const digest = "sha256:0461e986b67c86eb84c12c72153b60ee48a3f10478793cadb1c584853f519b39"
	// Pin the independently calculated fixture digest instead of deriving the
	// expected identity with the same helper being exercised.
	if value.Requests[0].ContentDigest != digest {
		t.Fatalf("unexpected fixture digest: %s", value.Requests[0].ContentDigest)
	}
	value.Source.Artifact.Name = "other-source"
	value.PreparationDigest = "sha256:" + strings.Repeat("a", 64)
	value.Requests[0].Origins[0].Pointer = "#/paths/~1other/get"
	if err := value.Validate(); err != nil {
		t.Fatalf("provenance changed request identity: %v", err)
	}
}

func TestHTTPRequestSetStructuralSchemaParity(t *testing.T) {
	schema := requestSetSchema(t)
	cases := map[string]func(map[string]any){
		"missing body":          func(v map[string]any) { delete(requestObject(v), "body") },
		"null body":             func(v map[string]any) { requestObject(v)["body"] = nil },
		"null headers":          func(v map[string]any) { requestObject(v)["headers"] = nil },
		"unknown request field": func(v map[string]any) { requestObject(v)["testParameters"] = []any{} },
		"wrong casing":          func(v map[string]any) { v["SchemaVersion"] = v["schemaVersion"]; delete(v, "schemaVersion") },
		"missing complete":      func(v map[string]any) { delete(v["coverage"].(map[string]any), "complete") },
		"missing revision":      func(v map[string]any) { delete(v["source"].(map[string]any)["artifact"].(map[string]any), "revision") },
		"null revision":         func(v map[string]any) { v["source"].(map[string]any)["artifact"].(map[string]any)["revision"] = nil },
		"blank revision":        func(v map[string]any) { v["source"].(map[string]any)["artifact"].(map[string]any)["revision"] = "  " },
		"unknown source":        func(v map[string]any) { v["source"].(map[string]any)["url"] = "https://example.test" },
		"unsupported version":   func(v map[string]any) { v["schemaVersion"] = 2 },
		"bad digest":            func(v map[string]any) { v["preparationDigest"] = "sha256:abc" },
		"empty origins":         func(v map[string]any) { v["requests"].([]any)[0].(map[string]any)["origins"] = []any{} },
		"null gaps":             func(v map[string]any) { v["gaps"] = nil },
		"TRACE method":          func(v map[string]any) { requestObject(v)["method"] = "TRACE" },
		"non-ASCII URL":         func(v map[string]any) { requestObject(v)["url"] = "https://example.test/é" },
		"fragment URL":          func(v map[string]any) { requestObject(v)["url"] = "https://example.test/#fragment" },
		"uppercase header": func(v map[string]any) {
			requestObject(v)["headers"] = []any{map[string]any{"name": "Accept", "value": "application/json"}}
		},
		"newline header": func(v map[string]any) {
			requestObject(v)["headers"] = []any{map[string]any{"name": "accept", "value": "x\ny"}}
		},
		"invalid pointer escape": func(v map[string]any) {
			v["gaps"] = []any{map[string]any{"pointer": "#/paths/~2", "code": "unsupported"}}
		},
		"negative coverage": func(v map[string]any) { v["coverage"].(map[string]any)["skipped"] = -1 },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			var value map[string]any
			if err := json.Unmarshal(requestSetFixture(t), &value); err != nil {
				t.Fatal(err)
			}
			mutate(value)
			raw, err := json.Marshal(value)
			if err != nil {
				t.Fatal(err)
			}
			if schemaAccepts(t, schema, raw) {
				t.Fatal("schema accepted invalid structure")
			}
			if _, err := DecodeHTTPRequestSet(raw); err == nil {
				t.Fatal("Go accepted invalid structure")
			}
		})
	}
}

func requestObject(value map[string]any) map[string]any {
	return value["requests"].([]any)[0].(map[string]any)["request"].(map[string]any)
}

func TestHTTPRequestSetSemanticTampering(t *testing.T) {
	cases := map[string]func(*HTTPRequestSet){
		"request changed":   func(v *HTTPRequestSet) { v.Requests[0].Request.Body = "tampered" },
		"digest changed":    func(v *HTTPRequestSet) { v.Requests[0].ContentDigest = "sha256:" + strings.Repeat("a", 64) },
		"id changed":        func(v *HTTPRequestSet) { v.Requests[0].ID = "request-" + strings.Repeat("b", 64) },
		"duplicate request": func(v *HTTPRequestSet) { v.Requests = append(v.Requests, v.Requests[0]) },
		"duplicate origin": func(v *HTTPRequestSet) {
			v.Requests[0].Origins = append(v.Requests[0].Origins, v.Requests[0].Origins[0])
		},
		"unsorted origins": func(v *HTTPRequestSet) {
			v.Requests[0].Origins = []RequestOrigin{{"#/z"}, {"#/a"}}
			v.Coverage.Operations = 2
			v.Coverage.Prepared = 2
		},
		"duplicate gap": func(v *HTTPRequestSet) {
			v.Gaps = []PreparationGap{{"#", "unsupported"}, {"#", "unsupported"}}
			v.Coverage.Complete = false
		},
		"unsorted gaps": func(v *HTTPRequestSet) {
			v.Gaps = []PreparationGap{{"#/z", "unsupported"}, {"#/a", "unsupported"}}
			v.Coverage.Complete = false
		},
		"false completeness": func(v *HTTPRequestSet) { v.Gaps = []PreparationGap{{"#", "unsupported"}} },
		"false incomplete":   func(v *HTTPRequestSet) { v.Coverage.Complete = false },
		"false prepared":     func(v *HTTPRequestSet) { v.Coverage.Prepared = 0 },
		"false skipped":      func(v *HTTPRequestSet) { v.Coverage.Skipped = 1 },
		"excess operations": func(v *HTTPRequestSet) {
			v.Coverage.Operations = 1001
			v.Coverage.Skipped = 1000
			v.Coverage.Complete = false
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			value := validRequestSet(t)
			mutate(&value)
			if err := value.Validate(); err == nil {
				t.Fatal("accepted semantic tampering")
			}
			if _, err := MarshalHTTPRequestSet(value); err == nil {
				t.Fatal("marshaled semantic tampering")
			}
			raw, err := json.Marshal(value)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := DecodeHTTPRequestSet(raw); err == nil {
				t.Fatal("decoded semantic tampering")
			}
		})
	}
}

func TestHTTPRequestSetRejectsLenientJSON(t *testing.T) {
	raw := string(requestSetFixture(t))
	cases := map[string]string{
		"duplicate key":        strings.Replace(raw, `"schemaVersion":1`, `"schemaVersion":1,"schemaVersion":1`, 1),
		"nested duplicate":     strings.Replace(raw, `"body":""`, `"body":"","body":""`, 1),
		"trailing JSON":        raw + " {}",
		"null document":        "null",
		"array document":       "[]",
		"invalid UTF-8":        strings.Replace(raw, `"body":""`, "\"body\":\"\xff\"", 1),
		"lone high surrogate":  strings.Replace(raw, `"body":""`, `"body":"\ud800"`, 1),
		"lone low surrogate":   strings.Replace(raw, `"body":""`, `"body":"\udc00"`, 1),
		"wrong surrogate pair": strings.Replace(raw, `"body":""`, `"body":"\ud800\u0061"`, 1),
		"oversized artifact":   raw + strings.Repeat(" ", MaxHTTPRequestSetBytes),
	}
	for name, input := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := DecodeHTTPRequestSet([]byte(input)); err == nil {
				t.Fatal("accepted invalid JSON")
			}
		})
	}
}

func TestPreparedHTTPRequestValidationAndBounds(t *testing.T) {
	cases := map[string]func(*PreparedHTTPRequest){
		"URL credentials":        func(r *PreparedHTTPRequest) { r.URL = "https://user:password@example.test/" },
		"invalid URL escape":     func(r *PreparedHTTPRequest) { r.URL = "https://example.test/%zz" },
		"invalid query escape":   func(r *PreparedHTTPRequest) { r.URL = "https://example.test/?key=%zz" },
		"truncated query escape": func(r *PreparedHTTPRequest) { r.URL = "https://example.test/?key=%2" },
		"zero port":              func(r *PreparedHTTPRequest) { r.URL = "https://example.test:0/" },
		"excess port":            func(r *PreparedHTTPRequest) { r.URL = "https://example.test:65536/" },
		"overflowing port":       func(r *PreparedHTTPRequest) { r.URL = "https://example.test:999999999999999999999999/" },
		"empty port":             func(r *PreparedHTTPRequest) { r.URL = "https://example.test:/" },
		"URL backslash":          func(r *PreparedHTTPRequest) { r.URL = "https://example.test/\\foo" },
		"URL bytes": func(r *PreparedHTTPRequest) {
			r.URL = "https://example.test/" + strings.Repeat("a", MaxHTTPRequestURLBytes)
		},
		"body bytes":         func(r *PreparedHTTPRequest) { r.Body = strings.Repeat("é", MaxHTTPRequestBodyBytes/2+1) },
		"invalid body UTF-8": func(r *PreparedHTTPRequest) { r.Body = "\xff" },
		"header name bytes":  func(r *PreparedHTTPRequest) { r.Headers = []HTTPRequestHeader{{strings.Repeat("x", 129), "value"}} },
		"header value bytes": func(r *PreparedHTTPRequest) { r.Headers = []HTTPRequestHeader{{"x", strings.Repeat("é", 4097)}} },
		"header aggregate": func(r *PreparedHTTPRequest) {
			r.Headers = []HTTPRequestHeader{{"a", strings.Repeat("x", 8192)}, {"b", strings.Repeat("x", 8192)}, {"c", strings.Repeat("x", 8192)}, {"d", strings.Repeat("x", 8192)}}
		},
		"header controls":  func(r *PreparedHTTPRequest) { r.Headers = []HTTPRequestHeader{{"x", "secret\r\ninjected: value"}} },
		"header duplicate": func(r *PreparedHTTPRequest) { r.Headers = []HTTPRequestHeader{{"x", "a"}, {"x", "b"}} },
		"header ordering":  func(r *PreparedHTTPRequest) { r.Headers = []HTTPRequestHeader{{"z", "a"}, {"a", "b"}} },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			request := validRequestSet(t).Requests[0].Request
			mutate(&request)
			if err := request.Validate(); err == nil {
				t.Fatal("accepted invalid request")
			}
			if _, err := RequestContentDigest(request); err == nil {
				t.Fatal("hashed invalid request")
			}
		})
	}
	request := validRequestSet(t).Requests[0].Request
	request.Body = strings.Repeat("é", MaxHTTPRequestBodyBytes/2)
	request.Headers = []HTTPRequestHeader{{"x", "tab\tallowed"}}
	if _, err := RequestContentDigest(request); err != nil {
		t.Fatalf("rejected exact body bound: %v", err)
	}
	request.Body = "ordinary scanner marker * and FFUFHASH stay unchanged"
	if _, err := RequestContentDigest(request); err != nil {
		t.Fatalf("applied scanner-specific validation: %v", err)
	}
}

func TestHTTPRequestSetEmptyAndPartialCoverage(t *testing.T) {
	value := validRequestSet(t)
	value.Requests = []RequestSetEntry{}
	value.Coverage = RequestSetCoverage{Complete: true}
	if _, err := MarshalHTTPRequestSet(value); err != nil {
		t.Fatal(err)
	}
	value.Coverage = RequestSetCoverage{Operations: 2, Skipped: 2, Complete: false}
	value.Gaps = []PreparationGap{{"#/paths/~1pets/get", "missing_parameter"}}
	raw, err := MarshalHTTPRequestSet(value)
	if err != nil {
		t.Fatal(err)
	}
	if !schemaAccepts(t, requestSetSchema(t), raw) {
		t.Fatal("schema rejected partial coverage")
	}
	if _, err := DecodeHTTPRequestSet(raw); err != nil {
		t.Fatal(err)
	}
}

func TestHTTPRequestSetJSONStringUnicode(t *testing.T) {
	for _, value := range []string{`"\ud83d\ude00"`, `"literal \\ud800"`, `"escaped \" quotation"`, `"é"`} {
		if !requestSetJSONStringUnicode([]byte(value)) {
			t.Fatalf("rejected valid Unicode %s", value)
		}
	}
}

func TestHTTPRequestSetArtifactAndArrayBounds(t *testing.T) {
	value := validRequestSet(t)
	entry := value.Requests[0]
	value.Requests = make([]RequestSetEntry, MaxHTTPRequestSetRequests+1)
	if err := value.Validate(); err == nil {
		t.Fatal("accepted excess requests")
	}
	value.Requests = []RequestSetEntry{entry}
	value.Gaps = make([]PreparationGap, MaxHTTPRequestSetGaps+1)
	if err := value.Validate(); err == nil {
		t.Fatal("accepted excess gaps")
	}
	value.Gaps = []PreparationGap{}
	value.Requests[0].Origins = make([]RequestOrigin, MaxHTTPRequestSetOrigins+1)
	if err := value.Validate(); err == nil {
		t.Fatal("accepted excess origins")
	}

	// Individual requests fit their limits but their canonical artifact does not.
	value.Requests = []RequestSetEntry{}
	for i := 0; i < 70; i++ {
		request := PreparedHTTPRequest{Method: "POST", URL: "https://example.test/" + strconv.Itoa(i), Headers: []HTTPRequestHeader{}, Body: strings.Repeat("a", MaxHTTPRequestBodyBytes)}
		digest, err := RequestContentDigest(request)
		if err != nil {
			t.Fatal(err)
		}
		value.Requests = append(value.Requests, RequestSetEntry{ID: "request-" + strings.TrimPrefix(digest, "sha256:"), ContentDigest: digest, Request: request, Origins: []RequestOrigin{{Pointer: "#/paths/~1" + strconv.Itoa(i) + "/post"}}})
	}
	sort.Slice(value.Requests, func(i, j int) bool { return value.Requests[i].ID < value.Requests[j].ID })
	value.Coverage = RequestSetCoverage{Operations: 70, Prepared: 70, Complete: true}
	if err := value.Validate(); err != nil {
		t.Fatal(err)
	}
	if _, err := MarshalHTTPRequestSet(value); err == nil {
		t.Fatal("marshaled oversized aggregate artifact")
	}
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := DecodeHTTPRequestSet(raw); err == nil {
		t.Fatal("decoded oversized aggregate artifact")
	}
}

func TestHTTPRequestSetOriginsAreUniqueAcrossRequests(t *testing.T) {
	value := validRequestSet(t)
	second := value.Requests[0]
	second.Request.URL += "&second=true"
	digest, err := RequestContentDigest(second.Request)
	if err != nil {
		t.Fatal(err)
	}
	second.ContentDigest = digest
	second.ID = "request-" + strings.TrimPrefix(digest, "sha256:")
	value.Requests = append(value.Requests, second)
	sort.Slice(value.Requests, func(i, j int) bool { return value.Requests[i].ID < value.Requests[j].ID })
	value.Coverage = RequestSetCoverage{Operations: 2, Prepared: 2, Complete: true}
	if err := value.Validate(); err == nil {
		t.Fatal("accepted same operation twice")
	}
	value.Requests[1].Origins = []RequestOrigin{{Pointer: "#/paths/~1other/get"}}
	if err := value.Validate(); err != nil {
		t.Fatal(err)
	}
	value.Requests[0], value.Requests[1] = value.Requests[1], value.Requests[0]
	if err := value.Validate(); err == nil {
		t.Fatal("accepted unsorted requests")
	}
}

func TestHTTPRequestSetInvalidFixtures(t *testing.T) {
	for _, name := range []string{"invalid-missing-body.json", "invalid-content-digest.json", "invalid-duplicate-key.json"} {
		t.Run(name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("../../api/scan/v1/testdata", name))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := DecodeHTTPRequestSet(raw); err == nil {
				t.Fatal("accepted invalid fixture")
			}
		})
	}
}
