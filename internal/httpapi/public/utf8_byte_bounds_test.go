package public

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp/syntax"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"
	"unicode"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/artifactpreview"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/gitimport"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"go.yaml.in/yaml/v4"
)

// utf8ByteBoundOwners binds each shared byte-bounded rule to the Go bound
// that enforces it. A rule marked -1 is bound field by field by its owner's
// own test: LIM-AUD-29 is unexported in auditservice
// (TestSharedAuditTextBoundsCountUTF8Bytes), and LIM-RCF-23 and LIM-SCN-94
// carry two bounds each (credentials and scanservice tests).
var utf8ByteBoundOwners = map[string]int{
	"LIM-SMT-2":  projectsitemap.MaxMemberKeyBytes,
	"LIM-SMT-6":  projectsitemap.MaxNoteBytes,
	"LIM-SMT-11": projectsitemap.MaxTagBytes,
	"LIM-SMT-16": projectsitemap.MaxPointerBytes,
	"LIM-RUN-10": maxCancellationReasonBytes,
	"LIM-PRJ-3":  projectstore.MaxNameBytes,
	"LIM-PRJ-5":  projectstore.MaxDescriptionBytes,
	"LIM-REV-11": auditservice.MaxReviewRationale,
	"LIM-RUN-5":  runlabels.MaxRunMetadataLabelValue,
	"LIM-AUD-29": -1,
	"LIM-AUD-39": auditstore.MaxArtifactLogicalKeyBytes,
	"LIM-AUD-40": auditstore.MaxSubjectKeyBytes,
	"LIM-GIT-3":  credentials.MaximumGitKeyBytes,
	"LIM-GIT-22": gitimport.MaximumRefBytes,
	"LIM-GIT-23": gitimport.MaximumRemoteURLBytes,
	"LIM-RCF-1":  llmgateway.MaximumGatewayFailureSignatureText,
	"LIM-RCF-23": -1,
	"LIM-SCN-94": -1,
	"LIM-API-27": artifactpreview.MaximumPathBytes,
}

// requestStringsWithoutByteBound lists each public request string with a
// maxLength, neither an ASCII-only grammar nor a byte bound, and the reason
// the Server does not reject a value within maxLength for its UTF-8 size
// (API-R4). Request strings of Eval schemas need no entry: evaldomain.Validate
// checks Eval documents against the embedded Eval JSON Schemas, whose
// maxLength counts code points as OpenAPI does.
var requestStringsWithoutByteBound = map[string]string{
	"CreateCredentialRequest/properties/label": "credentials.MaximumCredentialLabel counts code points",
	"ModelPolicyBody/properties/model":         "config.Manager bounds the model name in code points",
	"parameters/CatalogQuery":                  "maxCatalogSearchRunes counts code points",
	// A Run label selector key=value has no bound of its own: its key is
	// ASCII by the RunMetadataLabels key grammar (63 bytes) and its value
	// counts bytes (LIM-RUN-5), which one string schema cannot both express.
	"GET /v1/runs query label/items":                                  "composite: key grammar and LIM-RUN-5 value bytes",
	"GET /v1/projects/{projectId}/runs query label/items":             "composite: key grammar and LIM-RUN-5 value bytes",
	"SitemapEditorRule/properties/origin":                             "composite: origin and path share one 8 KiB byte bound (LIM-SMT-2)",
	"SitemapEditorRule/properties/path":                               "composite: origin and path share one 8 KiB byte bound (LIM-SMT-2)",
	"GET /v1/eval-experiments query projectId":                        "compared with stored Project IDs; no size bound",
	"GET /v1/eval-experiments/{id}/charts/{chart} query suiteId":      "compared with the pinned suites; no size bound",
	"GET /v1/eval-experiments/{id}/members query suiteId":             "compared with the pinned suites; no size bound",
	"GET /v1/eval-experiments/{id}/pairs query suiteId":               "compared with the pinned suites; no size bound",
	"GET /v1/eval-experiments/{id}/members query variantId":           "compared with the pinned variants; no size bound",
	"GET /v1/eval-experiments/{id}/charts/{chart} query viewSnapshot": "compared with the current view snapshot; no size bound",
	"GET /v1/eval-experiments/{id}/members query viewSnapshot":        "compared with the current view snapshot; no size bound",
	"GET /v1/eval-experiments/{id}/pairs query viewSnapshot":          "compared with the current view snapshot; no size bound",
	"GET /v1/eval-experiments/{id}/pairs/{pairId} query viewSnapshot": "compared with the current view snapshot; no size bound",
	"GET /v1/eval-experiments/{id}/report query viewSnapshot":         "compared with the current view snapshot; no size bound",
}

// TestSharedUTF8ByteBoundsAreDeclared holds every public string that the
// Server bounds in UTF-8 bytes to the component schema that declares it: the
// schema carries x-max-utf8-bytes equal to its maxLength and to the Go bound,
// and no other place in the document than these and the login password
// declares a byte bound. Every other request string with a maxLength, in a
// request body or a parameter, declares an ASCII-only grammar or is listed in
// requestStringsWithoutByteBound.
func TestSharedUTF8ByteBoundsAreDeclared(t *testing.T) {
	document := loadPublicOpenAPI(t)
	raw, err := os.ReadFile(publicOpenAPIPath)
	if err != nil {
		t.Fatal(err)
	}
	var source map[string]any
	if err := yaml.Unmarshal(raw, &source); err != nil {
		t.Fatal(err)
	}
	_, fields := contractstest.ReadUTF8ByteBoundCases(t)
	for _, field := range fields {
		bound, owned := utf8ByteBoundOwners[field.Rule]
		if !owned {
			t.Errorf("%s: rule %s has no Go owner", field.Schema+"/"+strings.Join(field.Path, "/"), field.Rule)
		} else if bound >= 0 && bound != field.MaxBytes {
			t.Errorf("%s: shared bound %d, Go bound %d", field.Schema+"/"+strings.Join(field.Path, "/"), field.MaxBytes, bound)
		}
	}
	for _, violation := range utf8ByteBoundViolations(document, source, fields, requestStringsWithoutByteBound) {
		t.Error(violation)
	}
}

// TestUTF8ByteBoundCheckFindsProbes pins that the check above fails for an
// undeclared byte bound, an unlisted or inline declaration, an unclassified
// request string in a body or a parameter and a stale exception.
func TestUTF8ByteBoundCheckFindsProbes(t *testing.T) {
	const probe = `openapi: 3.1.0
info: {title: probe, version: "1"}
paths:
  /probe:
    post:
      parameters:
        - {name: cursor, in: query, schema: {type: string, maxLength: 64, pattern: '^[A-Za-z0-9_-]+$'}}
        - {name: q, in: query, schema: {type: string, maxLength: 64}}
        - {name: inline, in: query, schema: {type: string, maxLength: 64, x-max-utf8-bytes: 64}}
        - {name: any, in: query, schema: {type: string, maxLength: 64, pattern: '^[^\x00]+$'}}
        - {name: state, in: query, schema: {type: string, enum: [a, b]}}
        - {name: target, in: query, schema: {type: string, format: uri, maxLength: 64}}
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                ref: {type: string, maxLength: 16, x-max-utf8-bytes: 16}
                body: {$ref: '#/components/schemas/Probe'}
      responses:
        '204': {description: none}
components:
  schemas:
    Probe:
      type: object
      properties:
        declared: {type: string, maxLength: 8, x-max-utf8-bytes: 8}
        unlisted: {type: string, maxLength: 8, x-max-utf8-bytes: 8}
        wrong: {type: string, maxLength: 8, x-max-utf8-bytes: 4}
        missing: {type: string, maxLength: 8}
        excused: {type: string, maxLength: 8}
`
	loader := openapi3.NewLoader()
	document, err := loader.LoadFromData([]byte(probe))
	if err != nil {
		t.Fatal(err)
	}
	var source map[string]any
	if err := yaml.Unmarshal([]byte(probe), &source); err != nil {
		t.Fatal(err)
	}
	fields := []contractstest.UTF8ByteBound{
		{Schema: "Probe", Path: []string{"properties", "declared"}, MaxBytes: 8, Rule: "LIM-PROBE-1"},
		{Schema: "Probe", Path: []string{"properties", "wrong"}, MaxBytes: 8, Rule: "LIM-PROBE-1"},
		{Schema: "Probe", Path: []string{"properties", "missing"}, MaxBytes: 8, Rule: "LIM-PROBE-1"},
	}
	violations := strings.Join(utf8ByteBoundViolations(document, source, fields, map[string]string{
		"Probe/properties/excused": "probe",
		"Probe/properties/gone":    "stale",
	}), "\n")
	for _, want := range []string{
		"Probe/properties/wrong: x-max-utf8-bytes = 4, want 8",
		"Probe/properties/missing: x-max-utf8-bytes = <nil>, want 8",
		"$/components/schemas/Probe/properties/unlisted declares x-max-utf8-bytes",
		"$/paths/~1probe/post/parameters/2/schema declares x-max-utf8-bytes",
		"$/paths/~1probe/post/requestBody/content/application~1json/schema/properties/ref declares x-max-utf8-bytes",
		"POST /probe query q has a maxLength",
		"POST /probe query any has a maxLength",
		"Probe/properties/missing has a maxLength",
		"Probe/properties/gone is listed",
	} {
		if !strings.Contains(violations, want) {
			t.Errorf("the check missed %q in:\n%s", want, violations)
		}
	}
	for _, clean := range []string{"query cursor", "query state", "query target", "Probe/properties/declared", "Probe/properties/excused has"} {
		if strings.Contains(violations, clean) {
			t.Errorf("the check reported %q in:\n%s", clean, violations)
		}
	}
}

// utf8ByteBoundViolations checks one document against the shared byte
// bounds and the request strings excused from them.
func utf8ByteBoundViolations(document *openapi3.T, source map[string]any, fields []contractstest.UTF8ByteBound, excused map[string]string) []string {
	var violations []string
	listed := map[string]bool{"$/components/schemas/LoginRequest/properties/password": true}
	for _, field := range fields {
		location := strings.Join(append([]string{field.Schema}, field.Path...), "/")
		listed["$/components/schemas/"+location] = true
		schema := schemaAtPath(document, field.Schema, field.Path)
		if schema == nil {
			violations = append(violations, location+": no such schema")
			continue
		}
		if declared := fmt.Sprint(schema.Extensions["x-max-utf8-bytes"]); declared != strconv.Itoa(field.MaxBytes) {
			violations = append(violations, fmt.Sprintf("%s: x-max-utf8-bytes = %s, want %d", location, declared, field.MaxBytes))
		}
		if schema.MaxLength == nil {
			violations = append(violations, fmt.Sprintf("%s: no maxLength, want %d next to the byte bound", location, field.MaxBytes))
		} else if *schema.MaxLength != uint64(field.MaxBytes) {
			violations = append(violations, fmt.Sprintf("%s: maxLength = %d, want %d next to the byte bound", location, *schema.MaxLength, field.MaxBytes))
		}
	}
	for _, location := range byteBoundDeclarations(source, "$") {
		if !listed[location] {
			violations = append(violations, location+" declares x-max-utf8-bytes, but the shared UTF-8 byte bound cases do not list it as a component schema")
		}
	}
	used := map[string]bool{}
	for _, location := range unclassifiedRequestStrings(document) {
		if _, ok := excused[location]; ok {
			used[location] = true
			continue
		}
		violations = append(violations, location+" has a maxLength but neither x-max-utf8-bytes nor an ASCII-only pattern; declare one or list why the Server does not count its bytes")
	}
	for location := range excused {
		if !used[location] {
			violations = append(violations, location+" is listed without a byte bound, but no request string there needs the exception")
		}
	}
	sort.Strings(violations)
	return violations
}

// byteBoundDeclarations returns the location of every schema in the raw
// document that declares x-max-utf8-bytes, wherever it appears.
func byteBoundDeclarations(value any, location string) []string {
	var found []string
	switch current := value.(type) {
	case map[string]any:
		if _, declared := current["x-max-utf8-bytes"]; declared {
			found = append(found, location)
		}
		for key, child := range current {
			escaped := strings.ReplaceAll(strings.ReplaceAll(key, "~", "~0"), "/", "~1")
			found = append(found, byteBoundDeclarations(child, location+"/"+escaped)...)
		}
	case []any:
		for index, child := range current {
			found = append(found, byteBoundDeclarations(child, location+"/"+strconv.Itoa(index))...)
		}
	}
	return found
}

// byteCountingFormats are the formats under which maxLength already counts
// bytes: an RFC 3986 URI is ASCII (an IRI would be format iri), and a binary
// body is a sequence of octets.
var byteCountingFormats = map[string]bool{"uri": true, "binary": true}

// unclassifiedRequestStrings returns each request string, in a request body
// or a parameter and in the component schemas they reference, that has a
// maxLength but no enum, const, byte bound, byte-counting format or ASCII-only
// pattern. A string of an Eval schema is checked by the embedded Eval JSON
// Schemas instead.
func unclassifiedRequestStrings(document *openapi3.T) []string {
	walk := requestStringWalk{seen: map[string]bool{}}
	for name, parameter := range document.Components.Parameters {
		if parameter.Value != nil {
			walk.visit(parameter.Value.Schema, "parameters/"+name)
		}
	}
	for path, item := range document.Paths.Map() {
		for method, operation := range item.Operations() {
			at := method + " " + path
			for _, parameter := range append(slices.Clone(item.Parameters), operation.Parameters...) {
				if parameter.Ref == "" && parameter.Value != nil {
					walk.visit(parameter.Value.Schema, at+" "+parameter.Value.In+" "+parameter.Value.Name)
				}
			}
			if operation.RequestBody != nil && operation.RequestBody.Value != nil {
				for mediaType, content := range operation.RequestBody.Value.Content {
					walk.visit(content.Schema, at+" body "+mediaType)
				}
			}
		}
	}
	sort.Strings(walk.found)
	return walk.found
}

type requestStringWalk struct {
	seen  map[string]bool
	found []string
}

func (w *requestStringWalk) visit(reference *openapi3.SchemaRef, location string) {
	if reference == nil || reference.Value == nil {
		return
	}
	if name, ok := strings.CutPrefix(reference.Ref, "#/components/schemas/"); ok {
		if w.seen[name] {
			return
		}
		w.seen[name] = true
		location = name
	}
	schema := reference.Value
	if schema.Type != nil && slices.Contains(*schema.Type, "string") && schema.MaxLength != nil &&
		len(schema.Enum) == 0 && schema.Const == nil && schema.Extensions["x-max-utf8-bytes"] == nil && !byteCountingFormats[schema.Format] &&
		(schema.Pattern == "" || !asciiOnlyPattern(schema.Pattern)) && !strings.HasPrefix(location, "Eval") {
		w.found = append(w.found, location)
	}
	for name, child := range schema.Properties {
		w.visit(child, location+"/properties/"+name)
	}
	w.visit(schema.AdditionalProperties.Schema, location+"/additionalProperties")
	w.visit(schema.PropertyNames, location+"/propertyNames")
	w.visit(schema.Items, location+"/items")
	for keyword, children := range map[string]openapi3.SchemaRefs{"allOf": schema.AllOf, "oneOf": schema.OneOf, "anyOf": schema.AnyOf} {
		for index, child := range children {
			w.visit(child, location+"/"+keyword+"/"+strconv.Itoa(index))
		}
	}
}

// asciiOnlyPattern reports whether every string a pattern matches is ASCII,
// so that its maxLength counts bytes: no construct of the parsed pattern
// admits a character above U+007F. A pattern Go cannot parse is not.
func asciiOnlyPattern(pattern string) bool {
	parsed, err := syntax.Parse(pattern, syntax.Perl)
	if err != nil {
		return false
	}
	return asciiOnlyRegexp(parsed)
}

func asciiOnlyRegexp(parsed *syntax.Regexp) bool {
	switch parsed.Op {
	case syntax.OpAnyChar, syntax.OpAnyCharNotNL:
		return false
	case syntax.OpLiteral, syntax.OpCharClass:
		if parsed.Flags&syntax.FoldCase != 0 {
			return false
		}
		for _, char := range parsed.Rune {
			if char > unicode.MaxASCII {
				return false
			}
		}
	}
	for _, sub := range parsed.Sub {
		if !asciiOnlyRegexp(sub) {
			return false
		}
	}
	return true
}

// TestSharedCancellationReasonCountsUTF8Bytes sends each shared character's
// reason at and one byte over the bound: the first cancels the Run and is
// stored exactly, the second is 400 invalid_request and changes nothing.
func TestSharedCancellationReasonCountsUTF8Bytes(t *testing.T) {
	characters, fields := contractstest.ReadUTF8ByteBoundCases(t)
	var bound contractstest.UTF8ByteBound
	for _, field := range fields {
		if field.Schema == "CancelRunRequest" {
			bound = field
		}
	}
	if bound.MaxBytes == 0 {
		t.Fatal("no shared CancelRunRequest bound")
	}
	for index, value := range bound.Values(characters) {
		fixture := newHandlerFixture(t)
		runID := "run-reason-" + strconv.Itoa(index)
		fixture.runs.runs[runID] = runstore.WorkflowRun{
			RunID: runID, OwnerID: "user-1", WorkflowName: "artifact-copy", WorkflowVersion: "1", State: runstore.RunRunning,
		}
		body, err := json.Marshal(map[string]string{"reason": value.Value})
		if err != nil {
			t.Fatal(err)
		}
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, authenticatedRequest(http.MethodPost, "/v1/runs/"+runID+"/cancel", bytes.NewReader(body)))
		stored := fixture.runs.runs[runID].Cancellation
		switch {
		case value.Within && (response.Code != http.StatusAccepted || stored == nil || stored.Reason == nil || *stored.Reason != value.Value):
			t.Errorf("%q reason of %d bytes = %d %s, want it stored", value.Character, len(value.Value), response.Code, response.Body.String())
		case !value.Within && (response.Code != http.StatusBadRequest || stored != nil):
			t.Errorf("%q reason of %d bytes = %d %s, want 400 and no cancellation", value.Character, len(value.Value), response.Code, response.Body.String())
		}
	}
}

// TestSharedAuditArtifactKeyCountsUTF8Bytes checks the logical key of an
// Audit artifact read (LIM-AUD-39) with each shared character at and one byte
// over the declared bound.
func TestSharedAuditArtifactKeyCountsUTF8Bytes(t *testing.T) {
	characters, fields := contractstest.ReadUTF8ByteBoundCases(t)
	checked := false
	for _, field := range fields {
		if field.Rule != "LIM-AUD-39" {
			continue
		}
		checked = true
		for _, value := range field.Values(characters) {
			if validAuditArtifactSelector(value.Value, "revision-1", "") != value.Within {
				t.Errorf("key of %d bytes of %q: within %v", len(value.Value), value.Character, value.Within)
			}
		}
	}
	if !checked {
		t.Fatal("no shared Audit artifact key bound")
	}
}

// schemaAtPath follows properties, additionalProperties, propertyNames and
// items from a component schema; an empty path is the component itself.
func schemaAtPath(document *openapi3.T, name string, path []string) *openapi3.Schema {
	reference := document.Components.Schemas[name]
	for index := 0; reference != nil && index < len(path); index++ {
		if reference.Value == nil {
			return nil
		}
		switch path[index] {
		case "properties":
			if index+1 == len(path) {
				return nil
			}
			index++
			reference = reference.Value.Properties[path[index]]
		case "additionalProperties":
			reference = reference.Value.AdditionalProperties.Schema
		case "propertyNames":
			reference = reference.Value.PropertyNames
		case "items":
			reference = reference.Value.Items
		default:
			return nil
		}
	}
	if reference == nil {
		return nil
	}
	return reference.Value
}
