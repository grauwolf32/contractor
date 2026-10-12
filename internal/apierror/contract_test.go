package apierror

import (
	"encoding/json"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
)

// The registry is checked against the public contracts and the browser
// client that consume its codes (ERR-R14).
const (
	publicOpenAPI = "../../api/openapi/contractor-public-v1.yaml"
	eventSchema   = "../../api/events/contractor-events-v1.schema.json"
	browserSource = "../../ui/src"
)

// uiLocalCodes are the codes the browser client gives its own failures
// (ui/src/api/error.ts and client.ts): they never come from the Server, so
// they must not be registered.
var uiLocalCodes = []string{"incompatible_api", "invalid_api_response", "network_error", "request_failed"}

func loadOpenAPI(t *testing.T) *openapi3.T {
	t.Helper()
	loader := openapi3.NewLoader()
	loader.IsExternalRefsAllowed = false
	document, err := loader.LoadFromFile(publicOpenAPI)
	if err != nil {
		t.Fatalf("load public OpenAPI: %v", err)
	}
	return document
}

func registeredCodes(include func(*entry) bool) []string {
	var codes []string
	for _, row := range registry {
		if include(row) {
			codes = append(codes, row.code)
		}
	}
	slices.Sort(codes)
	return codes
}

func sortedStrings(values []any) []string {
	result := make([]string, 0, len(values))
	for _, value := range values {
		text, _ := value.(string)
		result = append(result, text)
	}
	slices.Sort(result)
	return result
}

// TestOpenAPIDocumentsTheRegisteredCodes checks the managed Eval code enum,
// the error details kinds, the error statuses and the codes the response
// descriptions name against the registry.
func TestOpenAPIDocumentsTheRegisteredCodes(t *testing.T) {
	document := loadOpenAPI(t)
	schemas := document.Components.Schemas

	evalError := schemas["EvalAPIError"]
	if evalError == nil || evalError.Value.Properties["code"] == nil {
		t.Fatal("OpenAPI has no EvalAPIError code")
	}
	documented := sortedStrings(evalError.Value.Properties["code"].Value.Enum)
	registered := registeredCodes(func(row *entry) bool { return strings.HasPrefix(row.code, "eval_") })
	if !slices.Equal(documented, registered) {
		t.Errorf("EvalAPIError code enum = %v, registry Eval codes = %v", documented, registered)
	}

	envelope := schemas["Error"]
	if envelope == nil || envelope.Value.Properties["details"] == nil ||
		len(envelope.Value.Properties["details"].Value.OneOf) == 0 {
		t.Fatal("OpenAPI Error has no details variants")
	}
	for _, variant := range envelope.Value.Properties["details"].Value.OneOf {
		kind := variant.Value.Properties["kind"]
		if kind == nil {
			t.Errorf("error details %s has no kind", variant.Ref)
			continue
		}
		value, _ := kind.Value.Const.(string)
		if _, ok := Lookup(value); !ok && !(variant.Ref == "#/components/schemas/EvalErrorDetails" && value == "eval") {
			t.Errorf("error details %s has kind %q, which is not a registered code", variant.Ref, value)
		}
	}

	// Every registered HTTP status is answered by some operation, and every
	// error status an operation documents carries a registered code.
	statuses := map[int]bool{}
	for _, row := range registry {
		if row.status != 0 {
			statuses[row.status] = false
		}
	}
	codeInDescription := regexp.MustCompile("`([a-z][a-z0-9_]*)`")
	checkDescription := func(where string, description *string) {
		if description == nil {
			return
		}
		for _, match := range codeInDescription.FindAllStringSubmatch(*description, -1) {
			if _, ok := Lookup(match[1]); !ok {
				t.Errorf("%s names `%s`, which is not a registered code", where, match[1])
			}
		}
	}
	for path, item := range document.Paths.Map() {
		for method, operation := range item.Operations() {
			for status, response := range operation.Responses.Map() {
				code, err := strconv.Atoi(status)
				if err != nil || code < 400 {
					continue
				}
				if _, ok := statuses[code]; !ok {
					t.Errorf("%s %s documents %d, which no registered code carries", method, path, code)
				}
				statuses[code] = true
				if response.Ref == "" && response.Value != nil {
					checkDescription(method+" "+path+" "+status, response.Value.Description)
				}
			}
		}
	}
	for status, documented := range statuses {
		if !documented {
			t.Errorf("registered status %d is documented by no operation", status)
		}
	}
	for name, response := range document.Components.Responses {
		if response.Value != nil {
			checkDescription("response "+name, response.Value.Description)
		}
	}
}

// TestEventSchemaErrorFrameCodesAreTheFrameCodes keeps the WebSocket error
// frame enum equal to the registered frame codes.
func TestEventSchemaErrorFrameCodesAreTheFrameCodes(t *testing.T) {
	content, err := os.ReadFile(eventSchema)
	if err != nil {
		t.Fatal(err)
	}
	var schema struct {
		Defs map[string]struct {
			Properties map[string]struct {
				Enum []any `json:"enum"`
			} `json:"properties"`
		} `json:"$defs"`
	}
	if err := json.Unmarshal(content, &schema); err != nil {
		t.Fatal(err)
	}
	frame, ok := schema.Defs["ErrorFrame"]
	if !ok {
		t.Fatal("event schema has no ErrorFrame")
	}
	documented := sortedStrings(frame.Properties["code"].Enum)
	registered := registeredCodes(func(row *entry) bool { return row.frame })
	if !slices.Equal(documented, registered) {
		t.Errorf("ErrorFrame code enum = %v, registry frame codes = %v", documented, registered)
	}
}

// TestBrowserClientComparesOnlyRegisteredCodes reads the browser client's
// production sources: every code it compares a public API error or error
// frame with, and every error details kind it parses, is registered, unless
// it is one of the client's own failure codes.
func TestBrowserClientComparesOnlyRegisteredCodes(t *testing.T) {
	comparison := regexp.MustCompile(`\b(?:error|err|frame|parsed)\??\.code\s*(?:===|!==|==|!=)\s*"([^"]+)"`)
	membership := regexp.MustCompile(`\[([^\[\]]*)\]\s*\.includes\(\s*code\s*\)`)
	literal := regexp.MustCompile(`"([^"]+)"`)
	detailsKind := regexp.MustCompile(`\bkind: "([^"]+)"`)
	local := map[string]bool{}
	for _, code := range uiLocalCodes {
		if _, ok := Lookup(code); ok {
			t.Errorf("browser client code %s is also a registered Server code", code)
		}
		local[code] = false
	}
	compared := map[string][]string{}
	err := filepath.WalkDir(browserSource, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		name := entry.Name()
		if entry.IsDir() {
			if name == "generated" || name == "test" {
				return filepath.SkipDir
			}
			return nil
		}
		if !(strings.HasSuffix(name, ".ts") || strings.HasSuffix(name, ".tsx")) ||
			strings.Contains(name, ".test.") {
			return nil
		}
		content, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		text := string(content)
		for code := range local {
			if strings.Contains(text, `"`+code+`"`) {
				local[code] = true
			}
		}
		for _, match := range comparison.FindAllStringSubmatch(text, -1) {
			compared[match[1]] = append(compared[match[1]], path)
		}
		for _, match := range membership.FindAllStringSubmatch(text, -1) {
			for _, code := range literal.FindAllStringSubmatch(match[1], -1) {
				compared[code[1]] = append(compared[code[1]], path)
			}
		}
		if filepath.ToSlash(path) == browserSource+"/api/error.ts" {
			for _, match := range detailsKind.FindAllStringSubmatch(text, -1) {
				compared[match[1]] = append(compared[match[1]], path)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	// A scan that finds nothing proves nothing: the patterns must still match
	// the client's comparisons.
	if len(compared) == 0 {
		t.Fatal("the browser client scan found no error code comparison")
	}
	for _, code := range slices.Sorted(maps.Keys(compared)) {
		_, registered := Lookup(code)
		if _, own := local[code]; !registered && !own {
			t.Errorf("%v compare an error with %q, which is not a registered code", compared[code], code)
		}
	}
	for code, used := range local {
		if !used {
			t.Errorf("browser client code %s is no longer used; remove it from uiLocalCodes", code)
		}
	}
}
