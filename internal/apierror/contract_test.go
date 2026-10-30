package apierror

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
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
	eventsSource  = "../httpapi/public/events"
)

// privateAPISources are the private APIs that write their own error codes:
// the allocation-scoped Artifact API and the Control Plane API.
var privateAPISources = []string{"../httpapi/privateartifacts", "../controlplane"}

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

// TestEventsSendExactlyTheFrameCodes derives the codes the events package
// sends in error frames from its sendError and endAll calls: every frame
// code is sent, and every sent code is a frame code.
func TestEventsSendExactlyTheFrameCodes(t *testing.T) {
	kinds := registeredKindNames(t)
	sent := map[string][]string{}
	paths, err := filepath.Glob(filepath.Join(eventsSource, "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range paths {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		set := token.NewFileSet()
		file, err := parser.ParseFile(set, path, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(file, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			selector, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			argument := map[string]int{"sendError": 1, "endAll": 0}
			index, ok := argument[selector.Sel.Name]
			if !ok || index >= len(call.Args) {
				return true
			}
			where := set.Position(call.Pos()).String()
			kind, ok := call.Args[index].(*ast.SelectorExpr)
			if !ok || !isIdent(kind.X, "apierror") || kinds[kind.Sel.Name] == "" {
				t.Errorf("%s: an error frame must name a registered apierror Kind", where)
				return true
			}
			sent[kinds[kind.Sel.Name]] = append(sent[kinds[kind.Sel.Name]], where)
			return true
		})
	}
	if len(sent) == 0 {
		t.Fatal("the events scan found no error frame")
	}
	frames := registeredCodes(func(row *entry) bool { return row.frame })
	if codes := slices.Sorted(maps.Keys(sent)); !slices.Equal(codes, frames) {
		t.Errorf("the events package sends error frames %v, registry frame codes = %v", codes, frames)
	}
}

// TestPrivateAPIsAnswerSharedCodesLikeTheRegistry reads every error the
// private APIs write: a code that is also a public code keeps the registry's
// HTTP status and retryable flag there too (ERR-R14). Private codes stay
// outside the registry.
func TestPrivateAPIsAnswerSharedCodesLikeTheRegistry(t *testing.T) {
	statuses := httpStatusConstants(t)
	shared := 0
	for _, source := range privateAPISources {
		paths, err := filepath.Glob(filepath.Join(source, "*.go"))
		if err != nil {
			t.Fatal(err)
		}
		written := 0
		for _, path := range paths {
			if strings.HasSuffix(path, "_test.go") {
				continue
			}
			set := token.NewFileSet()
			file, err := parser.ParseFile(set, path, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			ast.Inspect(file, func(node ast.Node) bool {
				call, ok := node.(*ast.CallExpr)
				if !ok || len(call.Args) != 5 {
					return true
				}
				switch name := calledName(call.Fun); name {
				case "writeError", "writePrivateError":
				default:
					return true
				}
				written++
				where := set.Position(call.Pos()).String()
				if fromOneKind(call.Args) {
					return true
				}
				literal, ok := call.Args[2].(*ast.BasicLit)
				if !ok || literal.Kind != token.STRING {
					t.Errorf("%s: a private error code must be a literal", where)
					return true
				}
				code, _ := strconv.Unquote(literal.Value)
				kind, public := Lookup(code)
				if !public {
					return true
				}
				shared++
				status, retryable := -1, ""
				if selector, ok := call.Args[1].(*ast.SelectorExpr); ok && isIdent(selector.X, "http") {
					status = statuses[selector.Sel.Name]
				}
				if flag, ok := call.Args[4].(*ast.Ident); ok && (flag.Name == "true" || flag.Name == "false") {
					retryable = flag.Name
				}
				if status != kind.Status() || retryable != strconv.FormatBool(kind.Retryable()) {
					t.Errorf("%s: private %s answers status %d, retryable %q; the registry gives %d, retryable %t",
						where, code, status, retryable, kind.Status(), kind.Retryable())
				}
				return true
			})
		}
		if written == 0 {
			t.Errorf("the scan of %s found no private error", source)
		}
	}
	if shared == 0 {
		t.Fatal("the private API scan found no shared public code")
	}
}

// fromOneKind reports an error written as kind.Status(), kind.Code(), any
// message and kind.Retryable() of one Kind value, which agrees with the
// registry by construction.
func fromOneKind(arguments []ast.Expr) bool {
	receiver := func(expression ast.Expr, method string) string {
		call, ok := expression.(*ast.CallExpr)
		if !ok || len(call.Args) != 0 {
			return ""
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || selector.Sel.Name != method {
			return ""
		}
		ident, ok := selector.X.(*ast.Ident)
		if !ok {
			return ""
		}
		return ident.Name
	}
	kind := receiver(arguments[2], "Code")
	return kind != "" && receiver(arguments[1], "Status") == kind && receiver(arguments[4], "Retryable") == kind
}

// httpStatusConstants reads the net/http status constants of the Go
// toolchain that builds the Server.
func httpStatusConstants(t *testing.T) map[string]int {
	t.Helper()
	file, err := parser.ParseFile(token.NewFileSet(), filepath.Join(runtime.GOROOT(), "src", "net", "http", "status.go"), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	statuses := map[string]int{}
	ast.Inspect(file, func(node ast.Node) bool {
		value, ok := node.(*ast.ValueSpec)
		if !ok {
			return true
		}
		for index, name := range value.Names {
			if index < len(value.Values) {
				if literal, ok := value.Values[index].(*ast.BasicLit); ok && literal.Kind == token.INT {
					statuses[name.Name], _ = strconv.Atoi(literal.Value)
				}
			}
		}
		return true
	})
	if statuses["StatusConflict"] != 409 {
		t.Fatal("net/http status constants were not found")
	}
	return statuses
}

func calledName(function ast.Expr) string {
	switch value := function.(type) {
	case *ast.Ident:
		return value.Name
	case *ast.SelectorExpr:
		return value.Sel.Name
	}
	return ""
}

// registeredKindNames maps each exported Kind variable of the registry to its
// code, read from the first argument of its registration call.
func registeredKindNames(t *testing.T) map[string]string {
	t.Helper()
	paths, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	names := map[string]string{}
	for _, path := range paths {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(file, func(node ast.Node) bool {
			value, ok := node.(*ast.ValueSpec)
			if !ok {
				return true
			}
			for index, name := range value.Names {
				if index >= len(value.Values) || !name.IsExported() {
					continue
				}
				call, ok := value.Values[index].(*ast.CallExpr)
				if !ok || len(call.Args) == 0 {
					continue
				}
				if literal, ok := call.Args[0].(*ast.BasicLit); ok && literal.Kind == token.STRING {
					code, _ := strconv.Unquote(literal.Value)
					if _, registered := Lookup(code); registered {
						names[name.Name] = code
					}
				}
			}
			return true
		})
	}
	if names["NotFound"] != "not_found" {
		t.Fatal("the registry scan did not find the NotFound Kind")
	}
	return names
}

func isIdent(expression ast.Expr, name string) bool {
	ident, ok := expression.(*ast.Ident)
	return ok && ident.Name == name
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
