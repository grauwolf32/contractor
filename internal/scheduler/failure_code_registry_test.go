package scheduler

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// failureCodeSources are the packages whose code ends a Planner invocation
// or a StageExecution: the Planners and their Worker invocations, the Server
// operations and the Scheduler.
var failureCodeSources = []string{"../planner", "../operations", "."}

// constantSources declare the named string constants those packages use as
// failure codes.
var constantSources = []string{"../planner", "../operations", ".", "../runstore"}

// codeFields names, for each record that carries a Stage failure code to the
// public API, the field that holds the code: a Planner failure, a
// StageTermination, an execution report error (AttemptDiagnostic.code), a
// Planner telemetry span and a Run output publication result.
var codeFields = map[string]string{
	"Failure":                          "Code",
	"StageTermination":                 "Code",
	"ExecutionError":                   "Code",
	"PlannerSpanAttributes":            "ErrorCode",
	"RecordRunOutputPublicationParams": "ErrorCode",
}

// TestStageFailureCodesAreRegistered derives every code the Planners and the
// Scheduler write into a Planner failure, a StageTermination or another
// record of codeFields, and requires one errors.md registry row for each,
// owned by a requirement (ERR-R24). A code reaches those records as a
// literal or named constant in such a record, as the first argument of
// NewError, or through a helper that forwards one of its string parameters
// there.
func TestStageFailureCodesAreRegistered(t *testing.T) {
	codes := stageFailureCodes(t, failureCodeSources)
	// Each path by which a code reaches a record must keep being found, so a
	// refactoring cannot hide codes from the scan.
	for code, path := range map[string]string{
		"allocation_fence_failed":              "a Failure literal",
		"planner_cancelled":                    "NewError",
		"invalid_a2a_response":                 "NewError in a2a",
		"server_operation_timeout":             "a forwarding closure",
		"planner_execution_config_unavailable": "a forwarding method",
		"planner_tool_selection_invalid":       "a Failure literal wrapped into an ExecutionError",
		"allocation_report_unavailable":        "an ExecutionError literal",
		"publication_attempts_exhausted":       "an output publication result",
		"invalid_request":                      "an operation Failure literal",
	} {
		if len(codes[code]) == 0 {
			t.Errorf("the scan no longer finds %s, written through %s", code, path)
		}
	}
	registry, err := os.ReadFile("../../docs/spec/platform/errors.md")
	if err != nil {
		t.Fatal(err)
	}
	rows := registeredRows(t, string(registry))
	for _, code := range slices.Sorted(maps.Keys(codes)) {
		owners := rows[code]
		switch {
		case len(owners) != 1:
			t.Errorf("%s: %s has %d errors.md registry rows, want one", strings.Join(codes[code], ", "), code, len(owners))
		case !ownerPattern.MatchString(owners[0]):
			t.Errorf("%s: the errors.md row of %s names no owning requirement: %q", strings.Join(codes[code], ", "), code, owners[0])
		}
	}
}

// ownerPattern matches an Owner cell that links a requirement.
var ownerPattern = regexp.MustCompile(`\[[A-Z]+-R[0-9]+\]\(`)

// registeredRows maps each code of the errors.md Registry section to the
// Owner cells of its rows.
func registeredRows(t *testing.T, registry string) map[string][]string {
	t.Helper()
	_, section, found := strings.Cut(registry, "\n## Registry\n")
	if !found {
		t.Fatal("errors.md has no Registry section")
	}
	rows := map[string][]string{}
	for _, line := range strings.Split(section, "\n") {
		match := regexp.MustCompile("^\\| `([a-z][a-z0-9_]*)` \\|").FindStringSubmatch(line)
		if match == nil {
			continue
		}
		cells := strings.Split(strings.Trim(line, "| "), " | ")
		rows[match[1]] = append(rows[match[1]], cells[len(cells)-1])
	}
	return rows
}

type parsedFile struct {
	path string
	file *ast.File
	set  *token.FileSet
}

func parseTree(t *testing.T, roots []string) []parsedFile {
	t.Helper()
	var files []parsedFile
	for _, root := range roots {
		err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.IsDir() {
				if path != root && root == "." {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			set := token.NewFileSet()
			file, err := parser.ParseFile(set, path, nil, 0)
			if err != nil {
				return err
			}
			fillElidedTypes(file)
			files = append(files, parsedFile{path: path, file: file, set: set})
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	return files
}

// fillElidedTypes gives each composite literal whose type is elided inside a
// slice, array or map literal the element type of its container, so that
// []reporting.ExecutionError{{Code: ...}} is read like its explicit form.
func fillElidedTypes(file *ast.File) {
	ast.Inspect(file, func(node ast.Node) bool {
		container, ok := node.(*ast.CompositeLit)
		if !ok {
			return true
		}
		var element ast.Expr
		switch value := container.Type.(type) {
		case *ast.ArrayType:
			element = value.Elt
		case *ast.MapType:
			element = value.Value
		default:
			return true
		}
		if star, ok := element.(*ast.StarExpr); ok {
			element = star.X
		}
		for _, item := range container.Elts {
			if pair, ok := item.(*ast.KeyValueExpr); ok {
				item = pair.Value
			}
			if literal, ok := item.(*ast.CompositeLit); ok && literal.Type == nil {
				literal.Type = element
			}
		}
		return true
	})
}

// stageFailureCodes maps each code to the source positions that write it.
func stageFailureCodes(t *testing.T, roots []string) map[string][]string {
	t.Helper()
	constants := map[string]string{}
	for _, source := range parseTree(t, constantSources) {
		for _, declaration := range source.file.Decls {
			general, ok := declaration.(*ast.GenDecl)
			if !ok || general.Tok != token.CONST {
				continue
			}
			for _, spec := range general.Specs {
				value := spec.(*ast.ValueSpec)
				for index, name := range value.Names {
					if index < len(value.Values) {
						if literal, ok := value.Values[index].(*ast.BasicLit); ok && literal.Kind == token.STRING {
							text, _ := strconv.Unquote(literal.Value)
							constants[name.Name] = text
						}
					}
				}
			}
		}
	}
	files := parseTree(t, roots)
	// forwarders maps a helper's name to the index of the string parameter it
	// writes as a failure code; NewError is the first.
	forwarders := map[string]int{"NewError": 0}
	codeTarget := func(node ast.Node, parameter string) bool {
		switch value := node.(type) {
		case *ast.CompositeLit:
			name := codeField(value.Type)
			if name == "" {
				return false
			}
			for _, element := range value.Elts {
				if field, ok := element.(*ast.KeyValueExpr); ok && isIdent(field.Key, name) && isIdent(field.Value, parameter) {
					return true
				}
			}
		case *ast.CallExpr:
			if index, ok := forwarders[calledName(value.Fun)]; ok && index < len(value.Args) && isIdent(value.Args[index], parameter) {
				return true
			}
		}
		return false
	}
	for changed := true; changed; {
		changed = false
		for _, source := range files {
			ast.Inspect(source.file, func(node ast.Node) bool {
				name, function := "", (*ast.FuncType)(nil)
				var body ast.Node
				switch value := node.(type) {
				case *ast.FuncDecl:
					name, function, body = value.Name.Name, value.Type, value.Body
				case *ast.AssignStmt:
					if len(value.Lhs) == 1 && len(value.Rhs) == 1 {
						if literal, ok := value.Rhs[0].(*ast.FuncLit); ok {
							if ident, ok := value.Lhs[0].(*ast.Ident); ok {
								name, function, body = ident.Name, literal.Type, literal.Body
							}
						}
					}
				}
				if function == nil || body == nil {
					return true
				}
				if _, known := forwarders[name]; known {
					return true
				}
				index := 0
				for _, field := range function.Params.List {
					for _, parameter := range field.Names {
						if isIdent(field.Type, "string") {
							forwards := false
							ast.Inspect(body, func(inner ast.Node) bool {
								forwards = forwards || codeTarget(inner, parameter.Name)
								return !forwards
							})
							if forwards {
								forwarders[name] = index
								changed = true
								return true
							}
						}
						index++
					}
					if len(field.Names) == 0 {
						index++
					}
				}
				return true
			})
		}
	}
	codes := map[string][]string{}
	for _, source := range files {
		record := func(expression ast.Expr) {
			code := ""
			switch value := expression.(type) {
			case *ast.BasicLit:
				if value.Kind == token.STRING {
					code, _ = strconv.Unquote(value.Value)
				}
			case *ast.Ident:
				code = constants[value.Name]
			case *ast.SelectorExpr:
				code = constants[value.Sel.Name]
			}
			if code != "" {
				position := source.set.Position(expression.Pos())
				codes[code] = append(codes[code], filepath.ToSlash(position.Filename)+":"+strconv.Itoa(position.Line))
			}
		}
		ast.Inspect(source.file, func(node ast.Node) bool {
			switch value := node.(type) {
			case *ast.CompositeLit:
				if name := codeField(value.Type); name != "" {
					for _, element := range value.Elts {
						if field, ok := element.(*ast.KeyValueExpr); ok && isIdent(field.Key, name) {
							record(field.Value)
						}
					}
				}
			case *ast.CallExpr:
				if index, ok := forwarders[calledName(value.Fun)]; ok && index < len(value.Args) {
					record(value.Args[index])
				}
			}
			return true
		})
	}
	return codes
}

// codeField returns the code field of a record type of codeFields, or "".
func codeField(expression ast.Expr) string {
	switch value := expression.(type) {
	case *ast.Ident:
		return codeFields[value.Name]
	case *ast.SelectorExpr:
		return codeFields[value.Sel.Name]
	}
	return ""
}

func isIdent(expression ast.Expr, name string) bool {
	ident, ok := expression.(*ast.Ident)
	return ok && ident.Name == name
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
