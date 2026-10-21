package public

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// contextFiles assigns every production file of this package to the context
// whose handler struct its methods use. A new file must be classified here or
// in sharedFiles.
var contextFiles = map[string][]string{
	"sessionHandlers": {"auth_handlers.go", "event_handlers.go"},
	"catalogHandlers": {
		"workflow_handlers.go", "configuration_handlers.go", "agent_instructions.go",
		"agent_template_usage.go", "catalog_discovery.go",
	},
	"projectHandlers":   {"project_handlers.go"},
	"runHandlers":       {"run_handlers.go", "run_recovery.go"},
	"runRepeatHandlers": {"run_repeat_handlers.go"},
	"queueHandlers":     {"queue_handlers.go"},
	"artifactHandlers": {
		"artifact_handlers.go", "project_artifact_handlers.go", "artifact_archive_handlers.go",
		"git_import_handlers.go",
	},
	"auditHandlers": {
		"audit_handlers.go", "audit_models.go", "audit_preparation.go", "audit_review_handlers.go",
		"audit_standard_handlers.go", "audit_event_handlers.go", "audit_types.go", "owner_audit_lists.go", "audit_artifact_handlers.go",
	},
	"findingHandlers": {"finding_handlers.go", "finding_collection_handlers.go"},
	"evalHandlers": {
		"eval_handlers.go", "eval_helpers.go", "eval_records.go", "eval_comparison.go", "eval_bins.go",
		"eval_types.go",
	},
	"operationsHandlers":           {"operations_handlers.go", "performance_handlers.go", "scheduler_settings_handlers.go", "model_selection_handlers.go"},
	"credentialHandlers":           {"credential_handlers.go"},
	"runtimeConfigurationHandlers": {"runtime_configuration_handlers.go", "runtime_agent_principal_handlers.go"},
	"gitKeyHandlers":               {"git_key_handlers.go"},
}

// sharedFiles hold the transport, the request, response and error helpers and
// the read models every context may use. Only the composition files among them
// name context handler structs.
var sharedFiles = []string{
	"artifact_response.go", "authorization.go", "error_diagnostics.go", "errors.go", "pagination.go", "request.go",
	"types.go", "types_configuration.go", "types_operations.go", "types_run_project.go", "version.go",
}

// compositionFiles assemble the contexts and register their routes.
var compositionFiles = []string{"router.go", "routes.go"}

// TestContextFilesReachOnlyTheirOwnDependencies keeps every context on its own
// handler struct: a context file declares methods only on its own struct and
// names no other context's struct, dependencies or helpers; every dependency
// a context declares is used by its files, and every method of a dependency
// interface the package declares is called, so a context's dependencies stay
// as narrow as its routes.
func TestContextFilesReachOnlyTheirOwnDependencies(t *testing.T) {
	pkg := parsePublicPackage(t)
	owner := classifyPublicFiles(t, pkg)

	// The context of each handler struct and dependency type, and the file
	// that declares every package-level name.
	contextTypes := map[string]string{}
	for context := range contextFiles {
		contextTypes[context] = context
		dependencies, ok := pkg.dependencyTypeOf(context)
		if !ok {
			t.Errorf("%s embeds no dependencies type declared in its own files", context)
			continue
		}
		contextTypes[dependencies] = context
	}
	for _, aggregate := range []string{"Dependencies", "contextHandlers"} {
		contextTypes[aggregate] = "composition"
	}

	for name, file := range pkg.files {
		context := owner[name]
		if context == "composition" {
			continue
		}
		ast.Inspect(file, func(node ast.Node) bool {
			switch current := node.(type) {
			case *ast.FuncDecl:
				if current.Recv == nil || len(current.Recv.List) == 0 {
					return true
				}
				receiver := receiverType(current.Recv.List[0].Type)
				if receiverContext, ok := contextTypes[receiver]; ok && receiverContext != context {
					t.Errorf("%s declares %s on %s, a type of the %s context", name, current.Name.Name, receiver, receiverContext)
				}
			case *ast.Ident:
				if pkg.isSelector(current) {
					return true
				}
				if referenced, ok := contextTypes[current.Name]; ok && referenced != context {
					t.Errorf("%s:%d names %s of the %s context", name, pkg.line(current), current.Name, referenced)
				}
				declaringFile, declared := pkg.declaredIn[current.Name]
				if declaringContext := owner[declaringFile]; declared && declaringContext != context &&
					declaringContext != "shared" {
					t.Errorf("%s:%d uses %s, which the %s context declares in %s",
						name, pkg.line(current), current.Name, declaringContext, declaringFile)
				}
			}
			return true
		})
	}

	// Every field of a handler struct is used by its context, and every method
	// of a package-declared dependency interface is called.
	for context, files := range contextFiles {
		fields := pkg.handlerFields(context)
		if len(fields) == 0 {
			t.Errorf("%s declares no dependency", context)
		}
		used := map[string]map[string]bool{}
		for _, name := range files {
			for field, methods := range pkg.fieldUse(name, context) {
				if used[field] == nil {
					used[field] = map[string]bool{}
				}
				for method := range methods {
					used[field][method] = true
				}
			}
		}
		for _, field := range fields {
			methods, ok := used[field.name]
			if !ok {
				t.Errorf("%s never uses its dependency %s", context, field.name)
				continue
			}
			for _, method := range pkg.interfaceMethods(field.typeName) {
				if !methods[method] && !methods["*"] {
					t.Errorf("%s depends on %s.%s through %s but never calls it", context, field.typeName, method, field.name)
				}
			}
		}
	}
}

// TestNewHandlerRequiresEveryService removes each service of each context in
// turn: NewHandler refuses every incomplete set and names the missing service,
// so no handler needs a nil check.
func TestNewHandlerRequiresEveryService(t *testing.T) {
	var services testServices
	fixture := newHandlerFixtureWithAuth(
		t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(s *testServices) { services = *s },
	)
	services.RunCreator = newTestRunCreator(t, services, fixture.runs, fixture.unit, true)
	complete := services.dependencies()
	if _, err := NewHandler(complete); err != nil {
		t.Fatalf("complete dependencies rejected: %v", err)
	}
	aggregate := reflect.ValueOf(complete)
	checked := 0
	for group := range aggregate.NumField() {
		groupField := aggregate.Type().Field(group)
		if groupField.Type.Kind() != reflect.Struct || !strings.HasSuffix(groupField.Type.Name(), "Dependencies") {
			continue
		}
		for field := range groupField.Type.NumField() {
			service := groupField.Type.Field(field)
			switch service.Type.Kind() {
			case reflect.Interface, reflect.Pointer, reflect.Func, reflect.Map, reflect.Slice, reflect.Chan:
			default:
				continue
			}
			candidate := complete
			value := reflect.ValueOf(&candidate).Elem().Field(group).Field(field)
			value.Set(reflect.Zero(value.Type()))
			_, err := NewHandler(candidate)
			if err == nil || !strings.Contains(err.Error(), " "+service.Name+" is missing") {
				t.Errorf("NewHandler without %s.%s = %v", groupField.Name, service.Name, err)
			}
			checked++
		}
	}
	if checked < 50 {
		t.Fatalf("checked %d services; the dependency groups were not found", checked)
	}
	for name, remove := range map[string]func(*Dependencies){
		"NewRequestID":   func(d *Dependencies) { d.NewRequestID = nil },
		"Logger":         func(d *Dependencies) { d.Logger = nil },
		"BrowserOrigins": func(d *Dependencies) { d.Session.BrowserOrigins = auth.OriginPolicy{} },
		"BearerToken":    func(d *Dependencies) { d.BearerToken = contracts.NewSecretString("") },
	} {
		candidate := complete
		remove(&candidate)
		if _, err := NewHandler(candidate); err == nil {
			t.Errorf("NewHandler without %s succeeded", name)
		}
	}
}

type publicPackage struct {
	fileSet    *token.FileSet
	files      map[string]*ast.File
	declaredIn map[string]string
	types      map[string]*ast.TypeSpec
	selectors  map[*ast.Ident]bool
}

func parsePublicPackage(t *testing.T) publicPackage {
	t.Helper()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	pkg := publicPackage{
		fileSet: token.NewFileSet(), files: map[string]*ast.File{}, declaredIn: map[string]string{},
		types: map[string]*ast.TypeSpec{}, selectors: map[*ast.Ident]bool{},
	}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || filepath.Ext(name) != ".go" || strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(pkg.fileSet, name, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		pkg.files[name] = file
		for _, declaration := range file.Decls {
			switch current := declaration.(type) {
			case *ast.FuncDecl:
				if current.Recv == nil {
					pkg.declaredIn[current.Name.Name] = name
				}
			case *ast.GenDecl:
				for _, spec := range current.Specs {
					switch value := spec.(type) {
					case *ast.TypeSpec:
						pkg.declaredIn[value.Name.Name] = name
						pkg.types[value.Name.Name] = value
					case *ast.ValueSpec:
						for _, identifier := range value.Names {
							pkg.declaredIn[identifier.Name] = name
						}
					}
				}
			}
		}
		ast.Inspect(file, func(node ast.Node) bool {
			switch current := node.(type) {
			case *ast.SelectorExpr:
				pkg.selectors[current.Sel] = true
			case *ast.KeyValueExpr:
				if key, ok := current.Key.(*ast.Ident); ok {
					pkg.selectors[key] = true
				}
			case *ast.Field:
				// Struct fields, interface methods, parameters and results.
				for _, name := range current.Names {
					pkg.selectors[name] = true
				}
			}
			return true
		})
	}
	if len(pkg.files) == 0 {
		t.Fatal("no production files parsed")
	}
	return pkg
}

// classifyPublicFiles maps each production file to its context, "shared" or
// "composition", and fails for an unclassified or doubly classified file.
func classifyPublicFiles(t *testing.T, pkg publicPackage) map[string]string {
	t.Helper()
	owner := map[string]string{}
	assign := func(name, context string) {
		if previous, ok := owner[name]; ok {
			t.Errorf("%s is classified as both %s and %s", name, previous, context)
		}
		if _, ok := pkg.files[name]; !ok {
			t.Errorf("classified file %s does not exist", name)
		}
		owner[name] = context
	}
	for context, files := range contextFiles {
		for _, name := range files {
			assign(name, context)
		}
	}
	for _, name := range sharedFiles {
		assign(name, "shared")
	}
	for _, name := range compositionFiles {
		assign(name, "composition")
	}
	for name := range pkg.files {
		if _, ok := owner[name]; !ok {
			t.Errorf("%s belongs to no context; classify it in contextFiles, sharedFiles or compositionFiles", name)
		}
	}
	return owner
}

func (p publicPackage) line(node ast.Node) int {
	return p.fileSet.Position(node.Pos()).Line
}

// isSelector reports an identifier that names a field, method, parameter or
// struct key rather than a package-level declaration.
func (p publicPackage) isSelector(identifier *ast.Ident) bool {
	return p.selectors[identifier]
}

// dependencyTypeOf names the dependencies struct a handler struct embeds.
func (p publicPackage) dependencyTypeOf(context string) (string, bool) {
	spec, ok := p.types[context]
	if !ok {
		return "", false
	}
	structure, ok := spec.Type.(*ast.StructType)
	if !ok {
		return "", false
	}
	for _, field := range structure.Fields.List {
		if len(field.Names) == 0 && strings.HasSuffix(typeName(field.Type), "Dependencies") {
			name := typeName(field.Type)
			return name, slices.Contains(contextFiles[context], p.declaredIn[name])
		}
	}
	return "", false
}

type handlerField struct{ name, typeName string }

// handlerFields lists the fields a context's handlers can reach: those of its
// embedded dependencies and its own other fields.
func (p publicPackage) handlerFields(context string) []handlerField {
	var result []handlerField
	var collect func(string)
	collect = func(typeName string) {
		spec, ok := p.types[typeName]
		if !ok {
			return
		}
		structure, ok := spec.Type.(*ast.StructType)
		if !ok {
			return
		}
		for _, field := range structure.Fields.List {
			fieldType := typeNameOf(field.Type)
			if len(field.Names) == 0 {
				collect(fieldType)
				continue
			}
			for _, name := range field.Names {
				result = append(result, handlerField{name: name.Name, typeName: fieldType})
			}
		}
	}
	collect(context)
	sort.Slice(result, func(i, j int) bool { return result[i].name < result[j].name })
	return result
}

// fieldUse lists, for each handler field a file reaches through the receiver
// of a method of context, the methods it calls on it. A field handed to a
// function of this package counts as calling every method of that
// parameter's interface; a field used otherwise is recorded with "*" only
// when it is not an interface declared by this package.
func (p publicPackage) fieldUse(name, context string) map[string]map[string]bool {
	used := map[string]map[string]bool{}
	mark := func(field, method string) {
		if used[field] == nil {
			used[field] = map[string]bool{}
		}
		if method != "" {
			used[field][method] = true
		}
	}
	parameters := p.parameterTypes()
	for _, declaration := range p.files[name].Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if !ok || function.Recv == nil || len(function.Recv.List[0].Names) == 0 ||
			receiverType(function.Recv.List[0].Type) != context {
			continue
		}
		receiver := function.Recv.List[0].Names[0].Name
		fieldOf := func(expression ast.Expr) string {
			selector, ok := expression.(*ast.SelectorExpr)
			if !ok {
				return ""
			}
			if base, ok := selector.X.(*ast.Ident); ok && base.Name == receiver {
				return selector.Sel.Name
			}
			return ""
		}
		ast.Inspect(function.Body, func(node ast.Node) bool {
			switch current := node.(type) {
			case *ast.CallExpr:
				if selector, ok := current.Fun.(*ast.SelectorExpr); ok {
					if field := fieldOf(selector.X); field != "" {
						mark(field, selector.Sel.Name)
					}
				}
				callee := ""
				switch fun := current.Fun.(type) {
				case *ast.Ident:
					callee = fun.Name
				case *ast.SelectorExpr:
					if base, ok := fun.X.(*ast.Ident); ok && base.Name == receiver {
						callee = context + "." + fun.Sel.Name
					}
				case *ast.IndexExpr:
					if identifier, ok := fun.X.(*ast.Ident); ok {
						callee = identifier.Name
					}
				}
				for index, argument := range current.Args {
					field := fieldOf(argument)
					if field == "" {
						continue
					}
					mark(field, "")
					types := parameters[callee]
					if index < len(types) {
						for _, method := range p.interfaceMethods(types[index]) {
							mark(field, method)
						}
						if len(p.interfaceMethods(types[index])) == 0 {
							mark(field, "*")
						}
					} else {
						mark(field, "*")
					}
				}
			case *ast.SelectorExpr:
				if field := fieldOf(current); field != "" {
					mark(field, "")
				}
			}
			return true
		})
	}
	return used
}

// parameterTypes lists the parameter type names of every function and method
// of this package, methods keyed as receiver.method.
func (p publicPackage) parameterTypes() map[string][]string {
	result := map[string][]string{}
	for _, file := range p.files {
		for _, declaration := range file.Decls {
			function, ok := declaration.(*ast.FuncDecl)
			if !ok {
				continue
			}
			key := function.Name.Name
			if function.Recv != nil && len(function.Recv.List) != 0 {
				key = receiverType(function.Recv.List[0].Type) + "." + key
			}
			var types []string
			for _, parameter := range function.Type.Params.List {
				count := max(len(parameter.Names), 1)
				for range count {
					types = append(types, typeNameOf(parameter.Type))
				}
			}
			result[key] = types
		}
	}
	return result
}

// interfaceMethods lists the methods of an interface declared by this
// package, those of embedded package interfaces included; any other type has
// none.
func (p publicPackage) interfaceMethods(name string) []string {
	spec, ok := p.types[name]
	if !ok {
		return nil
	}
	structure, ok := spec.Type.(*ast.InterfaceType)
	if !ok {
		return nil
	}
	var methods []string
	for _, method := range structure.Methods.List {
		if len(method.Names) == 0 {
			methods = append(methods, p.interfaceMethods(typeNameOf(method.Type))...)
			continue
		}
		for _, identifier := range method.Names {
			methods = append(methods, identifier.Name)
		}
	}
	return methods
}

func receiverType(expression ast.Expr) string {
	if star, ok := expression.(*ast.StarExpr); ok {
		expression = star.X
	}
	if identifier, ok := expression.(*ast.Ident); ok {
		return identifier.Name
	}
	return ""
}

// typeNameOf names a package-local type expression, or returns "" for any
// other type.
func typeNameOf(expression ast.Expr) string {
	if identifier, ok := expression.(*ast.Ident); ok {
		return identifier.Name
	}
	return ""
}

func typeName(expression ast.Expr) string {
	switch current := expression.(type) {
	case *ast.Ident:
		return current.Name
	case *ast.StarExpr:
		return typeName(current.X)
	case *ast.SelectorExpr:
		return typeName(current.X) + "." + current.Sel.Name
	}
	return ""
}
