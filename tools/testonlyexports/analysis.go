package main

import (
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"path/filepath"
	"slices"
	"strings"

	"golang.org/x/tools/go/packages"
)

// Finding is an exported identifier that no production code references.
type Finding struct {
	// Key names the identifier relative to the module path:
	// internal/runstore.PostgresStore, internal/runstore.PostgresStore.GetRun
	// for a method, or internal/scheduler.Allocator.GetGrant for the method of
	// an interface.
	Key string
	// Kind is func, method, interface method, type, const or var.
	Kind string
	// Position is the declaration, with a module-relative file name.
	Position token.Position
	// TestRefs counts the references from test code; zero means that nothing
	// refers to the identifier.
	TestRefs int
	// TestFiles are the module-relative files of those references, sorted.
	TestFiles []string
}

// HelperImport is a production package that imports a test helper package,
// which breaks the allowlist rule: production code must not depend on code
// that the check treats as test code.
type HelperImport struct {
	// Importer and Helper are package paths relative to the module path.
	Importer, Helper string
}

// Result is what Analyze reports.
type Result struct {
	Findings      []Finding
	HelperImports []HelperImport
}

// Options select the module that Analyze loads and the build tags of its
// tests.
type Options struct {
	// Dir is the root directory of the module; Analyze loads Dir/... .
	Dir string
	// Tags are build tags that select further test files, such as
	// integration and e2e.
	Tags []string
}

// dynamicMethodNames are methods that the standard library finds through
// anonymous interfaces, which export data does not describe: errors.Is,
// errors.As and errors.Unwrap look for Unwrap, Is and As; os.IsTimeout and
// net look for Timeout and Temporary. A method of one of these names is
// never reported.
var dynamicMethodNames = map[string]bool{
	"As": true, "Is": true, "Temporary": true, "Timeout": true, "Unwrap": true,
}

// candidate is an exported declaration in production code of the module.
type candidate struct {
	key      string
	kind     string
	obj      types.Object
	position token.Position
	// extents are the source ranges of the declaration itself. A reference
	// inside them, such as a recursive call or a method's receiver, is not a
	// use; a type's extents include its methods.
	extents   []extent
	prodRefs  int
	testRefs  int
	testFiles map[string]bool
}

type extent struct {
	file       string
	start, end int
}

func (e extent) contains(position token.Position) bool {
	return e.file == position.Filename && e.start <= position.Offset && position.Offset < e.end
}

type analysis struct {
	module     string
	dir        string
	candidates map[string]*candidate
	// assertions are the compile-time interface checks of production code,
	// var _ I = (*T)(nil): they keep a type honest but use neither I nor T.
	assertions []extent
	// interfaceMethods are the methods of every interface, named or
	// anonymous, that production code of the module declares.
	interfaceMethods []*types.Func
	// usedInterfaceMethods holds the interface methods that production code
	// calls, directly or through a wider interface.
	usedInterfaceMethods map[*types.Func]bool
	// namedTypes are the named non-interface types that production code of
	// the module declares at package level.
	namedTypes []*types.Named
	// owners maps a method to the named types whose method sets hold it.
	owners map[*types.Func][]*types.Named
}

// Analyze loads every package of the module with its tests and returns the
// exported identifiers that production code does not reference, sorted by
// key, and the production packages that import a test helper package.
//
// Production code is every file that is not test code. Test code is a
// _test.go file, a file of a test helper package, whose name ends in "test"
// (like artifactstest), or a file under the module's tests/ directory, which
// holds the process tests. Declarations in test code and in generated files
// (the "Code generated ... DO NOT EDIT." header) are not reported, but
// generated files are production code: their references count.
//
// Reported are exported package-level functions, types, constants and
// variables, the exported methods of exported types, and the exported
// methods of exported interfaces. Struct fields are not: encoding/json and
// database scans use them without a reference. A method counts as used when
// a production reference reaches it, or when its type, or a type that embeds
// it, implements an interface whose method of that name is used: every
// interface declared outside the module, the error type, and every interface
// of the module whose method production code calls.
func Analyze(options Options) (Result, error) {
	dir, err := filepath.Abs(options.Dir)
	if err != nil {
		return Result{}, err
	}
	config := &packages.Config{
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedImports | packages.NeedTypes |
			packages.NeedTypesInfo | packages.NeedSyntax | packages.NeedModule,
		Dir:   dir,
		Tests: true,
	}
	if len(options.Tags) > 0 {
		config.BuildFlags = []string{"-tags=" + strings.Join(options.Tags, ",")}
	}
	loaded, err := packages.Load(config, "./...")
	if err != nil {
		return Result{}, err
	}
	var problems []string
	packages.Visit(loaded, nil, func(p *packages.Package) {
		for _, e := range p.Errors {
			problems = append(problems, e.Error())
		}
	})
	if len(problems) > 0 {
		return Result{}, fmt.Errorf("load packages:\n%s", strings.Join(problems, "\n"))
	}
	a := &analysis{
		dir:                  dir,
		candidates:           map[string]*candidate{},
		usedInterfaceMethods: map[*types.Func]bool{},
	}
	var production, variants []*packages.Package
	for _, p := range loaded {
		if p.Module != nil && p.Module.Main && a.module == "" {
			a.module = p.Module.Path
		}
		switch {
		case p.Name == "main" && strings.HasSuffix(p.PkgPath, ".test"):
			// The generated test main refers to nothing of interest.
		case p.ID == p.PkgPath:
			production = append(production, p)
		default:
			// A test variant, "p [p.test]" or "p_test [p.test]".
			variants = append(variants, p)
		}
	}
	if a.module == "" {
		return Result{}, errors.New("no package of the main module was loaded")
	}
	for _, p := range production {
		a.collect(p)
	}
	for _, p := range production {
		a.count(p, false)
	}
	for _, p := range variants {
		a.count(p, true)
	}
	used := a.usedInterfaces(production)
	findings := []Finding{}
	for _, c := range a.candidates {
		if c.prodRefs > 0 || a.satisfiesUsedInterface(c, used) {
			continue
		}
		files := make([]string, 0, len(c.testFiles))
		for file := range c.testFiles {
			files = append(files, a.relative(file))
		}
		slices.Sort(files)
		position := c.position
		position.Filename = a.relative(position.Filename)
		findings = append(findings, Finding{
			Key: c.key, Kind: c.kind, Position: position, TestRefs: c.testRefs, TestFiles: files,
		})
	}
	slices.SortFunc(findings, func(x, y Finding) int { return strings.Compare(x.Key, y.Key) })
	return Result{Findings: findings, HelperImports: a.helperImports(production)}, nil
}

// helperImports returns the imports of a test helper package by production
// packages, sorted.
func (a *analysis) helperImports(production []*packages.Package) []HelperImport {
	result := []HelperImport{}
	for _, p := range production {
		if len(p.GoFiles) == 0 || a.testCode(p, p.GoFiles[0]) {
			continue
		}
		importer, _ := a.relativePackage(p.PkgPath)
		for path, imported := range p.Imports {
			if helper, ok := a.relativePackage(path); ok && strings.HasSuffix(imported.Name, "test") {
				result = append(result, HelperImport{Importer: importer, Helper: helper})
			}
		}
	}
	slices.SortFunc(result, func(x, y HelperImport) int {
		return strings.Compare(x.Importer+" "+x.Helper, y.Importer+" "+y.Helper)
	})
	return result
}

// relativePackage returns a package path of the module relative to the
// module path, "." for the root package, and false for another module.
func (a *analysis) relativePackage(path string) (string, bool) {
	if path != a.module && !strings.HasPrefix(path, a.module+"/") {
		return "", false
	}
	relative := strings.TrimPrefix(strings.TrimPrefix(path, a.module), "/")
	if relative == "" {
		relative = "."
	}
	return relative, true
}

// relative returns file relative to the module root, with forward slashes.
func (a *analysis) relative(file string) string {
	relative, err := filepath.Rel(a.dir, file)
	if err != nil {
		return file
	}
	return filepath.ToSlash(relative)
}

// testCode reports whether a file of package p is test code.
func (a *analysis) testCode(p *packages.Package, file string) bool {
	if strings.HasSuffix(file, "_test.go") || strings.HasSuffix(p.Name, "test") {
		return true
	}
	return strings.HasPrefix(a.relative(file)+"/", "tests/")
}

// collect records the exported declarations in the production files of p,
// and the interface methods and named types that interface satisfaction
// needs.
func (a *analysis) collect(p *packages.Package) {
	methods := map[string][]extent{}
	var named []*candidate
	for _, file := range p.Syntax {
		name := p.Fset.Position(file.Package).Filename
		if a.testCode(p, name) {
			continue
		}
		generated := ast.IsGenerated(file)
		for _, decl := range file.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				kind := "func"
				if d.Recv != nil {
					kind = "method"
					base := receiverBase(d.Recv.List[0].Type)
					methods[base] = append(methods[base], a.extentOf(p, d))
					if !ast.IsExported(base) {
						// Only an interface reaches a method of an unexported
						// type from another package.
						continue
					}
				}
				if !generated && d.Name.IsExported() {
					a.add(p, d.Name, kind, a.extentOf(p, d))
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					if c := a.collectSpec(p, spec, d.Tok, generated); c != nil && c.kind == "type" {
						named = append(named, c)
					}
				}
			}
		}
	}
	for _, c := range named {
		c.extents = append(c.extents, methods[c.obj.Name()]...)
	}
	for id, obj := range p.TypesInfo.Defs {
		if obj == nil || a.testCode(p, p.Fset.Position(id.Pos()).Filename) {
			continue
		}
		switch o := obj.(type) {
		case *types.Func:
			if recv := o.Type().(*types.Signature).Recv(); recv != nil && types.IsInterface(recv.Type()) {
				a.interfaceMethods = append(a.interfaceMethods, o)
			}
		case *types.TypeName:
			if named, ok := o.Type().(*types.Named); ok && !types.IsInterface(named) && packageLevel(o) {
				a.namedTypes = append(a.namedTypes, named)
			}
		}
	}
}

// collectSpec records the exported names of one type, const or var spec and
// returns the candidate of a type spec. A compile-time interface check is
// recorded as an assertion instead.
func (a *analysis) collectSpec(p *packages.Package, spec ast.Spec, tok token.Token, generated bool) *candidate {
	switch s := spec.(type) {
	case *ast.TypeSpec:
		if generated || !s.Name.IsExported() {
			return nil
		}
		c := a.add(p, s.Name, "type", a.extentOf(p, s))
		if iface, ok := s.Type.(*ast.InterfaceType); ok {
			for _, field := range iface.Methods.List {
				for _, method := range field.Names {
					if method.IsExported() {
						a.add(p, method, "interface method", a.extentOf(p, field))
					}
				}
			}
		}
		return c
	case *ast.ValueSpec:
		if s.Type != nil && len(s.Names) == 1 && s.Names[0].Name == "_" {
			a.assertions = append(a.assertions, a.extentOf(p, s))
			return nil
		}
		if generated {
			return nil
		}
		for _, value := range s.Names {
			if value.IsExported() {
				a.add(p, value, tok.String(), a.extentOf(p, s))
			}
		}
	}
	return nil
}

func (a *analysis) extentOf(p *packages.Package, node ast.Node) extent {
	start, end := p.Fset.Position(node.Pos()), p.Fset.Position(node.End())
	return extent{file: start.Filename, start: start.Offset, end: end.Offset}
}

func (a *analysis) add(p *packages.Package, id *ast.Ident, kind string, at extent) *candidate {
	obj := p.TypesInfo.Defs[id]
	key, ok := a.keyOf(obj)
	if !ok {
		return nil
	}
	c := &candidate{key: key, kind: kind, obj: obj, position: p.Fset.Position(id.Pos()), extents: []extent{at}}
	a.candidates[key] = c
	return c
}

// count records the references in the files of p. A test variant compiles
// the production files of its package again, so only its test files count.
func (a *analysis) count(p *packages.Package, variant bool) {
	for id, obj := range p.TypesInfo.Uses {
		position := p.Fset.Position(id.Pos())
		if variant && !strings.HasSuffix(position.Filename, "_test.go") {
			continue
		}
		test := a.testCode(p, position.Filename)
		if !test && slices.ContainsFunc(a.assertions, func(e extent) bool { return e.contains(position) }) {
			continue
		}
		if fn, ok := obj.(*types.Func); ok && !test {
			fn = fn.Origin()
			if recv := fn.Type().(*types.Signature).Recv(); recv != nil && types.IsInterface(recv.Type()) {
				a.usedInterfaceMethods[fn] = true
			}
		}
		key, ok := a.keyOf(obj)
		if !ok {
			continue
		}
		c := a.candidates[key]
		switch {
		case c == nil:
		case test:
			c.testRefs++
			if c.testFiles == nil {
				c.testFiles = map[string]bool{}
			}
			c.testFiles[position.Filename] = true
		case !slices.ContainsFunc(c.extents, func(e extent) bool { return e.contains(position) }):
			c.prodRefs++
		}
	}
}

// keyOf names a package-level object or a method of a package-level type of
// the module. It reports false for anything else, such as a field, a local
// declaration or an object of another module. Keys do not depend on the
// package variant, so the objects of a test variant map to the same
// candidates.
func (a *analysis) keyOf(obj types.Object) (string, bool) {
	if obj == nil || obj.Pkg() == nil {
		return "", false
	}
	relative, ok := a.relativePackage(obj.Pkg().Path())
	if !ok {
		return "", false
	}
	switch o := obj.(type) {
	case *types.Func:
		o = o.Origin()
		if recv := o.Type().(*types.Signature).Recv(); recv != nil {
			named := namedOf(recv.Type())
			if named == nil || !packageLevel(named.Obj()) {
				return "", false
			}
			return relative + "." + named.Obj().Name() + "." + o.Name(), true
		}
	case *types.Var:
		if o.Origin().IsField() {
			return "", false
		}
	case *types.TypeName, *types.Const:
	default:
		return "", false
	}
	if !packageLevel(obj) {
		return "", false
	}
	return relative + "." + obj.Name(), true
}

func packageLevel(obj types.Object) bool {
	return obj.Pkg() != nil && obj.Parent() == obj.Pkg().Scope()
}

func namedOf(t types.Type) *types.Named {
	if pointer, ok := types.Unalias(t).(*types.Pointer); ok {
		t = pointer.Elem()
	}
	named, _ := types.Unalias(t).(*types.Named)
	if named != nil {
		named = named.Origin()
	}
	return named
}

func receiverBase(expr ast.Expr) string {
	for {
		switch e := expr.(type) {
		case *ast.StarExpr:
			expr = e.X
		case *ast.ParenExpr:
			expr = e.X
		case *ast.IndexExpr:
			expr = e.X
		case *ast.IndexListExpr:
			expr = e.X
		case *ast.Ident:
			return e.Name
		default:
			return ""
		}
	}
}

// usedInterfaces returns, by method name, the interfaces whose methods count
// as used: every interface declared outside the module, the error type, and
// each interface of the module whose method production code calls, directly
// or through a wider interface that the interface satisfies.
func (a *analysis) usedInterfaces(production []*packages.Package) map[string][]types.Type {
	used := map[string][]types.Type{}
	addAll := func(t types.Type) {
		iface, ok := t.Underlying().(*types.Interface)
		if !ok {
			return
		}
		for method := range iface.Methods() {
			used[method.Name()] = append(used[method.Name()], t)
		}
	}
	addAll(types.Universe.Lookup("error").Type())
	seen := map[*types.Package]bool{}
	var visit func(*types.Package)
	visit = func(pkg *types.Package) {
		if seen[pkg] {
			return
		}
		seen[pkg] = true
		if path := pkg.Path(); path != a.module && !strings.HasPrefix(path, a.module+"/") {
			scope := pkg.Scope()
			for _, name := range scope.Names() {
				if typeName, ok := scope.Lookup(name).(*types.TypeName); ok && types.IsInterface(typeName.Type()) {
					addAll(typeName.Type())
				}
			}
		}
		for _, imported := range pkg.Imports() {
			visit(imported)
		}
	}
	for _, p := range production {
		visit(p.Types)
	}
	for method := range a.usedInterfaceMethods {
		used[method.Name()] = append(used[method.Name()], method.Type().(*types.Signature).Recv().Type())
	}
	for changed := true; changed; {
		changed = false
		for _, method := range a.interfaceMethods {
			if a.usedInterfaceMethods[method] {
				continue
			}
			owner := method.Type().(*types.Signature).Recv().Type()
			for _, wider := range used[method.Name()] {
				if !types.Identical(owner, wider) && implements(owner, wider) {
					a.usedInterfaceMethods[method] = true
					used[method.Name()] = append(used[method.Name()], owner)
					changed = true
					break
				}
			}
		}
	}
	return used
}

// satisfiesUsedInterface reports whether c is a method that a used interface
// reaches: an interface method that production code calls, or the method of
// a type that implements, itself or through a type that embeds it, an
// interface with a used method of that name.
func (a *analysis) satisfiesUsedInterface(c *candidate, used map[string][]types.Type) bool {
	method, ok := c.obj.(*types.Func)
	if !ok {
		return false
	}
	recv := method.Type().(*types.Signature).Recv()
	if recv == nil {
		return false
	}
	if types.IsInterface(recv.Type()) {
		return a.usedInterfaceMethods[method]
	}
	if dynamicMethodNames[method.Name()] {
		return true
	}
	interfaces := used[method.Name()]
	if len(interfaces) == 0 {
		return false
	}
	for _, owner := range a.ownersOf(method) {
		if owner.TypeParams().Len() > 0 {
			// Implements is unspecified for an uninstantiated generic type,
			// so a method named like a used interface method counts.
			return true
		}
		for _, iface := range interfaces {
			if implements(owner, iface) || implements(types.NewPointer(owner), iface) {
				return true
			}
		}
	}
	return false
}

// ownersOf returns the named types of the module whose method sets hold
// method: its receiver type and every type that embeds it.
func (a *analysis) ownersOf(method *types.Func) []*types.Named {
	if a.owners == nil {
		a.owners = map[*types.Func][]*types.Named{}
		for _, named := range a.namedTypes {
			for selection := range types.NewMethodSet(types.NewPointer(named)).Methods() {
				if fn, ok := selection.Obj().(*types.Func); ok {
					a.owners[fn.Origin()] = append(a.owners[fn.Origin()], named)
				}
			}
		}
	}
	return a.owners[method]
}

func implements(v, t types.Type) bool {
	iface, ok := t.Underlying().(*types.Interface)
	return ok && types.Implements(v, iface)
}
