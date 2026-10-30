package archtest

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/constant"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// This file is the secret-holder guard of SEC-R88
// (docs/spec/platform/security.md). fmt calls a value's own Format, String or
// GoString method only while it reaches the value through exported struct
// fields. Through an unexported field it prints the raw fields for every verb,
// and a verb that does not fit a pointer (%t, %s, %c, ...) makes it print the
// target of a nested pointer as well. log/slog's text handler formats with
// %+v, so it behaves the same way. The guard checks every named type of the
// production tree (api/, cmd/ and internal/) against two rules:
//
//   - A secret type redacts everywhere: it implements fmt.Formatter,
//     fmt.Stringer, fmt.GoStringer, slog.LogValuer and json.Marshaler. A secret
//     type is one of the rules' leaf types or a type whose String or GoString
//     method writes a redaction marker.
//   - A holder redacts as a whole: a type implements fmt.Formatter itself when
//     it reaches a secret type through an unexported struct field, a secret
//     type of another module (which cannot redact itself) through any field,
//     or a plain-text field named like a secret that is unexported or is not a
//     wire field (a field without a json or yaml name). The walk follows
//     exported fields until a type that formats itself, slices, arrays, map
//     keys and values, and at most one pointer below the type, as fmt does. It
//     does not look inside an interface or a type of another module.
//
// The methods must be in the value method set, so that a copied value redacts
// too; a type that holds a lock, which vet's copylocks check keeps from being
// copied, may declare them on its pointer. encoding/json never reads an
// unexported field, so a holder's Format also covers slog's JSON handler; the
// secret types' own MarshalJSON covers exported paths.

type secretHolderRules struct {
	// leaves are the module's types that hold one secret value, named by
	// package path relative to the module and type name. They are secret
	// even if their String changes.
	leaves []string
	// external are types of other modules or the standard library that hold
	// key material and print it for some verb, with the reason.
	external map[string]string
	// nonSecret are types that redact themselves but hold no secret, so a
	// holder need not redact them, with the reason. They still redact
	// everywhere.
	nonSecret map[string]string
	// exceptions are holders the guard does not check, with the reason.
	exceptions map[string]string
	// plainFields are fields, named type.field, whose secret-like name holds
	// no secret value, with the reason.
	plainFields map[string]string
}

var productionSecretHolderRules = secretHolderRules{
	leaves: []string{
		"internal/contracts.SecretString",
		"internal/credentials.RuntimeCredentialMaterial",
		"internal/credentials.Token",
		"internal/credentials/litellm.adminKey",
	},
	external: map[string]string{
		"crypto/ecdh.PrivateKey":    "keeps the private key bytes in unexported fields",
		"crypto/ecdsa.PrivateKey":   "its D formats as the private scalar",
		"crypto/ed25519.PrivateKey": "is the private key bytes",
		"crypto/rsa.PrivateKey":     "its D and Primes format as the private exponent and factors",
		"crypto/tls.Certificate":    "holds its private key, which %t and similar verbs print",
		"crypto/tls.Config":         "holds certificate private keys and, on a server, session ticket keys",
	},
	nonSecret: map[string]string{
		"internal/auth.SessionHandle": "a SHA-256 digest of a 256-bit session token: it names a session " +
			"in the Server's map but cannot authenticate one",
	},
	exceptions: map[string]string{
		"internal/contracts/runtimesettings.revealedCaido": "allocation-wire form built and encoded " +
			"inside RuntimeSettings.RevealJSON only (SEC-R88); it carries the plaintext by design",
		"internal/contracts/runtimesettings.revealedHTTPOriginTarget": "allocation-wire form built and " +
			"encoded inside RuntimeSettings.RevealJSON only (SEC-R88); it carries the plaintext by design",
		"internal/contracts/runtimesettings.revealedHTTPProxy": "allocation-wire form built and encoded " +
			"inside RuntimeSettings.RevealJSON only (SEC-R88); it carries the plaintext by design",
		"internal/contracts/runtimesettings.revealedInteractsh": "allocation-wire form built and encoded " +
			"inside RuntimeSettings.RevealJSON only (SEC-R88); it carries the plaintext by design",
		"internal/contracts/runtimesettings.revealedTelemetry": "allocation-wire form built and encoded " +
			"inside RuntimeSettings.RevealJSON only (SEC-R88); it carries the plaintext by design",
		"internal/scanplan.revealedOptions": "in-process JSON form of the normalization round trip, " +
			"built and encoded inside one call; it carries the plaintext by design",
	},
	plainFields: map[string]string{
		"internal/localpki.Paths.PrivateKey": "the path of the private key file, not the key",
		"internal/mtls.Files.PrivateKey":     "the path of the private key file, not the key",
		"internal/mtlstest.Files.PrivateKey": "the path of the private key file, not the key",
	},
}

// redactionMarkers are the string fragments by which a String or GoString
// method declares its type secret.
var redactionMarkers = []string{"[REDACTED", "[ENCRYPTED"}

// secretFieldWords mark a field of plain text (a string, a byte slice or
// array, a string slice or a string map) whose value is a secret; a field
// matches when its lower-cased name contains one of them.
var secretFieldWords = []string{
	"secret", "password", "passwd", "token", "cookie", "csrf", "bearer",
	"apikey", "adminkey", "privatekey", "mackey", "masterkey", "signingkey",
	"databaseurl",
}

// secretFieldReferenceSuffixes mark a field that names, locates or counts a
// secret instead of holding it, such as tokenID, passwordFile or
// csrfTokenDigest; a digest of a high-entropy secret does not reveal it.
var secretFieldReferenceSuffixes = []string{
	"id", "ids", "ref", "refs", "file", "files", "path", "paths", "kind", "kinds",
	"name", "names", "digest", "digests", "count", "bytes", "header", "headers",
}

// secretHolderFixture is the fixture package whose findings
// TestSecretHolderGuardReportsEachRule pins.
const secretHolderFixture = "internal/archtest/testdata/secretholders"

func TestSecretHoldersRedactForEveryVerb(t *testing.T) {
	packages, err := loadTypedPackages(moduleRoot(t), "./api/...", "./cmd/...", "./internal/...")
	if err != nil {
		t.Fatal(err)
	}
	if len(packages.checked) < 100 {
		t.Fatalf("type-checked only %d packages of the module", len(packages.checked))
	}
	if problems := checkSecretHolders(packages, productionSecretHolderRules); len(problems) > 0 {
		t.Fatal("secret-holder check failed (SEC-R88).\n\n" + strings.Join(problems, "\n") + secretHolderAdvice)
	}
}

// The fixture holds one type per rule and per walk boundary; the guard must
// report exactly the types that the fixture marks as findings, and every
// stale table entry.
func TestSecretHolderGuardReportsEachRule(t *testing.T) {
	packages, err := loadTypedPackages(moduleRoot(t), "./"+secretHolderFixture)
	if err != nil {
		t.Fatal(err)
	}
	fixture := func(name string) string { return secretHolderFixture + "." + name }
	problems := checkSecretHolders(packages, secretHolderRules{
		leaves:     []string{"internal/contracts.SecretString", fixture("leaf"), fixture("absentLeaf")},
		external:   map[string]string{"crypto/tls.Config": "fixture"},
		nonSecret:  map[string]string{fixture("handle"): "fixture", fixture("absentHandle"): "fixture"},
		exceptions: map[string]string{fixture("excepted"): "fixture", fixture("absentException"): ""},
	})
	want := []string{
		"secret leaf " + fixture("absentLeaf") + " names no type",
		"non-secret type " + fixture("absentHandle") + " names no type",
		"exception " + fixture("absentException") + " needs a reason",
		"exception " + fixture("absentException") + " names no type",
		fixture("ExportedHolder") + " reaches internal/contracts.SecretString through .Inner.token",
		fixture("ExportedPlain") + " reaches the plain-text secret field .Password",
		fixture("boxedSecret") + " reaches internal/contracts.SecretString through .box.value",
		fixture("embedsHolder") + " reaches internal/contracts.SecretString through .unexportedSecret.token",
		fixture("heldStringOnly") + " reaches " + fixture("stringOnlySecret") + " through .secret",
		fixture("leaf") + " is a secret leaf but does not implement fmt.Formatter, slog.LogValuer, json.Marshaler",
		fixture("nestedInMap") + " reaches internal/contracts.SecretString through .values[][]",
		fixture("onePointer") + " reaches " + fixture("RedactingHolder") + " through .inner*",
		fixture("plainPassword") + " reaches the plain-text secret field .password",
		fixture("plainSecretFields") + " reaches the plain-text secret field .SecretFields",
		fixture("pointerFormatted") + " reaches internal/contracts.SecretString through .token",
		fixture("stringOnlySecret") + " writes a redaction marker in String or GoString but does not implement " +
			"fmt.Formatter, slog.LogValuer, json.Marshaler",
		fixture("tlsHolder") + " reaches crypto/tls.Config through .Config*",
		fixture("unexportedSecret") + " reaches internal/contracts.SecretString through .token",
	}
	if len(problems) != len(want) {
		t.Errorf("got %d findings, want %d:\n%s", len(problems), len(want), strings.Join(problems, "\n"))
	}
	for index := range min(len(problems), len(want)) {
		if !strings.HasPrefix(problems[index], want[index]) {
			t.Errorf("finding %d = %q\nwant prefix  %q", index, problems[index], want[index])
		}
	}
}

const secretHolderAdvice = `

Give a reported holder a Format method with a value receiver that writes only
its redacted form for every verb, as internal/scheduler/report_secrets.go does
(add String, GoString, MarshalJSON and LogValue when the type is itself logged
or encoded), or keep the secret in an exported field of a type that redacts
itself. A new kind of secret value belongs in contracts.SecretString or another
secret type, not in a plain string. The rules and tables are in
internal/archtest/secret_holders_test.go.`

func moduleRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

// typedPackages are the main module's packages that the listed patterns
// matched, with their module dependencies, type-checked from source.
type typedPackages struct {
	root    string
	module  string
	fset    *token.FileSet
	checked []*typedPackage
	std     map[string]*types.Package
}

type typedPackage struct {
	path    string
	files   []*ast.File
	info    *types.Info
	matched bool
}

type listedPackage struct {
	ImportPath string
	Dir        string
	GoFiles    []string
	CgoFiles   []string
	Export     string
	DepOnly    bool
	Module     *struct {
		Path string
		Main bool
	}
	Error *struct{ Err string }
}

// loadTypedPackages lists patterns and their dependencies with `go list
// -export`, type-checks every package of the main module from its non-test
// files in dependency order, and imports every other package from the
// compiler's export data.
func loadTypedPackages(root string, patterns ...string) (*typedPackages, error) {
	goTool, err := exec.LookPath("go")
	if err != nil {
		return nil, fmt.Errorf("the go command is required: %w", err)
	}
	// The guard needs the interfaces it checks for, whatever the patterns import.
	required := []string{"fmt", "log/slog", "encoding/json"}
	arguments := append([]string{
		"list", "-deps", "-export",
		"-json=ImportPath,Dir,GoFiles,CgoFiles,Export,DepOnly,Module,Error",
	}, required...)
	command := exec.Command(goTool, append(arguments, patterns...)...)
	command.Dir = root
	var stderr bytes.Buffer
	command.Stderr = &stderr
	output, err := command.Output()
	if err != nil {
		return nil, fmt.Errorf("go list: %w\n%s", err, stderr.String())
	}
	exports := map[string]string{}
	var local []listedPackage
	module := ""
	decoder := json.NewDecoder(bytes.NewReader(output))
	for {
		var listed listedPackage
		if err := decoder.Decode(&listed); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return nil, fmt.Errorf("decode go list output: %w", err)
		}
		if listed.Error != nil {
			return nil, fmt.Errorf("go list %s: %s", listed.ImportPath, listed.Error.Err)
		}
		if listed.Module != nil && listed.Module.Main {
			module = listed.Module.Path
			local = append(local, listed)
			continue
		}
		exports[listed.ImportPath] = listed.Export
	}
	if module == "" {
		return nil, fmt.Errorf("go list %s matched no package of the main module", strings.Join(patterns, " "))
	}

	result := &typedPackages{root: root, module: module, fset: token.NewFileSet(), std: map[string]*types.Package{}}
	compiled := importer.ForCompiler(result.fset, "gc", func(path string) (io.ReadCloser, error) {
		file := exports[path]
		if file == "" {
			return nil, fmt.Errorf("no export data for %s", path)
		}
		return os.Open(file)
	})
	checked := map[string]*types.Package{}
	imports := importerFunc(func(path string) (*types.Package, error) {
		if pkg := checked[path]; pkg != nil {
			return pkg, nil
		}
		return compiled.Import(path)
	})
	for _, path := range required {
		pkg, err := imports.Import(path)
		if err != nil {
			return nil, err
		}
		result.std[path] = pkg
	}
	// go list -deps prints every package after its dependencies.
	for _, listed := range local {
		if len(listed.CgoFiles) > 0 {
			return nil, fmt.Errorf("%s uses cgo, which the secret-holder guard cannot type-check", listed.ImportPath)
		}
		files := make([]*ast.File, 0, len(listed.GoFiles))
		for _, name := range listed.GoFiles {
			file, err := parser.ParseFile(result.fset, filepath.Join(listed.Dir, name), nil, parser.SkipObjectResolution)
			if err != nil {
				return nil, err
			}
			files = append(files, file)
		}
		info := &types.Info{Defs: map[*ast.Ident]types.Object{}, Uses: map[*ast.Ident]types.Object{}}
		config := types.Config{Importer: imports}
		pkg, err := config.Check(listed.ImportPath, result.fset, files, info)
		if err != nil {
			return nil, fmt.Errorf("type-check %s: %w", listed.ImportPath, err)
		}
		checked[listed.ImportPath] = pkg
		result.checked = append(result.checked, &typedPackage{
			path: listed.ImportPath, files: files, info: info, matched: !listed.DepOnly,
		})
	}
	return result, nil
}

type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }

type secretHolderChecker struct {
	packages *typedPackages
	rules    secretHolderRules
	// secret holds the module's secret types: the leaves and every type whose
	// String or GoString writes a redaction marker.
	secret     map[*types.TypeName]bool
	declared   map[*types.TypeName]bool
	formatter  *types.Interface
	stringer   *types.Interface
	goStringer *types.Interface
	logValuer  *types.Interface
	marshaler  *types.Interface
	locker     *types.Interface
}

// checkSecretHolders returns one message per stale table entry and then one
// per offending named type of the matched packages, sorted by type.
func checkSecretHolders(packages *typedPackages, rules secretHolderRules) []string {
	checker := &secretHolderChecker{
		packages:   packages,
		rules:      rules,
		secret:     map[*types.TypeName]bool{},
		declared:   map[*types.TypeName]bool{},
		formatter:  lookupInterface(packages.std["fmt"], "Formatter"),
		stringer:   lookupInterface(packages.std["fmt"], "Stringer"),
		goStringer: lookupInterface(packages.std["fmt"], "GoStringer"),
		logValuer:  lookupInterface(packages.std["log/slog"], "LogValuer"),
		marshaler:  lookupInterface(packages.std["encoding/json"], "Marshaler"),
		locker:     lockerInterface(),
	}
	defined := map[string]*types.TypeName{}
	var roots []*types.TypeName
	for _, pkg := range packages.checked {
		for _, object := range pkg.info.Defs {
			name, ok := object.(*types.TypeName)
			if !ok || name.IsAlias() {
				continue
			}
			defined[checker.key(name)] = name
			if pkg.matched {
				roots = append(roots, name)
			}
		}
		for _, file := range pkg.files {
			for _, declaration := range file.Decls {
				if name := redactingReceiver(pkg.info, declaration); name != nil {
					checker.secret[name] = true
					checker.declared[name] = true
				}
			}
		}
	}

	var problems []string
	for _, key := range rules.leaves {
		if name := defined[key]; name != nil {
			checker.secret[name] = true
		} else {
			problems = append(problems, fmt.Sprintf("secret leaf %s names no type; fix or delete the entry", key))
		}
	}
	for _, key := range sortedKeys(rules.nonSecret) {
		if name := defined[key]; name != nil {
			delete(checker.secret, name)
			checker.declared[name] = true
		} else {
			problems = append(problems, fmt.Sprintf("non-secret type %s names no type; fix or delete the entry", key))
		}
	}
	for _, key := range sortedKeys(rules.plainFields) {
		typeKey := key[:max(strings.LastIndex(key, "."), 0)]
		if rules.plainFields[key] == "" {
			problems = append(problems, fmt.Sprintf("plain field %s needs a reason", key))
		}
		if defined[typeKey] == nil {
			problems = append(problems, fmt.Sprintf("plain field %s names no type; fix or delete the entry", key))
		}
	}
	for _, key := range sortedKeys(rules.exceptions) {
		if rules.exceptions[key] == "" {
			problems = append(problems, fmt.Sprintf("exception %s needs a reason", key))
		}
		if defined[key] == nil {
			problems = append(problems, fmt.Sprintf("exception %s names no type; fix or delete the entry", key))
		}
	}

	type finding struct{ key, message string }
	var findings []finding
	for _, name := range roots {
		named, ok := name.Type().(*types.Named)
		if !ok {
			continue
		}
		key := checker.key(name)
		if checker.secret[name] || checker.declared[name] {
			if missing := checker.missingMethods(named); len(missing) > 0 {
				reason := "is a secret leaf"
				if checker.declared[name] {
					reason = "writes a redaction marker in String or GoString"
				}
				findings = append(findings, finding{key, fmt.Sprintf("%s %s but does not implement %s (%s)",
					key, reason, strings.Join(missing, ", "), checker.position(name))})
				continue
			}
		}
		if _, excepted := rules.exceptions[key]; excepted || checker.implements(named, checker.formatter) {
			continue
		}
		if path, target := checker.holderPath(named); target != "" {
			findings = append(findings, finding{key, fmt.Sprintf("%s reaches %s without implementing fmt.Formatter (%s)",
				key, target+path, checker.position(name))})
		}
	}
	slices.SortFunc(findings, func(a, b finding) int { return strings.Compare(a.key, b.key) })
	for _, item := range findings {
		problems = append(problems, item.message)
	}
	return problems
}

// redactingReceiver returns the receiver type of a String or GoString method
// that writes a redaction marker.
func redactingReceiver(info *types.Info, declaration ast.Decl) *types.TypeName {
	function, ok := declaration.(*ast.FuncDecl)
	if !ok || function.Recv == nil || function.Body == nil ||
		(function.Name.Name != "String" && function.Name.Name != "GoString") ||
		!writesRedactionMarker(info, function.Body) {
		return nil
	}
	method, ok := info.Defs[function.Name].(*types.Func)
	if !ok {
		return nil
	}
	receiver := method.Signature().Recv().Type()
	if pointer, ok := receiver.(*types.Pointer); ok {
		receiver = pointer.Elem()
	}
	if named, ok := receiver.(*types.Named); ok {
		return named.Origin().Obj()
	}
	return nil
}

func (c *secretHolderChecker) key(name *types.TypeName) string {
	if name.Pkg() == nil {
		return name.Name()
	}
	path := name.Pkg().Path()
	if relative, ok := strings.CutPrefix(path, c.packages.module+"/"); ok {
		path = relative
	}
	return path + "." + name.Name()
}

func (c *secretHolderChecker) inModule(name *types.TypeName) bool {
	return name.Pkg() != nil && strings.HasPrefix(name.Pkg().Path()+"/", c.packages.module+"/")
}

func (c *secretHolderChecker) position(name *types.TypeName) string {
	position := c.packages.fset.Position(name.Pos())
	file := position.Filename
	if relative, err := filepath.Rel(c.packages.root, file); err == nil {
		file = filepath.ToSlash(relative)
	}
	return file + ":" + strconv.Itoa(position.Line)
}

// missingMethods returns the interfaces a secret type lacks.
func (c *secretHolderChecker) missingMethods(named *types.Named) []string {
	var missing []string
	for _, required := range []struct {
		name  string
		iface *types.Interface
	}{
		{"fmt.Formatter", c.formatter},
		{"fmt.Stringer", c.stringer},
		{"fmt.GoStringer", c.goStringer},
		{"slog.LogValuer", c.logValuer},
		{"json.Marshaler", c.marshaler},
	} {
		if !c.implements(named, required.iface) {
			missing = append(missing, required.name)
		}
	}
	return missing
}

// implements reports whether a value of named implements iface, or its pointer
// does when named holds a lock.
func (c *secretHolderChecker) implements(named *types.Named, iface *types.Interface) bool {
	if types.Implements(named, iface) {
		return true
	}
	return c.holdsLock(named, map[types.Type]bool{}) && types.Implements(types.NewPointer(named), iface)
}

// holdsLock reports whether a value of t contains a lock that vet's copylocks
// check forbids copying: a type whose pointer, but not its value, has Lock and
// Unlock methods.
func (c *secretHolderChecker) holdsLock(t types.Type, seen map[types.Type]bool) bool {
	if seen[t] {
		return false
	}
	seen[t] = true
	if _, ok := t.Underlying().(*types.Interface); !ok &&
		types.Implements(types.NewPointer(t), c.locker) && !types.Implements(t, c.locker) {
		return true
	}
	switch underlying := t.Underlying().(type) {
	case *types.Struct:
		for index := range underlying.NumFields() {
			if c.holdsLock(underlying.Field(index).Type(), seen) {
				return true
			}
		}
	case *types.Array:
		return c.holdsLock(underlying.Elem(), seen)
	}
	return false
}

// holderPath returns what root reaches and the field path by which fmt would
// print it, or an empty target when root reaches no secret.
func (c *secretHolderChecker) holderPath(root *types.Named) (path, target string) {
	seen := map[string]bool{}
	var walk func(t types.Type, exported bool, pointers int, path string, top bool) (string, string)
	walk = func(t types.Type, exported bool, pointers int, path string, top bool) (string, string) {
		t = types.Unalias(t)
		state := fmt.Sprintf("%s|%t|%d", types.TypeString(t, nil), exported, pointers)
		if seen[state] || len(seen) > 100_000 {
			return "", ""
		}
		seen[state] = true
		if named, ok := t.(*types.Named); ok && !top {
			name := named.Origin().Obj()
			if _, ok := c.rules.external[c.key(name)]; ok {
				return path, c.key(name) + " through "
			}
			if !c.inModule(name) {
				return "", ""
			}
			if c.secret[name] {
				if exported {
					// fmt calls the secret type's own methods, which the
					// first rule requires.
					return "", ""
				}
				return path, c.key(name) + " through "
			}
		}
		if !top && exported && types.Implements(t, c.formatter) {
			return "", ""
		}
		owner := ""
		if named, ok := t.(*types.Named); ok {
			owner = c.key(named.Origin().Obj())
		}
		switch underlying := t.Underlying().(type) {
		case *types.Struct:
			for index := range underlying.NumFields() {
				field := underlying.Field(index)
				fieldExported := exported && field.Exported()
				fieldPath := path + "." + field.Name()
				_, notSecret := c.rules.plainFields[owner+"."+field.Name()]
				if !notSecret && isPlainSecretField(field) && (!fieldExported || !isWireField(underlying.Tag(index))) {
					return fieldPath, "the plain-text secret field "
				}
				if found, target := walk(field.Type(), fieldExported, pointers, fieldPath, false); target != "" {
					return found, target
				}
			}
		case *types.Pointer:
			// fmt prints the address of a nested pointer for %v; a verb that
			// does not fit a pointer prints one level of its target instead.
			if pointers >= 1 {
				return "", ""
			}
			return walk(underlying.Elem(), exported, pointers+1, path+"*", false)
		case *types.Slice:
			return walk(underlying.Elem(), exported, pointers, path+"[]", false)
		case *types.Array:
			return walk(underlying.Elem(), exported, pointers, path+"[]", false)
		case *types.Map:
			if found, target := walk(underlying.Key(), exported, pointers, path+"[key]", false); target != "" {
				return found, target
			}
			return walk(underlying.Elem(), exported, pointers, path+"[]", false)
		}
		return "", ""
	}
	return walk(root, true, 0, "", true)
}

// isPlainSecretField reports whether field holds plain text under a name that
// marks a secret value.
func isPlainSecretField(field *types.Var) bool {
	if !isPlainText(field.Type()) {
		return false
	}
	name := strings.ToLower(field.Name())
	for _, suffix := range secretFieldReferenceSuffixes {
		if strings.HasSuffix(name, suffix) {
			return false
		}
	}
	return slices.ContainsFunc(secretFieldWords, func(word string) bool { return strings.Contains(name, word) })
}

// isWireField reports whether a struct tag names the field for a JSON or YAML
// document: a wire form carries the plaintext by contract.
func isWireField(tag string) bool {
	for _, key := range []string{"json", "yaml"} {
		value, ok := reflect.StructTag(tag).Lookup(key)
		if name, _, _ := strings.Cut(value, ","); ok && name != "-" {
			return true
		}
	}
	return false
}

func isPlainText(t types.Type) bool {
	switch underlying := t.Underlying().(type) {
	case *types.Basic:
		return underlying.Info()&types.IsString != 0
	case *types.Pointer:
		return isPlainText(underlying.Elem())
	case *types.Slice:
		return isByte(underlying.Elem()) || isPlainText(underlying.Elem())
	case *types.Array:
		return isByte(underlying.Elem())
	case *types.Map:
		return isPlainText(underlying.Key()) && isPlainText(underlying.Elem())
	}
	return false
}

func isByte(t types.Type) bool {
	basic, ok := t.Underlying().(*types.Basic)
	return ok && basic.Kind() == types.Byte
}

// writesRedactionMarker reports whether a method body contains a string
// literal, or names a string constant, with a redaction marker.
func writesRedactionMarker(info *types.Info, body *ast.BlockStmt) bool {
	hasMarker := func(text string) bool {
		return slices.ContainsFunc(redactionMarkers, func(marker string) bool { return strings.Contains(text, marker) })
	}
	found := false
	ast.Inspect(body, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.BasicLit:
			if node.Kind == token.STRING && hasMarker(node.Value) {
				found = true
			}
		case *ast.Ident:
			if value, ok := info.Uses[node].(*types.Const); ok && value.Val().Kind() == constant.String &&
				hasMarker(constant.StringVal(value.Val())) {
				found = true
			}
		}
		return !found
	})
	return found
}

func lookupInterface(pkg *types.Package, name string) *types.Interface {
	return pkg.Scope().Lookup(name).Type().Underlying().(*types.Interface)
}

func lockerInterface() *types.Interface {
	signature := types.NewSignatureType(nil, nil, nil, nil, nil, false)
	return types.NewInterfaceType([]*types.Func{
		types.NewFunc(token.NoPos, nil, "Lock", signature),
		types.NewFunc(token.NoPos, nil, "Unlock", signature),
	}, nil).Complete()
}
