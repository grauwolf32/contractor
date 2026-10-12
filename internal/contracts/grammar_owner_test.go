package contracts_test

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"regexp/syntax"
	"strconv"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// canonicalRegexp returns the simplified syntax tree of pattern, in which
// equal character classes and groupings print alike.
func canonicalRegexp(pattern string) (string, bool) {
	parsed, err := syntax.Parse(pattern, syntax.Perl)
	if err != nil {
		return "", false
	}
	return parsed.Simplify().String(), true
}

// TestPublicIdentifierGrammarsHaveOneGoDefinition keeps the public ResourceId
// and ConfigId grammars in one Go definition each, contracts.ValidResourceID
// and contracts.ValidName: no non-test Go source outside internal/contracts
// may hold a regular expression that accepts exactly one of those bounded
// languages. Such copies drifted from the public schemas before. A pattern
// with another bound is another rule and is not reported.
func TestPublicIdentifierGrammarsHaveOneGoDefinition(t *testing.T) {
	t.Parallel()
	grammars := map[string]string{}
	for name, pattern := range map[string]string{
		"ResourceId (use contracts.ValidResourceID)": `^[A-Za-z0-9][A-Za-z0-9_.:-]{0,` + strconv.Itoa(contracts.MaxResourceIDLength-1) + `}$`,
		"ConfigId (use contracts.ValidName)":         `^[A-Za-z0-9][A-Za-z0-9_.-]{0,` + strconv.Itoa(contracts.MaxNameLength-1) + `}$`,
	} {
		canonical, ok := canonicalRegexp(pattern)
		if !ok {
			t.Fatalf("reference pattern %q does not parse", pattern)
		}
		grammars[canonical] = name
	}
	// The comparison must see through reordered classes and groups.
	for pattern, want := range map[string]string{
		`^(?:[0-9A-Za-z])[-:._0-9a-zA-Z]{0,255}$`: "ResourceId (use contracts.ValidResourceID)",
		`^[a-zA-Z0-9][-_.a-zA-Z0-9]{0,127}$`:      "ConfigId (use contracts.ValidName)",
	} {
		if canonical, _ := canonicalRegexp(pattern); grammars[canonical] != want {
			t.Fatalf("probe %q is not recognized as %s", pattern, want)
		}
	}

	root := contractstest.Path(t)
	owner := filepath.Join(root, "internal", "contracts")
	files := token.NewFileSet()
	var copies []string
	scanned := 0
	for _, top := range []string{"api", "cmd", "internal", "tests", "tools"} {
		err := filepath.WalkDir(filepath.Join(root, top), func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				if path == owner || entry.Name() == "testdata" || strings.HasPrefix(entry.Name(), ".") {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			file, err := parser.ParseFile(files, path, nil, parser.SkipObjectResolution)
			if err != nil {
				return err
			}
			scanned++
			ast.Inspect(file, func(node ast.Node) bool {
				literal, ok := node.(*ast.BasicLit)
				if !ok || literal.Kind != token.STRING {
					return true
				}
				value, err := strconv.Unquote(literal.Value)
				if err != nil || !strings.Contains(value, "[") {
					return true
				}
				if canonical, ok := canonicalRegexp(value); ok && grammars[canonical] != "" {
					position := files.Position(literal.Pos())
					relative, _ := filepath.Rel(root, position.Filename)
					copies = append(copies, fmt.Sprintf("%s:%d: %s", relative, position.Line, grammars[canonical]))
				}
				return true
			})
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if scanned < 500 {
		t.Fatalf("scanned only %d Go files; the repository walk is broken", scanned)
	}
	if len(copies) > 0 {
		t.Fatalf("Go copies of a public identifier grammar outside internal/contracts:\n%s", strings.Join(copies, "\n"))
	}
}
