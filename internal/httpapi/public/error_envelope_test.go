package public

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestErrorEnvelopesComeFromTheRegistry keeps every public error answer on
// the error registry (ERR-R14): the HTTP error envelope and the WebSocket
// error frame are each built in one function that takes an apierror.Kind, so
// no handler can answer with an unregistered code or another retryable flag.
func TestErrorEnvelopesComeFromTheRegistry(t *testing.T) {
	for _, check := range []struct{ directory, envelope, builder string }{
		{".", "errorResponse", "writeErrorDetails"},
		{"events", "errorFrame", "newErrorFrame"},
	} {
		entries, err := os.ReadDir(check.directory)
		if err != nil {
			t.Fatal(err)
		}
		built := 0
		for _, entry := range entries {
			name := entry.Name()
			if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
				continue
			}
			path := filepath.Join(check.directory, name)
			file, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			for _, declaration := range file.Decls {
				function, isFunction := declaration.(*ast.FuncDecl)
				ast.Inspect(declaration, func(node ast.Node) bool {
					literal, ok := node.(*ast.CompositeLit)
					if !ok {
						return true
					}
					if identifier, ok := literal.Type.(*ast.Ident); !ok || identifier.Name != check.envelope {
						return true
					}
					if !isFunction || function.Name.Name != check.builder {
						t.Errorf("%s builds a %s outside %s", path, check.envelope, check.builder)
						return true
					}
					built++
					return true
				})
			}
		}
		if built != 1 {
			t.Errorf("%s builds %d %s values, want exactly one", check.builder, built, check.envelope)
		}
	}
}
