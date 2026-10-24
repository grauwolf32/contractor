package e2e

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func TestProcessServerBuildsStayInPackageCache(t *testing.T) {
	for _, directory := range []string{".", "../ui-stack"} {
		files, err := os.ReadDir(directory)
		if err != nil {
			t.Fatal(err)
		}
		for _, file := range files {
			if filepath.Ext(file.Name()) != ".go" {
				continue
			}
			path := filepath.Join(directory, file.Name())
			tree, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			ast.Inspect(tree, func(node ast.Node) bool {
				call, ok := node.(*ast.CallExpr)
				if !ok {
					return true
				}
				build, server := false, false
				for _, arg := range call.Args {
					literal, ok := arg.(*ast.BasicLit)
					if !ok || literal.Kind != token.STRING {
						continue
					}
					value, err := strconv.Unquote(literal.Value)
					if err != nil {
						t.Fatal(err)
					}
					build = build || value == "build"
					server = server || value == "./cmd/contractor-server"
				}
				if build && server {
					t.Errorf("%s rebuilds the Server; use the package build cache", path)
				}
				return true
			})
		}
	}
}
