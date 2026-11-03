package archtest

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
)

// Domain commands and namespace identities belong to their domain or policy.
// Shared storage, receipts and lifecycle must not dispatch on sitemap.
// Layer dependencies bar imports of projectsitemap; table ownership bars its
// direct artifact-table SQL. This check only keeps the reserved namespace at
// its policy boundary, without banning unrelated prose or artifact names.
func TestSitemapUsesSharedArtifactMechanisms(t *testing.T) {
	files := 0
	for _, directory := range []string{
		"internal/artifacts", "internal/persistence/postgres", "internal/projectlifecycle",
		"internal/httpapi/privateartifacts/artifact_handlers.go",
		"internal/httpapi/privateartifacts/artifact_read_access.go",
	} {
		err := filepath.WalkDir(filepath.Join("..", "..", directory), func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				if entry.Name() == "testdata" || path == filepath.Join("..", "..", "internal", "artifacts", "policy") {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			positions := token.NewFileSet()
			file, err := parser.ParseFile(positions, path, nil, 0)
			if err != nil {
				return err
			}
			files++
			ast.Inspect(file, func(node ast.Node) bool {
				var value string
				switch node := node.(type) {
				case *ast.Ident:
					value = node.Name
				case *ast.BasicLit:
					if node.Kind == token.STRING {
						value, _ = strconv.Unquote(node.Value)
					}
				}
				if value == artifactpolicy.ProjectSitemapNamespace || value == "ProjectSitemapNamespace" {
					t.Errorf("%s: shared artifact mechanism references sitemap: %q", positions.Position(node.Pos()), value)
				}
				return true
			})
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if files == 0 {
		t.Fatal("no shared artifact mechanism files checked")
	}
}
