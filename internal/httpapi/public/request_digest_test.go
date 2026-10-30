package public

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

const contentDigestImport = "github.com/grauwolf32/contractor/internal/contentdigest"

// TestHandlersComputeNoRequestDigest keeps the idempotency request digests
// with the packages that own the requests (runservice, projectstore,
// auditservice): what a key binds is their rule, so another caller of the
// same service reproduces it without copying edge code (API-R69).
func TestHandlersComputeNoRequestDigest(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for _, name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		source, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		for _, violation := range requestDigestViolations(t, name, string(source)) {
			t.Errorf("%s computes a request digest; move it to the service that owns the request", violation)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("no handler source was checked")
	}
}

// TestRequestDigestCheckFindsProbes pins that the check above can fail.
func TestRequestDigestCheckFindsProbes(t *testing.T) {
	for name, source := range map[string]string{
		"digest function": `package public
func probeRequestDigest() string { return "" }`,
		"contentdigest call": `package public
import "github.com/grauwolf32/contractor/internal/contentdigest"
func probe(value any) (string, error) { return contentdigest.JSON(value) }`,
		"aliased contentdigest call": `package public
import digests "github.com/grauwolf32/contractor/internal/contentdigest"
func probe(data []byte) string { return digests.Bytes(data) }`,
		"contentdigest value": `package public
import "github.com/grauwolf32/contractor/internal/contentdigest"
var probe = contentdigest.CanonicalJSON`,
	} {
		if violations := requestDigestViolations(t, "probe.go", source); len(violations) == 0 {
			t.Errorf("%s: the check missed it", name)
		}
	}
	clean := `package public
import "github.com/grauwolf32/contractor/internal/contentdigest"
func validDigest(value string) bool { return contentdigest.Valid(value) }`
	if violations := requestDigestViolations(t, "clean.go", clean); len(violations) != 0 {
		t.Errorf("a digest validation was reported: %v", violations)
	}
}

// requestDigestViolations reports each function whose name ends in
// RequestDigest and each use of contentdigest other than Valid in one file.
func requestDigestViolations(t *testing.T, name, source string) []string {
	t.Helper()
	fileSet := token.NewFileSet()
	file, err := parser.ParseFile(fileSet, name, source, 0)
	if err != nil {
		t.Fatal(err)
	}
	digestPackage := ""
	for _, spec := range file.Imports {
		path, err := strconv.Unquote(spec.Path.Value)
		if err != nil {
			t.Fatal(err)
		}
		if path == contentDigestImport {
			digestPackage = "contentdigest"
			if spec.Name != nil {
				digestPackage = spec.Name.Name
			}
		}
	}
	var violations []string
	ast.Inspect(file, func(node ast.Node) bool {
		switch typed := node.(type) {
		case *ast.FuncDecl:
			if strings.HasSuffix(strings.ToLower(typed.Name.Name), "requestdigest") {
				violations = append(violations, fileSet.Position(typed.Pos()).String()+" "+typed.Name.Name)
			}
		case *ast.SelectorExpr:
			if ident, ok := typed.X.(*ast.Ident); ok && digestPackage != "" && ident.Name == digestPackage && typed.Sel.Name != "Valid" {
				violations = append(violations, fileSet.Position(typed.Pos()).String()+" contentdigest."+typed.Sel.Name)
			}
		}
		return true
	})
	return violations
}
