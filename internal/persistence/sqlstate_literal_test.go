package persistence

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// classifierFiles are the only Go files that may spell an SQLSTATE: the named
// constants with Classify, and the table test that checks them (PER-R13).
var classifierFiles = map[string]bool{
	"internal/persistence/postgres/sqlstate.go":      true,
	"internal/persistence/postgres/sqlstate_test.go": true,
}

// postgresSQLStateClasses are the classes of PostgreSQL's SQLSTATE appendix.
var postgresSQLStateClasses = strings.Fields(`00 01 02 03 08 09 0A 0B 0F 0L 0P 0Z
20 21 22 23 24 25 26 27 28 2B 2D 2F 34 38 39 3B 3D 3F 40 42 44 53 54 55 57 58 72 F0 HV P0 XX`)

// nonSQLStateLiterals are values that only look like SQLSTATE codes, keyed by
// file and value, each with the reason it is not one.
var nonSQLStateLiterals = map[string]string{
	"internal/gitimport/analysis_test.go:40000":               "Git directory tree mode in the exact archive-selection fixture",
	"internal/gitimport/archive.go:40000":                     "Git tree mode of a directory entry",
	"internal/artifacts/query_indexes_postgres_test.go:25000": "payload bytes hashed by an index probe",
	"internal/persistence/sqlstate_literal_test.go:P0001":     "sample SQL that this scanner's own test must report",
	"internal/persistence/sqlstate_literal_test.go:23505":     "sample SQL that this scanner's own test must report",
}

var sqlStateShape = regexp.MustCompile(`^[0-9A-Z]{5}$`)

// sqlConditionCode finds an SQLSTATE that SQL text inside a Go string names by
// code: RAISE ... USING ERRCODE = '40001' or EXCEPTION WHEN SQLSTATE '23505'.
// SQL embedded in Go names the condition instead (serialization_failure).
var sqlConditionCode = regexp.MustCompile(`(?i)(?:ERRCODE\s*=\s*|SQLSTATE\s+)'([0-9A-Z]{5})'`)

func TestSQLStateLiteralsLiveOnlyInTheClassifier(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	var violations []string
	unused := map[string]bool{}
	for key := range nonSQLStateLiterals {
		unused[key] = true
	}
	files := 0
	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			name := entry.Name()
			if path != root && (strings.HasPrefix(name, ".") || name == "node_modules" || name == "testdata") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		relative = filepath.ToSlash(relative)
		if classifierFiles[relative] {
			return nil
		}
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			return err
		}
		files++
		for _, found := range sqlStateLiterals(fset, file, relative) {
			key := relative + ":" + found.value
			if _, allowed := nonSQLStateLiterals[key]; allowed {
				delete(unused, key)
				continue
			}
			violations = append(violations, found.position)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if files < 100 {
		t.Fatalf("scanned only %d Go files; the repository walk is broken", files)
	}
	for _, violation := range violations {
		t.Errorf("%s: name this SQLSTATE in internal/persistence/postgres/sqlstate.go and compare Classify results", violation)
	}
	for key := range unused {
		t.Errorf("%s: allowlisted non-SQLSTATE literal no longer exists; remove it from nonSQLStateLiterals", key)
	}
}

func TestSQLStateLiteralScanReportsCodesNotOtherWords(t *testing.T) {
	source := "package sample\n\nfunc f(code string) bool {\n\treturn code == \"55000\" ||\n\t\tcode == `40P01` ||\n" +
		"\t\tcode == \"PATCH\" || code == \"10000\" || code == \"2300\"\n}\n\n" +
		"const raise = `RAISE EXCEPTION 'x' USING ERRCODE = 'P0001'`\n" +
		"const handle = `EXCEPTION WHEN sqlstate '23505' THEN NULL`\n" +
		"const named = `RAISE EXCEPTION 'x' USING ERRCODE = 'serialization_failure'`\n"
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "sample.go", source, 0)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, found := range sqlStateLiterals(fset, file, "sample.go") {
		got = append(got, found.position)
	}
	want := []string{
		`sample.go:4: "55000"`, `sample.go:5: "40P01"`, `sample.go:9: "P0001"`, `sample.go:10: "23505"`,
	}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("SQLSTATE literals = %q, want %q", got, want)
	}
}

type sqlStateLiteral struct {
	value    string
	position string
}

// sqlStateLiterals reports every string literal whose whole value is a code of
// a PostgreSQL SQLSTATE class, and every such code that SQL text inside a
// string literal names in an ERRCODE or SQLSTATE clause.
func sqlStateLiterals(fset *token.FileSet, file *ast.File, path string) []sqlStateLiteral {
	var found []sqlStateLiteral
	ast.Inspect(file, func(node ast.Node) bool {
		literal, ok := node.(*ast.BasicLit)
		if !ok || literal.Kind != token.STRING {
			return true
		}
		value, err := strconv.Unquote(literal.Value)
		if err != nil {
			return true
		}
		codes := []string{value}
		if !sqlStateShape.MatchString(value) {
			codes = codes[:0]
			for _, match := range sqlConditionCode.FindAllStringSubmatch(value, -1) {
				codes = append(codes, strings.ToUpper(match[1]))
			}
		}
		position := fset.Position(literal.Pos())
		for _, code := range codes {
			if isSQLStateClass(code[:2]) {
				found = append(found, sqlStateLiteral{
					value: code, position: fmt.Sprintf("%s:%d: %q", path, position.Line, code),
				})
			}
		}
		return true
	})
	return found
}

func isSQLStateClass(class string) bool {
	for _, known := range postgresSQLStateClasses {
		if class == known {
			return true
		}
	}
	return false
}
