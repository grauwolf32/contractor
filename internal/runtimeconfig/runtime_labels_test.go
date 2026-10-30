package runtimeconfig

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedRuntimeLabelSetCases holds both Go forms of the Runtime label set
// rule to the shared cases: Run selection and Agent label replacement
// normalize a set, and the stored Agent labels and a Run snapshot's label
// pins accept only a set already in canonical order.
func TestSharedRuntimeLabelSetCases(t *testing.T) {
	t.Parallel()
	var fixture struct {
		Description   string `json:"description"`
		MaxLabels     int    `json:"maxLabels"`
		MaxLabelBytes int    `json:"maxLabelBytes"`
		Cases         []struct {
			Name      string   `json:"name"`
			Labels    []string `json:"labels"`
			Valid     bool     `json:"valid"`
			Canonical []string `json:"canonical"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "runtime-label-set-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("the shared Runtime label set cases are empty")
	}
	for _, test := range fixture.Cases {
		normalized, err := NormalizeLabels(test.Labels)
		if (err == nil) != test.Valid || test.Valid && !slices.Equal(normalized, test.Canonical) {
			t.Errorf("%s: NormalizeLabels = (%q, %v), want valid %t and %q", test.Name, normalized, err, test.Valid, test.Canonical)
		}
		if err != nil && !strings.Contains(err.Error(), ErrInvalid.Error()) {
			t.Errorf("%s: NormalizeLabels error %v is not ErrInvalid", test.Name, err)
		}
		stored := test.Valid && slices.Equal(test.Labels, test.Canonical)
		if err := ValidateAgentLabels(test.Labels); (err == nil) != stored {
			t.Errorf("%s: ValidateAgentLabels = %v, want accepted %t", test.Name, err, stored)
		}
		snapshot := builtInRunSnapshot()
		for _, label := range test.Labels {
			snapshot.Labels = append(snapshot.Labels, PinnedLabel{
				Label: label, Explicit: true, BindingRevision: 1,
				Config: Ref{Name: BuiltInName, Version: BuiltInVersion, Digest: BuiltInDigest},
			})
		}
		if err := snapshot.Validate(); (err == nil) != stored {
			t.Errorf("%s: Run snapshot label pins accepted = %v, want %t", test.Name, err, stored)
		}
	}
	if ValidateAgentLabels(nil) == nil {
		t.Error("ValidateAgentLabels accepted a null label set")
	}
}

// runtimeLabelRuleHome is the one definition of the Runtime label set rule.
const runtimeLabelRuleHome = "internal/contracts/runtimesettings/runtime_labels.go"

// TestRuntimeLabelSetRuleHasOneDefinition scans the Server's Go source for
// copies of the Runtime label set rule (RCF-R118 to RCF-R120, LIM-RCF-26)
// outside its definition: a loop that rejects a member equal to the default
// label, a bound of a label slice's length in such a function, or another
// constant for the bound. Every caller uses
// runtimesettings.CanonicalRuntimeLabels, directly or through
// NormalizeLabels and ValidateAgentLabels.
func TestRuntimeLabelSetRuleHasOneDefinition(t *testing.T) {
	for _, finding := range runtimeLabelRuleCopies(t, contractstest.Path(t, "internal")) {
		t.Errorf("%s", finding)
	}
	// The scan must find a copy where there is one: the public handler's
	// former copy and the second bound.
	probe := t.TempDir()
	if err := writeProbe(probe, "httpapi/public/labels.go", `package public

func normalize(labels []string) error {
	if len(labels) > 32 {
		return errInvalidRequest
	}
	for _, label := range labels {
		if label == runtimeconfig.DefaultLabel {
			return errInvalidRequest
		}
	}
	return nil
}

func pin(labels []string) (defaultPin string) {
	for _, label := range labels {
		if label == runtimeconfig.DefaultLabel {
			defaultPin = label
			continue
		}
	}
	return defaultPin
}
`); err != nil {
		t.Fatal(err)
	}
	if err := writeProbe(probe, "runtimeconfig/bounds.go", "package runtimeconfig\n\nconst MaximumRunLabels = 32\n"); err != nil {
		t.Fatal(err)
	}
	if got := runtimeLabelRuleCopies(t, probe); len(got) != 3 {
		t.Errorf("the scan reports %d copies in the probe, want the bound, the default exclusion and the constant: %q", len(got), got)
	}
}

func writeProbe(root, name, source string) error {
	path := filepath.Join(root, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(source), 0o644)
}

// runtimeLabelRuleCopies returns the copies of the label set rule in the
// non-test Go files below root.
func runtimeLabelRuleCopies(t *testing.T, root string) []string {
	t.Helper()
	labelSlice := regexp.MustCompile(`(?i)labels$`)
	boundName := regexp.MustCompile(`(?i)(run|agent|runtime)labels$`)
	found := map[string]bool{}
	var findings []string
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil || entry.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") ||
			strings.HasSuffix(filepath.ToSlash(path), runtimeLabelRuleHome) {
			return walkErr
		}
		set := token.NewFileSet()
		file, err := parser.ParseFile(set, path, nil, 0)
		if err != nil {
			return err
		}
		// The literal "default" is the default label only where Runtime
		// labels are in reach; elsewhere it names other defaults.
		literalDefault := strings.Contains(filepath.ToSlash(path), "/runtimeconfig/") ||
			strings.Contains(filepath.ToSlash(path), "/runtimesettings/")
		for _, spec := range file.Imports {
			if strings.HasSuffix(spec.Path.Value, `/runtimeconfig"`) || strings.HasSuffix(spec.Path.Value, `/runtimesettings"`) {
				literalDefault = true
			}
		}
		isDefault := func(expression ast.Expr) bool {
			if literal, ok := expression.(*ast.BasicLit); ok && literal.Kind == token.STRING {
				text, _ := strconv.Unquote(literal.Value)
				return literalDefault && text == DefaultLabel
			}
			return isName(expression, "DefaultLabel") || isName(expression, "DefaultRuntimeLabel")
		}
		report := func(node ast.Node, what string) {
			if where := set.Position(node.Pos()).String(); !found[where] {
				found[where] = true
				findings = append(findings, where+": "+what)
			}
		}
		ast.Inspect(file, func(node ast.Node) bool {
			if value, ok := node.(*ast.ValueSpec); ok {
				for index, name := range value.Names {
					if literal, ok := valueAt(value, index).(*ast.BasicLit); ok && literal.Kind == token.INT && boundName.MatchString(name.Name) {
						report(name, "declares another Runtime label bound "+name.Name)
					}
				}
			}
			var body *ast.BlockStmt
			switch function := node.(type) {
			case *ast.FuncDecl:
				body = function.Body
			case *ast.FuncLit:
				body = function.Body
			}
			if body == nil {
				return true
			}
			// A loop that rejects a label equal to the default label.
			excludes := false
			ast.Inspect(body, func(inner ast.Node) bool {
				loop, ok := inner.(*ast.RangeStmt)
				if !ok {
					return true
				}
				ast.Inspect(loop.Body, func(statement ast.Node) bool {
					check, ok := statement.(*ast.IfStmt)
					if !ok || len(check.Body.List) == 0 {
						return true
					}
					if _, rejects := check.Body.List[0].(*ast.ReturnStmt); !rejects {
						return true
					}
					ast.Inspect(check.Cond, func(condition ast.Node) bool {
						comparison, ok := condition.(*ast.BinaryExpr)
						if ok && (comparison.Op == token.EQL || comparison.Op == token.NEQ) &&
							(isDefault(comparison.X) || isDefault(comparison.Y)) {
							excludes = true
							report(comparison, "rejects the default label from a label set")
						}
						return true
					})
					return true
				})
				return true
			})
			if !excludes {
				return true
			}
			ast.Inspect(body, func(inner ast.Node) bool {
				comparison, ok := inner.(*ast.BinaryExpr)
				if !ok {
					return true
				}
				for _, side := range [][2]ast.Expr{{comparison.X, comparison.Y}, {comparison.Y, comparison.X}} {
					call, ok := side[0].(*ast.CallExpr)
					if !ok || !isName(call.Fun, "len") || len(call.Args) != 1 || !labelSlice.MatchString(lastName(call.Args[0])) {
						continue
					}
					if literal, ok := side[1].(*ast.BasicLit); ok && literal.Kind == token.INT || isName(side[1], "MaxRuntimeLabels") {
						report(comparison, "bounds the length of a label set")
					}
				}
				return true
			})
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return findings
}

func valueAt(value *ast.ValueSpec, index int) ast.Expr {
	if index < len(value.Values) {
		return value.Values[index]
	}
	return nil
}

func isName(expression ast.Expr, name string) bool {
	return lastName(expression) == name
}

// lastName returns the identifier an expression ends with.
func lastName(expression ast.Expr) string {
	switch value := expression.(type) {
	case *ast.Ident:
		return value.Name
	case *ast.SelectorExpr:
		return value.Sel.Name
	}
	return ""
}
