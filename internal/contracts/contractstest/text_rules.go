package contractstest

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// WorkspacePathRule is one rule's bounds in
// api/testdata/v1alpha1/workspace-path-cases.json.
type WorkspacePathRule struct {
	EmptyAllowed  bool `json:"emptyAllowed"`
	MaxBytes      int  `json:"maxBytes"`
	MaxComponents int  `json:"maxComponents"`
}

// WorkspacePathCase is one case of the same fixture with its value resolved
// and the verdict of each rule.
type WorkspacePathCase struct {
	Name            string
	Value           string
	WorkspaceTarget bool
	WorkerStatePath bool
}

// WorkspacePathCases strictly reads the relative workspace path cases that
// the Python Runtime reads too (runtime/tests/test_shared_text_rules.py). An
// unknown key, a missing verdict or a case with both or neither of value and
// repeat fails the test instead of defaulting a verdict to false.
func WorkspacePathCases(t testing.TB) (WorkspacePathRule, WorkspacePathRule, []WorkspacePathCase) {
	t.Helper()
	var fixture struct {
		Description string `json:"description"`
		Rules       struct {
			WorkspaceTarget *WorkspacePathRule `json:"workspaceTarget"`
			WorkerStatePath *WorkspacePathRule `json:"workerStatePath"`
		} `json:"rules"`
		Cases []struct {
			Name   string  `json:"name"`
			Value  *string `json:"value"`
			Repeat *struct {
				Unit   string `json:"unit"`
				Count  int    `json:"count"`
				Suffix string `json:"suffix"`
			} `json:"repeat"`
			WorkspaceTarget *bool `json:"workspaceTarget"`
			WorkerStatePath *bool `json:"workerStatePath"`
		} `json:"cases"`
	}
	decoder := json.NewDecoder(bytes.NewReader(ReadFile(t, "api", "testdata", "v1alpha1", "workspace-path-cases.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatalf("decode workspace-path-cases.json: %v", err)
	}
	if fixture.Rules.WorkspaceTarget == nil || fixture.Rules.WorkerStatePath == nil || len(fixture.Cases) == 0 {
		t.Fatal("workspace-path-cases.json needs both rules and at least one case")
	}
	cases := make([]WorkspacePathCase, 0, len(fixture.Cases))
	for _, item := range fixture.Cases {
		if item.Name == "" || (item.Value == nil) == (item.Repeat == nil) ||
			item.WorkspaceTarget == nil || item.WorkerStatePath == nil {
			t.Fatalf("workspace path case %q needs a name, exactly one of value and repeat, and both verdicts", item.Name)
		}
		value := ""
		if item.Value != nil {
			value = *item.Value
		} else {
			value = strings.Repeat(item.Repeat.Unit, item.Repeat.Count) + item.Repeat.Suffix
		}
		cases = append(cases, WorkspacePathCase{
			Name: item.Name, Value: value,
			WorkspaceTarget: *item.WorkspaceTarget, WorkerStatePath: *item.WorkerStatePath,
		})
	}
	return *fixture.Rules.WorkspaceTarget, *fixture.Rules.WorkerStatePath, cases
}
