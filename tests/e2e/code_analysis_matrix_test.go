package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type codeAnalysisMatrix struct {
	SchemaVersion string                   `yaml:"schema_version"`
	Policy        codeAnalysisMatrixPolicy `yaml:"policy"`
	Cases         []codeAnalysisMatrixCase `yaml:"cases"`
	Gates         []matrixGate             `yaml:"gates"`
}

type codeAnalysisMatrixPolicy struct {
	Snapshot  string `yaml:"snapshot"`
	Child     string `yaml:"child"`
	Retention string `yaml:"retention"`
}

type codeAnalysisMatrixCase struct {
	ID         string                      `yaml:"id"`
	Category   string                      `yaml:"category"`
	Acceptance []string                    `yaml:"acceptance"`
	Faults     []string                    `yaml:"faults"`
	Expected   string                      `yaml:"expected"`
	Test       codeAnalysisMatrixTestOwner `yaml:"test"`
}

type codeAnalysisMatrixTestOwner struct {
	Source string `yaml:"source"`
	Symbol string `yaml:"symbol"`
}

var (
	codeAnalysisAcceptance = []string{"A1", "A2", "A3", "A4", "A5"}
	codeAnalysisCategories = []string{
		"cancellation", "capacity", "concurrency", "lifecycle", "protocol", "redaction",
	}
	codeAnalysisFaults = []string{
		"F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10",
		"F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19",
	}
)

func TestCodeAnalysisHardeningMatrixIsComplete(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	data, err := os.ReadFile("code_analysis_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := decodeCodeAnalysisMatrix(data)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateCodeAnalysisMatrix(repositoryRoot, matrix); err != nil {
		t.Fatal(err)
	}

	t.Run("missing acceptance", func(t *testing.T) {
		broken := mustDecodeCodeAnalysisMatrix(t, data)
		for index := range broken.Cases {
			broken.Cases[index].Acceptance = slices.DeleteFunc(
				broken.Cases[index].Acceptance,
				func(value string) bool { return value == "A1" },
			)
			if len(broken.Cases[index].Acceptance) == 0 {
				broken.Cases[index].Acceptance = []string{"A2"}
			}
		}
		if err := validateCodeAnalysisMatrix(repositoryRoot, broken); err == nil ||
			!strings.Contains(err.Error(), "A1") {
			t.Fatalf("missing acceptance validation error = %v", err)
		}
	})
	t.Run("missing fault", func(t *testing.T) {
		broken := mustDecodeCodeAnalysisMatrix(t, data)
		for index := range broken.Cases {
			broken.Cases[index].Faults = slices.DeleteFunc(
				broken.Cases[index].Faults,
				func(value string) bool { return value == "F19" },
			)
		}
		if err := validateCodeAnalysisMatrix(repositoryRoot, broken); err == nil ||
			!strings.Contains(err.Error(), "F19") {
			t.Fatalf("missing fault validation error = %v", err)
		}
	})
	t.Run("renamed test owner", func(t *testing.T) {
		broken := mustDecodeCodeAnalysisMatrix(t, data)
		broken.Cases[0].Test.Symbol += "Renamed"
		if err := validateCodeAnalysisMatrix(repositoryRoot, broken); err == nil ||
			!strings.Contains(err.Error(), "does not define") {
			t.Fatalf("renamed owner validation error = %v", err)
		}
	})
}

func decodeCodeAnalysisMatrix(data []byte) (codeAnalysisMatrix, error) {
	return decodeStrictMatrix[codeAnalysisMatrix](data, "code-analysis")
}

func mustDecodeCodeAnalysisMatrix(t *testing.T, data []byte) codeAnalysisMatrix {
	t.Helper()
	return mustDecodeStrictMatrix[codeAnalysisMatrix](t, data, "code-analysis")
}

func validateCodeAnalysisMatrix(repositoryRoot string, matrix codeAnalysisMatrix) error {
	if matrix.SchemaVersion != "1.0" || strings.TrimSpace(matrix.Policy.Snapshot) == "" ||
		strings.TrimSpace(matrix.Policy.Child) == "" ||
		strings.TrimSpace(matrix.Policy.Retention) == "" {
		return fmt.Errorf("incomplete code-analysis matrix policy: %+v", matrix.Policy)
	}
	seenCases := map[string]bool{}
	seenAcceptance := map[string]bool{}
	seenCategories := map[string]bool{}
	seenFaults := map[string]bool{}
	seenOwners := map[string]string{}
	for _, item := range matrix.Cases {
		if item.ID == "" || item.Category == "" || item.Expected == "" || seenCases[item.ID] ||
			len(item.Acceptance) == 0 {
			return fmt.Errorf("invalid or duplicate code-analysis case: %+v", item)
		}
		if !slices.Contains(codeAnalysisCategories, item.Category) {
			return fmt.Errorf("case %q has unknown category %q", item.ID, item.Category)
		}
		seenCases[item.ID] = true
		seenCategories[item.Category] = true
		for _, acceptance := range item.Acceptance {
			if !slices.Contains(codeAnalysisAcceptance, acceptance) {
				return fmt.Errorf("case %q has unknown acceptance %q", item.ID, acceptance)
			}
			seenAcceptance[acceptance] = true
		}
		for _, fault := range item.Faults {
			if !slices.Contains(codeAnalysisFaults, fault) || seenFaults[fault] {
				return fmt.Errorf("case %q has unknown or duplicate fault %q", item.ID, fault)
			}
			seenFaults[fault] = true
		}
		owner := item.Test.Source + "#" + item.Test.Symbol
		if previous, exists := seenOwners[owner]; exists {
			return fmt.Errorf("duplicate test owner %q in cases %q and %q", owner, previous, item.ID)
		}
		seenOwners[owner] = item.ID
		if err := validateMatrixSymbolOwner(repositoryRoot, item.Test.Source, item.Test.Symbol); err != nil {
			return fmt.Errorf("case %q: %w", item.ID, err)
		}
	}
	for _, acceptance := range codeAnalysisAcceptance {
		if !seenAcceptance[acceptance] {
			return fmt.Errorf("required code-analysis acceptance %q is absent", acceptance)
		}
	}
	for _, category := range codeAnalysisCategories {
		if !seenCategories[category] {
			return fmt.Errorf("required code-analysis category %q is absent", category)
		}
	}
	for _, fault := range codeAnalysisFaults {
		if !seenFaults[fault] {
			return fmt.Errorf("required code-analysis fault %q is absent", fault)
		}
	}

	return validateMatrixGates(
		"code-analysis", matrix.Gates, []string{"e2e", "hardening", "matrix", "runtime"}, true,
	)
}
