package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type taintAnnotationMatrix struct {
	SchemaVersion string                      `yaml:"schema_version"`
	Policy        taintAnnotationMatrixPolicy `yaml:"policy"`
	Cases         []taintAnnotationMatrixCase `yaml:"cases"`
	Gates         []matrixGate                `yaml:"gates"`
}

type taintAnnotationMatrixPolicy struct {
	Mutation  string `yaml:"mutation"`
	Lifecycle string `yaml:"lifecycle"`
	Retention string `yaml:"retention"`
}

type taintAnnotationMatrixCase struct {
	ID         string                         `yaml:"id"`
	Category   string                         `yaml:"category"`
	Acceptance []string                       `yaml:"acceptance"`
	Faults     []string                       `yaml:"faults"`
	Expected   string                         `yaml:"expected"`
	Test       taintAnnotationMatrixTestOwner `yaml:"test"`
}

type taintAnnotationMatrixTestOwner struct {
	Source string `yaml:"source"`
	Symbol string `yaml:"symbol"`
}

var (
	taintAnnotationAcceptance = []string{"A1", "A2", "A3", "A4"}
	taintAnnotationCategories = []string{
		"capacity", "concurrency", "lifecycle", "redaction", "resolution", "validation",
	}
	taintAnnotationFaults = []string{
		"F01", "F02", "F03", "F04", "F05", "F06", "F07",
		"F08", "F09", "F10", "F11", "F12", "F13", "F14",
	}
)

func TestTaintAnnotationHardeningMatrixIsComplete(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	data, err := os.ReadFile("taint_annotations_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := decodeTaintAnnotationMatrix(data)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateTaintAnnotationMatrix(repositoryRoot, matrix); err != nil {
		t.Fatal(err)
	}

	t.Run("missing acceptance", func(t *testing.T) {
		broken := mustDecodeTaintAnnotationMatrix(t, data)
		for index := range broken.Cases {
			broken.Cases[index].Acceptance = slices.DeleteFunc(
				broken.Cases[index].Acceptance,
				func(value string) bool { return value == "A1" },
			)
			if len(broken.Cases[index].Acceptance) == 0 {
				broken.Cases[index].Acceptance = []string{"A2"}
			}
		}
		if err := validateTaintAnnotationMatrix(repositoryRoot, broken); err == nil ||
			!strings.Contains(err.Error(), "A1") {
			t.Fatalf("missing acceptance validation error = %v", err)
		}
	})
	t.Run("missing fault", func(t *testing.T) {
		broken := mustDecodeTaintAnnotationMatrix(t, data)
		for index := range broken.Cases {
			broken.Cases[index].Faults = slices.DeleteFunc(
				broken.Cases[index].Faults,
				func(value string) bool { return value == "F14" },
			)
		}
		if err := validateTaintAnnotationMatrix(repositoryRoot, broken); err == nil ||
			!strings.Contains(err.Error(), "F14") {
			t.Fatalf("missing fault validation error = %v", err)
		}
	})
	t.Run("renamed test owner", func(t *testing.T) {
		broken := mustDecodeTaintAnnotationMatrix(t, data)
		broken.Cases[0].Test.Symbol += "Renamed"
		if err := validateTaintAnnotationMatrix(repositoryRoot, broken); err == nil ||
			!strings.Contains(err.Error(), "does not define") {
			t.Fatalf("renamed owner validation error = %v", err)
		}
	})
}

func decodeTaintAnnotationMatrix(data []byte) (taintAnnotationMatrix, error) {
	return decodeStrictMatrix[taintAnnotationMatrix](data, "taint-annotation")
}

func mustDecodeTaintAnnotationMatrix(t *testing.T, data []byte) taintAnnotationMatrix {
	t.Helper()
	return mustDecodeStrictMatrix[taintAnnotationMatrix](t, data, "taint-annotation")
}

func validateTaintAnnotationMatrix(repositoryRoot string, matrix taintAnnotationMatrix) error {
	if matrix.SchemaVersion != "1.0" || strings.TrimSpace(matrix.Policy.Mutation) == "" ||
		strings.TrimSpace(matrix.Policy.Lifecycle) == "" ||
		strings.TrimSpace(matrix.Policy.Retention) == "" {
		return fmt.Errorf("incomplete taint-annotation matrix policy: %+v", matrix.Policy)
	}
	seenCases := map[string]bool{}
	seenAcceptance := map[string]bool{}
	seenCategories := map[string]bool{}
	seenFaults := map[string]bool{}
	seenOwners := map[string]string{}
	for _, item := range matrix.Cases {
		if item.ID == "" || item.Category == "" || item.Expected == "" || seenCases[item.ID] ||
			len(item.Acceptance) == 0 {
			return fmt.Errorf("invalid or duplicate taint-annotation case: %+v", item)
		}
		if !slices.Contains(taintAnnotationCategories, item.Category) {
			return fmt.Errorf("case %q has unknown category %q", item.ID, item.Category)
		}
		seenCases[item.ID] = true
		seenCategories[item.Category] = true
		for _, acceptance := range item.Acceptance {
			if !slices.Contains(taintAnnotationAcceptance, acceptance) {
				return fmt.Errorf("case %q has unknown acceptance %q", item.ID, acceptance)
			}
			seenAcceptance[acceptance] = true
		}
		for _, fault := range item.Faults {
			if !slices.Contains(taintAnnotationFaults, fault) || seenFaults[fault] {
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
	for _, acceptance := range taintAnnotationAcceptance {
		if !seenAcceptance[acceptance] {
			return fmt.Errorf("required taint-annotation acceptance %q is absent", acceptance)
		}
	}
	for _, category := range taintAnnotationCategories {
		if !seenCategories[category] {
			return fmt.Errorf("required taint-annotation category %q is absent", category)
		}
	}
	for _, fault := range taintAnnotationFaults {
		if !seenFaults[fault] {
			return fmt.Errorf("required taint-annotation fault %q is absent", fault)
		}
	}

	return validateMatrixGates(
		"taint-annotation", matrix.Gates, []string{"e2e", "hardening", "matrix", "runtime"}, true,
	)
}
