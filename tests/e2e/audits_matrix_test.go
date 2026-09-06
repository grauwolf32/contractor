package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type auditsMatrix struct {
	SchemaVersion string             `yaml:"schema_version"`
	Policy        auditsMatrixPolicy `yaml:"policy"`
	Cases         []auditsMatrixCase `yaml:"cases"`
	Gates         []matrixGate       `yaml:"gates"`
}

type auditsMatrixPolicy struct {
	Orchestration string `yaml:"orchestration"`
	Settlement    string `yaml:"settlement"`
	Retention     string `yaml:"retention"`
	Reporting     string `yaml:"reporting"`
	Isolation     string `yaml:"isolation"`
}

type auditsMatrixCase struct {
	ID         string         `yaml:"id"`
	Category   string         `yaml:"category"`
	Acceptance []string       `yaml:"acceptance"`
	Faults     []string       `yaml:"faults"`
	Expected   string         `yaml:"expected"`
	Test       auditTestOwner `yaml:"test"`
}

type auditTestOwner struct {
	Source string `yaml:"source"`
	Kind   string `yaml:"kind"`
	Name   string `yaml:"name"`
}

var (
	auditsAcceptance = []string{"A1", "A2", "A3", "A4", "A5"}
	auditsCategories = []string{
		"browser", "cancellation", "capability", "concurrency", "configuration",
		"findings", "input_security", "inventory", "isolation", "matrix", "package",
		"process", "recovery", "regression", "reporting", "retention", "retry",
		"review", "scheduler", "settlement",
	}
	auditsFaults = []string{
		"F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10",
		"F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20",
		"F21", "F22", "F23", "F24", "F25", "F26", "F27", "F28",
	}
)

func TestAuditsHardeningMatrixIsComplete(t *testing.T) {
	repositoryRoot := filepath.Join("..", "..")
	data, err := os.ReadFile("audits_matrix.yml")
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := decodeStrictMatrix[auditsMatrix](data, "Audits")
	if err != nil {
		t.Fatal(err)
	}
	if err := validateAuditsMatrix(repositoryRoot, matrix); err != nil {
		t.Fatal(err)
	}

	t.Run("missing policy", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditsMatrix](t, data, "Audits")
		broken.Policy.Settlement = ""
		requireAuditsMatrixError(t, repositoryRoot, broken, "policy")
	})
	t.Run("missing acceptance", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditsMatrix](t, data, "Audits")
		for index := range broken.Cases {
			broken.Cases[index].Acceptance = slices.DeleteFunc(
				broken.Cases[index].Acceptance, func(value string) bool { return value == "A5" },
			)
		}
		requireAuditsMatrixError(t, repositoryRoot, broken, "A5")
	})
	t.Run("missing category", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditsMatrix](t, data, "Audits")
		broken.Cases = slices.DeleteFunc(broken.Cases, func(value auditsMatrixCase) bool {
			return value.Category == "capability"
		})
		requireAuditsMatrixError(t, repositoryRoot, broken, `category "capability"`)
	})
	t.Run("missing fault", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditsMatrix](t, data, "Audits")
		for index := range broken.Cases {
			broken.Cases[index].Faults = slices.DeleteFunc(
				broken.Cases[index].Faults, func(value string) bool { return value == "F28" },
			)
		}
		requireAuditsMatrixError(t, repositoryRoot, broken, "F28")
	})
	t.Run("duplicate owner", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditsMatrix](t, data, "Audits")
		broken.Cases[1].Test = broken.Cases[0].Test
		requireAuditsMatrixError(t, repositoryRoot, broken, "duplicate test owner")
	})
	t.Run("renamed owner", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditsMatrix](t, data, "Audits")
		broken.Cases[0].Test.Name += "Renamed"
		requireAuditsMatrixError(t, repositoryRoot, broken, "does not define")
	})
	t.Run("missing release gate", func(t *testing.T) {
		broken := mustDecodeStrictMatrix[auditsMatrix](t, data, "Audits")
		broken.Gates = slices.DeleteFunc(broken.Gates, func(value matrixGate) bool {
			return value.ID == "release"
		})
		requireAuditsMatrixError(t, repositoryRoot, broken, `gate "release"`)
	})
}

func requireAuditsMatrixError(t *testing.T, repositoryRoot string, matrix auditsMatrix, want string) {
	t.Helper()
	err := validateAuditsMatrix(repositoryRoot, matrix)
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("matrix error = %v, want %q", err, want)
	}
}

func validateAuditsMatrix(repositoryRoot string, matrix auditsMatrix) error {
	if matrix.SchemaVersion != "1.0" || strings.TrimSpace(matrix.Policy.Orchestration) == "" ||
		strings.TrimSpace(matrix.Policy.Settlement) == "" || strings.TrimSpace(matrix.Policy.Retention) == "" ||
		strings.TrimSpace(matrix.Policy.Reporting) == "" || strings.TrimSpace(matrix.Policy.Isolation) == "" {
		return fmt.Errorf("incomplete Audits policy: %+v", matrix.Policy)
	}
	seenCases := map[string]bool{}
	seenAcceptance := map[string]bool{}
	seenCategories := map[string]bool{}
	seenFaults := map[string]bool{}
	seenOwners := map[string]string{}
	for _, item := range matrix.Cases {
		if item.ID == "" || item.Category == "" || strings.TrimSpace(item.Expected) == "" ||
			seenCases[item.ID] || len(item.Acceptance) == 0 {
			return fmt.Errorf("invalid or duplicate Audits case: %+v", item)
		}
		if !slices.Contains(auditsCategories, item.Category) {
			return fmt.Errorf("case %q has unknown category %q", item.ID, item.Category)
		}
		seenCases[item.ID] = true
		seenCategories[item.Category] = true
		for _, acceptance := range item.Acceptance {
			if !slices.Contains(auditsAcceptance, acceptance) {
				return fmt.Errorf("case %q has unknown acceptance %q", item.ID, acceptance)
			}
			seenAcceptance[acceptance] = true
		}
		for _, fault := range item.Faults {
			if !slices.Contains(auditsFaults, fault) || seenFaults[fault] {
				return fmt.Errorf("case %q has unknown or duplicate fault %q", item.ID, fault)
			}
			seenFaults[fault] = true
		}
		owner := item.Test.Source + "#" + item.Test.Kind + "#" + item.Test.Name
		if previous, exists := seenOwners[owner]; exists {
			return fmt.Errorf("duplicate test owner %q in cases %q and %q", owner, previous, item.ID)
		}
		seenOwners[owner] = item.ID
		if err := validateMatrixNamedOwner(repositoryRoot, item.Test.Source, item.Test.Kind, item.Test.Name); err != nil {
			return fmt.Errorf("case %q: %w", item.ID, err)
		}
	}
	for _, acceptance := range auditsAcceptance {
		if !seenAcceptance[acceptance] {
			return fmt.Errorf("required Audits acceptance %q is absent", acceptance)
		}
	}
	for _, category := range auditsCategories {
		if !seenCategories[category] {
			return fmt.Errorf("required Audits category %q is absent", category)
		}
	}
	for _, fault := range auditsFaults {
		if !seenFaults[fault] {
			return fmt.Errorf("required Audits fault %q is absent", fault)
		}
	}
	return validateMatrixGates(
		"Audits", matrix.Gates,
		[]string{"browser", "hardening", "matrix", "process", "release"}, true,
	)
}
